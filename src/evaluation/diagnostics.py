"""
src/evaluation/diagnostics.py - Phân loại nguyên nhân lỗi tự động (Error Taxonomy) & Báo cáo E2E.

Tự động gắn 6 nhãn lỗi cho từng mẫu:
- OK: Đạt chuẩn cả Retrieval lẫn Generation.
- RETRIEVAL_MISS: Không kéo được chunk vàng / ngữ cảnh thiếu trầm trọng.
- FALSE_REFUSAL: Ngữ cảnh đã có thông tin nhưng bot vẫn từ chối trả lời.
- HALLUCINATION: Bịa số liệu / khẳng định ngoài ngữ cảnh.
- INCOMPLETE_OR_WRONG: Ngữ cảnh đủ nhưng trả lời thiếu hoặc sai nội dung.
- OOD_FAILURE: Câu ngoài phạm vi nhưng bot không từ chối mà bịa câu trả lời.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.agent_eval import (
    evaluate_agent_decisions,
    get_agent_info,
    has_agent_data,
)

ERROR_LABELS: Tuple[str, ...] = (
    "OK",
    "ROUTER_FALSE_OOS",
    "RETRIEVAL_MISS",
    "FALSE_REFUSAL",
    "HALLUCINATION",
    "INCOMPLETE_OR_WRONG",
    "OOD_FAILURE",
    "GENERATION_ERROR",
)


def classify_sample_error(sample_record: Dict[str, Any]) -> Tuple[str, str]:
    """
    Gắn nhãn nguyên nhân lỗi cho 1 mẫu dựa trên Retrieval + Offline Metrics + LLM Judge (nếu có).
    Trả về `(error_label, reason)`.
    """
    is_ood = bool(sample_record.get("is_out_of_domain"))
    offline = sample_record.get("offline_metrics") or {}
    retrieval = sample_record.get("retrieval_metrics") or {}
    judge = sample_record.get("judge") or {}

    if offline.get("is_generation_error"):
        return (
            "GENERATION_ERROR",
            "Lỗi dịch vụ hoặc chạm Rate Limit LLM khi sinh câu trả lời (chạy lại với --resume để bù).",
        )

    judge_ok = judge.get("status") == "ok"
    raw_scores = judge.get("raw_scores") or {}
    judge_details = judge.get("details") or {}

    if is_ood:
        is_refusal = bool(offline.get("is_refusal", False))
        if judge_ok:
            refused_ok = bool(judge_details.get("refused_properly", False))
            hallucinated = bool(judge_details.get("hallucinated_info", False))
            if not refused_ok or hallucinated:
                return (
                    "OOD_FAILURE",
                    judge_details.get("reason") or "Bot không từ chối đúng chuẩn hoặc có thông tin bịa đặt cho câu OOD.",
                )
            return ("OK", judge_details.get("reason") or "Từ chối lịch sự đúng nguyên tắc cho câu OOD.")

        if not is_refusal:
            return ("OOD_FAILURE", "Bot trả lời câu hỏi ngoài phạm vi UET thay vì từ chối.")
        return ("OK", "Phát hiện mẫu câu từ chối lịch sự cho câu OOD.")

    # 0. ROUTER_FALSE_OOS: Câu hỏi trong phạm vi UET nhưng bị JEV Router chặn nhầm thành OUT_OF_SCOPE
    if has_agent_data(sample_record):
        agent_info = get_agent_info(sample_record)
        if agent_info.get("route") == "OUT_OF_SCOPE" or agent_info.get("status") == "out_of_scope":
            return (
                "ROUTER_FALSE_OOS",
                "Câu hỏi thuộc phạm vi UET nhưng JEV Router phân loại nhầm thành OUT_OF_SCOPE và chặn trả lời.",
            )

    chunk_hit = retrieval.get("chunk_hit_at_k")
    doc_hit = retrieval.get("doc_hit_at_k")
    is_refusal = bool(offline.get("is_refusal", False))
    num_faith = offline.get("numeric_faithfulness", 1.0)
    ungrounded_nums = offline.get("ungrounded_entities") or []
    off_faith = offline.get("faithfulness", 1.0)
    off_corr = offline.get("answer_correctness")
    ctx_rel = offline.get("context_relevance", 0.0)

    j_faith = raw_scores.get("faithfulness") if judge_ok else None
    j_corr = raw_scores.get("answer_correctness") if judge_ok else None
    j_suff = raw_scores.get("context_sufficiency") if judge_ok else None

    has_context_evidence = (
        chunk_hit == 1.0
        or doc_hit == 1.0
        or (j_suff is not None and j_suff >= 4)
        or (chunk_hit is None and ctx_rel >= 0.5)
    )

    # 1. FALSE_REFUSAL: Context có thông tin nhưng bot từ chối
    if is_refusal and has_context_evidence:
        return (
            "FALSE_REFUSAL",
            "Ngữ cảnh truy xuất có thông tin liên quan (chunk/doc hit hoặc context_sufficiency >= 4) nhưng bot vẫn từ chối.",
        )

    # 2. RETRIEVAL_MISS: Không trúng chunk vàng hoặc Judge đánh giá context thiếu, dẫn tới sai/thiếu/từ chối
    retrieval_missed = (chunk_hit == 0.0) or (j_suff is not None and j_suff <= 2)
    answer_not_good = (
        is_refusal
        or (j_corr is not None and j_corr <= 3)
        or (j_corr is None and off_corr is not None and off_corr < 0.70)
    )
    if retrieval_missed and answer_not_good:
        return (
            "RETRIEVAL_MISS",
            "Không kéo được chunk vàng lên Top-K hoặc ngữ cảnh thiếu dữ kiện khiến câu trả lời chưa đạt.",
        )

    # 3. HALLUCINATION: Judge phát hiện bịa hoặc Numeric Grounding Guard bắt số liệu không có trong context
    if (j_faith is not None and j_faith <= 3) or (num_faith < 1.0 and len(ungrounded_nums) > 0) or (
        j_faith is None and off_faith < 0.55
    ):
        extra = f" Số liệu/mã ngoài ngữ cảnh: {ungrounded_nums}" if ungrounded_nums else ""
        return (
            "HALLUCINATION",
            f"Phát hiện khẳng định hoặc số liệu không có căn cứ trong ngữ cảnh.{extra}",
        )

    # 4. INCOMPLETE_OR_WRONG: Ngữ cảnh đủ nhưng câu trả lời thiếu/sai ý
    if (j_corr is not None and j_corr <= 3) or (j_corr is None and off_corr is not None and off_corr < 0.65) or is_refusal:
        return (
            "INCOMPLETE_OR_WRONG",
            (judge_details.get("answer_correctness", {}) or {}).get("reason")
            or "Ngữ cảnh có dữ kiện nhưng câu trả lời thiếu ý hoặc sai lệch so với Ground Truth.",
        )

    return ("OK", "Đáp ứng tốt cả độ chính xác thực tế và tính trung thực với ngữ cảnh.")


def _safe_mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def build_e2e_report(
    scored_samples: List[Dict[str, Any]],
    top_k: int,
    judge_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tổng hợp báo cáo E2E đầy đủ: Retrieval Metrics, Offline Metrics, LLM Judge,
    OOD & False Refusal Rates, Latency, Error Taxonomy, Breakdowns & Top-10 Worst Samples.
    """
    in_domain = [s for s in scored_samples if not s.get("is_out_of_domain")]
    ood_samples = [s for s in scored_samples if s.get("is_out_of_domain")]
    gen_errors = [s for s in scored_samples if s.get("error_label") == "GENERATION_ERROR"]
    valid_in_domain = [s for s in in_domain if s.get("error_label") != "GENERATION_ERROR"]
    valid_ood_samples = [s for s in ood_samples if s.get("error_label") != "GENERATION_ERROR"]

    # 1. Trung bình Retrieval metrics trên in-domain
    retrieval_keys = [
        "doc_hit_at_k",
        "doc_recall_at_k",
        "chunk_hit_at_k",
        "chunk_recall_at_k",
        "chunk_ndcg_at_k",
        "chunk_mrr",
    ]
    retrieval_summary: Dict[str, float] = {}
    for key in retrieval_keys:
        vals = [
            float(s["retrieval_metrics"][key])
            for s in in_domain
            if (s.get("retrieval_metrics") or {}).get(key) is not None
        ]
        avg = _safe_mean(vals)
        if avg is not None:
            retrieval_summary[key] = avg

    # 2. Trung bình Offline metrics trên in-domain
    offline_keys = [
        "context_relevance",
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
        "token_precision",
        "token_recall",
        "token_f1",
        "keyword_recall",
        "rouge_l_recall",
        "rouge_l_f1",
        "numeric_faithfulness",
        "numeric_correctness",
        "semantic_answer_correctness",
        "context_hit",
        "overall_score",
    ]
    offline_summary: Dict[str, float] = {}
    for key in offline_keys:
        vals = [
            float(s["offline_metrics"][key])
            for s in valid_in_domain
            if (s.get("offline_metrics") or {}).get(key) is not None
        ]
        avg = _safe_mean(vals)
        if avg is not None:
            offline_summary[key] = avg

    # 3. Trung bình LLM Judge (CHỈ tính trên valid_judged_samples, KHÔNG bao giờ ép error -> 0.0)
    valid_in_judge = [s for s in in_domain if (s.get("judge") or {}).get("status") == "ok"]
    err_in_judge = [s for s in in_domain if (s.get("judge") or {}).get("status") == "error"]
    valid_ood_judge = [s for s in ood_samples if (s.get("judge") or {}).get("status") == "ok"]
    err_ood_judge = [s for s in ood_samples if (s.get("judge") or {}).get("status") == "error"]

    judge_norm_keys = [
        "faithfulness",
        "answer_correctness",
        "answer_relevance",
        "context_sufficiency",
        "overall_score",
    ]
    judge_summary: Dict[str, Any] = {
        "enabled": bool(judge_info and judge_info.get("enabled")),
        "judge_provider": (judge_info or {}).get("provider"),
        "judge_model": (judge_info or {}).get("model"),
        "valid_judged_samples": len(valid_in_judge),
        "judge_error_count": len(err_in_judge),
        "ood_valid_judged_samples": len(valid_ood_judge),
        "ood_judge_error_count": len(err_ood_judge),
        "average_normalized_scores": {},
        "average_raw_scores": {},
    }
    for key in judge_norm_keys:
        vals = [
            float(s["judge"]["normalized_scores"][key])
            for s in valid_in_judge
            if (s["judge"].get("normalized_scores") or {}).get(key) is not None
        ]
        avg = _safe_mean(vals)
        if avg is not None:
            judge_summary["average_normalized_scores"][key] = avg
    for key in ("faithfulness", "answer_correctness", "answer_relevance", "context_sufficiency"):
        vals = [
            float(s["judge"]["raw_scores"][key])
            for s in valid_in_judge
            if (s["judge"].get("raw_scores") or {}).get(key) is not None
        ]
        avg = _safe_mean(vals)
        if avg is not None:
            judge_summary["average_raw_scores"][key] = avg

    # 4. OOD Refusal Accuracy & In-Domain False Refusal Rate
    ood_ok_count = sum(1 for s in valid_ood_samples if s.get("error_label") == "OK")
    ood_refusal_acc = round(ood_ok_count / len(valid_ood_samples), 4) if valid_ood_samples else None

    false_refusals = [s for s in valid_in_domain if s.get("error_label") == "FALSE_REFUSAL"]
    false_refusal_rate = round(len(false_refusals) / len(valid_in_domain), 4) if valid_in_domain else 0.0

    # 5. Latency
    ret_lats = [float(s["retrieval_latency_s"]) for s in scored_samples if s.get("retrieval_latency_s") is not None]
    llm_lats = [float(s["llm_latency_s"]) for s in scored_samples if s.get("llm_latency_s") is not None]
    tot_lats = [float(s["total_latency_s"]) for s in scored_samples if s.get("total_latency_s") is not None]

    latency_summary = {
        "avg_retrieval_latency_s": _safe_mean(ret_lats),
        "avg_llm_latency_s": _safe_mean(llm_lats),
        "avg_total_latency_s": _safe_mean(tot_lats),
    }

    # 6. Error Taxonomy
    error_taxonomy: Dict[str, int] = {lbl: 0 for lbl in ERROR_LABELS}
    for s in scored_samples:
        lbl = s.get("error_label", "OK")
        error_taxonomy[lbl] = error_taxonomy.get(lbl, 0) + 1

    # 7. Breakdowns theo type, unit, category
    def _build_group_breakdown(group_field: str) -> Dict[str, Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for s in scored_samples:
            val = str(s.get(group_field) or "unknown")
            buckets.setdefault(val, []).append(s)

        out: Dict[str, Dict[str, Any]] = {}
        for g_name, items in buckets.items():
            in_items = [x for x in items if not x.get("is_out_of_domain")]
            valid_in_items = [x for x in in_items if x.get("error_label") != "GENERATION_ERROR"]
            g_judge = [x for x in valid_in_items if (x.get("judge") or {}).get("status") == "ok"]
            g_err_counts: Dict[str, int] = {}
            for x in items:
                el = x.get("error_label", "OK")
                g_err_counts[el] = g_err_counts.get(el, 0) + 1

            out[g_name] = {
                "count": len(items),
                "in_domain_count": len(in_items),
                "valid_in_domain_count": len(valid_in_items),
                "offline_overall": _safe_mean([
                    float(x["offline_metrics"]["overall_score"])
                    for x in valid_in_items
                    if (x.get("offline_metrics") or {}).get("overall_score") is not None
                ]),
                "answer_correctness": _safe_mean([
                    float(x["offline_metrics"]["answer_correctness"])
                    for x in valid_in_items
                    if (x.get("offline_metrics") or {}).get("answer_correctness") is not None
                ]),
                "faithfulness": _safe_mean([
                    float(x["offline_metrics"]["faithfulness"])
                    for x in valid_in_items
                    if (x.get("offline_metrics") or {}).get("faithfulness") is not None
                ]),
                "chunk_recall": _safe_mean([
                    float(x["retrieval_metrics"]["chunk_recall_at_k"])
                    for x in in_items
                    if (x.get("retrieval_metrics") or {}).get("chunk_recall_at_k") is not None
                ]),
                "judge_overall": _safe_mean([
                    float(x["judge"]["normalized_scores"]["overall_score"])
                    for x in g_judge
                    if (x["judge"].get("normalized_scores") or {}).get("overall_score") is not None
                ]),
                "error_counts": g_err_counts,
            }
        return out

    # 8. Top-10 Worst Samples (ưu tiên lỗi nặng + điểm thấp nhất)
    def _rank_worst_key(s: Dict[str, Any]) -> Tuple[int, float]:
        lbl = s.get("error_label", "OK")
        severity = 0 if lbl != "OK" else 1
        j = s.get("judge") or {}
        if j.get("status") == "ok" and (j.get("normalized_scores") or {}).get("overall_score") is not None:
            score = float(j["normalized_scores"]["overall_score"])
        elif s.get("is_out_of_domain"):
            score = 0.0 if lbl == "OOD_FAILURE" else 1.0
        else:
            score = float((s.get("offline_metrics") or {}).get("overall_score", 0.0))
        return (severity, score)

    worst_sorted = sorted(scored_samples, key=_rank_worst_key)[:10]
    worst_samples = []
    for s in worst_sorted:
        ws_item: Dict[str, Any] = {
            "sample_id": s.get("sample_id"),
            "type": s.get("type"),
            "unit": s.get("unit"),
            "error_label": s.get("error_label"),
            "error_reason": s.get("error_reason"),
            "offline_overall": (s.get("offline_metrics") or {}).get("overall_score"),
            "answer_correctness": (s.get("offline_metrics") or {}).get("answer_correctness"),
            "faithfulness": (s.get("offline_metrics") or {}).get("faithfulness"),
            "judge_scores": (s.get("judge") or {}).get("raw_scores") if (s.get("judge") or {}).get("status") == "ok" else None,
            "query": s.get("query"),
        }
        if s.get("agent_flags"):
            ws_item["agent_flags"] = list(s["agent_flags"])
        worst_samples.append(ws_item)

    agent_metrics = evaluate_agent_decisions(scored_samples)

    report_out: Dict[str, Any] = {
        "summary": {
            "total_samples": len(scored_samples),
            "in_domain_samples": len(in_domain),
            "valid_generated_samples": len(valid_in_domain),
            "generation_error_count": len(gen_errors),
            "ood_samples": len(ood_samples),
            "top_k": top_k,
            "ood_refusal_accuracy": ood_refusal_acc,
            "false_refusal_count": len(false_refusals),
            "false_refusal_rate": false_refusal_rate,
            "valid_judged_samples": len(valid_in_judge),
            "judge_error_count": len(err_in_judge),
        },
        "retrieval_metrics": retrieval_summary,
        "offline_metrics": offline_summary,
        "judge_metrics": judge_summary,
        "latency_metrics": latency_summary,
        "error_taxonomy": error_taxonomy,
        "by_type": _build_group_breakdown("type"),
        "by_unit": _build_group_breakdown("unit"),
        "by_category": _build_group_breakdown("category"),
        "worst_samples": worst_samples,
        "samples": scored_samples,
    }
    if agent_metrics is not None:
        report_out["agent_metrics"] = agent_metrics
    return report_out


def print_e2e_diagnostic_report(report: Dict[str, Any]) -> None:
    """In báo cáo tổng hợp E2E + Error Taxonomy gọn gàng trên Terminal."""
    s = report.get("summary", {})
    ret = report.get("retrieval_metrics", {})
    off = report.get("offline_metrics", {})
    judge = report.get("judge_metrics", {})
    lat = report.get("latency_metrics", {})
    tax = report.get("error_taxonomy", {})

    print("\n" + "=" * 78)
    print("📊 BÁO CÁO ĐÁNH GIÁ E2E RAG & PHÂN TÍCH NGUYÊN NHÂN LỖI (DIAGNOSTIC REPORT)")
    print("=" * 78)
    print(
        f"• Tổng mẫu: {s.get('total_samples', 0)} "
        f"(In-domain: {s.get('in_domain_samples', 0)} [Hợp lệ: {s.get('valid_generated_samples', s.get('in_domain_samples', 0))}]"
        f" | OOD: {s.get('ood_samples', 0)}) | Top-K={s.get('top_k')}"
    )
    if s.get("generation_error_count", 0) > 0:
        print(f"• Generation Errors    : {s['generation_error_count']} mẫu lỗi API/429 (không tính vào offline_metrics; chạy --resume để bù)")
    if s.get("ood_refusal_accuracy") is not None:
        print(f"• OOD Refusal Accuracy : {s['ood_refusal_accuracy']:.4f} ({s.get('ood_samples', 0)} câu OOD)")
    print(f"• False Refusal Rate   : {s.get('false_refusal_rate', 0.0):.4f} ({s.get('false_refusal_count', 0)} câu in-domain từ chối nhầm)")

    if ret:
        print("\n[1] Tầng Retrieval (từ snapshot Top-K):")
        for k, v in ret.items():
            print(f"    - {k:<24}: {v:.4f}")

    if off:
        print("\n[2] Offline Vietnamese RAG Metrics (0-LLM):")
        for k in [
            "overall_score",
            "answer_correctness",
            "faithfulness",
            "answer_relevance",
            "context_relevance",
            "token_recall",
            "keyword_recall",
            "token_f1",
            "rouge_l_f1",
            "numeric_faithfulness",
            "numeric_correctness",
            "semantic_answer_correctness",
        ]:
            if k in off:
                print(f"    - {k:<28}: {off[k]:.4f}")

    if judge.get("enabled"):
        print(
            f"\n[3] LLM-as-a-Judge ({judge.get('judge_provider')}:{judge.get('judge_model')}) "
            f"[Hợp lệ: {judge.get('valid_judged_samples', 0)} | Lỗi: {judge.get('judge_error_count', 0)}]:"
        )
        for k, v in (judge.get("average_normalized_scores") or {}).items():
            raw_v = (judge.get("average_raw_scores") or {}).get(k)
            raw_str = f" (raw={raw_v:.2f}/5)" if raw_v is not None else ""
            print(f"    - {k:<24}: {v:.4f}{raw_str}")

    if any(v is not None for v in lat.values()):
        print(
            f"\n[4] Độ trễ trung bình: Retrieval={lat.get('avg_retrieval_latency_s')}s | "
            f"LLM={lat.get('avg_llm_latency_s')}s | Total={lat.get('avg_total_latency_s')}s"
        )

    print("\n[5] Phân bố nguyên nhân lỗi (Error Taxonomy):")
    for lbl in ERROR_LABELS:
        print(f"    - {lbl:<22}: {tax.get(lbl, 0)}")

    by_type = report.get("by_type") or {}
    if by_type:
        print("\n[6] Breakdown theo loại câu hỏi (`type`):")
        for t_name, td in by_type.items():
            print(
                f"    - {t_name:<15} (n={td['count']:<2}) | "
                f"offline={td.get('offline_overall')} | corr={td.get('answer_correctness')} | "
                f"faith={td.get('faithfulness')} | chunkR={td.get('chunk_recall')}"
            )

    worst = report.get("worst_samples") or []
    failing = [w for w in worst if w.get("error_label") != "OK"] or worst[:5]
    if failing:
        print("\n[7] Top mẫu cần ưu tiên kiểm tra:")
        for w in failing[:10]:
            q_short = (w.get("query") or "")[:65]
            flags_str = f" flags={w['agent_flags']}" if w.get("agent_flags") else ""
            print(
                f"    • [{w.get('sample_id')}] ({w.get('error_label')}, {w.get('type')}/{w.get('unit')}) "
                f"overall={w.get('offline_overall')}{flags_str} | {q_short}..."
            )

    agent_m = report.get("agent_metrics")
    if isinstance(agent_m, dict):
        print("\n[8] 🤖 ĐÁNH GIÁ QUYẾT ĐỊNH AGENT (JEV / LLM):")
        rt = agent_m.get("router") or {}
        if rt:
            rdist = rt.get("route_distribution") or {}
            print(
                f"    • Router JEV (n={rt.get('total_routed', 0)}): "
                f"QUALIFIED={rdist.get('QUALIFIED_QUESTION', 0)}, "
                f"OUT_OF_SCOPE={rdist.get('OUT_OF_SCOPE', 0)}"
                f" | parse_fallback={rt.get('parse_fallback_count', 0)}"
            )
            if rt.get("false_oos_rate") is not None:
                print(
                    f"      - False OOS Rate (chặn nhầm in-domain): {rt['false_oos_rate']:.4f} "
                    f"({rt.get('false_oos_count', 0)}/{rt.get('in_domain_total', 0)})"
                )
            if rt.get("ood_caught_at_router_rate") is not None:
                print(
                    f"      - OOD bắt tại Router (tham khảo)      : {rt['ood_caught_at_router_rate']:.4f} "
                    f"({rt.get('ood_caught_at_router_count', 0)}/{rt.get('ood_total', 0)})"
                )
            if rt.get("explicit_labeled_total", 0) > 0 and rt.get("explicit_accuracy") is not None:
                print(
                    f"      - Explicit Route Accuracy             : {rt['explicit_accuracy']:.4f} "
                    f"(n={rt['explicit_labeled_total']})"
                )

        ae = agent_m.get("answer_eval") or {}
        if ae.get("evaluated_samples", 0) > 0:
            refs = ae.get("reference_source_counts") or {}
            print(
                f"    • Answer Eval JEV (n={ae['evaluated_samples']}, "
                f"ref=judge:{refs.get('judge', 0)}/offline:{refs.get('offline', 0)}): "
                f"Acc={ae.get('accuracy')} | P={ae.get('precision')} | R={ae.get('recall')} | "
                f"F1={ae.get('f1')} | ROC-AUC={ae.get('roc_auc')}"
            )
            print(
                f"      - False Accept (cho qua câu sai/thiếu): {ae.get('false_accept_rate')} "
                f"(n={ae.get('false_accept_count', 0)}: thiếu ý={ae.get('false_accept_incomplete_count', 0)}, "
                f"bịa đặt={ae.get('false_accept_hallucination_count', 0)})"
            )
            print(
                f"      - False Reject (đánh thiếu oan)       : {ae.get('false_reject_rate')} "
                f"(n={ae.get('false_reject_count', 0)})"
            )
            sweep = ae.get("threshold_sweep") or []
            if sweep:
                print("      - Quét ngưỡng noul_prob (không gọi lại pipeline):")
                print("        thr   |   P    |   R    |   F1   | FalseAcc | FalseRej | EstRetries")
                for row in sweep:
                    p_s = f"{row['precision']:.4f}" if row.get("precision") is not None else "  N/A "
                    r_s = f"{row['recall']:.4f}" if row.get("recall") is not None else "  N/A "
                    f_s = f"{row['f1']:.4f}" if row.get("f1") is not None else "  N/A "
                    fa_s = f"{row['false_accept_rate']:.4f}" if row.get("false_accept_rate") is not None else "  N/A "
                    fr_s = f"{row['false_reject_rate']:.4f}" if row.get("false_reject_rate") is not None else "  N/A "
                    er_s = f"{row['est_avg_retries']:.2f}" if row.get("est_avg_retries") is not None else " N/A"
                    print(
                        f"        {row['threshold']:<5.2f} | {p_s} | {r_s} | {f_s} |  {fa_s}  |  {fr_s}  |   {er_s}"
                    )

        tr = agent_m.get("trajectory") or {}
        if tr.get("total_agent_samples", 0) > 0:
            sdist = tr.get("status_distribution") or {}
            print(
                f"    • Trajectory (n={tr['total_agent_samples']}): "
                f"ok={sdist.get('ok', 0)}, partial={sdist.get('partial', 0)}, "
                f"out_of_scope={sdist.get('out_of_scope', 0)}, fail={sdist.get('fail', 0)}"
            )
            if tr.get("fail_reason_counts"):
                print(f"      - Fail reasons   : {tr['fail_reason_counts']}")
            print(
                f"      - Retries (RAG)  : avg={tr.get('avg_retries_rag')} | "
                f"triggered={tr.get('retry_triggered_rate')} | dist={tr.get('retry_distribution')}"
            )
            if tr.get("avg_jev_calls") is not None:
                print(
                    f"      - Avg calls/mẫu  : JEV={tr.get('avg_jev_calls')} | "
                    f"LLM={tr.get('avg_llm_calls')} | Retrieval={tr.get('avg_retrieval_calls')}"
                )
            if tr.get("node_latency_avg_s"):
                nlat = ", ".join(f"{k}={v}s" for k, v in tr["node_latency_avg_s"].items())
                print(f"      - Latency/node   : {nlat}")
            if tr.get("latency_by_route_s"):
                rlat = ", ".join(f"{k}={v}s" for k, v in tr["latency_by_route_s"].items())
                print(f"      - Latency/route  : {rlat}")

        lp = agent_m.get("loop")
        if isinstance(lp, dict) and lp.get("samples_with_retry", 0) > 0:
            src_str = f", điểm={lp['score_source']}" if lp.get("score_source") else ""
            print(f"    • Vòng lặp Follow-up (n={lp['samples_with_retry']} mẫu có retry{src_str}):")
            if lp.get("samples_with_gold", 0) > 0:
                print(
                    f"      - Gold recall (n={lp['samples_with_gold']}) : "
                    f"r0={lp.get('gold_recall_round0')} -> final={lp.get('gold_recall_final')} | "
                    f"Gold bị loại (max_chunks): {lp.get('gold_evicted_rate')}"
                )
            if lp.get("evaluated_samples", 0) > 0:
                delta_v = lp.get("avg_delta")
                delta_s = f"{delta_v:+.4f}" if delta_v is not None else "N/A"
                print(
                    f"      - Chất lượng vòng lặp: Win={lp.get('win_rate')} | "
                    f"Harm={lp.get('harm_rate')} | ΔTB={delta_s}"
                )
            curve = lp.get("correctness_by_round") or {}
            curve_parts = []
            if curve:
                curve_parts.append(", ".join(f"{k}={v}" for k, v in curve.items()))
            if lp.get("partial_samples", 0) > 0:
                curve_parts.append(
                    f"Partial chọn đúng best: {lp.get('partial_best_answer_accuracy')} (n={lp['partial_samples']})"
                )
            if curve_parts:
                print(f"      - Correctness/vòng   : {' | '.join(curve_parts)}")
    print("=" * 78 + "\n")

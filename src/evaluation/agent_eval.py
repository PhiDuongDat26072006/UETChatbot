"""
src/evaluation/agent_eval.py - Đánh giá các quyết định trung gian của Agentic RAG (JEV / LLM).

Cung cấp các hàm thuần (0-API, 0-mạng) chạy trên snapshot / báo cáo E2E để đo:
1. Router JEV (`compute_router_metrics`):
   - Phân bố 2 nhãn (OUT_OF_SCOPE, QUALIFIED_QUESTION).
   - Tỉ lệ chặn nhầm câu in-domain (`false_oos_rate`).
   - Tỉ lệ bắt OOD tại router (`ood_caught_at_router_rate`, không phạt nếu OOD đi qua RAG và vẫn từ chối đúng).
   - Ma trận nhầm lẫn 2x2 & Precision/Recall/F1 trên tập có nhãn `expected_route` tường minh (ví dụ golden_questions.json).
   - Số lần rơi vào parse fallback của JEV Router (`parse_fallback_count`).
2. Answer Eval JEV (`compute_answer_eval_metrics`):
   - So verdict của JEV với nhãn tham chiếu (LLM Judge `answer_correctness >= 4`, fallback Offline `>= 0.70`).
   - Ma trận nhầm lẫn, Accuracy, Precision, Recall, F1, False Accept Rate (tách lỗi thiếu ý vs giới hạn hallucination),
     False Reject Rate.
   - ROC-AUC của `noul_prob` và bảng quét ngưỡng (0.3 -> 0.8) không cần chạy lại pipeline.
3. Trajectory (`compute_trajectory_metrics`):
   - Phân bố trạng thái (`ok`, `partial`, `out_of_scope`, `fail`) và `fail_reason`.
   - Phân bố số vòng `retry`, số lần gọi JEV / LLM / Retrieval trung bình.
   - Độ trễ trung bình theo từng node và theo từng route.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.evaluation.agent_trace import (
    VALID_ROUTES,
    _LEGACY_ROUTE_MAP,
    _safe_mean,
    _safe_ratio,
    _to_json_safe,
    expected_route_of,
    get_agent_info,
    has_agent_data,
    normalize_route_label,
    sanitize_trace,
)
from src.evaluation.agent_rounds import (
    DEFAULT_JUDGE_WIN_THRESHOLD,
    DEFAULT_OFFLINE_WIN_THRESHOLD,
    _determine_chosen_round,
    _extract_target_chunk_set,
    _get_or_compute_rounds_eval,
    compute_loop_metrics,
    extract_rounds,
    score_sample_rounds,
)
from src.evaluation.metrics import (
    compute_answer_correctness_offline,
    strip_agent_notes,
)

DEFAULT_SWEEP_THRESHOLDS: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


# ============================================================================
# 1. ĐÁNH GIÁ ROUTER (JEV)
# ============================================================================

def compute_router_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tính chỉ số đánh giá JEV Router:
    - Phân bố route thực tế.
    - False OOS Rate trên câu in-domain (chỉ số quan trọng nhất: chặn nhầm câu hợp lệ).
    - OOD Caught at Router Rate (tỉ lệ câu OOD được bắt ngay ở router; mang tính tham khảo).
    - Ma trận nhầm lẫn 3x3, Accuracy & Per-class Precision/Recall/F1 trên tập có `expected_route` tường minh.
    - Số lần JEV trả nhãn lạ rơi vào parse fallback.
    """
    route_dist: Dict[str, int] = {r: 0 for r in VALID_ROUTES}
    route_dist["UNKNOWN"] = 0

    in_domain_total = 0
    false_oos_count = 0
    ood_total = 0
    ood_caught_at_router = 0
    parse_fallback_count = 0
    router_error_count = 0

    # Cho tập có nhãn tường minh (explicit expected_route)
    explicit_total = 0
    explicit_correct = 0
    confusion: Dict[str, Dict[str, int]] = {
        exp: {act: 0 for act in VALID_ROUTES} for exp in VALID_ROUTES
    }

    for rec in records:
        if not has_agent_data(rec):
            continue
        info = get_agent_info(rec)
        trace = info["trace"]

        for t in trace:
            if isinstance(t, dict) and t.get("node") == "router_node":
                if t.get("parse_fallback") is True:
                    parse_fallback_count += 1
                if t.get("error"):
                    router_error_count += 1

        actual = info["route"]
        if actual in VALID_ROUTES:
            route_dist[actual] += 1
        else:
            route_dist["UNKNOWN"] += 1
            continue

        exp_label, exp_source = expected_route_of(rec)
        is_ood = bool(rec.get("is_out_of_domain")) or (exp_source == "explicit" and exp_label == "OUT_OF_SCOPE")

        if is_ood:
            ood_total += 1
            if actual == "OUT_OF_SCOPE":
                ood_caught_at_router += 1
        else:
            in_domain_total += 1
            if actual == "OUT_OF_SCOPE":
                false_oos_count += 1

        if exp_source == "explicit" and exp_label in VALID_ROUTES:
            explicit_total += 1
            confusion[exp_label][actual] += 1
            if actual == exp_label:
                explicit_correct += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    if explicit_total > 0:
        for lbl in VALID_ROUTES:
            tp = confusion[lbl][lbl]
            support = sum(confusion[lbl].values())
            predicted = sum(confusion[exp][lbl] for exp in VALID_ROUTES)
            if support == 0 and predicted == 0:
                continue
            prec = _safe_ratio(tp, predicted)
            rec_v = _safe_ratio(tp, support)
            if prec is not None and rec_v is not None and (prec + rec_v) > 0:
                f1 = round(2 * prec * rec_v / (prec + rec_v), 4)
            else:
                f1 = 0.0 if (support > 0 or predicted > 0) else None
            per_class[lbl] = {
                "support": support,
                "predicted": predicted,
                "precision": prec,
                "recall": rec_v,
                "f1": f1,
            }

    routed_total = sum(route_dist[r] for r in VALID_ROUTES)
    result: Dict[str, Any] = {
        "total_routed": routed_total,
        "route_distribution": route_dist,
        "in_domain_total": in_domain_total,
        "false_oos_count": false_oos_count,
        "false_oos_rate": _safe_ratio(false_oos_count, in_domain_total),
        "ood_total": ood_total,
        "ood_caught_at_router_count": ood_caught_at_router,
        "ood_caught_at_router_rate": _safe_ratio(ood_caught_at_router, ood_total),
        "parse_fallback_count": parse_fallback_count,
        "router_error_count": router_error_count,
        "explicit_labeled_total": explicit_total,
        "explicit_accuracy": _safe_ratio(explicit_correct, explicit_total),
    }
    if explicit_total > 0:
        result["confusion_matrix"] = confusion
        result["per_class_metrics"] = per_class
    return result


# ============================================================================
# 2. ĐÁNH GIÁ ANSWER_EVAL (JEV) + ROC-AUC + QUÉT NGƯỠNG noul_prob
# ============================================================================

def answer_eval_reference(
    record: Dict[str, Any],
    judge_threshold: int = 4,
    offline_threshold: float = 0.70,
) -> Tuple[Optional[str], str]:
    """
    Suy ra nhãn tham chiếu ('SUFFICIENT' / 'INSUFFICIENT') cho câu trả lời cuối:
    - Ưu tiên LLM Judge: `raw_scores.answer_correctness >= judge_threshold` (mặc định 4/5).
    - Fallback Offline Metrics: `offline_metrics.answer_correctness >= offline_threshold` (mặc định 0.70).
    - Bỏ qua câu OOD, lỗi sinh, hoặc không đi qua nhánh RAG (`out_of_scope` / `fail`).
    """
    if bool(record.get("is_out_of_domain")):
        return None, "none"
    offline = record.get("offline_metrics") or {}
    if offline.get("is_generation_error"):
        return None, "none"

    info = get_agent_info(record)
    if info["status"] in ("out_of_scope", "fail"):
        return None, "none"

    judge = record.get("judge") or {}
    if judge.get("status") == "ok":
        raw_corr = (judge.get("raw_scores") or {}).get("answer_correctness")
        if raw_corr is not None:
            ref = "SUFFICIENT" if float(raw_corr) >= float(judge_threshold) else "INSUFFICIENT"
            return ref, "judge"

    off_corr = offline.get("answer_correctness")
    if off_corr is not None:
        ref = "SUFFICIENT" if float(off_corr) >= float(offline_threshold) else "INSUFFICIENT"
        return ref, "offline"

    return None, "none"


def _extract_answer_eval_summary(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Lấy phán quyết answer_eval tương ứng với câu trả lời cuối cùng được trả cho người dùng:
    - `status == 'ok'`: entry `answer_eval_node` cuối cùng (verdict = SUFFICIENT).
    - `status == 'partial'`: mọi entry đều là INSUFFICIENT; câu trả lời cuối là `best_answer`
      (vòng có `noul_prob` cao nhất, nếu hòa lấy sau cùng).
    """
    info = get_agent_info(record)
    if info["status"] in ("out_of_scope", "fail"):
        return None

    eval_entries = [
        t for t in info["trace"]
        if isinstance(t, dict) and t.get("node") == "answer_eval_node" and not t.get("error")
    ]
    if not eval_entries:
        return None

    all_probs: List[float] = []
    for e in eval_entries:
        p = e.get("noul_prob")
        if p is not None:
            try:
                all_probs.append(float(p))
            except (TypeError, ValueError):
                pass

    if info["status"] == "partial":
        chosen = eval_entries[-1]
        best_p: Optional[float] = None
        for e in eval_entries:
            p = e.get("noul_prob")
            if p is not None:
                try:
                    pf = float(p)
                    if best_p is None or pf >= best_p:
                        best_p = pf
                        chosen = e
                except (TypeError, ValueError):
                    pass
    else:
        chosen = eval_entries[-1]

    verdict_raw = str(chosen.get("result") or "").strip().upper()
    if verdict_raw not in ("SUFFICIENT", "INSUFFICIENT"):
        return None

    final_prob: Optional[float] = None
    if chosen.get("noul_prob") is not None:
        try:
            final_prob = float(chosen["noul_prob"])
        except (TypeError, ValueError):
            final_prob = None

    return {
        "verdict": verdict_raw,
        "noul_prob": final_prob,
        "all_noul_probs": all_probs,
        "eval_calls": len(eval_entries),
    }


def _is_hallucination_dominant(record: Dict[str, Any]) -> bool:
    """Kiểm tra lỗi của mẫu có xuất phát từ bịa đặt (hallucination — ngoài tầm nhìn của answer_eval) không."""
    if record.get("error_label") == "HALLUCINATION":
        return True
    judge = record.get("judge") or {}
    if judge.get("status") == "ok":
        raw = judge.get("raw_scores") or {}
        if raw.get("faithfulness") is not None and float(raw["faithfulness"]) <= 3:
            return True
    offline = record.get("offline_metrics") or {}
    if float(offline.get("numeric_faithfulness", 1.0)) < 1.0 and bool(offline.get("ungrounded_entities")):
        return True
    return False


def compute_roc_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> Optional[float]:
    """
    Tính ROC-AUC (Mann-Whitney U trên thứ hạng trung bình khi đồng hạng, không cần scikit-learn).
    Trả về None nếu dữ liệu chỉ có 1 lớp hoặc rỗng.
    """
    if len(y_true) != len(y_scores) or len(y_true) < 2:
        return None
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    paired = sorted(zip(y_scores, y_true), key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    pos_rank_sum = sum(ranks[idx] for idx, (_, y) in enumerate(paired) if y == 1)
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    return round(u_stat / (n_pos * n_neg), 4)


def compute_threshold_sweep(
    eval_items: List[Dict[str, Any]],
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
) -> List[Dict[str, Any]]:
    """
    Quét nhiều ngưỡng `noul_prob` trực tiếp trên snapshot đã lưu mà không cần gọi lại JEV/LLM.
    Trả về bảng: threshold, precision, recall, f1, false_accept_rate, false_reject_rate, est_avg_retries.
    """
    prob_items = [it for it in eval_items if it.get("noul_prob") is not None]
    if not prob_items:
        return []

    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        tp = fp = tn = fn = 0
        est_retries: List[int] = []

        for it in prob_items:
            ref_pos = bool(it["ref_is_sufficient"])
            pred_pos = float(it["noul_prob"]) >= float(thr)
            if ref_pos and pred_pos:
                tp += 1
            elif not ref_pos and pred_pos:
                fp += 1
            elif not ref_pos and not pred_pos:
                tn += 1
            else:
                fn += 1

            # Ước lượng cận dưới số vòng retry dựa trên chuỗi noul_prob đã quan sát
            all_p = it.get("all_noul_probs") or []
            if all_p:
                stopped_at = len(all_p)
                for idx_r, p_val in enumerate(all_p):
                    if p_val >= float(thr):
                        stopped_at = idx_r
                        break
                est_retries.append(stopped_at)

        prec = _safe_ratio(tp, tp + fp)
        rec = _safe_ratio(tp, tp + fn)
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = round(2 * prec * rec / (prec + rec), 4)
        else:
            f1 = 0.0

        rows.append({
            "threshold": round(float(thr), 2),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_accept_count": fp,
            "false_accept_rate": _safe_ratio(fp, fp + tn),
            "false_reject_count": fn,
            "false_reject_rate": _safe_ratio(fn, tp + fn),
            "est_avg_retries": _safe_mean(est_retries),
        })
    return rows


def compute_answer_eval_metrics(
    records: List[Dict[str, Any]],
    judge_threshold: int = 4,
    offline_threshold: float = 0.70,
    sweep_thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Đánh giá chất lượng bộ phân loại `answer_eval_node` của JEV:
    - Confusion matrix (reference vs JEV verdict).
    - False Accept Rate (tách `incomplete` vs `hallucination` vì JEV không nhìn thấy chunk).
    - False Reject Rate.
    - ROC-AUC và bảng quét ngưỡng `noul_prob`.
    """
    tp = fp = tn = fn = 0
    fp_hallucination = 0
    fp_incomplete = 0
    ref_sources: Dict[str, int] = {"judge": 0, "offline": 0}
    eval_items: List[Dict[str, Any]] = []

    for rec in records:
        if not has_agent_data(rec):
            continue
        ae = _extract_answer_eval_summary(rec)
        if ae is None:
            continue
        ref_label, ref_source = answer_eval_reference(
            rec,
            judge_threshold=judge_threshold,
            offline_threshold=offline_threshold,
        )
        if ref_label is None:
            continue

        ref_sources[ref_source] = ref_sources.get(ref_source, 0) + 1
        ref_pos = (ref_label == "SUFFICIENT")
        pred_pos = (ae["verdict"] == "SUFFICIENT")

        if ref_pos and pred_pos:
            tp += 1
        elif not ref_pos and pred_pos:
            fp += 1
            if _is_hallucination_dominant(rec):
                fp_hallucination += 1
            else:
                fp_incomplete += 1
        elif not ref_pos and not pred_pos:
            tn += 1
        else:
            fn += 1

        eval_items.append({
            "sample_id": rec.get("sample_id"),
            "ref_is_sufficient": ref_pos,
            "pred_is_sufficient": pred_pos,
            "noul_prob": ae["noul_prob"],
            "all_noul_probs": ae["all_noul_probs"],
        })

    total = tp + fp + tn + fn
    prec = _safe_ratio(tp, tp + fp)
    rec_v = _safe_ratio(tp, tp + fn)
    if prec is not None and rec_v is not None and (prec + rec_v) > 0:
        f1 = round(2 * prec * rec_v / (prec + rec_v), 4)
    else:
        f1 = 0.0 if total > 0 else None

    prob_items = [it for it in eval_items if it["noul_prob"] is not None]
    y_true = [1 if it["ref_is_sufficient"] else 0 for it in prob_items]
    y_scores = [float(it["noul_prob"]) for it in prob_items]
    auc = compute_roc_auc(y_true, y_scores)
    sweep = compute_threshold_sweep(eval_items, thresholds=sweep_thresholds)

    return {
        "evaluated_samples": total,
        "reference_source_counts": ref_sources,
        "confusion_matrix": {
            "SUFFICIENT": {"SUFFICIENT": tp, "INSUFFICIENT": fn},
            "INSUFFICIENT": {"SUFFICIENT": fp, "INSUFFICIENT": tn},
        },
        "accuracy": _safe_ratio(tp + tn, total),
        "precision": prec,
        "recall": rec_v,
        "f1": f1,
        "false_accept_count": fp,
        "false_accept_rate": _safe_ratio(fp, fp + tn),
        "false_accept_incomplete_count": fp_incomplete,
        "false_accept_hallucination_count": fp_hallucination,
        "false_reject_count": fn,
        "false_reject_rate": _safe_ratio(fn, tp + fn),
        "noul_prob_samples": len(prob_items),
        "roc_auc": auc,
        "threshold_sweep": sweep,
    }


# ============================================================================
# 3. THỐNG KÊ LUỒNG THỰC THI (TRAJECTORY)
# ============================================================================

def compute_trajectory_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Thống kê hành vi thực thi theo luồng của Agent:
    - Tỉ lệ trạng thái (ok / partial / out_of_scope / fail) và phân tích fail_reason.
    - Phân bố số vòng retry, tỉ lệ kích hoạt retry.
    - Số lần gọi JEV / LLM / Retrieval trung bình mỗi mẫu.
    - Độ trễ trung bình theo từng node và theo từng route.
    """
    agent_records = [r for r in records if has_agent_data(r)]
    total = len(agent_records)

    status_dist: Dict[str, int] = {"ok": 0, "partial": 0, "out_of_scope": 0, "fail": 0}
    fail_reasons: Dict[str, int] = {}
    retry_counts: Dict[int, int] = {}

    rag_retries: List[int] = []
    all_retries: List[int] = []
    jev_calls_list: List[int] = []
    llm_calls_list: List[int] = []
    ret_calls_list: List[int] = []

    node_latencies: Dict[str, List[float]] = {}
    route_latencies: Dict[str, List[float]] = {}

    for rec in agent_records:
        info = get_agent_info(rec)
        status = info["status"] or "unknown"
        status_dist[status] = status_dist.get(status, 0) + 1

        if status == "fail":
            fr = info["fail_reason"] or "unknown"
            fail_reasons[fr] = fail_reasons.get(fr, 0) + 1

        retry_v = info["retry"]
        retry_counts[retry_v] = retry_counts.get(retry_v, 0) + 1
        all_retries.append(retry_v)
        if status in ("ok", "partial"):
            rag_retries.append(retry_v)

        trace = info["trace"]
        if trace:
            jev_calls = sum(1 for t in trace if isinstance(t, dict) and t.get("source") == "jev")
            llm_calls = sum(
                1 for t in trace
                if isinstance(t, dict)
                and (
                    t.get("source") == "llm"
                    or (t.get("node") == "generation_node" and not t.get("error"))
                )
            )
            ret_calls = sum(1 for t in trace if isinstance(t, dict) and t.get("node") == "retrieval_node")
            jev_calls_list.append(jev_calls)
            llm_calls_list.append(llm_calls)
            ret_calls_list.append(ret_calls)

            for t in trace:
                if isinstance(t, dict) and t.get("node") and t.get("latency_s") is not None:
                    try:
                        node_latencies.setdefault(str(t["node"]), []).append(float(t["latency_s"]))
                    except (TypeError, ValueError):
                        pass

        route = info["route"]
        tot_lat = rec.get("total_latency_s")
        if route and tot_lat is not None:
            try:
                route_latencies.setdefault(route, []).append(float(tot_lat))
            except (TypeError, ValueError):
                pass

    status_rates = {k: _safe_ratio(v, total) for k, v in status_dist.items() if total > 0}
    retry_dist = {str(k): retry_counts[k] for k in sorted(retry_counts.keys())}
    rag_retry_triggered = sum(1 for r in rag_retries if r >= 1)

    return {
        "total_agent_samples": total,
        "status_distribution": status_dist,
        "status_rates": status_rates,
        "fail_reason_counts": fail_reasons,
        "retry_distribution": retry_dist,
        "avg_retries_rag": _safe_mean(rag_retries),
        "avg_retries_all": _safe_mean(all_retries),
        "retry_triggered_count": rag_retry_triggered,
        "retry_triggered_rate": _safe_ratio(rag_retry_triggered, len(rag_retries)),
        "avg_jev_calls": _safe_mean(jev_calls_list),
        "avg_llm_calls": _safe_mean(llm_calls_list),
        "avg_retrieval_calls": _safe_mean(ret_calls_list),
        "node_latency_avg_s": {
            node: _safe_mean(vals) for node, vals in node_latencies.items() if vals
        },
        "latency_by_route_s": {
            rt: _safe_mean(vals) for rt, vals in route_latencies.items() if vals
        },
    }


# ============================================================================
# 5. CỜ CHẨN ĐOÁN THEO MẪU & TỔNG HỢP
# ============================================================================

def compute_sample_agent_flags(
    record: Dict[str, Any],
    judge_threshold: int = 4,
    offline_threshold: float = 0.70,
) -> List[str]:
    """Gắn danh sách cờ quyết định agent cho từng mẫu (bổ sung cho `error_label`, không ghi đè)."""
    if not has_agent_data(record):
        return []

    flags: List[str] = []
    info = get_agent_info(record)
    is_ood = bool(record.get("is_out_of_domain"))

    if not is_ood and (info["route"] == "OUT_OF_SCOPE" or info["status"] == "out_of_scope"):
        flags.append("ROUTER_FALSE_OOS")

    for t in info["trace"]:
        if isinstance(t, dict) and t.get("node") == "router_node" and t.get("parse_fallback") is True:
            flags.append("ROUTER_PARSE_FALLBACK")
            break

    ae = _extract_answer_eval_summary(record)
    if ae is not None:
        ref_label, _ = answer_eval_reference(
            record,
            judge_threshold=judge_threshold,
            offline_threshold=offline_threshold,
        )
        if ref_label is not None:
            if ae["verdict"] == "SUFFICIENT" and ref_label == "INSUFFICIENT":
                flags.append("EVAL_FALSE_ACCEPT")
            elif ae["verdict"] == "INSUFFICIENT" and ref_label == "SUFFICIENT":
                flags.append("EVAL_FALSE_REJECT")

    if info["status"] == "partial":
        flags.append("PARTIAL_EXHAUSTED")
    elif info["status"] == "fail":
        flags.append("AGENT_SERVICE_FAIL")

    rounds_eval = record.get("agent_rounds_eval")
    if isinstance(rounds_eval, dict):
        if rounds_eval.get("outcome") == "harmed":
            flags.append("LOOP_HARMED")
        if rounds_eval.get("gold_evicted") is True:
            flags.append("GOLD_EVICTED")

    return flags


def evaluate_agent_decisions(
    records: List[Dict[str, Any]],
    judge_threshold: int = 4,
    offline_threshold: float = 0.70,
    sweep_thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
) -> Optional[Dict[str, Any]]:
    """
    Tổng hợp toàn bộ đánh giá quyết định Agent (Router + Answer Eval + Trajectory + Loop).
    Trả về `None` nếu snapshot không chứa dữ liệu agent (đảm bảo tương thích ngược với chế độ modular).
    """
    if not any(has_agent_data(r) for r in records):
        return None

    result: Dict[str, Any] = {
        "router": compute_router_metrics(records),
        "answer_eval": compute_answer_eval_metrics(
            records,
            judge_threshold=judge_threshold,
            offline_threshold=offline_threshold,
            sweep_thresholds=sweep_thresholds,
        ),
        "trajectory": compute_trajectory_metrics(records),
    }
    loop_metrics = compute_loop_metrics(records)
    if loop_metrics is not None:
        result["loop"] = loop_metrics
    return result

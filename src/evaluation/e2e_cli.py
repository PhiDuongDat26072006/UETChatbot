"""
src/evaluation/e2e_cli.py - CLI runners (`run_eval_e2e_cli`, `run_router_eval_cli`)
for End-to-End RAG Evaluation and standalone JEV Router Evaluation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.evaluation.agent_eval import (
    compute_router_metrics,
    normalize_route_label,
    sanitize_trace,
)
from src.evaluation.diagnostics import print_e2e_diagnostic_report
from src.evaluation.ragas_eval import UETRagasEvaluator, _atomic_write_json
from src.evaluation.snapshot import (
    _is_sample_ood,
    generate_predictions,
    score_predictions,
)
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("evaluation")
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def run_eval_e2e_cli(
    dataset_path: Optional[Union[str, Path]] = None,
    domain: Optional[str] = None,
    use_ragas: bool = False,
    use_judge: bool = False,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_rounds: bool = False,
    num_samples: Optional[int] = None,
    top_k: Optional[int] = None,
    model_name: str = "deepseek/deepseek-v4.1-flash-1017",
    output_path: Optional[Union[str, Path]] = None,
    delay: float = 1.2,
    generate_only: bool = False,
    score_from: Optional[Union[str, Path]] = None,
    save_predictions: Optional[Union[str, Path]] = None,
    enable_semantic: bool = True,
    use_batch_llm: bool = False,
    resume: bool = True,
):
    """Chạy đánh giá toàn trình End-to-End RAG trên CLI (hỗ trợ 2 pha, Offline, UET Judge & Ragas)."""
    from src.evaluation.evaluator import DOMAIN_FILE_MAP, UETEvaluator

    base_dir = BASE_DIR
    cfg = load_yaml_config()
    if top_k is None:
        top_k = cfg.get("retrieval", {}).get("final_top_k", 10)

    # Pha 2 độc lập: Chấm điểm từ file snapshot predictions.json có sẵn
    if score_from is not None:
        score_path = Path(score_from)
        if not score_path.exists():
            print(f"❌ Không tìm thấy file snapshot predictions tại: {score_path}")
            return None
        out_p = Path(output_path) if output_path else (base_dir / "data" / "eval_data" / "e2e_eval_results.json")
        print("\n" + "=" * 70)
        print("🧮 [PHA 2] CHẤM ĐIỂM TỪ SNAPSHOT PREDICTIONS CÓ SẴN")
        print(f"• Snapshot đầu vào : {score_path}")
        print(f"• Chế độ Judge     : {'BẬT' if use_judge else 'TẮT (Offline Metrics)'}")
        print(f"• Semantic (bge-m3): {'BẬT' if enable_semantic else 'TẮT'}")
        print("=" * 70)
        report_dict = score_predictions(
            predictions_source=score_path,
            top_k=top_k,
            use_judge=use_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
            enable_semantic=enable_semantic,
            output_path=out_p,
            judge_rounds=judge_rounds,
        )
        print_e2e_diagnostic_report(report_dict)
        print(f"📁 Đã lưu báo cáo E2E chi tiết vào: {out_p}\n")
        return report_dict

    if not dataset_path:
        cfg_file = cfg.get("paths", {}).get("eval_dataset_file")
        if cfg_file:
            target_path = base_dir / cfg_file if not Path(cfg_file).is_absolute() else Path(cfg_file)
        else:
            default_annotated = base_dir / "data" / "eval_data" / "e2e_eval_100_annotated.json"
            if default_annotated.exists():
                target_path = default_annotated
            else:
                target_path = base_dir / "data" / "eval_data" / "uet_rag_eval_100" / "e2e_eval_100.json"
    else:
        target_path = Path(dataset_path)

    dataset = UETEvaluator.load_dataset(target_path, domain=domain)
    domain_str = f" [Domain/Unit: {domain.upper()}]" if domain else ""
    target_name = target_path.name if target_path.is_file() else f"uet_domain_qa{domain_str}"

    if not dataset:
        print(f"❌ Không tìm thấy tập dữ liệu đánh giá hợp lệ tại: {target_path}")
        if domain:
            print(f"   Domain hỗ trợ: {', '.join(DOMAIN_FILE_MAP.keys())}, all")
        return None

    from src.pipeline import describe_pipeline_mode, get_active_pipeline

    rag_pipeline = get_active_pipeline()
    active_mode_label = describe_pipeline_mode(rag_pipeline)
    actual_samples = num_samples if (num_samples and num_samples > 0) else len(dataset)

    if generate_only:
        pred_out = Path(
            save_predictions
            or output_path
            or (base_dir / "data" / "eval_data" / "e2e_predictions.json")
        )
        print("\n" + "=" * 70)
        print("📸 [PHA 1] CHẠY PIPELINE & LƯU SNAPSHOT PREDICTIONS")
        print(f"• Chế độ RAG  : {active_mode_label}")
        print(f"• Tập dữ liệu : {target_name} ({actual_samples}/{len(dataset)} câu)")
        print(f"• Top-K       : {top_k}")
        print(f"• Snapshot    : {pred_out}")
        print("=" * 70)
        snapshot = generate_predictions(
            pipeline=rag_pipeline,
            dataset=dataset,
            top_k=top_k,
            num_samples=num_samples,
            delay_between_queries=delay,
            output_path=pred_out,
            use_batch_llm=use_batch_llm,
            resume=resume,
        )
        print(f"\n✅ Đã tạo snapshot {snapshot['total_samples']} mẫu tại: {pred_out}\n")
        return snapshot

    if use_ragas:
        print("\n" + "=" * 70)
        print("🤖 ĐANG CHẠY ĐÁNH GIÁ RAG BẰNG RAGAS & LLM-AS-A-JUDGE")
        print(f"• Tập dữ liệu: {target_name} (Tổng: {len(dataset)} câu)")
        print(f"• Giám khảo LLM: {model_name}")
        print(f"• Số lượng câu hỏi đánh giá: {actual_samples}")
        print(f"• Top-K ngữ cảnh: {top_k}")
        print("=" * 70)

        out_p = Path(output_path) if output_path else (base_dir / "data" / "eval_data" / "ragas_eval_results.json")
        ragas_eval = UETRagasEvaluator(
            model_name=model_name,
            embedder=getattr(rag_pipeline, "embedding_model", None),
        )
        report = ragas_eval.evaluate_pipeline(
            pipeline=rag_pipeline,
            dataset=dataset,
            top_k=top_k,
            num_samples=num_samples,
            output_path=out_p,
            delay_between_queries=delay,
        )
        print("\n📈 KẾT QUẢ RAGAS:")
        for metric, score in report["average_metrics"].items():
            val_str = f"{score:.4f}" if score is not None else "N/A (all NaN)"
            print(f"  • {metric:<20}: {val_str} (valid={report['valid_counts'].get(metric, 0)}, nan={report['nan_counts'].get(metric, 0)})")
        print(f"📁 Đã lưu báo cáo Ragas vào: {out_p}\n")
        return report

    out_p = Path(output_path) if output_path else (base_dir / "data" / "eval_data" / "e2e_eval_results.json")
    pred_out = Path(save_predictions) if save_predictions else out_p.with_name(f"{out_p.stem}_predictions.json")

    print("\n" + "=" * 70)
    print("📊 ĐANG CHẠY ĐÁNH GIÁ TOÀN TRÌNH E2E RAG (OFFLINE + DIAGNOSTICS)")
    print(f"• Chế độ RAG  : {active_mode_label}")
    print(f"• Tập dữ liệu : {target_name} ({actual_samples}/{len(dataset)} câu)")
    print(f"• Top-K       : {top_k}")
    print(f"• LLM Judge   : {'BẬT (UETLLMJudgeEvaluator)' if use_judge else 'TẮT'}")
    print(f"• Semantic    : {'BẬT (bge-m3)' if enable_semantic else 'TẮT'}")
    print("=" * 70)

    evaluator = UETEvaluator(
        use_llm_judge=use_judge,
        embedder=getattr(rag_pipeline, "embedding_model", None),
        enable_semantic_metrics=enable_semantic,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )
    eval_report = evaluator.evaluate_pipeline(
        pipeline=rag_pipeline,
        dataset=dataset,
        top_k=top_k,
        num_samples=num_samples,
        delay_between_queries=delay,
        output_path=out_p,
        predictions_path=pred_out,
        use_batch_llm=use_batch_llm,
        resume=resume,
        judge_rounds=judge_rounds,
    )
    if hasattr(evaluator, "last_diagnostic_report"):
        print_e2e_diagnostic_report(evaluator.last_diagnostic_report)
    print(f"📁 Đã lưu snapshot predictions : {pred_out}")
    print(f"📁 Đã lưu báo cáo E2E chi tiết : {out_p}\n")
    return eval_report


def run_router_eval_cli(
    dataset_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    decision_client: Optional[Any] = None,
    num_samples: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Đánh giá riêng JEV Router (chỉ gọi `decision_client.route()`, không chạy Retrieval/LLM).
    Mặc định dùng `data/golden_questions.json` (35 câu có nhãn `expected_route`).
    """
    from src.evaluation.evaluator import UETEvaluator

    target_path = Path(dataset_path) if dataset_path else (BASE_DIR / "data" / "golden_questions.json")
    if not target_path.is_absolute():
        target_path = BASE_DIR / target_path
    if not target_path.exists():
        print(f"❌ Không tìm thấy tập dữ liệu router eval tại: {target_path}")
        return None

    samples = UETEvaluator.load_dataset(target_path)
    if num_samples and num_samples > 0:
        samples = samples[:num_samples]
    if not samples:
        print(f"❌ Không đọc được mẫu hợp lệ từ: {target_path}")
        return None

    client = decision_client
    if client is None:
        from src.decision import JevDecisionClient

        cfg = load_yaml_config()
        jev_cfg = cfg.get("jev", {})
        client = JevDecisionClient(
            endpoint=os.getenv("JEV_ENDPOINT") or jev_cfg.get("endpoint"),
            api_key=os.getenv("JEV_API_KEY") or os.getenv("OPENROUTER_API_KEY") or jev_cfg.get("api_key"),
            model=jev_cfg.get("model", "strands-decider-12B-gemma4-v1-2610"),
            timeout_s=float(jev_cfg.get("timeout_s", 30.0)),
            max_retries=int(jev_cfg.get("max_retries", 1)),
        )

    print("\n" + "=" * 72)
    print(f"🧭 ĐÁNH GIÁ JEV ROUTER ĐỘC LẬP ({len(samples)} câu từ {target_path.name})")
    print("=" * 72)

    records: List[Dict[str, Any]] = []
    latencies: List[float] = []
    mismatches: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, 1):
        meta = dict(sample.metadata or {})
        expected_raw = meta.get("expected_route")
        t0 = time.perf_counter()
        actual_route_val: Optional[str] = None
        err_str: Optional[str] = None
        trace_entry: Dict[str, Any] = {"node": "router_node", "decision": "route", "source": "jev"}

        try:
            try:
                route_obj = client.route(sample.query, session_id=f"router_eval_{sample.id}")
            except TypeError:
                route_obj = client.route(sample.query)
            actual_route_val = route_obj.value if hasattr(route_obj, "value") else str(route_obj)
            lat = round(time.perf_counter() - t0, 3)
            latencies.append(lat)
            trace_entry["result"] = actual_route_val
            trace_entry["latency_s"] = lat
            last_meta = getattr(client, "last_trace_meta", None)
            if isinstance(last_meta, dict):
                for k in ("server_latency_s", "payload_bytes", "raw_label", "parse_fallback"):
                    if k in last_meta and last_meta[k] is not None:
                        trace_entry[k] = last_meta[k]
        except Exception as exc:
            lat = round(time.perf_counter() - t0, 3)
            err_str = str(exc)
            trace_entry["error"] = err_str
            trace_entry["latency_s"] = lat

        rec: Dict[str, Any] = {
            "sample_id": sample.id,
            "query": sample.query,
            "category": meta.get("category", ""),
            "is_out_of_domain": _is_sample_ood(meta) or str(meta.get("category", "")).lower() == "out_of_scope",
            "expected_route": expected_raw,
            "agent_route": actual_route_val,
            "agent_status": "fail" if err_str else ("out_of_scope" if actual_route_val == "OUT_OF_SCOPE" else "ok"),
            "total_latency_s": lat,
            "error": err_str,
            "agent": {
                "route": actual_route_val,
                "status": "fail" if err_str else ("out_of_scope" if actual_route_val == "OUT_OF_SCOPE" else "ok"),
                "retry": 0,
                "fail_reason": "jev_unavailable" if err_str else None,
                "trace": sanitize_trace([trace_entry]),
            },
        }
        records.append(rec)

        exp_norm = normalize_route_label(expected_raw)
        is_match = (actual_route_val == exp_norm) if exp_norm else True
        if exp_norm and not is_match:
            mismatches.append({
                "sample_id": sample.id,
                "expected": exp_norm,
                "actual": actual_route_val or "ERROR",
                "query": sample.query,
            })
        mark = "✅" if is_match else "⚠️"
        print(
            f"[{idx:02d}/{len(samples):02d}] {sample.id:<6} | "
            f"Actual: {(actual_route_val or 'ERROR'):<18} | Expected: {(exp_norm or 'N/A'):<18} {mark} ({lat:.2f}s)"
        )

    router_summary = compute_router_metrics(records)
    avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else None

    report: Dict[str, Any] = {
        "dataset": str(target_path),
        "total_samples": len(records),
        "avg_latency_s": avg_lat,
        "router_metrics": router_summary,
        "mismatches": mismatches,
        "samples": records,
    }

    print("\n" + "=" * 72)
    print("📈 TỔNG KẾT JEV ROUTER:")
    print(f"  • Tổng câu hỏi        : {len(records)} | Avg latency: {avg_lat}s")
    print(f"  • Phân bố thực tế     : {router_summary.get('route_distribution')}")
    if router_summary.get("explicit_accuracy") is not None:
        print(
            f"  • Explicit Accuracy   : {router_summary['explicit_accuracy']:.4f} "
            f"(n={router_summary['explicit_labeled_total']})"
        )
    if router_summary.get("false_oos_rate") is not None:
        print(
            f"  • False OOS Rate      : {router_summary['false_oos_rate']:.4f} "
            f"({router_summary['false_oos_count']}/{router_summary['in_domain_total']})"
        )
    print(f"  • Parse Fallback Count: {router_summary.get('parse_fallback_count', 0)}")
    if router_summary.get("confusion_matrix"):
        print("  • Ma trận nhầm lẫn (Hàng = Expected, Cột = Actual):")
        cm = router_summary["confusion_matrix"]
        header_lbl = "Expected / Actual"
        print(f"    {header_lbl:<20} | {'OUT_OF_SCOPE':>12} | {'QUALIFIED':>10}")
        for exp_lbl in ("OUT_OF_SCOPE", "QUALIFIED_QUESTION"):
            row = cm.get(exp_lbl, {})
            print(
                f"    {exp_lbl:<20} | {row.get('OUT_OF_SCOPE', 0):>12} | "
                f"{row.get('QUALIFIED_QUESTION', 0):>10}"
            )
    if mismatches:
        print(f"  • Các câu phân loại sai ({len(mismatches)}):")
        for mm in mismatches[:10]:
            print(f"    - [{mm['sample_id']}] expected={mm['expected']} -> actual={mm['actual']} | {mm['query'][:60]}")
    print("=" * 72 + "\n")

    if output_path:
        _atomic_write_json(Path(output_path), report)
        logger.info(f"Đã lưu báo cáo router eval vào: {output_path}")
    return report

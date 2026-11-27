"""
src/evaluation/retrieval_cli.py - CLI table formatters and runners
(`run_eval_retrieval_cli`, `main`) for Retrieval Evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.helpers import load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def print_metrics_table(eval_result_dict: Dict[str, Any], k_values: List[int], title: str):
    """In bảng tổng hợp các chỉ số IR chuẩn theo các ngưỡng K."""
    agg = eval_result_dict.get("aggregated_metrics", {})
    print(f"\n{title}")
    header_cols = "  ".join([f"{f'K={k}':<10}" for k in k_values])
    print(f"{'Chỉ số (Metric)':<25} {header_cols}")
    print("-" * 65)
    for m in ["precision", "recall", "ndcg", "hit_rate"]:
        row = f"{m.replace('_', ' ').capitalize():<25} "
        for k in k_values:
            val = agg.get(f"mean_{m}@{k}", 0.0)
            row += f"{val:<10.4f}  "
        print(row.strip())
    print("-" * 65)
    print(f"  • MRR (Mean Reciprocal Rank)   : {agg.get('mean_mrr', 0.0):.4f}")
    print(f"  • MAP (Mean Average Precision) : {agg.get('mean_map', 0.0):.4f}")


def print_query_type_matrix(eval_result_dict: Dict[str, Any], target_k: int, ordered_types: List[str], title: str):
    """In ma trận chi tiết các chỉ số theo từng dạng câu hỏi (Query Type Matrix)."""
    by_type = eval_result_dict.get("by_query_type", {})
    matrix_types = [t for t in ordered_types if t in by_type]
    if not matrix_types:
        matrix_types = list(by_type.keys())
    if not matrix_types:
        return
    _print_breakdown_table(by_type, matrix_types, target_k, title, col_w=17)


def print_unit_matrix(eval_result_dict: Dict[str, Any], target_k: int, title: str = "MA TRẬN ĐÁNH GIÁ THEO KHOA/ĐƠN VỊ (UNIT)"):
    """In ma trận chi tiết các chỉ số theo từng đơn vị/khoa (Unit Matrix)."""
    by_unit = eval_result_dict.get("by_unit", {})
    if not by_unit or len(by_unit) <= 1:
        return
    _print_breakdown_table(by_unit, list(by_unit.keys()), target_k, title, col_w=16)


def print_category_matrix(eval_result_dict: Dict[str, Any], target_k: int, title: str = "MA TRẬN ĐÁNH GIÁ THEO LĨNH VỰC (CATEGORY)"):
    """In ma trận chi tiết các chỉ số theo từng lĩnh vực (Category Matrix)."""
    by_cat = eval_result_dict.get("by_category", {})
    if not by_cat or len(by_cat) <= 1:
        return
    _print_breakdown_table(by_cat, list(by_cat.keys()), target_k, title, col_w=20)


def _print_breakdown_table(group_data: Dict[str, Any], keys: List[str], target_k: int, title: str, col_w: int = 17):
    """Hàm in bảng ma trận tổng quát."""
    line_len = 14 + len(keys) * col_w
    sep = "=" * line_len

    print("\n" + sep)
    print(f"📊 {title} @ K={target_k}")
    print(sep)

    col_headers = [f"{k}({group_data[k].get('count', 0)})" for k in keys]
    header_line = f"{'':<14}" + "".join([f"{ch:<{col_w}}" for ch in col_headers])
    print(header_line)
    print("-" * line_len)

    for metric_label, metric_key in [("Precision", "precision"), ("Recall", "recall")]:
        row_str = f"{metric_label:<14}"
        for k in keys:
            val = group_data[k].get(f"mean_{metric_key}@{target_k}", 0.0)
            row_str += f"{val:<{col_w}.4f}"
        print(row_str)

    print("-" * line_len)
    for metric_label, metric_key in [("MRR", "mean_mrr"), ("MAP", "mean_map")]:
        row_str = f"{metric_label:<14}"
        for k in keys:
            val = group_data[k].get(metric_key, 0.0)
            row_str += f"{val:<{col_w}.4f}"
        print(row_str)

    print("-" * line_len)
    for metric_label, metric_key in [("NDCG", "ndcg"), ("Hit Rate", "hit_rate")]:
        row_str = f"{metric_label:<14}"
        for k in keys:
            val = group_data[k].get(f"mean_{metric_key}@{target_k}", 0.0)
            row_str += f"{val:<{col_w}.4f}"
        print(row_str)

    print(sep)


def run_eval_retrieval_cli(
    top_k: Optional[int] = None,
    eval_dir: Optional[str | Path] = None,
    dedup: bool = True,
    k_values: Optional[List[int]] = None,
    output: Optional[Union[str, Path]] = None,
    level: str = "both",
    mode: str = "live",
    domain: Optional[str] = None,
    use_cache: bool = False,
):
    """
    Chạy đánh giá hiệu quả phân hệ Retrieval (Precision, Recall, MRR, NDCG) và hiển thị trên CLI.
    Hỗ trợ level = "doc", "chunk", hoặc "both".
    Hỗ trợ mode = "live" (chạy thật) hoặc "mock" (chạy nhanh với ground-truth).
    Hỗ trợ domain = Lọc theo khoa/đơn vị (unit/domain).
    """
    from src.evaluation.evaluate_retrieval import RetrievalEvaluator

    cfg = load_yaml_config()
    target_k = top_k if top_k is not None else cfg.get("retrieval", {}).get("final_top_k", 5)

    if k_values is None:
        k_values = sorted({k for k in (1, 3, 5, target_k) if k <= target_k})

    if eval_dir is None:
        cfg_file = cfg.get("paths", {}).get("eval_dataset_file")
        if cfg_file:
            eval_dir = PROJECT_ROOT / cfg_file if not Path(cfg_file).is_absolute() else Path(cfg_file)
        else:
            default_new = PROJECT_ROOT / "data" / "eval_data" / "uet_rag_eval_100"
            if default_new.exists():
                eval_dir = default_new
            else:
                eval_dir = PROJECT_ROOT / "data" / "eval_data" / "e2e_eval_100_annotated.json"
    else:
        eval_dir = Path(eval_dir)

    has_data = (
        eval_dir.is_file()
        or (eval_dir / "e2e_eval_100_annotated.json").exists()
        or (eval_dir / "e2e_eval_100.json").exists()
        or (eval_dir / "e2e_eval_100.jsonl").exists()
        or (eval_dir / "retrieval_eval_queries.json").exists()
    )
    if not has_data:
        print(f"❌ Không tìm thấy dữ liệu đánh giá tại: {eval_dir}")
        return

    level_display = {
        "doc": "TÀI LIỆU (DOCUMENT-LEVEL)",
        "chunk": "ĐOẠN CẮT (CHUNK-LEVEL)",
        "both": "SONG SONG CẢ TÀI LIỆU VÀ ĐOẠN CẮT (DOC & CHUNK DUAL-LEVEL)",
    }.get(level, level.upper())

    label_type = "Tập dữ liệu:" if eval_dir.is_file() else "Thư mục đánh giá:"
    print("\n" + "=" * 65)
    print("🔬 ĐANG CHẠY ĐÁNH GIÁ PHÂN HỆ TRUY XUẤT (RETRIEVAL EVALUATION)")
    print(f"• {label_type:<21} {eval_dir.name}")
    print(f"• Chế độ (Mode):       {mode.upper()}")
    print(f"• Cấu hình K đánh giá: K={target_k} (Lấy từ config.yaml: retrieval.final_top_k)")
    print(f"• Cấp độ đánh giá:     {level_display}")
    if domain:
        print(f"• Lọc Domain/Unit:     {domain.upper()}")
    print("=" * 65)

    evaluator = RetrievalEvaluator(eval_data_dir=eval_dir)
    if domain:
        evaluator.filter_by_domain(domain)

    if mode == "mock":
        mock_doc_results = {}
        mock_chunk_results = {}
        for q in evaluator.queries:
            qid = q["query_id"]
            gold_docs = list(evaluator.qrels_doc.get(qid, {}).keys())
            gold_chunks = list(evaluator.qrels_chunk.get(qid, {}).keys())
            mock_doc_results[qid] = gold_docs[:target_k]
            mock_chunk_results[qid] = gold_chunks[:target_k]

        if level == "doc":
            eval_result = evaluator.evaluate_batch(mock_doc_results, k_values=k_values, level="doc")
        elif level == "chunk":
            eval_result = evaluator.evaluate_batch(mock_chunk_results, k_values=k_values, level="chunk")
        else:
            doc_res = evaluator.evaluate_batch(mock_doc_results, k_values=k_values, level="doc")
            chunk_res = evaluator.evaluate_batch(mock_chunk_results, k_values=k_values, level="chunk")
            eval_result = {
                "summary": doc_res["summary"],
                "level": "both",
                "doc_level": doc_res,
                "chunk_level": chunk_res,
            }
    else:
        from src.pipeline import get_retriever
        retriever = get_retriever()
        eval_result = evaluator.evaluate_retriever(
            retriever,
            top_k=target_k,
            k_values=k_values,
            dedup_documents=dedup,
            level=level,
            use_cache=use_cache,
        )

    meta_base = eval_dir.parent if eval_dir.is_file() else eval_dir
    meta_file = meta_base / "dataset_metadata.json"
    ordered_types = []
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                ordered_types = list(json.load(mf).get("query_types", {}).keys())
        except Exception:
            pass

    reranker_label = ""
    if mode != "mock":
        r_type = cfg.get("retrieval", {}).get("reranker_type", "")
        if r_type:
            reranker_label = f" với reranker={r_type}"

    print("\n" + "=" * 65)
    print(f"📈 BÁO CÁO KẾT QUẢ PHÂN HỆ RETRIEVAL{reranker_label} (TOP-K = {target_k})")
    print("=" * 65)
    summary = eval_result.get("summary", {})
    ood_count = summary.get("ood_queries", 0)
    print(f"Tổng số queries đánh giá: {summary.get('in_domain_queries', 0)} in-domain + {ood_count} OOD queries")

    ood_stats = eval_result.get("ood_stats") or {}
    if ood_stats.get("ood_queries", 0) > 0:
        print(
            f"OOD Score Calibration   : OOD avg top1={ood_stats.get('ood_avg_top1_score')} | "
            f"In-domain avg top1={ood_stats.get('in_domain_avg_top1_score')} | "
            f"Gap={ood_stats.get('score_gap')} | Suggested midpoint={ood_stats.get('suggested_threshold_midpoint')}"
        )

    if level == "both":
        doc_res = eval_result["doc_level"]
        chunk_res = eval_result["chunk_level"]

        print_metrics_table(doc_res, k_values, "1️⃣ ĐÁNH GIÁ MỨC TÀI LIỆU (DOCUMENT-LEVEL)")
        print_metrics_table(chunk_res, k_values, "2️⃣ ĐÁNH GIÁ MỨC ĐOẠN CẮT (CHUNK-LEVEL)")
        print("=" * 65 + "\n")

        print_query_type_matrix(
            doc_res, target_k, ordered_types, "MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI - MỨC TÀI LIỆU (DOC-LEVEL)"
        )
        print_query_type_matrix(
            chunk_res, target_k, ordered_types, "MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI - MỨC ĐOẠN CẮT (CHUNK-LEVEL)"
        )
        print_unit_matrix(
            chunk_res, target_k, "MA TRẬN ĐÁNH GIÁ THEO KHOA/ĐƠN VỊ (UNIT) - MỨC ĐOẠN CẮT (CHUNK-LEVEL)"
        )
        print_category_matrix(
            chunk_res, target_k, "MA TRẬN ĐÁNH GIÁ THEO LĨNH VỰC (CATEGORY) - MỨC ĐOẠN CẮT (CHUNK-LEVEL)"
        )
    elif level == "chunk":
        chunk_res = eval_result
        print_metrics_table(chunk_res, k_values, "🎯 ĐÁNH GIÁ MỨC ĐOẠN CẮT (CHUNK-LEVEL)")
        print("=" * 65 + "\n")
        print_query_type_matrix(
            chunk_res, target_k, ordered_types, "MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI - MỨC ĐOẠN CẮT (CHUNK-LEVEL)"
        )
        print_unit_matrix(
            chunk_res, target_k, "MA TRẬN ĐÁNH GIÁ THEO KHOA/ĐƠN VỊ (UNIT) - MỨC ĐOẠN CẮT (CHUNK-LEVEL)"
        )
        print_category_matrix(
            chunk_res, target_k, "MA TRẬN ĐÁNH GIÁ THEO LĨNH VỰC (CATEGORY) - MỨC ĐOẠN CẮT (CHUNK-LEVEL)"
        )
    else:
        doc_res = eval_result
        print_metrics_table(doc_res, k_values, "📄 ĐÁNH GIÁ MỨC TÀI LIỆU (DOCUMENT-LEVEL)")
        print("=" * 65 + "\n")
        print_query_type_matrix(
            doc_res, target_k, ordered_types, "MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI - MỨC TÀI LIỆU (DOC-LEVEL)"
        )
        print_unit_matrix(
            doc_res, target_k, "MA TRẬN ĐÁNH GIÁ THEO KHOA/ĐƠN VỊ (UNIT) - MỨC TÀI LIỆU (DOC-LEVEL)"
        )
        print_category_matrix(
            doc_res, target_k, "MA TRẬN ĐÁNH GIÁ THEO LĨNH VỰC (CATEGORY) - MỨC TÀI LIỆU (DOC-LEVEL)"
        )

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu kết quả chi tiết: {output_path}\n")


def main():
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from src.evaluation.evaluate_retrieval import RetrievalEvaluator

    cfg = load_yaml_config()
    default_k = cfg.get("retrieval", {}).get("final_top_k", 5)

    cfg_file = cfg.get("paths", {}).get("eval_dataset_file")
    if cfg_file:
        default_eval_dir = PROJECT_ROOT / cfg_file if not Path(cfg_file).is_absolute() else Path(cfg_file)
    else:
        default_eval_dir = PROJECT_ROOT / "data" / "eval_data" / "e2e_eval_100_annotated.json"

    parser = argparse.ArgumentParser(description="Đánh giá hiệu quả Retrieval cho UET RAG Chatbot")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=str(default_eval_dir),
        help=f"Thư mục chứa dữ liệu đánh giá (mặc định: {default_eval_dir.name})",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Lọc đánh giá theo khoa/đơn vị (unit/domain, ví dụ: FET, FIT, TUYENSINH)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=default_k,
        help=f"Số kết quả trả về top-K (mặc định: lấy từ config.yaml = {default_k})",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["doc", "chunk", "both"],
        default="both",
        help="Cấp độ đánh giá: 'doc', 'chunk', hoặc 'both' (mặc định: both)",
    )
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file lưu kết quả JSON")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mock", "live"],
        default="mock",
        help="'mock' = test với ground truth, 'live' = gọi UETRetriever thật",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Tắt tính năng khử trùng lặp cấp Document (áp dụng cho doc level)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Bật cache retrieval theo hash cấu hình (mặc định: tắt để tránh đo nhầm khi đổi config)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Tắt cache retrieval (mặc định)",
    )
    args = parser.parse_args()
    use_cache = bool(args.use_cache and not args.no_cache)

    eval_k = args.top_k if args.top_k is not None else default_k
    k_values = [1, 3, 5, 10, 20] if args.mode == "mock" else [eval_k]

    print(f"\n{'='*70}")
    print(f"🔬 ĐÁNH GIÁ RETRIEVAL MODULE - UET RAG CHATBOT")
    print(f"{'='*70}")
    print(f"  Vị trí script: src/evaluation/evaluate_retrieval.py")
    print(f"  Mode:          {args.mode}")
    print(f"  Level:         {args.level}")
    print(f"  Top-K:         {eval_k}")
    print(f"  Doc Dedup:     {not args.no_dedup}")
    print(f"  Eval dir:      {args.eval_dir}")
    print()

    evaluator = RetrievalEvaluator(eval_data_dir=args.eval_dir)
    if args.domain:
        evaluator.filter_by_domain(args.domain)

    if args.mode == "mock":
        print("📝 Đang chạy mock evaluation (giả lập retriever với ground truth)...")
        doc_mock_results: Dict[str, List[str]] = {}
        chunk_mock_results: Dict[str, List[str]] = {}

        for query in evaluator.queries:
            qid = query["query_id"]
            gold_docs = list(evaluator.qrels_doc.get(qid, {}).keys())
            doc_mock_results[qid] = gold_docs[:eval_k]

            gold_chunks = list(evaluator.qrels_chunk.get(qid, {}).keys())
            chunk_mock_results[qid] = gold_chunks[:eval_k]

        if args.level == "doc":
            eval_result = evaluator.evaluate_batch(doc_mock_results, k_values=k_values, level="doc")
        elif args.level == "chunk":
            eval_result = evaluator.evaluate_batch(chunk_mock_results, k_values=k_values, level="chunk")
        else:
            doc_res = evaluator.evaluate_batch(doc_mock_results, k_values=k_values, level="doc")
            chunk_res = evaluator.evaluate_batch(chunk_mock_results, k_values=k_values, level="chunk")
            eval_result = {
                "summary": doc_res["summary"],
                "level": "both",
                "doc_level": doc_res,
                "chunk_level": chunk_res,
                "aggregated_metrics": doc_res["aggregated_metrics"],
                "by_difficulty": doc_res["by_difficulty"],
                "by_query_type": doc_res["by_query_type"],
                "per_query_results": doc_res["per_query_results"],
            }

    elif args.mode == "live":
        print("🚀 Đang khởi tạo UETRetriever từ src/pipeline.py...")
        try:
            from src.pipeline import get_retriever

            retriever = get_retriever()
            print("  ✅ Khởi tạo UETRetriever thành công!")
            eval_result = evaluator.evaluate_retriever(
                retriever,
                top_k=eval_k,
                k_values=k_values,
                dedup_documents=not args.no_dedup,
                level=args.level,
                use_cache=use_cache,
            )
        except Exception as e:
            print(f"❌ Không thể chạy live mode: {e}")
            import traceback
            traceback.print_exc()
            return

    print(f"\n{'='*70}")
    print(f"📈 KẾT QUẢ ĐÁNH GIÁ RETRIEVAL")
    print(f"{'='*70}")

    def _print_level_stats(res_dict: Dict[str, Any], label: str):
        agg = res_dict.get("aggregated_metrics", {})
        print(f"\n  📌 METRICS TỔNG HỢP - {label} ({res_dict['summary']['in_domain_queries']} in-domain queries):")
        print(f"  {'─'*55}")
        for k in [k for k in [1, 3, 5, 10] if f'mean_precision@{k}' in agg]:
            p = agg.get(f"mean_precision@{k}", 0.0)
            r = agg.get(f"mean_recall@{k}", 0.0)
            n = agg.get(f"mean_ndcg@{k}", 0.0)
            h = agg.get(f"mean_hit_rate@{k}", 0.0)
            print(f"    @{k:>2}: Precision={p:.4f} | Recall={r:.4f} | NDCG={n:.4f} | HitRate={h:.4f}")
        print(f"\n    MRR  = {agg.get('mean_mrr', 0.0):.4f}")
        print(f"    MAP  = {agg.get('mean_map', 0.0):.4f}")

    if args.level == "both":
        _print_level_stats(eval_result["doc_level"], "MỨC TÀI LIỆU (DOCUMENT-LEVEL)")
        _print_level_stats(eval_result["chunk_level"], "MỨC ĐOẠN CẮT (CHUNK-LEVEL)")
        print_query_type_matrix(eval_result["chunk_level"], eval_k, [], "MA TRẬN DẠNG CÂU HỎI (CHUNK-LEVEL)")
        print_unit_matrix(eval_result["chunk_level"], eval_k, "MA TRẬN KHOA/ĐƠN VỊ (UNIT)")
        print_category_matrix(eval_result["chunk_level"], eval_k, "MA TRẬN LĨNH VỰC (CATEGORY)")
    elif args.level == "chunk":
        _print_level_stats(eval_result, "MỨC ĐOẠN CẮT (CHUNK-LEVEL)")
        print_query_type_matrix(eval_result, eval_k, [], "MA TRẬN DẠNG CÂU HỎI (CHUNK-LEVEL)")
        print_unit_matrix(eval_result, eval_k, "MA TRẬN KHOA/ĐƠN VỊ (UNIT)")
        print_category_matrix(eval_result, eval_k, "MA TRẬN LĨNH VỰC (CATEGORY)")
    else:
        _print_level_stats(eval_result, "MỨC TÀI LIỆU (DOCUMENT-LEVEL)")
        print_query_type_matrix(eval_result, eval_k, [], "MA TRẬN DẠNG CÂU HỎI (DOC-LEVEL)")
        print_unit_matrix(eval_result, eval_k, "MA TRẬN KHOA/ĐƠN VỊ (UNIT)")
        print_category_matrix(eval_result, eval_k, "MA TRẬN LĨNH VỰC (CATEGORY)")

    eval_path = Path(args.eval_dir)
    out_dir = eval_path.parent if eval_path.is_file() else eval_path
    output_path = args.output or str(
        out_dir / f"eval_results_{args.mode}_{args.level}_top{eval_k}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã lưu kết quả chi tiết: {output_path}\n")
    print(f"\n✅ HOÀN TẤT ĐÁNH GIÁ RETRIEVAL!\n")

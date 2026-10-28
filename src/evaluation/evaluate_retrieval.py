"""
src/evaluation/evaluate_retrieval.py - Module & Script đánh giá hiệu quả truy xuất (Retrieval Evaluation).

Sử dụng bộ dữ liệu đánh giá trong data/eval_data/retrieval_eval_data/
để đo lường hiệu quả của Retriever qua các metrics chuẩn IR:
  - Precision@K, Recall@K (K = 1, 3, 5, 10, 20)
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)
  - MAP (Mean Average Precision)
  - Hit Rate@K

Cách sử dụng CLI:
    python src/evaluation/evaluate_retrieval.py --mode mock --top_k 5
    python src/evaluation/evaluate_retrieval.py --mode live --top_k 5

Sử dụng trong Code:
    from src.evaluation.evaluate_retrieval import RetrievalEvaluator
    evaluator = RetrievalEvaluator()
    results = evaluator.evaluate_retriever(retriever, top_k=5)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Đảm bảo root project có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr,
    average_precision,
    hit_rate_at_k,
)
from src.utils.helpers import get_logger

logger = get_logger("retrieval_evaluation")


class RetrievalEvaluator:
    """
    Bộ đánh giá hiệu quả Retrieval (IR Evaluator) cho UET RAG Chatbot.
    Tải queries, qrels từ tập benchmark và tính toán các metrics IR chuẩn.
    """

    def __init__(self, eval_data_dir: Optional[str | Path] = None):
        """
        :param eval_data_dir: Thư mục chứa retrieval_eval_queries.json và retrieval_eval_qrels.json.
                              Mặc định: data/eval_data/retrieval_eval_data/
        """
        if eval_data_dir is None:
            self.eval_data_dir = PROJECT_ROOT / "data" / "eval_data" / "retrieval_eval_data"
        else:
            self.eval_data_dir = Path(eval_data_dir)

        self.queries: List[Dict[str, Any]] = []
        self.qrels: Dict[str, Dict[str, int]] = {}
        self._load_data()

    def _load_data(self):
        """Tải queries và qrels từ file JSON."""
        queries_path = self.eval_data_dir / "retrieval_eval_queries.json"
        qrels_path = self.eval_data_dir / "retrieval_eval_qrels.json"

        if not queries_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file queries: {queries_path}")
        if not qrels_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file qrels: {qrels_path}")

        with open(queries_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        with open(qrels_path, "r", encoding="utf-8") as f:
            self.qrels = json.load(f)

        logger.info(f"Đã tải {len(self.queries)} queries và {len(self.qrels)} qrels entries từ {self.eval_data_dir}")

    def get_relevant_ids(self, query_id: str) -> Set[str]:
        """Lấy tập hợp tất cả document IDs liên quan cho một query."""
        return set(self.qrels.get(query_id, {}).keys())

    def evaluate_single_query(
        self,
        query_id: str,
        retrieved_doc_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10, 20],
        is_out_of_domain: bool = False,
        refused_to_answer: bool = True,
    ) -> Dict[str, Any]:
        """
        Đánh giá kết quả retrieval cho một câu hỏi duy nhất.

        :param query_id: ID câu hỏi
        :param retrieved_doc_ids: Danh sách doc IDs trả về bởi retriever (theo thứ tự rank)
        :param k_values: Các ngưỡng K để tính Precision@K, Recall@K, etc.
        :param is_out_of_domain: Câu hỏi ngoài phạm vi UET (không có tài liệu ground truth)
        :param refused_to_answer: Hệ thống có từ chối trả lời đúng chuẩn hay không
        :return: Dict chứa tất cả metrics cho query này
        """
        relevant_ids = self.get_relevant_ids(query_id)
        query_qrels = self.qrels.get(query_id, {})

        result: Dict[str, Any] = {
            "query_id": query_id,
            "num_retrieved": len(retrieved_doc_ids),
            "num_relevant_in_ground_truth": len(relevant_ids),
            "is_out_of_domain": is_out_of_domain,
        }

        # Quy tắc đánh giá out_of_domain:
        # Nếu đúng là từ chối trả lời (hệ thống nhận diện chuẩn ngoài phạm vi) -> 1.0000
        # Nếu cố tình trả lời -> 0.0000
        if is_out_of_domain:
            ood_score = 1.0 if refused_to_answer else 0.0
            for k in k_values:
                result[f"precision@{k}"] = ood_score
                result[f"recall@{k}"] = ood_score
                result[f"ndcg@{k}"] = ood_score
                result[f"hit_rate@{k}"] = ood_score
            result["mrr"] = ood_score
            result["map"] = ood_score
            return result

        # Tính Precision@K, Recall@K, NDCG@K, HitRate@K cho từng K
        for k in k_values:
            result[f"precision@{k}"] = round(precision_at_k(retrieved_doc_ids, relevant_ids, k), 4)
            result[f"recall@{k}"] = round(recall_at_k(retrieved_doc_ids, relevant_ids, k), 4)
            result[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_doc_ids, query_qrels, k), 4)
            result[f"hit_rate@{k}"] = round(hit_rate_at_k(retrieved_doc_ids, relevant_ids, k), 4)

        # MRR và MAP
        result["mrr"] = round(mrr(retrieved_doc_ids, relevant_ids), 4)
        result["map"] = round(average_precision(retrieved_doc_ids, relevant_ids), 4)

        return result

    def evaluate_batch(
        self,
        retrieval_results: Dict[str, List[str]],
        k_values: List[int] = [1, 3, 5, 10, 20],
    ) -> Dict[str, Any]:
        """
        Đánh giá toàn bộ tập câu hỏi.

        :param retrieval_results: Dict {query_id: [retrieved_doc_ids]}
        :param k_values: Các ngưỡng K
        :return: Dict chứa chi tiết per-query results và aggregated metrics
        """
        per_query_results = []

        for query in self.queries:
            qid = query["query_id"]
            retrieved = retrieval_results.get(qid, [])
            q_types = query.get("query_types") or ([query.get("query_type")] if query.get("query_type") else [])
            is_ood = "out_of_domain" in q_types or (len(self.get_relevant_ids(qid)) == 0)
            res = self.evaluate_single_query(
                qid, retrieved, k_values, is_out_of_domain=is_ood, refused_to_answer=True
            )
            res["query_text"] = query.get("query_text", "")
            res["difficulty"] = query.get("difficulty", "unknown")
            res["query_types"] = q_types
            per_query_results.append(res)

        # Tách riêng OOD queries ra khỏi trung bình chính
        in_domain_results = [
            r for r in per_query_results
            if not any(q["query_id"] == r["query_id"] and q.get("query_type") == "out_of_domain"
                       for q in self.queries)
        ]

        aggregated: Dict[str, float] = {}
        if in_domain_results:
            metric_keys = [
                k for k in in_domain_results[0].keys()
                if isinstance(in_domain_results[0][k], (int, float))
                and k not in ("num_retrieved", "num_relevant_in_ground_truth")
            ]
            for key in metric_keys:
                vals = [r[key] for r in in_domain_results if key in r]
                aggregated[f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        # Phân tích theo difficulty (easy, medium, hard)
        difficulty_metrics: Dict[str, Dict[str, float]] = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in in_domain_results if r.get("difficulty") == diff]
            if diff_results:
                difficulty_metrics[diff] = {}
                for key in metric_keys:
                    vals = [r[key] for r in diff_results if key in r]
                    difficulty_metrics[diff][f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        # Phân tích theo toàn bộ query_types khai báo trong dataset_metadata.json (multi-label)
        meta_file = self.eval_data_dir / "dataset_metadata.json"
        target_types = []
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    target_types = list(json.load(mf).get("query_types", {}).keys())
            except Exception:
                pass
        if not target_types:
            target_types = ["factoid", "procedure", "list", "out_of_domain", "comparison", "informal", "multi_hop", "aggregation"]

        type_metrics: Dict[str, Dict[str, Any]] = {}
        for qt in target_types:
            qt_results = [
                r for q, r in zip(self.queries, per_query_results)
                if qt in (q.get("query_types") or ([q.get("query_type")] if q.get("query_type") else []))
            ]
            type_metrics[qt] = {"count": len(qt_results)}
            for key in metric_keys:
                vals = [r[key] for r in qt_results if key in r]
                type_metrics[qt][f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "summary": {
                "total_queries": len(self.queries),
                "evaluated_queries": len(retrieval_results),
                "in_domain_queries": len(in_domain_results),
            },
            "aggregated_metrics": aggregated,
            "by_difficulty": difficulty_metrics,
            "by_query_type": type_metrics,
            "per_query_results": per_query_results,
        }

    def evaluate_retriever(
        self,
        retriever: Any,
        top_k: int = 5,
        k_values: Optional[List[int]] = None,
        dedup_documents: bool = True,
    ) -> Dict[str, Any]:
        """
        Đánh giá trực tiếp một instance Retriever (kế thừa BaseRetriever).
        
        :param retriever: Đối tượng UETRetriever hoặc BaseRetriever
        :param top_k: Số lượng documents tối đa cần retrieve
        :param k_values: Các ngưỡng K (mặc định: chỉ test với đúng [top_k])
        :param dedup_documents: Khử trùng lặp ở cấp độ Document ID (tránh các chunk cùng 1 tài liệu chiếm chỗ Top-K)
        """
        if k_values is None:
            k_values = [top_k]
        from src.base import UserQuery

        retrieval_results: Dict[str, List[str]] = {}
        query_top_k = max(top_k * 4, 25) if dedup_documents else top_k

        # Tự động nạp cache kết quả retrieval nếu đã từng chạy
        cache_file = self.eval_data_dir / f".retrieval_cache_top{top_k}.json"
        cached: Dict[str, List[str]] = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
            except Exception:
                cached = {}

        needs_save = False
        for i, query in enumerate(self.queries):
            qid = query["query_id"]
            if qid in cached and len(cached[qid]) >= top_k:
                retrieval_results[qid] = cached[qid][:top_k]
                continue

            user_query = UserQuery(query_text=query["query_text"])
            try:
                contexts = retriever.retrieve(user_query, top_k=query_top_k)
                retrieved_ids = []
                seen_doc_ids = set()

                for c in contexts:
                    doc_id = None
                    if hasattr(c, "metadata") and isinstance(c.metadata, dict):
                        doc_id = c.metadata.get("document_id")
                    elif isinstance(c, dict) and isinstance(c.get("metadata"), dict):
                        doc_id = c["metadata"].get("document_id")
                    elif isinstance(c, dict):
                        doc_id = c.get("document_id")

                    if not doc_id:
                        if hasattr(c, "chunk_id"):
                            doc_id = c.chunk_id
                        elif hasattr(c, "id"):
                            doc_id = c.id
                        elif isinstance(c, dict):
                            doc_id = c.get("chunk_id", c.get("id", ""))
                        elif isinstance(c, str):
                            doc_id = c

                    if doc_id:
                        doc_id_str = str(doc_id)
                        if dedup_documents:
                            if doc_id_str not in seen_doc_ids:
                                seen_doc_ids.add(doc_id_str)
                                retrieved_ids.append(doc_id_str)
                                if len(retrieved_ids) >= top_k:
                                    break
                        else:
                            retrieved_ids.append(doc_id_str)
                            if len(retrieved_ids) >= top_k:
                                break

                retrieval_results[qid] = retrieved_ids
                cached[qid] = retrieved_ids
                needs_save = True
            except Exception as e:
                logger.error(f"Lỗi khi retrieve cho query {qid}: {e}")
                retrieval_results[qid] = []

            if (i + 1) % 10 == 0 or (i + 1) == len(self.queries):
                logger.info(f"Đã retrieve {i + 1}/{len(self.queries)} queries...")

        if needs_save:
            try:
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump(cached, cf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return self.evaluate_batch(retrieval_results, k_values=k_values)


def run_eval_retrieval_cli(
    top_k: Optional[int] = None,
    eval_dir: Optional[str | Path] = None,
    dedup: bool = True,
    k_values: Optional[List[int]] = None,
    output: Optional[Union[str, Path]] = None,
):
    """Chạy đánh giá hiệu quả phân hệ Retrieval (Precision, Recall, MRR, NDCG) và hiển thị trên CLI."""
    from src.utils.helpers import load_yaml_config
    cfg = load_yaml_config()
    target_k = top_k if top_k is not None else cfg.get("retrieval", {}).get("final_top_k", 5)

    if k_values is None:
        k_values = [target_k]

    print("\n" + "=" * 65)
    print("🔬 ĐANG CHẠY ĐÁNH GIÁ PHÂN HỆ TRUY XUẤT (RETRIEVAL EVALUATION)")
    print(f"• Cấu hình K đánh giá: K={target_k} (Lấy từ config.yaml: retrieval.final_top_k)")
    print("=" * 65)
    from src.pipeline import get_retriever

    if eval_dir is None:
        eval_dir = PROJECT_ROOT / "data" / "eval_data" / "retrieval_eval_data"
    else:
        eval_dir = Path(eval_dir)

    if not (eval_dir / "retrieval_eval_queries.json").exists():
        print(f"❌ Không tìm thấy dữ liệu đánh giá tại: {eval_dir}")
        return

    evaluator = RetrievalEvaluator(eval_data_dir=eval_dir)
    retriever = get_retriever()
    eval_result = evaluator.evaluate_retriever(
        retriever, top_k=target_k, k_values=k_values, dedup_documents=dedup
    )

    print("\n" + "=" * 65)
    print(f"📈 BÁO CÁO KẾT QUẢ PHÂN HỆ RETRIEVAL (TOP-K = {target_k})")
    print("=" * 65)
    agg = eval_result.get("aggregated_metrics", {})
    summary = eval_result.get("summary", {})
    print(f"Tổng số queries đánh giá: {summary.get('in_domain_queries', 0)} in-domain queries\n")
    
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
    print(f"  • MRR (Mean Reciprocal Rank)  : {agg.get('mean_mrr', 0.0):.4f}")
    print(f"  • MAP (Mean Average Precision) : {agg.get('mean_map', 0.0):.4f}")
    print("=" * 65 + "\n")

    # =========================================================================
    # MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI (QUERY TYPE MATRIX) - TOÀN BỘ QUERY TYPES
    # =========================================================================
    by_type = eval_result.get("by_query_type", {})

    meta_file = eval_dir / "dataset_metadata.json"
    ordered_types = []
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                ordered_types = list(json.load(mf).get("query_types", {}).keys())
        except Exception:
            pass
    if not ordered_types:
        ordered_types = list(by_type.keys())

    matrix_types = [t for t in ordered_types if t in by_type]

    if matrix_types:
        col_w = 17
        line_len = 14 + len(matrix_types) * col_w
        sep = "=" * line_len

        print(sep)
        print(f"📊 MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI (QUERY TYPE MATRIX @ K={target_k})")
        print(sep)

        # Tiêu đề cột: factoid(71), procedure(17), list(28)...
        col_headers = [f"{t}({by_type[t].get('count', 0)})" for t in matrix_types]
        header_line = f"{'':<14}" + "".join([f"{ch:<{col_w}}" for ch in col_headers])
        print(header_line)
        print("-" * line_len)

        # Hàng Precision và Recall (chuẩn xác theo mẫu ảnh)
        for metric_label, metric_key in [("Precision", "precision"), ("Recall", "recall")]:
            row_str = f"{metric_label:<14}"
            for t in matrix_types:
                val = by_type[t].get(f"mean_{metric_key}@{target_k}", 0.0)
                row_str += f"{val:<{col_w}.4f}"
            print(row_str)

        print("-" * line_len)
        # Hàng MRR và MAP
        for metric_label, metric_key in [("MRR", "mean_mrr"), ("MAP", "mean_map")]:
            row_str = f"{metric_label:<14}"
            for t in matrix_types:
                val = by_type[t].get(metric_key, 0.0)
                row_str += f"{val:<{col_w}.4f}"
            print(row_str)

        print("-" * line_len)
        # Các chỉ số bổ trợ (NDCG, Hit Rate)
        for metric_label, metric_key in [("NDCG", "ndcg"), ("Hit Rate", "hit_rate")]:
            row_str = f"{metric_label:<14}"
            for t in matrix_types:
                val = by_type[t].get(f"mean_{metric_key}@{target_k}", 0.0)
                row_str += f"{val:<{col_w}.4f}"
            print(row_str)

        print(sep + "\n")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu kết quả chi tiết: {output_path}\n")


# ============================================================================
# CLI RUNNER
# ============================================================================

def main():
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Đánh giá hiệu quả Retrieval cho UET RAG Chatbot")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "eval_data" / "retrieval_eval_data"),
        help="Thư mục chứa dữ liệu đánh giá (mặc định: data/eval_data/retrieval_eval_data)",
    )
    parser.add_argument("--top_k", type=int, default=None, help="Số kết quả trả về top-K (mặc định: lấy từ config.yaml)")
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
        help="Tắt tính năng khử trùng lặp cấp Document (mặc định luôn khử trùng lặp)",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"🔬 ĐÁNH GIÁ RETRIEVAL MODULE - UET RAG CHATBOT")
    print(f"{'='*70}")
    print(f"  Vị trí script: src/evaluation/evaluate_retrieval.py")
    print(f"  Mode:          {args.mode}")
    print(f"  Top-K:         {args.top_k}")
    print(f"  Doc Dedup:     {not args.no_dedup}")
    print(f"  Eval dir:      {args.eval_dir}")
    print()

    evaluator = RetrievalEvaluator(eval_data_dir=args.eval_dir)

    if args.mode == "mock":
        print("📝 Đang chạy mock evaluation (giả lập retriever với ground truth)...")
        retrieval_results: Dict[str, List[str]] = {}
        for query in evaluator.queries:
            qid = query["query_id"]
            # Lấy danh sách doc IDs liên quan (đảm bảo khử trùng lặp)
            seen_ids = set()
            relevant = []
            for did in query.get("highly_relevant_doc_ids", []) + query.get("relevant_doc_ids", []):
                if did not in seen_ids:
                    seen_ids.add(did)
                    relevant.append(did)
            retrieval_results[qid] = relevant[:args.top_k]

        eval_result = evaluator.evaluate_batch(retrieval_results, k_values=[1, 3, 5, 10, 20])

    elif args.mode == "live":
        print("🚀 Đang khởi tạo UETRetriever từ src/pipeline.py...")
        try:
            from src.pipeline import get_retriever

            retriever = get_retriever()
            print("  ✅ Khởi tạo UETRetriever thành công!")
            eval_result = evaluator.evaluate_retriever(
                retriever, top_k=args.top_k, k_values=[args.top_k], dedup_documents=not args.no_dedup
            )

        except Exception as e:
            print(f"❌ Không thể chạy live mode: {e}")
            import traceback
            traceback.print_exc()
            return

    # In kết quả tổng hợp
    print(f"\n{'='*70}")
    print(f"📈 KẾT QUẢ ĐÁNH GIÁ RETRIEVAL")
    print(f"{'='*70}")

    agg = eval_result["aggregated_metrics"]
    print(f"\n  📌 METRICS TỔNG HỢP ({eval_result['summary']['in_domain_queries']} in-domain queries):")
    print(f"  {'─'*55}")

    for k in [1, 3, 5, 10]:
        p = agg.get(f"mean_precision@{k}", 0.0)
        r = agg.get(f"mean_recall@{k}", 0.0)
        n = agg.get(f"mean_ndcg@{k}", 0.0)
        h = agg.get(f"mean_hit_rate@{k}", 0.0)
        print(f"    @{k:>2}: Precision={p:.4f} | Recall={r:.4f} | NDCG={n:.4f} | HitRate={h:.4f}")

    print(f"\n    MRR  = {agg.get('mean_mrr', 0.0):.4f}")
    print(f"    MAP  = {agg.get('mean_map', 0.0):.4f}")

    print(f"\n  📌 THEO ĐỘ KHÓ (P@5 / R@5 / NDCG@5):")
    for diff, metrics in eval_result.get("by_difficulty", {}).items():
        p5 = metrics.get("mean_precision@5", 0.0)
        r5 = metrics.get("mean_recall@5", 0.0)
        n5 = metrics.get("mean_ndcg@5", 0.0)
        print(f"    {diff:>8}: P@5={p5:.4f} | R@5={r5:.4f} | NDCG@5={n5:.4f}")

    # Ghi kết quả
    output_path = args.output or str(
        Path(args.eval_dir) / f"eval_results_{args.mode}_top{args.top_k}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã lưu kết quả chi tiết: {output_path}")
    print(f"\n✅ HOÀN TẤT ĐÁNH GIÁ RETRIEVAL!\n")


if __name__ == "__main__":
    main()

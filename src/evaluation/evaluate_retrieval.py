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
from typing import Any, Dict, List, Optional, Set

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
    ) -> Dict[str, Any]:
        """
        Đánh giá kết quả retrieval cho một câu hỏi duy nhất.

        :param query_id: ID câu hỏi
        :param retrieved_doc_ids: Danh sách doc IDs trả về bởi retriever (theo thứ tự rank)
        :param k_values: Các ngưỡng K để tính Precision@K, Recall@K, etc.
        :return: Dict chứa tất cả metrics cho query này
        """
        relevant_ids = self.get_relevant_ids(query_id)
        query_qrels = self.qrels.get(query_id, {})

        result: Dict[str, Any] = {
            "query_id": query_id,
            "num_retrieved": len(retrieved_doc_ids),
            "num_relevant_in_ground_truth": len(relevant_ids),
        }

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
            res = self.evaluate_single_query(qid, retrieved, k_values)
            res["query_text"] = query.get("query_text", "")
            res["difficulty"] = query.get("difficulty", "unknown")
            res["query_type"] = query.get("query_type", "unknown")
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

        # Phân tích theo query_type
        type_metrics: Dict[str, Dict[str, float]] = {}
        for qt in set(q.get("query_type", "unknown") for q in self.queries):
            qt_results = [r for r in per_query_results if r.get("query_type") == qt]
            if qt_results:
                type_metrics[qt] = {}
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
        k_values: List[int] = [1, 3, 5, 10, 20],
    ) -> Dict[str, Any]:
        """
        Đánh giá trực tiếp một instance Retriever (kế thừa BaseRetriever).
        
        :param retriever: Đối tượng UETRetriever hoặc BaseRetriever
        :param top_k: Số lượng documents tối đa cần retrieve
        :param k_values: Các ngưỡng K
        """
        from src.base import UserQuery

        retrieval_results: Dict[str, List[str]] = {}
        for i, query in enumerate(self.queries):
            qid = query["query_id"]
            user_query = UserQuery(query_text=query["query_text"])
            try:
                contexts = retriever.retrieve(user_query, top_k=top_k)
                # contexts có thể là List[RetrievedContext] hoặc danh sách dict
                retrieved_ids = []
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
                        retrieved_ids.append(str(doc_id))
                retrieval_results[qid] = retrieved_ids
            except Exception as e:
                logger.error(f"Lỗi khi retrieve cho query {qid}: {e}")
                retrieval_results[qid] = []

            if (i + 1) % 10 == 0 or (i + 1) == len(self.queries):
                logger.info(f"Đã retrieve {i + 1}/{len(self.queries)} queries...")

        return self.evaluate_batch(retrieval_results, k_values=k_values)


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
    parser.add_argument("--top_k", type=int, default=5, help="Số kết quả trả về top-K (mặc định: 5)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file lưu kết quả JSON")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mock", "live"],
        default="mock",
        help="'mock' = test với ground truth, 'live' = gọi UETRetriever thật",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"🔬 ĐÁNH GIÁ RETRIEVAL MODULE - UET RAG CHATBOT")
    print(f"{'='*70}")
    print(f"  Vị trí script: src/evaluation/evaluate_retrieval.py")
    print(f"  Mode:          {args.mode}")
    print(f"  Top-K:         {args.top_k}")
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
            eval_result = evaluator.evaluate_retriever(retriever, top_k=args.top_k)

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

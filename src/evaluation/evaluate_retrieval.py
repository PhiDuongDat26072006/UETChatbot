"""
src/evaluation/evaluate_retrieval.py - Module & Script đánh giá hiệu quả truy xuất (Retrieval Evaluation).

Sử dụng bộ dữ liệu đánh giá trong data/eval_data/retrieval_eval_data/
để đo lường hiệu quả của Retriever qua các metrics chuẩn IR:
  - Precision@K, Recall@K (K = 1, 3, 5, 10, 20)
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)
  - MAP (Mean Average Precision)
  - Hit Rate@K

Hỗ trợ đánh giá đa cấp độ (Multi-Level Retrieval Evaluation):
  - Document-level (--level doc): Đánh giá tìm đúng tài liệu chứa thông tin (document_id).
  - Chunk-level (--level chunk): Đánh giá tìm đúng chính xác đoạn cắt chứa câu trả lời (chunk_id).
  - Song song cả hai (--level both): Đánh giá và so sánh đồng thời Document và Chunk.

Cách sử dụng CLI:
    python main.py --eval --retrieval
    python main.py --eval --retrieval --level chunk
    python src/evaluation/evaluate_retrieval.py --mode mock --level both --top_k 5
    python src/evaluation/evaluate_retrieval.py --mode live --level both --top_k 5
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
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("retrieval_evaluation")


class RetrievalEvaluator:
    """
    Bộ đánh giá hiệu quả Retrieval (IR Evaluator) cho UET RAG Chatbot.
    Tải queries, qrels mức doc và qrels mức chunk từ tập benchmark để tính các chỉ số IR chuẩn.
    """

    def __init__(self, eval_data_dir: Optional[str | Path] = None):
        """
        :param eval_data_dir: Thư mục chứa retrieval_eval_queries.json, retrieval_eval_qrels.json,
                              và retrieval_eval_qrels_chunk.json.
                              Mặc định: data/eval_data/retrieval_eval_data/
        """
        if eval_data_dir is None:
            self.eval_data_dir = PROJECT_ROOT / "data" / "eval_data" / "retrieval_eval_data"
        else:
            self.eval_data_dir = Path(eval_data_dir)

        self.queries: List[Dict[str, Any]] = []
        self.qrels_doc: Dict[str, Dict[str, int]] = {}
        self.qrels_chunk: Dict[str, Dict[str, int]] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}  # alias backwards compatibility
        self._load_data()

    def _load_data(self):
        """Tải queries, qrels doc và qrels chunk từ file JSON."""
        queries_path = self.eval_data_dir / "retrieval_eval_queries.json"
        qrels_doc_path = self.eval_data_dir / "retrieval_eval_qrels.json"
        qrels_chunk_path = self.eval_data_dir / "retrieval_eval_qrels_chunk.json"

        if not queries_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file queries: {queries_path}")
        if not qrels_doc_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file qrels doc: {qrels_doc_path}")

        with open(queries_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        with open(qrels_doc_path, "r", encoding="utf-8") as f:
            self.qrels_doc = json.load(f)
            self.qrels = self.qrels_doc

        if qrels_chunk_path.exists():
            with open(qrels_chunk_path, "r", encoding="utf-8") as f:
                self.qrels_chunk = json.load(f)
        else:
            self.qrels_chunk = {}
            logger.warning(f"Không tìm thấy {qrels_chunk_path}, bỏ qua đánh giá mức chunk.")

        logger.info(
            f"Đã tải {len(self.queries)} queries, {len(self.qrels_doc)} qrels doc và "
            f"{len(self.qrels_chunk)} qrels chunk từ {self.eval_data_dir}"
        )

    def get_relevant_ids(self, query_id: str, level: str = "doc") -> Set[str]:
        """Lấy tập hợp tất cả IDs liên quan cho một query (mức doc hoặc chunk)."""
        if level == "chunk":
            return set(self.qrels_chunk.get(query_id, {}).keys())
        return set(self.qrels_doc.get(query_id, {}).keys())

    def evaluate_single_query(
        self,
        query_id: str,
        retrieved_ids: List[str],
        qrels: Optional[Dict[str, int]] = None,
        k_values: List[int] = [1, 3, 5, 10, 20],
        is_out_of_domain: bool = False,
        refused_to_answer: bool = True,
    ) -> Dict[str, Any]:
        """
        Đánh giá kết quả retrieval cho một câu hỏi duy nhất (hỗ trợ cả doc_id và chunk_id).

        :param query_id: ID câu hỏi
        :param retrieved_ids: Danh sách IDs trả về bởi retriever (theo thứ tự rank)
        :param qrels: Ground truth relevance mapping {id: score} (0, 1, 2)
        :param k_values: Các ngưỡng K để tính Precision@K, Recall@K, etc.
        :param is_out_of_domain: Câu hỏi ngoài phạm vi UET (không có tài liệu ground truth)
        :param refused_to_answer: Hệ thống có từ chối trả lời đúng chuẩn hay không
        :return: Dict chứa tất cả metrics cho query này
        """
        if qrels is None:
            qrels = self.qrels_doc.get(query_id, {})

        relevant_ids = set(qrels.keys())
        query_qrels = qrels

        result: Dict[str, Any] = {
            "query_id": query_id,
            "num_retrieved": len(retrieved_ids),
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
            result[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant_ids, k), 4)
            result[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant_ids, k), 4)
            result[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, query_qrels, k), 4)
            result[f"hit_rate@{k}"] = round(hit_rate_at_k(retrieved_ids, relevant_ids, k), 4)

        # MRR và MAP
        result["mrr"] = round(mrr(retrieved_ids, relevant_ids), 4)
        result["map"] = round(average_precision(retrieved_ids, relevant_ids), 4)

        return result

    def evaluate_batch(
        self,
        retrieval_results: Dict[str, List[str]],
        k_values: List[int] = [1, 3, 5, 10, 20],
        level: str = "doc",
    ) -> Dict[str, Any]:
        """
        Đánh giá toàn bộ tập câu hỏi theo cấp độ chỉ định (doc hoặc chunk).

        :param retrieval_results: Dict {query_id: [retrieved_ids]}
        :param k_values: Các ngưỡng K
        :param level: "doc" hoặc "chunk"
        :return: Dict chứa chi tiết per-query results và aggregated metrics
        """
        target_qrels = self.qrels_chunk if level == "chunk" else self.qrels_doc
        per_query_results = []

        for query in self.queries:
            qid = query["query_id"]
            retrieved = retrieval_results.get(qid, [])
            q_types = query.get("query_types") or ([query.get("query_type")] if query.get("query_type") else [])
            q_qrels = target_qrels.get(qid, {})
            is_ood = "out_of_domain" in q_types or (len(q_qrels) == 0)
            res = self.evaluate_single_query(
                qid, retrieved, qrels=q_qrels, k_values=k_values, is_out_of_domain=is_ood, refused_to_answer=True
            )
            res["query_text"] = query.get("query_text", "")
            res["difficulty"] = query.get("difficulty", "unknown")
            res["query_types"] = q_types
            per_query_results.append(res)

        # Tách riêng OOD queries ra khỏi trung bình chính
        in_domain_results = [
            r for r in per_query_results
            if not any(q["query_id"] == r["query_id"] and ("out_of_domain" in (q.get("query_types") or []))
                       for q in self.queries)
        ]

        aggregated: Dict[str, float] = {}
        metric_keys: List[str] = []
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
                "level": level,
            },
            "level": level,
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
        level: str = "both",
    ) -> Dict[str, Any]:
        """
        Đánh giá trực tiếp một instance Retriever (kế thừa BaseRetriever).
        
        :param retriever: Đối tượng UETRetriever hoặc BaseRetriever
        :param top_k: Số lượng documents/chunks tối đa cần retrieve
        :param k_values: Các ngưỡng K (mặc định: chỉ test với đúng [top_k])
        :param dedup_documents: Khử trùng lặp ở cấp độ Document ID (áp dụng cho doc_level)
        :param level: "doc", "chunk", hoặc "both"
        """
        if k_values is None:
            k_values = [top_k]
        from src.base import UserQuery

        doc_retrieval_results: Dict[str, List[str]] = {}
        chunk_retrieval_results: Dict[str, List[str]] = {}
        query_top_k = max(top_k * 4, 30) if dedup_documents else top_k

        cache_file = self.eval_data_dir / f".retrieval_cache_multilevel_top{top_k}.json"
        cached: Dict[str, Dict[str, List[str]]] = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
            except Exception:
                cached = {}

        needs_save = False
        for i, query in enumerate(self.queries):
            qid = query["query_id"]
            if qid in cached and "docs" in cached[qid] and "chunks" in cached[qid]:
                if len(cached[qid]["docs"]) >= top_k and len(cached[qid]["chunks"]) >= top_k:
                    doc_retrieval_results[qid] = cached[qid]["docs"][:top_k]
                    chunk_retrieval_results[qid] = cached[qid]["chunks"][:top_k]
                    continue

            user_query = UserQuery(query_text=query["query_text"])
            try:
                contexts = retriever.retrieve(user_query, top_k=query_top_k)

                # 1. Trích xuất doc_ids (có khử trùng nếu dedup_documents)
                retrieved_doc_ids = []
                seen_doc_ids = set()
                for c in contexts:
                    doc_id = None
                    if hasattr(c, "metadata") and isinstance(c.metadata, dict):
                        doc_id = c.metadata.get("document_id") or c.metadata.get("doc_id")
                    elif isinstance(c, dict) and isinstance(c.get("metadata"), dict):
                        doc_id = c["metadata"].get("document_id") or c["metadata"].get("doc_id")
                    elif isinstance(c, dict):
                        doc_id = c.get("document_id") or c.get("doc_id")
                    if not doc_id:
                        doc_id = getattr(c, "chunk_id", None) or getattr(c, "id", None)

                    if doc_id:
                        doc_id_str = str(doc_id)
                        if dedup_documents:
                            if doc_id_str not in seen_doc_ids:
                                seen_doc_ids.add(doc_id_str)
                                retrieved_doc_ids.append(doc_id_str)
                                if len(retrieved_doc_ids) >= top_k:
                                    break
                        else:
                            retrieved_doc_ids.append(doc_id_str)
                            if len(retrieved_doc_ids) >= top_k:
                                break

                # 2. Trích xuất chunk_ids (bảo toàn thứ hạng rank, không dedup theo doc)
                retrieved_chunk_ids = []
                seen_chunk_ids = set()
                for c in contexts:
                    chunk_id = getattr(c, "chunk_id", None)
                    if not chunk_id and hasattr(c, "metadata") and isinstance(c.metadata, dict):
                        chunk_id = c.metadata.get("chunk_id") or c.metadata.get("id")
                    if not chunk_id and hasattr(c, "id"):
                        chunk_id = c.id
                    if not chunk_id and isinstance(c, dict):
                        chunk_id = (
                            c.get("chunk_id")
                            or c.get("id")
                            or (c.get("metadata", {}).get("chunk_id") if isinstance(c.get("metadata"), dict) else None)
                        )

                    if chunk_id:
                        chunk_id_str = str(chunk_id)
                        if chunk_id_str not in seen_chunk_ids:
                            seen_chunk_ids.add(chunk_id_str)
                            retrieved_chunk_ids.append(chunk_id_str)
                            if len(retrieved_chunk_ids) >= top_k:
                                break

                doc_retrieval_results[qid] = retrieved_doc_ids
                chunk_retrieval_results[qid] = retrieved_chunk_ids
                cached[qid] = {"docs": retrieved_doc_ids, "chunks": retrieved_chunk_ids}
                needs_save = True
            except Exception as e:
                logger.error(f"Lỗi khi retrieve cho query {qid}: {e}")
                doc_retrieval_results[qid] = []
                chunk_retrieval_results[qid] = []

            if (i + 1) % 5 == 0 or (i + 1) == len(self.queries):
                logger.info(f"Đã retrieve {i + 1}/{len(self.queries)} queries...")
                if needs_save:
                    try:
                        with open(cache_file, "w", encoding="utf-8") as cf:
                            json.dump(cached, cf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

        if needs_save:
            try:
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump(cached, cf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if level == "doc":
            return self.evaluate_batch(doc_retrieval_results, k_values=k_values, level="doc")
        elif level == "chunk":
            return self.evaluate_batch(chunk_retrieval_results, k_values=k_values, level="chunk")
        else:
            # both
            doc_res = self.evaluate_batch(doc_retrieval_results, k_values=k_values, level="doc")
            chunk_res = self.evaluate_batch(chunk_retrieval_results, k_values=k_values, level="chunk")
            return {
                "summary": doc_res["summary"],
                "level": "both",
                "doc_level": doc_res,
                "chunk_level": chunk_res,
                # Aliases for backwards compatibility:
                "aggregated_metrics": doc_res["aggregated_metrics"],
                "by_difficulty": doc_res["by_difficulty"],
                "by_query_type": doc_res["by_query_type"],
                "per_query_results": doc_res["per_query_results"],
            }


# ============================================================================
# HELPER IN BẢNG BÁO CÁO (CLI DISPLAY)
# ============================================================================

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
        return

    col_w = 17
    line_len = 14 + len(matrix_types) * col_w
    sep = "=" * line_len

    print("\n" + sep)
    print(f"📊 {title} @ K={target_k}")
    print(sep)

    col_headers = [f"{t}({by_type[t].get('count', 0)})" for t in matrix_types]
    header_line = f"{'':<14}" + "".join([f"{ch:<{col_w}}" for ch in col_headers])
    print(header_line)
    print("-" * line_len)

    # Precision & Recall
    for metric_label, metric_key in [("Precision", "precision"), ("Recall", "recall")]:
        row_str = f"{metric_label:<14}"
        for t in matrix_types:
            val = by_type[t].get(f"mean_{metric_key}@{target_k}", 0.0)
            row_str += f"{val:<{col_w}.4f}"
        print(row_str)

    print("-" * line_len)
    # MRR & MAP
    for metric_label, metric_key in [("MRR", "mean_mrr"), ("MAP", "mean_map")]:
        row_str = f"{metric_label:<14}"
        for t in matrix_types:
            val = by_type[t].get(metric_key, 0.0)
            row_str += f"{val:<{col_w}.4f}"
        print(row_str)

    print("-" * line_len)
    # NDCG & Hit Rate
    for metric_label, metric_key in [("NDCG", "ndcg"), ("Hit Rate", "hit_rate")]:
        row_str = f"{metric_label:<14}"
        for t in matrix_types:
            val = by_type[t].get(f"mean_{metric_key}@{target_k}", 0.0)
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
):
    """
    Chạy đánh giá hiệu quả phân hệ Retrieval (Precision, Recall, MRR, NDCG) và hiển thị trên CLI.
    Hỗ trợ level = "doc", "chunk", hoặc "both".
    Hỗ trợ mode = "live" (chạy thật) hoặc "mock" (chạy nhanh với ground-truth).
    """
    cfg = load_yaml_config()
    target_k = top_k if top_k is not None else cfg.get("retrieval", {}).get("final_top_k", 5)

    if k_values is None:
        k_values = [target_k]

    if eval_dir is None:
        eval_dir = PROJECT_ROOT / "data" / "eval_data" / "retrieval_eval_data"
    else:
        eval_dir = Path(eval_dir)

    if not (eval_dir / "retrieval_eval_queries.json").exists():
        print(f"❌ Không tìm thấy dữ liệu đánh giá tại: {eval_dir}")
        return

    level_display = {
        "doc": "TÀI LIỆU (DOCUMENT-LEVEL)",
        "chunk": "ĐOẠN CẮT (CHUNK-LEVEL)",
        "both": "SONG SONG CẢ TÀI LIỆU VÀ ĐOẠN CẮT (DOC & CHUNK DUAL-LEVEL)",
    }.get(level, level.upper())

    print("\n" + "=" * 65)
    print("🔬 ĐANG CHẠY ĐÁNH GIÁ PHÂN HỆ TRUY XUẤT (RETRIEVAL EVALUATION)")
    print(f"• Chế độ (Mode):      {mode.upper()}")
    print(f"• Cấu hình K đánh giá: K={target_k} (Lấy từ config.yaml: retrieval.final_top_k)")
    print(f"• Cấp độ đánh giá:    {level_display}")
    print("=" * 65)

    evaluator = RetrievalEvaluator(eval_data_dir=eval_dir)

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
            retriever, top_k=target_k, k_values=k_values, dedup_documents=dedup, level=level
        )

    # Đọc ordered query types
    meta_file = eval_dir / "dataset_metadata.json"
    ordered_types = []
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                ordered_types = list(json.load(mf).get("query_types", {}).keys())
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(f"📈 BÁO CÁO KẾT QUẢ PHÂN HỆ RETRIEVAL (TOP-K = {target_k})")
    print("=" * 65)
    summary = eval_result.get("summary", {})
    print(f"Tổng số queries đánh giá: {summary.get('in_domain_queries', 0)} in-domain queries")

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
    elif level == "chunk":
        chunk_res = eval_result
        print_metrics_table(chunk_res, k_values, "🎯 ĐÁNH GIÁ MỨC ĐOẠN CẮT (CHUNK-LEVEL)")
        print("=" * 65 + "\n")
        print_query_type_matrix(
            chunk_res, target_k, ordered_types, "MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI - MỨC ĐOẠN CẮT (CHUNK-LEVEL)"
        )
    else:  # doc
        doc_res = eval_result
        print_metrics_table(doc_res, k_values, "📄 ĐÁNH GIÁ MỨC TÀI LIỆU (DOCUMENT-LEVEL)")
        print("=" * 65 + "\n")
        print_query_type_matrix(
            doc_res, target_k, ordered_types, "MA TRẬN ĐÁNH GIÁ THEO DẠNG CÂU HỎI - MỨC TÀI LIỆU (DOC-LEVEL)"
        )

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu kết quả chi tiết: {output_path}\n")


# ============================================================================
# CLI RUNNER DIRECT SCRIPT
# ============================================================================

def main():
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    cfg = load_yaml_config()
    default_k = cfg.get("retrieval", {}).get("final_top_k", 5)

    parser = argparse.ArgumentParser(description="Đánh giá hiệu quả Retrieval cho UET RAG Chatbot")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "eval_data" / "retrieval_eval_data"),
        help="Thư mục chứa dữ liệu đánh giá (mặc định: data/eval_data/retrieval_eval_data)",
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
    args = parser.parse_args()

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

    if args.mode == "mock":
        print("📝 Đang chạy mock evaluation (giả lập retriever với ground truth)...")
        doc_mock_results: Dict[str, List[str]] = {}
        chunk_mock_results: Dict[str, List[str]] = {}

        for query in evaluator.queries:
            qid = query["query_id"]
            # Doc mock results
            seen_docs = set()
            rel_docs = []
            for did in query.get("highly_relevant_doc_ids", []) + query.get("relevant_doc_ids", []):
                if did not in seen_docs:
                    seen_docs.add(did)
                    rel_docs.append(did)
            doc_mock_results[qid] = rel_docs[:eval_k]

            # Chunk mock results
            seen_chunks = set()
            rel_chunks = []
            for cid in query.get("highly_relevant_chunk_ids", []) + query.get("relevant_chunk_ids", []):
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    rel_chunks.append(cid)
            chunk_mock_results[qid] = rel_chunks[:eval_k]

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
                retriever, top_k=eval_k, k_values=k_values, dedup_documents=not args.no_dedup, level=args.level
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
    elif args.level == "chunk":
        _print_level_stats(eval_result, "MỨC ĐOẠN CẮT (CHUNK-LEVEL)")
    else:
        _print_level_stats(eval_result, "MỨC TÀI LIỆU (DOCUMENT-LEVEL)")

    # Ghi kết quả
    output_path = args.output or str(
        Path(args.eval_dir) / f"eval_results_{args.mode}_{args.level}_top{eval_k}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã lưu kết quả chi tiết: {output_path}")
    print(f"\n✅ HOÀN TẤT ĐÁNH GIÁ RETRIEVAL!\n")


if __name__ == "__main__":
    main()

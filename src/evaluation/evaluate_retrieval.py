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
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Đảm bảo root project có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
        :param eval_data_dir: Thư mục chứa tập đánh giá (uet_rag_eval_100 hoặc retrieval_eval_data).
                              Hỗ trợ định dạng thống nhất (e2e_eval_100.json) hoặc bộ 3 file cũ.
        """
        if eval_data_dir is None:
            cfg = load_yaml_config()
            cfg_file = cfg.get("paths", {}).get("eval_dataset_file")
            if cfg_file:
                target_path = PROJECT_ROOT / cfg_file if not Path(cfg_file).is_absolute() else Path(cfg_file)
                self.eval_data_dir = target_path
            else:
                default_new = PROJECT_ROOT / "data" / "eval_data" / "uet_rag_eval_100"
                if default_new.exists():
                    self.eval_data_dir = default_new
                else:
                    self.eval_data_dir = PROJECT_ROOT / "data" / "eval_data" / "e2e_eval_100_annotated.json"
        else:
            self.eval_data_dir = Path(eval_data_dir)

        self.queries: List[Dict[str, Any]] = []
        self.qrels_doc: Dict[str, Dict[str, int]] = {}
        self.qrels_chunk: Dict[str, Dict[str, int]] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}  # alias backwards compatibility
        self._load_data()

    def filter_by_domain(self, domain: Optional[str] = None):
        """Lọc danh sách queries theo khoa/đơn vị (unit/domain)."""
        if not domain or domain.lower().strip() == "all":
            return
        dom = domain.upper().strip()
        matched = [q for q in self.queries if q.get("unit", "").upper() == dom]
        if not matched:
            matched = [q for q in self.queries if dom in q.get("unit", "").upper()]
        if matched:
            self.queries = matched
            logger.info(f"Đã lọc danh sách queries theo domain '{dom}': còn {len(self.queries)} câu hỏi.")
        else:
            logger.warning(f"Không tìm thấy query nào thuộc domain '{dom}'. Giữ nguyên danh sách gốc.")

    def _load_data(self):
        """Tải queries, qrels doc và qrels chunk từ tập dữ liệu unified hoặc các file JSON chuẩn."""
        # 1. Kiểm tra trường hợp file thống nhất (Unified Dataset: e2e_eval_100.json/.jsonl)
        unified_file = None
        if self.eval_data_dir.is_file():
            unified_file = self.eval_data_dir
            self.eval_data_dir = self.eval_data_dir.parent
        else:
            for candidate in ["e2e_eval_100_annotated.json", "e2e_eval_100.json", "e2e_eval_100.jsonl"]:
                candidate_path = self.eval_data_dir / candidate
                if candidate_path.exists():
                    unified_file = candidate_path
                    break

        if unified_file is not None:
            items = []
            if unified_file.suffix == ".jsonl":
                with open(unified_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
            else:
                with open(unified_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]

            self.queries = []
            self.qrels_doc = {}
            self.qrels_chunk = {}

            for item in items:
                qid = str(item.get("id") or item.get("query_id"))
                qtext = item.get("query") or item.get("query_text", "")
                qtype = item.get("type") or item.get("query_type", "factoid")
                qtypes = item.get("query_types") or [qtype]
                unit = item.get("unit", "")
                category = item.get("category", "")

                # Doc relevance
                target_docs = item.get("target_document_ids") or []
                if isinstance(target_docs, list):
                    self.qrels_doc[qid] = {did: 2 for did in target_docs}
                elif isinstance(target_docs, dict):
                    self.qrels_doc[qid] = {str(k): int(v) for k, v in target_docs.items()}
                else:
                    self.qrels_doc[qid] = {}

                # Chunk relevance
                target_chunks = item.get("target_chunk_ids") or {}
                if isinstance(target_chunks, list):
                    self.qrels_chunk[qid] = {cid: 2 for cid in target_chunks}
                elif isinstance(target_chunks, dict):
                    self.qrels_chunk[qid] = {str(k): int(v) for k, v in target_chunks.items()}
                else:
                    self.qrels_chunk[qid] = {}

                self.queries.append({
                    "query_id": qid,
                    "query_text": qtext,
                    "query_types": qtypes,
                    "type": qtype,
                    "unit": unit,
                    "category": category,
                    "expected_unit": unit,
                    "expected_category": category,
                    "ground_truth": item.get("ground_truth"),
                    "highly_relevant_doc_ids": [did for did, g in self.qrels_doc[qid].items() if g >= 2],
                    "relevant_doc_ids": [did for did, g in self.qrels_doc[qid].items() if g == 1],
                    "highly_relevant_chunk_ids": [cid for cid, g in self.qrels_chunk[qid].items() if g >= 2],
                    "relevant_chunk_ids": [cid for cid, g in self.qrels_chunk[qid].items() if g == 1],
                })

            self.qrels = self.qrels_doc
            logger.info(
                f"Đã tải {len(self.queries)} queries, {len(self.qrels_doc)} qrels doc và "
                f"{len(self.qrels_chunk)} qrels chunk từ unified dataset: {unified_file.name}"
            )
            return

        # 2. Fallback: Đọc 3 file JSON theo định dạng cũ
        queries_path = self.eval_data_dir / "retrieval_eval_queries.json"
        qrels_doc_path = self.eval_data_dir / "retrieval_eval_qrels.json"
        qrels_chunk_path = self.eval_data_dir / "retrieval_eval_qrels_chunk.json"

        if not queries_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file queries hoặc unified dataset tại: {self.eval_data_dir}")
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
        refused_to_answer: Optional[bool] = None,
        top1_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Đánh giá kết quả retrieval cho một câu hỏi duy nhất (hỗ trợ cả doc_id và chunk_id).

        :param query_id: ID câu hỏi
        :param retrieved_ids: Danh sách IDs trả về bởi retriever (theo thứ tự rank)
        :param qrels: Ground truth relevance mapping {id: score} (0, 1, 2)
        :param k_values: Các ngưỡng K để tính Precision@K, Recall@K, etc.
        :param is_out_of_domain: Câu hỏi ngoài phạm vi UET (không có tài liệu ground truth)
        :param refused_to_answer: Chỉ truyền ở tầng E2E nếu muốn chấm trực tiếp OOD refusal;
                                  ở tầng Retrieval thuần (None) sẽ ghi nhận top1_score thay vì giả lập 1.0.
        :param top1_score: Điểm similarity/rerank của ứng viên rank #1 để hiệu chỉnh score_threshold cho OOD.
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
            "top1_score": round(float(top1_score), 4) if top1_score is not None else None,
        }

        # Câu hỏi out_of_domain không có tài liệu vàng để tính IR metrics:
        # Không hardcode 1.0 ở tầng Retrieval thuần (chỉ tính nếu caller truyền refused_to_answer rõ ràng).
        if is_out_of_domain:
            if refused_to_answer is not None:
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
        top1_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Đánh giá toàn bộ tập câu hỏi theo cấp độ chỉ định (doc hoặc chunk).

        :param retrieval_results: Dict {query_id: [retrieved_ids]}
        :param k_values: Các ngưỡng K
        :param level: "doc" hoặc "chunk"
        :param top1_scores: Dict {query_id: top1_similarity_score} (tùy chọn, dùng cho hiệu chỉnh OOD)
        :return: Dict chứa chi tiết per-query results, aggregated metrics và ood_stats
        """
        target_qrels = self.qrels_chunk if level == "chunk" else self.qrels_doc
        per_query_results = []

        for query in self.queries:
            qid = query["query_id"]
            retrieved = retrieval_results.get(qid, [])
            q_types = query.get("query_types") or (
                [query.get("type") or query.get("query_type")]
                if (query.get("type") or query.get("query_type"))
                else []
            )
            q_qrels = target_qrels.get(qid, {})
            is_ood = "out_of_domain" in q_types or (len(q_qrels) == 0)
            res = self.evaluate_single_query(
                qid,
                retrieved,
                qrels=q_qrels,
                k_values=k_values,
                is_out_of_domain=is_ood,
                refused_to_answer=None,
                top1_score=(top1_scores or {}).get(qid),
            )
            res["query_text"] = query.get("query_text", "")
            res["difficulty"] = query.get("difficulty", "unknown")
            res["query_types"] = q_types
            per_query_results.append(res)

        # Tách riêng OOD queries ra khỏi trung bình chính (tránh lẫn lộn vào IR metrics)
        in_domain_results = [r for r in per_query_results if not r.get("is_out_of_domain")]
        ood_results = [r for r in per_query_results if r.get("is_out_of_domain")]

        aggregated: Dict[str, float] = {}
        metric_keys: List[str] = []
        if in_domain_results:
            metric_keys = [
                k for k in in_domain_results[0].keys()
                if isinstance(in_domain_results[0][k], (int, float))
                and not isinstance(in_domain_results[0][k], bool)
                and k not in ("num_retrieved", "num_relevant_in_ground_truth", "top1_score")
            ]
            for key in metric_keys:
                vals = [r[key] for r in in_domain_results if key in r and r[key] is not None]
                aggregated[f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        # Thống kê riêng OOD để theo dõi ngưỡng score_threshold (không hardcode 1.0)
        ood_scores = [r["top1_score"] for r in ood_results if isinstance(r.get("top1_score"), (int, float))]
        in_scores = [r["top1_score"] for r in in_domain_results if isinstance(r.get("top1_score"), (int, float))]
        ood_avg = round(sum(ood_scores) / len(ood_scores), 4) if ood_scores else None
        in_avg = round(sum(in_scores) / len(in_scores), 4) if in_scores else None
        ood_stats = {
            "ood_queries": len(ood_results),
            "ood_avg_top1_score": ood_avg,
            "in_domain_avg_top1_score": in_avg,
            "score_gap": round(in_avg - ood_avg, 4) if (in_avg is not None and ood_avg is not None) else None,
            "suggested_threshold_midpoint": (
                round((in_avg + ood_avg) / 2, 4) if (in_avg is not None and ood_avg is not None) else None
            ),
        }

        # Phân tích theo difficulty (easy, medium, hard)
        difficulty_metrics: Dict[str, Dict[str, float]] = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in in_domain_results if r.get("difficulty") == diff]
            if diff_results:
                difficulty_metrics[diff] = {}
                for key in metric_keys:
                    vals = [r[key] for r in diff_results if key in r]
                    difficulty_metrics[diff][f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        # Phân tích theo toàn bộ query_types (chỉ trên các nhóm in-domain có IR metrics)
        meta_file = self.eval_data_dir / "dataset_metadata.json"
        target_types = []
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    target_types = list(json.load(mf).get("query_types", {}).keys())
            except Exception:
                pass
        if not target_types:
            target_types = ["factoid", "procedure", "list", "comparison", "summary", "informal", "multi_hop", "aggregation"]

        type_metrics: Dict[str, Dict[str, Any]] = {}
        for qt in target_types:
            if qt == "out_of_domain":
                continue
            qt_results = [
                r for q, r in zip(self.queries, per_query_results)
                if not r.get("is_out_of_domain")
                and qt in (q.get("query_types") or ([q.get("type") or q.get("query_type")] if (q.get("type") or q.get("query_type")) else []))
            ]
            if qt_results:
                type_metrics[qt] = {"count": len(qt_results)}
                for key in metric_keys:
                    vals = [r[key] for r in qt_results if key in r]
                    type_metrics[qt][f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        # Phân tích theo Unit (Khoa/Đơn vị) - chỉ in-domain
        unit_metrics: Dict[str, Dict[str, Any]] = {}
        units_present = []
        for q in self.queries:
            u = q.get("unit")
            if u and u != "out_of_domain" and u not in units_present:
                units_present.append(u)
        for u in units_present:
            u_results = [
                r for q, r in zip(self.queries, per_query_results)
                if not r.get("is_out_of_domain") and q.get("unit") == u
            ]
            if u_results:
                unit_metrics[u] = {"count": len(u_results)}
                for key in metric_keys:
                    vals = [r[key] for r in u_results if key in r]
                    unit_metrics[u][f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        # Phân tích theo Category (Lĩnh vực) - chỉ in-domain
        cat_metrics: Dict[str, Dict[str, Any]] = {}
        cats_present = []
        for q in self.queries:
            c = q.get("category")
            if c and c not in cats_present:
                cats_present.append(c)
        for c in cats_present:
            c_results = [
                r for q, r in zip(self.queries, per_query_results)
                if not r.get("is_out_of_domain") and q.get("category") == c
            ]
            if c_results:
                cat_metrics[c] = {"count": len(c_results)}
                for key in metric_keys:
                    vals = [r[key] for r in c_results if key in r]
                    cat_metrics[c][f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "summary": {
                "total_queries": len(self.queries),
                "evaluated_queries": len(retrieval_results),
                "in_domain_queries": len(in_domain_results),
                "ood_queries": len(ood_results),
                "level": level,
            },
            "level": level,
            "aggregated_metrics": aggregated,
            "ood_stats": ood_stats,
            "by_difficulty": difficulty_metrics,
            "by_query_type": type_metrics,
            "by_unit": unit_metrics,
            "by_category": cat_metrics,
            "per_query_results": per_query_results,
        }

    @staticmethod
    def _build_cache_signature(
        retriever: Any,
        top_k: int,
        dedup_documents: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Tạo chữ ký cấu hình đầy đủ cho cache Retrieval (tránh bẫy dùng nhầm cache khi đổi MMR/RRF/Graph/Index).
        """
        reranker = getattr(retriever, "reranker", None)
        reranker_cls = reranker.__class__.__name__ if reranker is not None else "none"
        reranker_model = (
            getattr(reranker, "model_name", None)
            or getattr(reranker, "onnx_filename", None)
            or reranker_cls
        )

        def _safe_count(obj: Any) -> Optional[int]:
            if obj is None or not hasattr(obj, "count"):
                return None
            try:
                val = obj.count()
                return int(val) if isinstance(val, (int, float)) else None
            except Exception:
                return None

        sig_dict: Dict[str, Any] = {
            "reranker_class": reranker_cls,
            "reranker_model": str(reranker_model),
            "enable_mmr": getattr(retriever, "enable_mmr", None),
            "mmr_lambda": getattr(retriever, "mmr_lambda", None),
            "mmr_top_n": getattr(retriever, "mmr_top_n", None),
            "enable_graph": getattr(retriever, "enable_graph", None),
            "graph_weight": getattr(retriever, "graph_weight", None),
            "graph_top_k": getattr(retriever, "graph_top_k", None),
            "graph_max_hops": getattr(retriever, "graph_max_hops", None),
            "graph_seed_top_m": getattr(retriever, "graph_seed_top_m", None),
            "bm25_weight": getattr(retriever, "bm25_weight", None),
            "vector_weight": getattr(retriever, "vector_weight", None),
            "rrf_k": getattr(retriever, "rrf_k", None),
            "bm25_top_k": getattr(retriever, "bm25_top_k", None),
            "vector_top_k": getattr(retriever, "vector_top_k", None),
            "candidate_pool_size": getattr(retriever, "candidate_pool_size", None),
            "rerank_top_n": getattr(retriever, "rerank_top_n", None),
            "score_threshold": getattr(retriever, "default_score_threshold", None),
            "top_k": top_k,
            "dedup_documents": dedup_documents,
            "vector_count": _safe_count(getattr(retriever, "vector_store", None)),
            "bm25_count": _safe_count(getattr(retriever, "bm25_index", None)),
            "graph_count": _safe_count(getattr(retriever, "graph_index", None)),
        }
        canonical = json.dumps(sig_dict, sort_keys=True, ensure_ascii=False, default=str)
        config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        sig_dict["config_hash"] = config_hash
        return config_hash, sig_dict

    def evaluate_retriever(
        self,
        retriever: Any,
        top_k: int = 5,
        k_values: Optional[List[int]] = None,
        dedup_documents: bool = True,
        level: str = "both",
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Đánh giá trực tiếp một instance Retriever (kế thừa BaseRetriever).

        :param retriever: Đối tượng UETRetriever hoặc BaseRetriever
        :param top_k: Số lượng documents/chunks tối đa cần retrieve
        :param k_values: Các ngưỡng K (mặc định: chỉ test với đúng [top_k])
        :param dedup_documents: Khử trùng lặp ở cấp độ Document ID (áp dụng cho doc_level)
        :param level: "doc", "chunk", hoặc "both"
        :param use_cache: Mặc định False (tắt cache để tránh đo nhầm kết quả cũ khi sửa config)
        """
        if k_values is None:
            k_values = [top_k]
        from src.base import UserQuery

        doc_retrieval_results: Dict[str, List[str]] = {}
        chunk_retrieval_results: Dict[str, List[str]] = {}
        top1_scores: Dict[str, float] = {}
        query_top_k = max(top_k * 4, 30) if dedup_documents else top_k

        reranker_name = "default"
        if hasattr(retriever, "reranker") and retriever.reranker is not None:
            reranker_name = getattr(retriever.reranker, "model_name", retriever.reranker.__class__.__name__)
            reranker_name = re.sub(r"[^\w\-]", "_", str(reranker_name))

        config_hash, config_meta = self._build_cache_signature(
            retriever=retriever,
            top_k=top_k,
            dedup_documents=dedup_documents,
        )
        cache_file = self.eval_data_dir / f".retrieval_cache_{reranker_name}_top{top_k}_{config_hash}.json"
        cached_entries: Dict[str, Dict[str, Any]] = {}

        if use_cache:
            logger.warning(
                "[CACHE ENABLED] RetrievalEvaluator đang dùng cache '%s' (config_hash=%s). "
                "Dùng --no-cache hoặc use_cache=False để chạy live toàn bộ.",
                cache_file.name,
                config_hash,
            )
            print(f"⚠️  [CACHE ENABLED] Đọc/ghi cache: {cache_file.name} (hash={config_hash})")
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as cf:
                        raw_cache = json.load(cf)
                    if isinstance(raw_cache, dict) and raw_cache.get("_meta", {}).get("config_hash") == config_hash:
                        cached_entries = raw_cache.get("entries", {})
                except Exception:
                    cached_entries = {}

        needs_save = False
        for i, query in enumerate(self.queries):
            qid = query["query_id"]
            if (
                use_cache
                and qid in cached_entries
                and "docs" in cached_entries[qid]
                and "chunks" in cached_entries[qid]
                and len(cached_entries[qid]["docs"]) >= min(top_k, 1)
                and len(cached_entries[qid]["chunks"]) >= min(top_k, 1)
            ):
                doc_retrieval_results[qid] = cached_entries[qid]["docs"][:top_k]
                chunk_retrieval_results[qid] = cached_entries[qid]["chunks"][:top_k]
                if cached_entries[qid].get("top1_score") is not None:
                    top1_scores[qid] = float(cached_entries[qid]["top1_score"])
                continue

            user_query = UserQuery(query_text=query["query_text"])
            try:
                contexts = retriever.retrieve(user_query, top_k=query_top_k)
                if contexts:
                    top1_val = getattr(contexts[0], "similarity_score", None)
                    if top1_val is None and isinstance(contexts[0], dict):
                        top1_val = contexts[0].get("similarity_score", contexts[0].get("score"))
                    if top1_val is not None:
                        top1_scores[qid] = float(top1_val)

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
                if use_cache:
                    cached_entries[qid] = {
                        "docs": retrieved_doc_ids,
                        "chunks": retrieved_chunk_ids,
                        "top1_score": top1_scores.get(qid),
                    }
                    needs_save = True
            except Exception as e:
                logger.error(f"Lỗi khi retrieve cho query {qid}: {e}")
                doc_retrieval_results[qid] = []
                chunk_retrieval_results[qid] = []

            if (i + 1) % 5 == 0 or (i + 1) == len(self.queries):
                logger.info(f"Đã retrieve {i + 1}/{len(self.queries)} queries...")
                if use_cache and needs_save:
                    try:
                        with open(cache_file, "w", encoding="utf-8") as cf:
                            json.dump({"_meta": config_meta, "entries": cached_entries}, cf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

        if use_cache and needs_save:
            try:
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump({"_meta": config_meta, "entries": cached_entries}, cf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if level == "doc":
            res = self.evaluate_batch(doc_retrieval_results, k_values=k_values, level="doc", top1_scores=top1_scores)
            res["cache_config"] = config_meta
            return res
        elif level == "chunk":
            res = self.evaluate_batch(chunk_retrieval_results, k_values=k_values, level="chunk", top1_scores=top1_scores)
            res["cache_config"] = config_meta
            return res
        else:
            doc_res = self.evaluate_batch(doc_retrieval_results, k_values=k_values, level="doc", top1_scores=top1_scores)
            chunk_res = self.evaluate_batch(chunk_retrieval_results, k_values=k_values, level="chunk", top1_scores=top1_scores)
            return {
                "summary": doc_res["summary"],
                "level": "both",
                "doc_level": doc_res,
                "chunk_level": chunk_res,
                "ood_stats": doc_res["ood_stats"],
                "cache_config": config_meta,
                # Aliases for backwards compatibility:
                "aggregated_metrics": doc_res["aggregated_metrics"],
                "by_difficulty": doc_res["by_difficulty"],
                "by_query_type": doc_res["by_query_type"],
                "per_query_results": doc_res["per_query_results"],
            }


from src.evaluation.retrieval_cli import (_print_breakdown_table, main, print_category_matrix, print_metrics_table, print_query_type_matrix, print_unit_matrix, run_eval_retrieval_cli)

if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()

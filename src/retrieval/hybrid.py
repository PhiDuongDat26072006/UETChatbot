"""
src/retrieval/hybrid.py - Bộ máy tìm kiếm kết hợp (Hybrid Search Fusion).
Hợp nhất kết quả từ BM25 (Lexical) và Vector Search (Dense Semantic) để tạo ra ~100 candidate chunks.
Hỗ trợ Reciprocal Rank Fusion (RRF) chuẩn công nghiệp và Relative Score Fusion.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.base import RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("hybrid")


class HybridSearcher:
    """
    Bộ hợp nhất kết quả tìm kiếm đa nhánh (Hybrid Search Fusion: BM25 + Vector + GraphRAG).
    """

    @staticmethod
    def _classify_search_source(has_bm25: bool, has_vector: bool, has_graph: bool) -> tuple[str, str]:
        if has_bm25 and has_vector and has_graph:
            return "BM25 + Vector + GraphRAG", "hybrid"
        if has_bm25 and has_vector:
            return "BM25 + Vector Search", "hybrid"
        if has_bm25 and has_graph:
            return "BM25 + GraphRAG", "hybrid"
        if has_vector and has_graph:
            return "Vector + GraphRAG", "hybrid"
        if has_bm25:
            return "BM25", "bm25"
        if has_vector:
            return "Vector Search", "vector"
        if has_graph:
            return "GraphRAG", "graph"
        return "Unknown", "unknown"

    @staticmethod
    def _merge_candidate_meta(target: Dict[str, Any], incoming: Optional[Dict[str, Any]]) -> None:
        if not incoming:
            return
        existing_triples = list(target.get("graph_triples") or [])
        existing_entities = list(target.get("graph_entities") or [])
        target.update(incoming)
        if existing_triples and not incoming.get("graph_triples"):
            target["graph_triples"] = existing_triples
        if existing_entities and not incoming.get("graph_entities"):
            target["graph_entities"] = existing_entities

    @staticmethod
    def reciprocal_rank_fusion(
        bm25_results: List[RetrievedContext],
        vector_results: List[RetrievedContext],
        top_n: int = 100,
        rrf_k: int = 60,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        graph_results: Optional[List[RetrievedContext]] = None,
        graph_weight: float = 0.35,
    ) -> List[RetrievedContext]:
        """
        Hợp nhất 2 hoặc 3 danh sách kết quả (BM25, Vector, GraphRAG) theo Reciprocal Rank Fusion (RRF).
        RRF(d) = w_bm25 / (k + rank_bm25) + w_vec / (k + rank_vec) + w_graph / (k + rank_graph)
        """
        graph_list = graph_results or []
        active_branches = [
            ("bm25", bm25_results or [], bm25_weight),
            ("vector", vector_results or [], vector_weight),
            ("graph", graph_list, graph_weight),
        ]
        non_empty = [(name, items, w) for name, items, w in active_branches if items]

        if not non_empty:
            logger.debug("Cả BM25, Vector và GraphRAG đều trả về rỗng. Hybrid search trả về danh sách rỗng.")
            return []

        if len(non_empty) == 1:
            b_name, items, _ = non_empty[0]
            label_map = {
                "bm25": ("BM25 Search", "BM25", "bm25", "bm25_rank"),
                "vector": ("Vector Search", "Vector Search", "vector", "vector_rank"),
                "graph": ("GraphRAG Search", "GraphRAG", "graph", "graph_rank"),
            }
            log_label, src_label, type_label, rank_key = label_map[b_name]
            logger.info(f"Chỉ có kết quả từ {log_label}, chuyển tiếp sang Reranker.")
            single_results: List[RetrievedContext] = []
            for item in items[:top_n]:
                meta = dict(item.metadata) if item.metadata else {}
                meta.setdefault("search_source", src_label)
                meta.setdefault("search_type", type_label)
                meta.setdefault(rank_key, item.rank)
                single_results.append(
                    RetrievedContext(
                        chunk_id=item.chunk_id,
                        text=item.text,
                        similarity_score=item.similarity_score,
                        rank=item.rank,
                        metadata=meta,
                    )
                )
            return single_results

        candidates: Dict[str, Dict[str, Any]] = {}
        for b_name, items, weight in non_empty:
            for rank_pos, item in enumerate(items, 1):
                cid = item.chunk_id
                rrf_contrib = weight / (rrf_k + rank_pos)
                if cid not in candidates:
                    candidates[cid] = {
                        "chunk_id": cid,
                        "text": item.text,
                        "rrf_score": 0.0,
                        "bm25_rank": None,
                        "bm25_score": None,
                        "vector_rank": None,
                        "vector_score": None,
                        "graph_rank": None,
                        "graph_score": None,
                        "metadata": {},
                    }
                entry = candidates[cid]
                entry["rrf_score"] += rrf_contrib
                entry[f"{b_name}_rank"] = rank_pos
                entry[f"{b_name}_score"] = item.similarity_score
                HybridSearcher._merge_candidate_meta(entry["metadata"], item.metadata)

        sorted_candidates = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
        top_candidates = sorted_candidates[:top_n]
        max_rrf = top_candidates[0]["rrf_score"] if top_candidates and top_candidates[0]["rrf_score"] > 0 else 1.0

        hybrid_results: List[RetrievedContext] = []
        for rank, item in enumerate(top_candidates, 1):
            normalized_score = round(min(1.0, max(0.01, item["rrf_score"] / max_rrf)), 4)
            enriched_meta = item["metadata"]
            enriched_meta["retrieval_method"] = "hybrid_rrf"
            enriched_meta["bm25_rank"] = item["bm25_rank"]
            enriched_meta["vector_rank"] = item["vector_rank"]
            if item["graph_rank"] is not None:
                enriched_meta["graph_rank"] = item["graph_rank"]
            if item["graph_score"] is not None:
                enriched_meta.setdefault("graph_score", round(float(item["graph_score"]), 4))
            enriched_meta["rrf_score"] = round(item["rrf_score"], 6)

            search_source, search_type = HybridSearcher._classify_search_source(
                item["bm25_rank"] is not None,
                item["vector_rank"] is not None,
                item["graph_rank"] is not None,
            )
            enriched_meta["search_source"] = search_source
            enriched_meta["search_type"] = search_type

            hybrid_results.append(
                RetrievedContext(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    similarity_score=normalized_score,
                    rank=rank,
                    metadata=enriched_meta,
                )
            )

        logger.info(
            f"Hybrid Search (RRF) đã hợp nhất {len(bm25_results)} BM25 + {len(vector_results)} Vector "
            f"+ {len(graph_list)} GraphRAG thành {len(hybrid_results)} candidate chunks."
        )
        return hybrid_results

    @staticmethod
    def linear_score_fusion(
        bm25_results: List[RetrievedContext],
        vector_results: List[RetrievedContext],
        top_n: int = 100,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        graph_results: Optional[List[RetrievedContext]] = None,
        graph_weight: float = 0.35,
    ) -> List[RetrievedContext]:
        """
        Hợp nhất theo tổ hợp tuyến tính điểm số tương đối (Relative Score Fusion).
        Score(d) = w_vec * Score_vec(d) + w_bm25 * Score_bm25(d) + w_graph * Score_graph(d)
        """
        graph_list = graph_results or []
        active_branches = [
            ("bm25", bm25_results or [], bm25_weight),
            ("vector", vector_results or [], vector_weight),
            ("graph", graph_list, graph_weight),
        ]
        non_empty = [(name, items, w) for name, items, w in active_branches if items]

        if not non_empty:
            return []

        if len(non_empty) == 1:
            b_name, items, _ = non_empty[0]
            src_map = {"bm25": ("BM25", "bm25"), "vector": ("Vector Search", "vector"), "graph": ("GraphRAG", "graph")}
            src_label, type_label = src_map[b_name]
            out: List[RetrievedContext] = []
            for item in items[:top_n]:
                meta = dict(item.metadata) if item.metadata else {}
                meta.setdefault("search_source", src_label)
                meta.setdefault("search_type", type_label)
                out.append(
                    RetrievedContext(
                        chunk_id=item.chunk_id,
                        text=item.text,
                        similarity_score=item.similarity_score,
                        rank=item.rank,
                        metadata=meta,
                    )
                )
            return out

        candidates: Dict[str, Dict[str, Any]] = {}
        for b_name, items, weight in non_empty:
            for item in items:
                cid = item.chunk_id
                if cid not in candidates:
                    candidates[cid] = {
                        "chunk_id": cid,
                        "text": item.text,
                        "bm25_score": 0.0,
                        "vector_score": 0.0,
                        "graph_score": 0.0,
                        "bm25_found": False,
                        "vector_found": False,
                        "graph_found": False,
                        "final_score": 0.0,
                        "metadata": {},
                    }
                entry = candidates[cid]
                entry[f"{b_name}_score"] = float(item.similarity_score)
                entry[f"{b_name}_found"] = True
                entry["final_score"] += weight * float(item.similarity_score)
                HybridSearcher._merge_candidate_meta(entry["metadata"], item.metadata)

        sorted_candidates = sorted(candidates.values(), key=lambda x: x["final_score"], reverse=True)
        top_candidates = sorted_candidates[:top_n]

        results: List[RetrievedContext] = []
        for rank, item in enumerate(top_candidates, 1):
            meta = item["metadata"]
            meta["retrieval_method"] = "hybrid_linear"
            meta["bm25_score"] = round(item["bm25_score"], 4)
            meta["vector_score"] = round(item["vector_score"], 4)
            if item["graph_found"]:
                meta["graph_score"] = round(item["graph_score"], 4)

            search_source, search_type = HybridSearcher._classify_search_source(
                item["bm25_found"],
                item["vector_found"],
                item["graph_found"],
            )
            meta["search_source"] = search_source
            meta["search_type"] = search_type

            results.append(
                RetrievedContext(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    similarity_score=round(item["final_score"], 4),
                    rank=rank,
                    metadata=meta,
                )
            )

        return results

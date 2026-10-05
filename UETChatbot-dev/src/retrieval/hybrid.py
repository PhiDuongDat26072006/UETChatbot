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
    Bộ hợp nhất kết quả tìm kiếm đa nhánh (Hybrid Search Fusion).
    """

    @staticmethod
    def reciprocal_rank_fusion(
        bm25_results: List[RetrievedContext],
        vector_results: List[RetrievedContext],
        top_n: int = 100,
        rrf_k: int = 60,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> List[RetrievedContext]:
        """
        Hợp nhất 2 danh sách kết quả theo thuật toán Reciprocal Rank Fusion (RRF).
        RRF(d) = w_bm25 / (k + rank_bm25) + w_vec / (k + rank_vec)

        :param bm25_results: Danh sách kết quả từ BM25 Search
        :param vector_results: Danh sách kết quả từ Vector Search
        :param top_n: Số lượng candidate chunks cần lấy (~100 candidate chunks)
        :param rrf_k: Hằng số làm mượt RRF (mặc định 60 theo tiêu chuẩn IR)
        :param bm25_weight: Trọng số của nhánh từ khóa BM25
        :param vector_weight: Trọng số của nhánh ngữ nghĩa Vector Search
        :return: Danh sách RetrievedContext kết hợp, sắp xếp giảm dần theo điểm RRF
        """
        # Nếu cả 2 danh sách đều rỗng
        if not bm25_results and not vector_results:
            logger.debug("Cả BM25 và Vector search đều trả về rỗng. Hybrid search trả về danh sách rỗng.")
            return []

        # Nếu một trong hai rỗng, trả về danh sách còn lại cắt theo top_n
        if not bm25_results:
            logger.info("Chỉ có kết quả từ Vector Search, chuyển tiếp sang Reranker.")
            return vector_results[:top_n]
        if not vector_results:
            logger.info("Chỉ có kết quả từ BM25 Search, chuyển tiếp sang Reranker.")
            return bm25_results[:top_n]

        # Bảng ánh xạ chunk_id -> thông tin hợp nhất
        candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Chấm điểm từ nhánh BM25
        for rank_bm25, item in enumerate(bm25_results, 1):
            cid = item.chunk_id
            rrf_score = bm25_weight / (rrf_k + rank_bm25)

            candidates[cid] = {
                "chunk_id": cid,
                "text": item.text,
                "rrf_score": rrf_score,
                "bm25_rank": rank_bm25,
                "bm25_score": item.similarity_score,
                "vector_rank": None,
                "vector_score": None,
                "metadata": dict(item.metadata) if item.metadata else {},
            }

        # 2. Chấm điểm từ nhánh Vector Search và cộng dồn RRF
        for rank_vec, item in enumerate(vector_results, 1):
            cid = item.chunk_id
            vec_rrf = vector_weight / (rrf_k + rank_vec)

            if cid in candidates:
                # Chunk xuất hiện ở CẢ HAI nhánh (cực kỳ liên quan!)
                candidates[cid]["rrf_score"] += vec_rrf
                candidates[cid]["vector_rank"] = rank_vec
                candidates[cid]["vector_score"] = item.similarity_score
                # Gộp thêm metadata nếu có
                if item.metadata:
                    candidates[cid]["metadata"].update(item.metadata)
            else:
                # Chunk chỉ xuất hiện ở nhánh Vector
                candidates[cid] = {
                    "chunk_id": cid,
                    "text": item.text,
                    "rrf_score": vec_rrf,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "vector_rank": rank_vec,
                    "vector_score": item.similarity_score,
                    "metadata": dict(item.metadata) if item.metadata else {},
                }

        # 3. Sắp xếp danh sách ứng viên giảm dần theo tổng điểm RRF
        sorted_candidates = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
        top_candidates = sorted_candidates[:top_n]

        # Chuẩn hóa điểm RRF về khoảng (0.0, 1.0] để đồng bộ
        max_rrf = top_candidates[0]["rrf_score"] if top_candidates and top_candidates[0]["rrf_score"] > 0 else 1.0

        hybrid_results: List[RetrievedContext] = []
        for rank, item in enumerate(top_candidates, 1):
            normalized_score = round(min(1.0, max(0.01, item["rrf_score"] / max_rrf)), 4)

            enriched_meta = item["metadata"]
            enriched_meta["retrieval_method"] = "hybrid_rrf"
            enriched_meta["bm25_rank"] = item["bm25_rank"]
            enriched_meta["vector_rank"] = item["vector_rank"]
            enriched_meta["rrf_score"] = round(item["rrf_score"], 6)

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
            f"thành {len(hybrid_results)} candidate chunks."
        )
        return hybrid_results

    @staticmethod
    def linear_score_fusion(
        bm25_results: List[RetrievedContext],
        vector_results: List[RetrievedContext],
        top_n: int = 100,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> List[RetrievedContext]:
        """
        Hợp nhất theo tổ hợp tuyến tính điểm số tương đối (Relative Score Fusion).
        Score(d) = w_vec * Score_vec(d) + w_bm25 * Score_bm25(d)
        """
        if not bm25_results and not vector_results:
            return []
        if not bm25_results:
            return vector_results[:top_n]
        if not vector_results:
            return bm25_results[:top_n]

        candidates: Dict[str, Dict[str, Any]] = {}

        for item in bm25_results:
            cid = item.chunk_id
            candidates[cid] = {
                "chunk_id": cid,
                "text": item.text,
                "bm25_score": item.similarity_score,
                "vector_score": 0.0,
                "metadata": dict(item.metadata) if item.metadata else {},
            }

        for item in vector_results:
            cid = item.chunk_id
            if cid in candidates:
                candidates[cid]["vector_score"] = item.similarity_score
                if item.metadata:
                    candidates[cid]["metadata"].update(item.metadata)
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "text": item.text,
                    "bm25_score": 0.0,
                    "vector_score": item.similarity_score,
                    "metadata": dict(item.metadata) if item.metadata else {},
                }

        # Tính tổng điểm
        for item in candidates.values():
            item["final_score"] = (bm25_weight * item["bm25_score"]) + (vector_weight * item["vector_score"])

        sorted_candidates = sorted(candidates.values(), key=lambda x: x["final_score"], reverse=True)
        top_candidates = sorted_candidates[:top_n]

        results: List[RetrievedContext] = []
        for rank, item in enumerate(top_candidates, 1):
            meta = item["metadata"]
            meta["retrieval_method"] = "hybrid_linear"
            meta["bm25_score"] = round(item["bm25_score"], 4)
            meta["vector_score"] = round(item["vector_score"], 4)

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

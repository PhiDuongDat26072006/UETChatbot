"""
src/retrieval/mmr.py - Thuật toán Maximal Marginal Relevance (MMR).
Chức năng: Đa dạng hóa tập ứng viên ngữ cảnh (Diversity Selection) sau bước Hybrid Search
và trước khi đưa vào mô hình Reranker.
Cân bằng giữa độ tương quan với câu hỏi (Relevance) và độ mới/khác biệt giữa các đoạn (Diversity).
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.base import RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("retrieval.mmr")


class MaximalMarginalRelevance:
    """
    Bộ chọn lọc đa dạng hóa ngữ cảnh Maximal Marginal Relevance (MMR).
    Dựa trên nghiên cứu Carbonell & Goldstein (1998):
        MMR = argmax_{D_i in R \\ S} [ lambda * Sim_1(D_i, Q) - (1 - lambda) * max_{D_j in S} Sim_2(D_i, D_j) ]
    Được tối ưu hóa bằng ma trận vectorized NumPy để đạt tốc độ xử lý <= 1ms.
    """

    @staticmethod
    def compute_mmr_indices(
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        lambda_mult: float = 0.6,
        top_n: int = 40,
    ) -> List[int]:
        """
        Tính toán danh sách chỉ mục (indices) của các ứng viên được chọn theo MMR.

        :param query_vector: Vector của câu hỏi (1D array shape (d,))
        :param candidate_vectors: Ma trận vector của các ứng viên (2D array shape (N, d))
        :param lambda_mult: Hệ số cân bằng giữa Relevance (1.0) và Diversity (0.0). Mặc định 0.6.
        :param top_n: Số lượng ứng viên cần trích xuất (M <= N)
        :return: Danh sách chỉ mục các ứng viên được chọn theo thứ tự ưu tiên MMR
        """
        num_candidates = candidate_vectors.shape[0]
        if num_candidates == 0:
            return []

        target_n = min(top_n, num_candidates)
        if target_n <= 0:
            return []

        # Chuẩn hóa L2 vector câu hỏi để tránh chia cho 0
        q_norm_val = np.linalg.norm(query_vector)
        q_norm = query_vector / (q_norm_val if q_norm_val > 1e-12 else 1e-12)

        # Chuẩn hóa L2 ma trận vector ứng viên
        doc_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        doc_norms[doc_norms < 1e-12] = 1e-12
        docs_norm = candidate_vectors / doc_norms

        # 1. Tính trước độ tương đồng Cosine giữa toàn bộ ứng viên và Query: shape (N,)
        sim_to_query = np.dot(docs_norm, q_norm)

        # 2. Khởi tạo danh sách ứng viên đã chọn và ứng viên còn lại
        selected_indices: List[int] = []
        remaining_indices = list(range(num_candidates))

        # 3. Chọn ứng viên đầu tiên: Ứng viên có độ liên quan tới Query cao nhất
        first_idx = int(np.argmax(sim_to_query))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # 4. Duy trì mảng max_sim_to_selected lưu độ tương đồng lớn nhất của mỗi ứng viên
        # với bất kỳ ứng viên nào đã có trong tập S
        max_sim_to_selected = np.dot(docs_norm, docs_norm[first_idx])

        # 5. Vòng lặp Greedy Selection cho các vị trí từ 2 đến target_n
        for _ in range(1, target_n):
            if not remaining_indices:
                break

            rem_arr = np.array(remaining_indices, dtype=np.int32)

            # Tính điểm MMR cho tất cả ứng viên còn lại
            query_rel = sim_to_query[rem_arr]
            doc_redundancy = max_sim_to_selected[rem_arr]
            mmr_scores = lambda_mult * query_rel - (1.0 - lambda_mult) * doc_redundancy

            best_rem_pos = int(np.argmax(mmr_scores))
            best_idx = remaining_indices[best_rem_pos]

            selected_indices.append(best_idx)
            remaining_indices.pop(best_rem_pos)

            # Cập nhật trực tuyến mảng max_sim_to_selected với ứng viên vừa được chọn
            new_sims = np.dot(docs_norm, docs_norm[best_idx])
            max_sim_to_selected = np.maximum(max_sim_to_selected, new_sims)

        return selected_indices

    @classmethod
    def select(
        cls,
        query_vector: Union[List[float], np.ndarray],
        candidates: List[RetrievedContext],
        candidate_vectors: Union[List[List[float]], np.ndarray],
        lambda_mult: float = 0.6,
        top_n: int = 40,
    ) -> List[RetrievedContext]:
        """
        Chọn lọc danh sách RetrievedContext đa dạng hóa ngữ cảnh bằng MMR.

        :param query_vector: Vector nhúng của câu hỏi
        :param candidates: Danh sách candidate chunks từ Hybrid Search
        :param candidate_vectors: Danh sách vector tương ứng với từng candidate chunk
        :param lambda_mult: Hệ số trade-off giữa Relevance và Diversity (0.0 đến 1.0)
        :param top_n: Số lượng candidate chunks tối đa giữ lại cho Reranker
        :return: Danh sách RetrievedContext đã được sắp xếp theo thứ tự MMR
        """
        if not candidates:
            return []

        if len(candidates) <= top_n:
            logger.debug(
                "Số lượng candidates (%d) <= top_n (%d), không cần rút gọn qua MMR.",
                len(candidates),
                top_n,
            )
            return candidates

        q_vec = np.asarray(query_vector, dtype=np.float32)
        c_vecs = np.asarray(candidate_vectors, dtype=np.float32)

        if c_vecs.shape[0] != len(candidates):
            logger.warning(
                "Kích thước candidate_vectors (%d) khác với số lượng candidates (%d). Bỏ qua MMR.",
                c_vecs.shape[0],
                len(candidates),
            )
            return candidates[:top_n]

        # Giới hạn lambda trong đoạn hợp lệ [0.0, 1.0]
        clamped_lambda = max(0.0, min(1.0, float(lambda_mult)))

        # Tính toán danh sách chỉ mục theo MMR
        selected_indices = cls.compute_mmr_indices(
            query_vector=q_vec,
            candidate_vectors=c_vecs,
            lambda_mult=clamped_lambda,
            top_n=top_n,
        )

        diverse_candidates: List[RetrievedContext] = []
        for rank, idx in enumerate(selected_indices, 1):
            item = candidates[idx]
            # Tạo bản sao shallow để không ảnh hưởng dữ liệu gốc nếu cần
            meta = dict(item.metadata) if item.metadata else {}
            meta["mmr_rank"] = rank
            meta["mmr_lambda"] = clamped_lambda

            diverse_candidates.append(
                RetrievedContext(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    similarity_score=item.similarity_score,
                    rank=rank,
                    metadata=meta,
                )
            )

        logger.debug(
            "MMR chọn ra %d chunks từ %d candidates (lambda=%.2f).",
            len(diverse_candidates),
            len(candidates),
            clamped_lambda,
        )
        return diverse_candidates

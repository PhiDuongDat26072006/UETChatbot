"""
src/retrieval/mmr.py - Thuật toán Maximal Marginal Relevance (MMR).
Chức năng: Đa dạng hóa tập ứng viên ngữ cảnh (Diversity Selection) bằng cách cân bằng
giữa độ tương quan với câu hỏi (Hybrid Relevance hoặc Dense Cosine) và độ khác biệt giữa các đoạn,
đồng thời hỗ trợ ngưỡng khử trùng lặp cao (`similarity_threshold`) và bảo vệ chunk bảng (`protect_tabular`).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from src.base import RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("retrieval.mmr")


class MaximalMarginalRelevance:
    """
    Bộ chọn lọc đa dạng hóa ngữ cảnh Maximal Marginal Relevance (MMR).
    Dựa trên nghiên cứu Carbonell & Goldstein (1998):
        MMR = argmax_{D_i in R \\ S} [ lambda * Rel(D_i, Q) - (1 - lambda) * max_{D_j in S} Sim(D_i, D_j) ]
    Được tối ưu hóa bằng ma trận vectorized NumPy để đạt tốc độ xử lý <= 1ms.
    """

    @staticmethod
    def compute_mmr_indices(
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        lambda_mult: float = 0.6,
        top_n: int = 40,
        relevance_scores: Optional[np.ndarray] = None,
        similarity_threshold: float = 0.0,
        tabular_doc_ids: Optional[List[Optional[str]]] = None,
    ) -> List[int]:
        """
        Tính toán danh sách chỉ mục (indices) của các ứng viên được chọn theo MMR.

        :param query_vector: Vector của câu hỏi (1D array shape (d,))
        :param candidate_vectors: Ma trận vector của các ứng viên (2D array shape (N, d))
        :param lambda_mult: Hệ số cân bằng giữa Relevance (1.0) và Diversity (0.0).
        :param top_n: Số lượng ứng viên cần trích xuất (M <= N)
        :param relevance_scores: Điểm relevance tùy chỉnh (VD: điểm Hybrid RRF/Rerank) thay cho thuần Dense Cosine.
        :param similarity_threshold: Ngưỡng tương đồng tối thiểu mới tính phạt redundancy (0.0 = phạt liên tục như chuẩn gốc).
        :param tabular_doc_ids: Danh sách document_id cho các chunk dạng bảng cần miễn phạt khi cùng document_id.
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

        # 1. Điểm liên quan tới Query (ưu tiên relevance_scores nếu được cung cấp)
        if relevance_scores is not None and len(relevance_scores) == num_candidates:
            sim_to_query = np.asarray(relevance_scores, dtype=np.float32)
        else:
            sim_to_query = np.dot(docs_norm, q_norm)

        def _compute_redundancy(anchor_idx: int) -> np.ndarray:
            sims = np.dot(docs_norm, docs_norm[anchor_idx]).astype(np.float32)
            if similarity_threshold > 0.0:
                sims = np.where(sims >= similarity_threshold, sims, 0.0)
            if tabular_doc_ids is not None and anchor_idx < len(tabular_doc_ids):
                anchor_doc = tabular_doc_ids[anchor_idx]
                if anchor_doc is not None:
                    for j, doc_id in enumerate(tabular_doc_ids):
                        if doc_id == anchor_doc:
                            sims[j] = 0.0
            return sims

        # 2. Khởi tạo danh sách ứng viên đã chọn và ứng viên còn lại
        selected_indices: List[int] = []
        remaining_indices = list(range(num_candidates))

        # 3. Chọn ứng viên đầu tiên có độ liên quan cao nhất
        first_idx = int(np.argmax(sim_to_query))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # 4. Duy trì mảng max_sim_to_selected lưu độ tương đồng lớn nhất với tập S
        max_sim_to_selected = _compute_redundancy(first_idx)

        # 5. Vòng lặp Greedy Selection cho các vị trí từ 2 đến target_n
        for _ in range(1, target_n):
            if not remaining_indices:
                break

            rem_arr = np.array(remaining_indices, dtype=np.int32)
            query_rel = sim_to_query[rem_arr]
            doc_redundancy = max_sim_to_selected[rem_arr]
            mmr_scores = lambda_mult * query_rel - (1.0 - lambda_mult) * doc_redundancy

            best_rem_pos = int(np.argmax(mmr_scores))
            best_idx = remaining_indices[best_rem_pos]

            selected_indices.append(best_idx)
            remaining_indices.pop(best_rem_pos)

            new_sims = _compute_redundancy(best_idx)
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
        use_hybrid_relevance: bool = False,
        similarity_threshold: float = 0.0,
        protect_tabular: bool = False,
    ) -> List[RetrievedContext]:
        """
        Chọn lọc danh sách RetrievedContext đa dạng hóa ngữ cảnh bằng MMR.

        :param query_vector: Vector nhúng của câu hỏi
        :param candidates: Danh sách candidate chunks
        :param candidate_vectors: Danh sách vector tương ứng với từng candidate chunk
        :param lambda_mult: Hệ số trade-off giữa Relevance và Diversity (0.0 đến 1.0)
        :param top_n: Số lượng candidate chunks tối đa giữ lại
        :param use_hybrid_relevance: Kết hợp điểm Hybrid/Rerank hiện có (0.80) với Vector Cosine (0.20)
        :param similarity_threshold: Ngưỡng cosine tối thiểu mới áp dụng phạt trùng lặp (0.0 = mặc định gốc)
        :param protect_tabular: Miễn phạt cho các chunk dạng bảng (`data_type == "tabular"` hoặc `[META:`) cùng document_id
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

        clamped_lambda = max(0.0, min(1.0, float(lambda_mult)))

        relevance_scores: Optional[np.ndarray] = None
        if use_hybrid_relevance:
            q_norm_val = float(np.linalg.norm(q_vec))
            q_unit = q_vec / (q_norm_val if q_norm_val > 1e-12 else 1e-12)
            c_norms = np.linalg.norm(c_vecs, axis=1, keepdims=True)
            c_norms[c_norms < 1e-12] = 1e-12
            vec_sims = np.dot(c_vecs / c_norms, q_unit)

            raw_scores = np.array([float(getattr(c, "similarity_score", 0.0) or 0.0) for c in candidates], dtype=np.float32)
            max_raw = float(np.max(raw_scores)) if len(raw_scores) > 0 else 0.0
            norm_hybrid = (raw_scores / max_raw) if max_raw > 1e-9 else raw_scores
            relevance_scores = np.clip(0.80 * norm_hybrid + 0.20 * vec_sims, 0.0, 1.0)

        tabular_doc_ids: Optional[List[Optional[str]]] = None
        if protect_tabular:
            tabular_doc_ids = []
            for c in candidates:
                meta = c.metadata if isinstance(c.metadata, dict) else {}
                is_tab = (meta.get("data_type") == "tabular") or ("[META:" in (c.text or ""))
                doc_id = meta.get("document_id") or meta.get("doc_id")
                tabular_doc_ids.append(str(doc_id) if (is_tab and doc_id) else None)

        selected_indices = cls.compute_mmr_indices(
            query_vector=q_vec,
            candidate_vectors=c_vecs,
            lambda_mult=clamped_lambda,
            top_n=top_n,
            relevance_scores=relevance_scores,
            similarity_threshold=similarity_threshold,
            tabular_doc_ids=tabular_doc_ids,
        )

        diverse_candidates: List[RetrievedContext] = []
        for rank, idx in enumerate(selected_indices, 1):
            item = candidates[idx]
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
            "MMR chọn ra %d chunks từ %d candidates (lambda=%.2f, hybrid=%s, thr=%.2f).",
            len(diverse_candidates),
            len(candidates),
            clamped_lambda,
            use_hybrid_relevance,
            similarity_threshold,
        )
        return diverse_candidates

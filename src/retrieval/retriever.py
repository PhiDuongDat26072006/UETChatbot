"""
src/retrieval/retriever.py - Phụ trách quy trình truy xuất ngữ cảnh (Retrieval).
Thành viên phụ trách: Người làm retrieval & similarity search.
Nhiệm vụ: Nhận câu hỏi, phối hợp với mô hình Embedding và VectorDB để tìm ra các đoạn văn bản phù hợp nhất (hỗ trợ reranking, score filtering).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from base import BaseEmbeddingModel, BaseVectorStore, UserQuery, RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("retrieval")


class UETRetriever:
    """
    Bộ điều phối truy xuất ngữ cảnh cho câu hỏi người dùng.
    """

    def __init__(self, embedding_model: BaseEmbeddingModel, vector_store: BaseVectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: UserQuery,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Truy xuất danh sách ngữ cảnh liên quan nhất cho câu hỏi.
        :param query: Câu hỏi người dùng (UserQuery)
        :param top_k: Số lượng văn bản cần lấy
        :param score_threshold: Ngưỡng điểm tương đồng tối thiểu (nếu có)
        :param filters: Bộ lọc metadata
        :return: Danh sách RetrievedContext
        """
        logger.info(f"Đang xử lý truy xuất cho câu hỏi: '{query.query_text}'")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT PHẦN NÀY:
        # Gợi ý:
        # 1. Nhúng câu hỏi thành vector: `query_vector = self.embedding_model.embed_query(query)`
        # 2. Tìm kiếm trong Vector DB: `contexts = self.vector_store.search(query_vector, top_k, filters)`
        # 3. Lọc theo điểm `score_threshold` (nếu có).
        # 4. (Nâng cao): Thêm bước Reranking bằng Cross-Encoder hoặc mô hình ColBERT nếu muốn điểm cao!
        # =========================================================================

        raise NotImplementedError(
            "TODO: Thành viên phụ trách Retrieval cần tự cài đặt logic hàm retrieve() trong src/retrieval/retriever.py!"
        )

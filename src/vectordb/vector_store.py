"""
src/vectordb/vector_store.py - Phụ trách lưu trữ và truy vấn Vector Database.
Thành viên phụ trách: Người quản lý Vector Database.
Nhiệm vụ: Quản lý kết nối cơ sở dữ liệu vector (ChromaDB, FAISS, Pinecone...), lưu trữ chunks & vectors, và tìm kiếm tương đồng.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.base import BaseVectorStore, DataChunk, EmbeddedVector, EmbeddedQueryVector, RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("vectordb")


class UETVectorStore(BaseVectorStore):
    """
    Quản lý Vector Database cho UET Chatbot.
    Thành viên phụ trách cần kế thừa BaseVectorStore từ base.py và tự cài đặt:
    - store()
    - search()
    - count()
    - clear()
    """

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "uet_knowledge_base"):
        self.persist_dir = persist_dir or Path("vector_db")
        self.collection_name = collection_name

    def store(
        self,
        chunks: List[DataChunk],
        vectors: Optional[List[EmbeddedVector]] = None,
    ) -> int:
        """
        Lưu danh sách DataChunk (kèm vector nếu có) vào Vector Database.
        :param chunks: Danh sách các đoạn văn bản
        :param vectors: Danh sách vector tương ứng
        :return: Số lượng chunk đã lưu thành công
        """
        logger.info(f"Đang lưu {len(chunks)} chunks vào collection '{self.collection_name}'...")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT PHẦN NÀY:
        # Gợi ý:
        # 1. Sử dụng ChromaDB: `chromadb.PersistentClient(path=str(self.persist_dir))`
        # 2. Tạo hoặc lấy collection: `client.get_or_create_collection(self.collection_name)`
        # 3. Nạp dữ liệu qua `collection.upsert(ids=..., documents=..., metadatas=...)`
        # 4. Trả về số lượng chunk lưu thành công.
        # =========================================================================

        raise NotImplementedError(
            "TODO: Thành viên phụ trách VectorDB cần tự cài đặt logic hàm store() trong src/vectordb/vector_store.py!"
        )

    def search(
        self,
        query_vector: EmbeddedQueryVector,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Tìm kiếm các đoạn văn bản có độ tương đồng cao nhất với câu hỏi.
        :param query_vector: Vector của câu hỏi
        :param top_k: Số lượng kết quả cần lấy
        :param filters: Bộ lọc metadata nếu có
        :return: Danh sách các đối tượng RetrievedContext
        """
        logger.info(f"Đang tìm kiếm top {top_k} đoạn liên quan cho query_id: {query_vector.query_id}...")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT PHẦN NÀY:
        # Gợi ý:
        # 1. Truy vấn collection bằng vector:
        #    `results = collection.query(query_embeddings=[query_vector.vector], n_results=top_k, where=filters)`
        # 2. Đóng gói kết quả thành danh sách `RetrievedContext(...)` theo base.py:
        #    - chunk_id, text, similarity_score, rank, metadata
        # =========================================================================

        raise NotImplementedError(
            "TODO: Thành viên phụ trách VectorDB cần tự cài đặt logic hàm search() trong src/vectordb/vector_store.py!"
        )

    def count(self) -> int:
        """Trả về tổng số chunk trong vector collection."""
        raise NotImplementedError("TODO: Cài đặt hàm count() trong src/vectordb/vector_store.py!")

    def clear(self) -> None:
        """Xóa toàn bộ dữ liệu trong collection."""
        raise NotImplementedError("TODO: Cài đặt hàm clear() trong src/vectordb/vector_store.py!")

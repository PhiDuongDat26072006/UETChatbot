"""
src/vectordb/vector_store.py - Phụ trách lưu trữ.
Thành viên phụ trách: Người quản lý cơ sở dữ liệu Vector.
Nhiệm vụ: Lưu trữ DataChunk + Vector vào ChromaDB.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Đảm bảo root directory có trong sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import chromadb
from chromadb.config import Settings

from src.base import (
    BaseVectorStore,
    BaseEmbeddingModel,
    DataChunk,
    EmbeddedVector,
    EmbeddedQueryVector,
    RetrievedContext,
)
from src.utils.helpers import get_logger

logger = get_logger("vectordb")


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn hóa metadata cho ChromaDB: ChromaDB chỉ chấp nhận các kiểu dữ liệu nguyên thủy
    (str, int, float, bool). Chuyển đổi dict, list, None thành string.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif value is None:
            cleaned[key] = ""
        else:
            cleaned[key] = str(value)
    return cleaned


class ChromaVectorStore(BaseVectorStore):
    """
    Lớp quản lý Vector Database dựa trên ChromaDB cho UET Chatbot.
    Thực thi lưu trữ các DataChunk + Vector và tìm kiếm tương đồng theo Cosine Distance.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: str = "uet_knowledge_base",
        embedding_model: Optional[BaseEmbeddingModel] = None,
    ):
        """
        Khởi tạo ChromaVectorStore kết nối tới thư mục lưu trữ ChromaDB.
        :param persist_dir: Đường dẫn thư mục lưu trữ ChromaDB (Mặc định: `vector_db/`)
        :param collection_name: Tên collection trong ChromaDB (Mặc định: `uet_knowledge_base`)
        :param embedding_model: Mô hình nhúng tùy chọn để tự nhúng chunk khi không truyền vectors
        """
        if persist_dir is None:
            self.persist_dir = BASE_DIR / "vector_db"
        else:
            self.persist_dir = Path(persist_dir)

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        logger.info(f"Đang kết nối tới ChromaDB tại thư mục: '{self.persist_dir}'...")
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Lấy hoặc tạo mới collection với cosine distance metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Đã mở collection '{self.collection_name}' (Đang có {self.collection.count()} vectors).")

    def store(
        self,
        chunks: List[DataChunk],
        vectors: Optional[List[EmbeddedVector]] = None,
    ) -> int:
        """
        Lưu danh sách chunks (và vectors tương ứng nếu có) vào ChromaDB.
        :param chunks: Danh sách các DataChunk
        :param vectors: Danh sách các EmbeddedVector tương ứng (nếu có)
        :return: Số lượng chunks đã lưu thành công
        """
        if not chunks:
            logger.warning("Danh sách chunks rỗng. Không có dữ liệu để lưu.")
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings: Optional[List[List[float]]] = None

        # Thu thập ID, nội dung document và metadata
        for idx, chunk in enumerate(chunks):
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            meta = dict(chunk.metadata or {})
            if chunk.document_id:
                meta["document_id"] = chunk.document_id
            meta["chunk_index"] = chunk.chunk_index
            
            metadatas.append(_clean_metadata(meta))

        # Xác định vectors
        if vectors is not None and len(vectors) == len(chunks):
            embeddings = [vec.vector for vec in vectors]
        elif self.embedding_model is not None:
            logger.info("Chưa có vectors đầu vào, tự động sinh vector từ embedding_model...")
            embedded_objs = self.embedding_model.embed_chunks(chunks)
            embeddings = [vec.vector for vec in embedded_objs]

        # Thêm/cập nhật dữ liệu vào ChromaDB bằng upsert
        try:
            if embeddings is not None:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
            saved_count = len(ids)
            logger.info(f"Đã lưu thành công {saved_count} chunks vào collection '{self.collection_name}'. Tổng số lượng hiện tại: {self.collection.count()}.")
            return saved_count
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu vào ChromaDB: {e}")
            raise e

    def search(
        self,
        query_vector: EmbeddedQueryVector,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Tìm kiếm top_k đoạn văn bản có độ tương đồng cao nhất với câu hỏi.
        :param query_vector: Vector của câu hỏi
        :param top_k: Số lượng văn bản liên quan cần lấy
        :param filters: Bộ lọc metadata nếu có
        :return: Danh sách RetrievedContext sắp xếp theo độ tương đồng giảm dần
        """
        if self.count() == 0:
            logger.warning(f"Collection '{self.collection_name}' đang rỗng, không thể tìm kiếm.")
            return []

        try:
            query_kwargs: Dict[str, Any] = {"n_results": min(top_k, self.count())}
            if filters:
                query_kwargs["where"] = filters

            if query_vector.vector:
                query_kwargs["query_embeddings"] = [query_vector.vector]
            elif query_vector.query_text:
                query_kwargs["query_texts"] = [query_vector.query_text]
            else:
                return []

            res = self.collection.query(**query_kwargs)
            contexts: List[RetrievedContext] = []

            if res and res.get("documents") and len(res["documents"]) > 0:
                docs = res["documents"][0]
                ids = res["ids"][0] if res.get("ids") else [f"chunk_{i}" for i in range(len(docs))]
                metas = res["metadatas"][0] if res.get("metadatas") else [{} for _ in docs]
                distances = res["distances"][0] if res.get("distances") else [0.5 for _ in docs]

                for rank, (cid, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1):
                    # Cosine distance sang cosine similarity: max(0, 1 - distance)
                    similarity = round(max(0.0, min(1.0, 1.0 - dist)), 4)
                    cleaned_meta = dict(meta or {})
                    if "title" not in cleaned_meta:
                        cleaned_meta["title"] = cleaned_meta.get("source") or f"Tài liệu UET ({cid})"

                    contexts.append(
                        RetrievedContext(
                            chunk_id=cid,
                            text=doc_text,
                            similarity_score=similarity,
                            rank=rank,
                            metadata=cleaned_meta,
                        )
                    )
            return contexts
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm trên ChromaDB: {e}")
            return []



    @property
    def _chroma_col(self):
        """Thuộc tính tương thích để truy cập collection ChromaDB nội bộ."""
        return self.collection

    def count(self) -> int:
        """Trả về tổng số chunk/vector trong collection ChromaDB."""
        return self.collection.count()

    def clear(self) -> None:
        """Xóa toàn bộ dữ liệu trong collection để reset lại database."""
        try:
            # Lấy tất cả các ID hiện có trong collection
            existing_data = self.collection.get(include=[])
            if existing_data and existing_data.get("ids"):
                # Xóa các vector theo ID thay vì xóa bỏ hoàn toàn collection
                self.collection.delete(ids=existing_data["ids"])
            logger.info(f"Đã xóa sạch dữ liệu trong collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Lỗi khi xóa dữ liệu collection '{self.collection_name}': {e}")


# Alias UETVectorStore tương thích quy chuẩn đặt tên các module UET
UETVectorStore = ChromaVectorStore




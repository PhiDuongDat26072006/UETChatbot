"""
src/embeddings/embedder.py - Phụ trách mô hình Vector Embedding (Task 3).
Thành viên phụ trách: Người quản lý mô hình Embedding.
Nhiệm vụ: Chuyển đổi văn bản thành các vector số thực 384 chiều bằng mô hình `all-MiniLM-L6-v2`.
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from typing import List, Optional

# Đảm bảo root directory có trong sys.path để import từ src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import (
    BaseEmbeddingModel,
    DataChunk,
    EmbeddedVector,
    UserQuery,
    EmbeddedQueryVector,
)
from src.utils.helpers import get_logger

logger = get_logger("embeddings")


class UETEmbedder(BaseEmbeddingModel):
    """
    Mô hình nhúng văn bản (Embedding Model) cho UET Chatbot.
    Sử dụng mô hình: `all-MiniLM-L6-v2` (384 chiều, chạy cục bộ offline, 0đ).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._dimension = 384
        self._embedding_fn = None

        try:
            import chromadb.utils.embedding_functions as ef
            self._embedding_fn = ef.DefaultEmbeddingFunction()
            logger.info(f"Đã khởi tạo thành công mô hình embedding '{self._model_name}' ({self._dimension} chiều).")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo DefaultEmbeddingFunction của Chroma: {e}. Sẽ dùng vector dự phòng.")

    @property
    def dimension(self) -> int:
        """Số chiều của vector embedding (384 chiều)."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Tên mô hình embedding."""
        return self._model_name

    def _fallback_vector(self, text: str) -> List[float]:
        """Tạo vector dự phòng 384 chiều dựa trên hash văn bản khi gặp sự cố."""
        hash_digest = hashlib.md5(text.encode("utf-8")).digest()
        vec = []
        for i in range(self._dimension):
            byte_val = hash_digest[i % len(hash_digest)]
            vec.append((byte_val / 255.0) * 2 - 1)
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Nhúng danh sách chuỗi văn bản thành danh sách vector số thực 384 chiều.
        :param texts: Danh sách các câu hoặc đoạn văn bản cần nhúng
        :return: Danh sách vector (List[List[float]])
        """
        if not texts:
            return []

        logger.info(f"Đang nhúng {len(texts)} đoạn văn bản bằng mô hình '{self.model_name}' ({self.dimension} chiều)...")

        if self._embedding_fn is not None:
            try:
                batch_size = 64
                all_vectors: List[List[float]] = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    raw_vectors = self._embedding_fn(batch)
                    all_vectors.extend([list(vec) for vec in raw_vectors])
                return all_vectors
            except Exception as e:
                logger.error(f"Lỗi khi nhúng văn bản bằng '{self.model_name}': {e}. Dùng vector dự phòng.")

        return [self._fallback_vector(t) for t in texts]




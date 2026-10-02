"""
src/embeddings/embedder.py - Phụ trách mô hình Vector Embedding.
Thành viên phụ trách: Người quản lý mô hình Embedding.
Nhiệm vụ: Chuyển đổi văn bản thành các vector số thực nhiều chiều (biểu diễn ngữ nghĩa).
"""

from __future__ import annotations
import os
from typing import List, Optional
from base import BaseEmbeddingModel
from src.utils.helpers import get_logger

logger = get_logger("embeddings")


class UETEmbedder(BaseEmbeddingModel):
    """
    Mô hình nhúng văn bản (Embedding Model).
    Thành viên phụ trách cần kế thừa BaseEmbeddingModel từ base.py và tự cài đặt:
    - property dimension
    - property model_name
    - embed_texts(texts: List[str]) -> List[List[float]]
    """

    def __init__(self, provider: str = "chroma", model_name: Optional[str] = None):
        self._provider = provider.lower()
        self._model_name = model_name or ("text-embedding-004" if provider == "gemini" else "all-MiniLM-L6-v2")

    @property
    def dimension(self) -> int:
        """Số chiều của vector embedding."""
        if self._provider == "gemini":
            return 768
        return 384  # Mặc định all-MiniLM-L6-v2 của ChromaDB là 384 chiều

    @property
    def model_name(self) -> str:
        """Tên mô hình embedding."""
        return self._model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Nhúng danh sách chuỗi văn bản thành danh sách vector số thực.
        :param texts: Danh sách các câu hoặc đoạn văn bản cần nhúng
        :return: Danh sách vector (List[List[float]])
        """
        logger.info(f"Đang nhúng {len(texts)} đoạn văn bản bằng mô hình '{self.model_name}'...")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT PHẦN NÀY:
        # Gợi ý:
        # 1. Nếu dùng Google Gemini API:
        #    - Dùng `google.genai.Client(api_key=...)`
        #    - Gọi `client.models.embed_content(model=self.model_name, contents=text)`
        # 2. Nếu dùng ChromaDB Default Embedding (Local, 0đ):
        #    - Dùng `chromadb.utils.embedding_functions.DefaultEmbeddingFunction()`
        # 3. Nếu dùng HuggingFace / SentenceTransformers (BGE-M3, PhoBERT...):
        #    - Dùng thư viện `sentence_transformers`
        # =========================================================================

        raise NotImplementedError(
            "TODO: Thành viên phụ trách Embedding cần tự cài đặt logic hàm embed_texts() trong src/embeddings/embedder.py!"
        )

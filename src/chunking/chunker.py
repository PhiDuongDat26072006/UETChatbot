"""
src/chunking/chunker.py - Phụ trách quy trình phân đoạn văn bản (Text Chunking).
Thành viên phụ trách: Người phân đoạn dữ liệu.
Nhiệm vụ: Cắt các tài liệu ProcessedData thành các đoạn DataChunk nhỏ hợp lý kèm overlap và metadata.
"""

from __future__ import annotations
from typing import List, Optional
from base import BaseChunker, ProcessedData, DataChunk
from src.utils.helpers import get_logger

logger = get_logger("chunking")


class UETChunker(BaseChunker):
    """
    Bộ phân đoạn văn bản theo quy chế đào tạo UET.
    Thành viên phụ trách cần kế thừa BaseChunker từ base.py và tự cài đặt hàm chunk().
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", " "]

    def chunk(self, doc: ProcessedData) -> List[DataChunk]:
        """
        Chia một tài liệu ProcessedData thành danh sách các đoạn DataChunk.
        :param doc: Đối tượng ProcessedData đã làm sạch
        :return: Danh sách các đối tượng DataChunk kèm metadata
        """
        logger.info(f"Bắt đầu phân đoạn tài liệu: '{doc.title}' (ID: {doc.id})")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT THUẬT TOÁN PHÂN ĐOẠN Ở ĐÂY:
        # Gợi ý:
        # 1. Có thể cài đặt phân đoạn đệ quy (Recursive Character Splitter) dựa trên self.separators.
        # 2. Hoặc phân đoạn theo ngữ nghĩa quy chế: Tách theo từng "Điều", "Khoản", "Chương".
        # 3. Đảm bảo mỗi chunk có kích thước xấp xỉ self.chunk_size và có độ gối đầu self.chunk_overlap.
        # 4. Gắn metadata đầy đủ vào từng DataChunk (title, doc_id, index, chương/điều nếu có).
        # =========================================================================

        raise NotImplementedError(
            "TODO: Thành viên phụ trách Chunking cần tự cài đặt logic hàm chunk() trong src/chunking/chunker.py!"
        )

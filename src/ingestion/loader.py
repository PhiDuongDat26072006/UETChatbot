"""
src/ingestion/loader.py - Phụ trách quy trình Data Ingestion & Crawling.
Thành viên phụ trách: Người thu thập & đọc dữ liệu.
Nhiệm vụ: Đọc và trích xuất dữ liệu từ PDF, Word (DOCX), Website UET, CSV... thành các đối tượng RawData.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from base import BaseDataCrawler, DataSource, RawData
from src.utils.helpers import get_logger

logger = get_logger("ingestion")


class UETDataLoader(BaseDataCrawler):
    """
    Bộ nạp và cào dữ liệu cho UET Chatbot.
    Thành viên phụ trách cần kế thừa BaseDataCrawler từ base.py và tự cài đặt hàm crawl().
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/raw")

    def crawl(self, source: DataSource) -> List[RawData]:
        """
        Thu thập dữ liệu từ một nguồn cụ thể (URL trang web hoặc đường dẫn file).
        :param source: Đối tượng DataSource định nghĩa nguồn cần tải
        :return: Danh sách các đối tượng RawData
        """
        logger.info(f"Bắt đầu thu thập dữ liệu từ nguồn: {source.uri} (loại: {source.source_type})")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT PHẦN NÀY:
        # Gợi ý:
        # 1. Nếu source.source_type == "web":
        #    - Sử dụng thư viện `requests` + `beautifulsoup4` để cào trang uet.vnu.edu.vn
        # 2. Nếu source.source_type == "file" hoặc "directory":
        #    - File PDF: dùng `pymupdf` (fitz)
        #    - File Word: dùng `python-docx`
        #    - File TXT/MD: đọc trực tiếp file UTF-8
        # 3. Đóng gói kết quả thành danh sách các đối tượng `RawData(...)` theo base.py
        # =========================================================================

        raise NotImplementedError(
            "TODO: Thành viên phụ trách Ingestion cần tự cài đặt logic hàm crawl() trong src/ingestion/loader.py!"
        )

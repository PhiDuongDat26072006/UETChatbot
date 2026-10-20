"""Module tải file tài liệu song song đa luồng."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

from ..config.session import DOWNLOAD_TIMEOUT

logger = logging.getLogger(__name__)
WORKERS_DOWNLOAD = 8


def download_file(session: requests.Session, file_url: str, output_folder: str) -> bool:
    """Tải một file từ URL và lưu vào thư mục đích dạng stream."""
    try:
        filename = os.path.basename(urlparse(file_url).path)
        if not filename:
            return False

        dest_path = os.path.join(output_folder, filename)
        # Bỏ qua nếu file đã tồn tại và có kích thước lớn hơn 0
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return True

        with session.get(file_url, stream=True, timeout=DOWNLOAD_TIMEOUT, verify=False) as r:
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                return True
            logger.warning("Download failed: %s (HTTP %s)", file_url, r.status_code)
    except Exception as error:
        logger.warning("Download failed: %s (%s)", file_url, error)
        logger.debug("Download failure", exc_info=True)
    return False


def download_files_parallel(
    session: requests.Session,
    file_list: list[str],
    output_folder: str,
    max_workers: int = WORKERS_DOWNLOAD,
) -> int:
    """Tải danh sách file song song với ThreadPoolExecutor."""
    total_files = len(file_list)
    if total_files == 0:
        return 0

    logger.info("[DOWNLOADING] %s attachments", total_files)
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, session, u, output_folder): u for u in file_list}
        for future in as_completed(futures):
            if future.result():
                success_count += 1

    return success_count

"""Module tải file tài liệu song song đa luồng."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import requests

from ..config import DOWNLOAD_TIMEOUT, WORKERS_DOWNLOAD


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
    except Exception:
        pass
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

    print(f"[*] Đang tải {total_files} file tài liệu song song ({max_workers} threads)...")
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, session, u, output_folder): u for u in file_list}
        for idx, future in enumerate(as_completed(futures), start=1):
            if future.result():
                success_count += 1
            if idx % 10 == 0 or idx == total_files:
                print(f"    -> Đã xử lý {idx}/{total_files} files (thành công: {success_count})...")

    return success_count

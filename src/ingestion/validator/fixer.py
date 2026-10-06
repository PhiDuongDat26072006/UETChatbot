"""Xác thực hàng loạt và cập nhật endpoints.txt cho các đơn vị."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import create_http_session
from .detector import check_and_fix_endpoint
from .runner import verify_all_endpoints

__all__ = ["verify_endpoint_file", "verify_all_endpoints"]


def verify_endpoint_file(file_path: str, max_workers: int = 10) -> dict:
    """Xác thực toàn bộ URL trong file data/raw_data/<domain>/endpoints/endpoints.txt."""
    if not os.path.exists(file_path):
        return {"total": 0, "valid": 0, "corrected": 0, "dropped": 0, "final": 0}

    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    session = create_http_session(pool_size=15, max_retries=2)
    final_urls = set()
    valid_count = corrected_count = dropped_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_and_fix_endpoint, u, session): u for u in urls}
        for future in as_completed(futures):
            orig_url, result_url, status = future.result()
            if result_url:
                final_urls.add(result_url)
                if status == "CORRECTED_SLASH":
                    corrected_count += 1
                else:
                    valid_count += 1
            else:
                dropped_count += 1

    sorted_urls = sorted(final_urls)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for u in sorted_urls:
            f.write(u + "\n")

    return {
        "total": len(urls),
        "valid": valid_count,
        "corrected": corrected_count,
        "dropped": dropped_count,
        "final": len(sorted_urls)
    }

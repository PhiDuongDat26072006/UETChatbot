"""Quét và trích xuất file tài liệu từ WordPress Media API."""
from __future__ import annotations

import requests

from ..config import REQUEST_TIMEOUT
from ..filters import clean_url, is_document_file


def fetch_wp_media_documents(session: requests.Session, base_url: str, max_pages: int = 5) -> set[str]:
    """Quét API /wp-json/wp/v2/media: CHỈ giữ lại các file tài liệu (.pdf, .docx...), bỏ qua ảnh."""
    files = set()
    page = 1
    clean_base = base_url.rstrip("/")
    while page <= max_pages:
        api_url = f"{clean_base}/wp-json/wp/v2/media?per_page=100&page={page}"
        try:
            res = session.get(api_url, timeout=REQUEST_TIMEOUT, verify=False)
            if res.status_code != 200 or "application/json" not in res.headers.get("Content-Type", ""):
                break
            data = res.json()
            if not data or not isinstance(data, list):
                break

            for item in data:
                source_url = item.get("source_url")
                if source_url:
                    cleaned = clean_url(source_url)
                    if is_document_file(cleaned):
                        files.add(cleaned)

            total_pages = int(res.headers.get("X-WP-TotalPages", 1))
            if page >= total_pages:
                break
            page += 1
        except Exception:
            break
    return files

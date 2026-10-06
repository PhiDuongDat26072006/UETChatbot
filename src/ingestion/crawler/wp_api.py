"""Module tương tác và trích xuất dữ liệu từ WordPress REST API."""
from __future__ import annotations

import requests

from ..config import REQUEST_TIMEOUT
from ..filters import (
    clean_url,
    is_document_file,
    is_valid_html_endpoint,
    normalize_endpoint_url,
)


def fetch_wp_media_documents(session: requests.Session, base_url: str, max_pages: int = 5) -> set[str]:
    """Quét API /wp-json/wp/v2/media: CHỈ giữ lại các file tài liệu (.pdf, .docx...), bỏ qua toàn bộ ảnh."""
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


def fetch_wp_posts_and_pages(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    target_domain: str,
    max_pages: int = 5
) -> set[str]:
    """Quét API /wp-json/wp/v2/posts và pages: chỉ lấy bài viết và trang HTML hợp lệ."""
    endpoints = set()
    page = 1
    clean_base = base_url.rstrip("/")
    while page <= max_pages:
        api_url = f"{clean_base}/wp-json/wp/v2/{endpoint}?per_page=100&page={page}"
        try:
            res = session.get(api_url, timeout=REQUEST_TIMEOUT, verify=False)
            if res.status_code != 200 or "application/json" not in res.headers.get("Content-Type", ""):
                break
            data = res.json()
            if not data or not isinstance(data, list):
                break

            for item in data:
                link = item.get("link")
                if link:
                    norm = normalize_endpoint_url(link)
                    if is_valid_html_endpoint(norm, target_domain):
                        endpoints.add(norm)

            total_pages = int(res.headers.get("X-WP-TotalPages", 1))
            if page >= total_pages:
                break
            page += 1
        except Exception:
            break
    return endpoints


def fetch_wp_category_posts(
    session: requests.Session,
    base_url: str,
    category_id: int,
    max_pages: int = 5
) -> set[str]:
    """Quét bài viết theo category ID cụ thể qua WordPress API (ví dụ Viện IAI category 146)."""
    endpoints = set()
    page = 1
    clean_base = base_url.rstrip("/")
    while page <= max_pages:
        api_url = f"{clean_base}/wp-json/wp/v2/posts?categories={category_id}&per_page=100&page={page}"
        try:
            res = session.get(api_url, timeout=REQUEST_TIMEOUT, verify=False)
            if res.status_code != 200 or "application/json" not in res.headers.get("Content-Type", ""):
                break
            data = res.json()
            if not data or not isinstance(data, list):
                break

            for item in data:
                link = item.get("link")
                if link:
                    norm = normalize_endpoint_url(link)
                    endpoints.add(norm)

            total_pages = int(res.headers.get("X-WP-TotalPages", 1))
            if page >= total_pages:
                break
            page += 1
        except Exception:
            break
    return endpoints

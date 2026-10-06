"""Module xử lý làm sạch và kiểm tra lọc URL (filters)."""
from __future__ import annotations

import os
from urllib.parse import urlparse

from .config import (
    CLOUD_STORAGE_DOMAINS,
    DOCUMENT_EXTENSIONS,
    IGNORED_EXTENSIONS,
    IGNORED_PATH_REGEX,
)
from .filter_helpers import clean_url, get_domain_folder_name, normalize_endpoint_url

__all__ = [
    "clean_url",
    "normalize_endpoint_url",
    "get_domain_folder_name",
    "is_document_file",
    "is_media_or_asset_file",
    "is_valid_html_endpoint",
    "is_allowed_file_domain",
    "is_cloud_storage_url",
]


def is_document_file(url: str) -> bool:
    """Kiểm tra xem URL có phải là file tài liệu cần tải (pdf, docx, ...) hay không."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)


def is_media_or_asset_file(url: str) -> bool:
    """Kiểm tra xem URL có phải là ảnh (png, svg, jpg...), css, js hoặc media rác."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IGNORED_EXTENSIONS)


def is_valid_html_endpoint(url: str, target_domain: str, subpath: str = None) -> bool:
    """Kiểm tra xem URL có phải là trang HTML bài viết / thông tin hợp lệ hay không."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    clean_netloc = parsed.netloc.lower().split(':')[0].removeprefix("www.")
    clean_target = target_domain.lower().split(':')[0].removeprefix("www.")

    if clean_netloc != clean_target:
        uet_aliases = ("uet.vnu.edu.vn", "uet.edu.vn")
        if not (clean_netloc in uet_aliases and clean_target in uet_aliases):
            return False

    if subpath:
        path_lower = parsed.path.lower()
        if subpath not in path_lower and f"category/{subpath}" not in path_lower:
            return False

    if is_document_file(url) or is_media_or_asset_file(url) or IGNORED_PATH_REGEX.search(parsed.path):
        return False

    last_segment = parsed.path.split('/')[-1]
    if '.' in last_segment:
        ext = os.path.splitext(last_segment)[1].lower()
        if ext not in (".html", ".htm", ".php"):
            return False

    return True


def is_allowed_file_domain(url: str, target_domain: str) -> bool:
    """Kiểm tra xem file tài liệu có thuộc domain mục tiêu hoặc CDN liên kết hay không."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().split(':')[0]
    clean_netloc = netloc.removeprefix("www.")
    clean_target = target_domain.lower().split(':')[0].removeprefix("www.")

    if clean_netloc == clean_target or clean_netloc.endswith("." + clean_target):
        return True
    return any(d in netloc for d in ("uet.vnu.edu.vn", "uet.edu.vn", "cdneverest.net"))


def is_cloud_storage_url(
    url: str,
    allowed_domains: tuple[str, ...] = CLOUD_STORAGE_DOMAINS,
) -> bool:
    """Kiểm tra URL có thuộc danh sách domain lưu trữ đám mây được phép hay không."""
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False
    netloc = parsed.netloc.lower().split(":")[0]
    return any(netloc == d or netloc.endswith("." + d) for d in allowed_domains)

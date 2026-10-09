"""Module xử lý làm sạch, chuẩn hóa và kiểm tra lọc URL (filters)."""
from __future__ import annotations

import os
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .config.faculties import faculty_storage_name, find_faculty, url_in_scope
from .config.ignore_rules import (
    CLOUD_STORAGE_DOMAINS,
    DOCUMENT_EXTENSIONS,
    IGNORED_EXTENSIONS,
    IGNORED_PATH_REGEX,
    IGNORED_QUERY_PARAMETERS,
)

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


def clean_url(url: str) -> str:
    """Xóa anchor (#), loại bỏ query cache/tracking rác (utm_*, fbclid, ...) và chuẩn hóa slash."""
    url = url.split("#")[0].strip()
    parsed = urlparse(url)

    if parsed.query:
        params = parse_qsl(parsed.query, keep_blank_values=False)
        kept_params = []
        for k, v in params:
            k_lower = k.lower()
            if k_lower.startswith("utm_") or k_lower in IGNORED_QUERY_PARAMETERS:
                continue
            kept_params.append((k, v))
        new_query = urlencode(kept_params)
        url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ""))

    parsed = urlparse(url)
    norm_path = re.sub(r'/{2,}', '/', parsed.path)
    return urlunparse((parsed.scheme, parsed.netloc, norm_path, parsed.params, parsed.query, ""))


def normalize_endpoint_url(url: str) -> str:
    """Chuẩn hóa endpoint: chuẩn hóa path, giữ nguyên trạng thái slash nguyên bản từ HTML hoặc link."""
    cleaned = clean_url(url)
    parsed = urlparse(cleaned)
    path = parsed.path or "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, ""))


def get_domain_folder_name(url: str) -> str:
    """Lấy tên thư mục phân loại theo domain."""
    parsed = urlparse(url)
    target = find_faculty(url)
    if target:
        return faculty_storage_name(target["faculty_id"])
    netloc = parsed.netloc or parsed.path
    return netloc.removeprefix("www.")


def is_document_file(url: str) -> bool:
    """Kiểm tra xem URL có phải là file tài liệu cần tải (pdf, docx, ...) hay không."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)


def is_media_or_asset_file(url: str) -> bool:
    """Kiểm tra xem URL có phải là ảnh (png, svg, jpg...), css, js hoặc media rác."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IGNORED_EXTENSIONS)


def is_valid_html_endpoint(
    url: str, target_domain: str, subpath: str | None = None,
    *, allowed_path_prefixes: list[str] | tuple[str, ...] = (),
) -> bool:
    """Kiểm tra xem URL có phải là trang HTML bài viết / thông tin hợp lệ hay không."""
    parsed = urlparse(url)
    if not url_in_scope(url, target_domain, allowed_path_prefixes):
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

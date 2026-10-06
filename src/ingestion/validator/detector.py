"""Phát hiện lỗi trang, Soft-404 và kiểm tra sửa lỗi trailing slash cho URL."""
from __future__ import annotations

from urllib.parse import urlparse, urlunparse
import requests


def get_alternative_slash_url(url: str) -> str:
    """Tạo URL đảo ngược trạng thái trailing slash (có / <-> không /)."""
    parsed = urlparse(url)
    path = parsed.path
    if not path or path == "/":
        return url
    alt_path = path.rstrip("/") if path.endswith("/") else (path + "/")
    return urlunparse((parsed.scheme, parsed.netloc, alt_path, parsed.params, parsed.query, parsed.fragment))


def is_page_not_found(r: requests.Response) -> bool:
    """Kiểm tra xem trang có phải là lỗi 404 hoặc Soft-404 hay không."""
    if r.status_code == 404:
        return True
    if r.status_code != 200:
        return False

    clean_path = urlparse(r.url).path.rstrip("/").lower()
    if clean_path in ("/page/404", "/404", "/error", "/not-found", "/error-404"):
        return True

    text_lower = r.text[:6000].lower()
    error_patterns = (
        "<title>page not found",
        "<title>404 not found",
        "<title>404 - not found",
        "<title>không tìm thấy",
        "<title>lỗi 404",
    )
    return any(pat in text_lower for pat in error_patterns)


# Re-export check_and_fix_endpoint for backwards compatibility
from .checker import check_and_fix_endpoint  # noqa: E402

__all__ = ["get_alternative_slash_url", "is_page_not_found", "check_and_fix_endpoint"]

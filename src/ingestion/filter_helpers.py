"""Hàm làm sạch chuỗi URL và xác định thư mục domain."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def clean_url(url: str) -> str:
    """Xóa anchor (#), loại bỏ query cache/tracking rác (ver, utm_*, fbclid, ...) và chuẩn hóa slash."""
    url = url.split("#")[0].strip()
    parsed = urlparse(url)

    if parsed.query:
        params = parse_qsl(parsed.query, keep_blank_values=False)
        kept_params = []
        for k, v in params:
            k_lower = k.lower()
            if k_lower.startswith("utm_") or k_lower in ("ver", "preview", "replytocom", "share", "ref", "fbclid"):
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
    if "vien-tri-tue-nhan-tao" in parsed.path:
        return "iai.uet.vnu.edu.vn"
    netloc = parsed.netloc or parsed.path
    return netloc.removeprefix("www.")

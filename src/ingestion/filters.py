"""Module xử lý làm sạch và kiểm tra lọc URL (filters)."""
import os
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .config import (
    CLOUD_STORAGE_DOMAINS,
    DOCUMENT_EXTENSIONS,
    IGNORED_EXTENSIONS,
    IGNORED_PATH_REGEX,
)


def clean_url(url: str) -> str:
    """Xóa anchor (#), loại bỏ query cache/tracking rác (ver, utm_*, fbclid, ...) và chuẩn hóa dấu gạch chéo."""
    url = url.split("#")[0].strip()
    parsed = urlparse(url)

    # Loại bỏ query parameters rác như ?ver=..., ?preview=..., ?utm_*
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

    # Chuẩn hóa path: loại bỏ các dấu gạch chéo liên tiếp (//)
    parsed = urlparse(url)
    norm_path = re.sub(r'/{2,}', '/', parsed.path)
    return urlunparse((parsed.scheme, parsed.netloc, norm_path, parsed.params, parsed.query, ""))


def normalize_endpoint_url(url: str) -> str:
    """Chuẩn hóa endpoint: chuẩn hóa path, giữ nguyên trạng thái slash nguyên bản từ HTML hoặc link."""
    cleaned = clean_url(url)
    parsed = urlparse(cleaned)
    path = parsed.path
    if not path:
        path = "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, ""))


def is_document_file(url: str) -> bool:
    """Kiểm tra xem URL có phải là file tài liệu cần tải (pdf, docx, ...) hay không."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)


def is_media_or_asset_file(url: str) -> bool:
    """Kiểm tra xem URL có phải là ảnh (png, svg, jpg...), css, js hoặc tài nguyên không phải bài viết."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IGNORED_EXTENSIONS)


def is_valid_html_endpoint(url: str, target_domain: str, subpath: str = None) -> bool:
    """Kiểm tra xem URL có phải là trang HTML bài viết / thông tin hợp lệ hay không."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    netloc = parsed.netloc.lower().split(':')[0]
    target_netloc = target_domain.lower().split(':')[0]
    clean_netloc = netloc.removeprefix("www.")
    clean_target = target_netloc.removeprefix("www.")

    # Domain phải khớp
    if clean_netloc != clean_target:
        # Hỗ trợ trường hợp uet.vnu.edu.vn và uet.edu.vn tương đương
        if not (clean_netloc in ("uet.vnu.edu.vn", "uet.edu.vn") and clean_target in ("uet.vnu.edu.vn", "uet.edu.vn")):
            return False

    # Nếu có subpath (ví dụ: Viện IAI tại /vien-tri-tue-nhan-tao/)
    if subpath:
        path_lower = parsed.path.lower()
        if subpath not in path_lower and f"category/{subpath}" not in path_lower:
            return False

    # Không phải tài liệu và không phải file media/asset
    if is_document_file(url) or is_media_or_asset_file(url):
        return False

    # Không nằm trong danh sách pattern rác (wp-json, wp-content, feed, login, base64 cache...)
    if IGNORED_PATH_REGEX.search(parsed.path):
        return False

    # Bỏ qua nếu có đuôi mở rộng lạ không phải trang web (.html, .htm, .php hoặc không có đuôi)
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
    target_netloc = target_domain.lower().split(':')[0]
    clean_netloc = netloc.removeprefix("www.")
    clean_target = target_netloc.removeprefix("www.")
    if clean_netloc == clean_target or clean_netloc.endswith("." + clean_target):
        return True
    if "uet.vnu.edu.vn" in netloc or "uet.edu.vn" in netloc or "cdneverest.net" in netloc:
        return True
    return False


def get_domain_folder_name(url: str) -> str:
    """Lấy tên thư mục phân loại theo domain."""
    parsed = urlparse(url)
    if "vien-tri-tue-nhan-tao" in parsed.path:
        return "iai.uet.vnu.edu.vn"
    netloc = parsed.netloc or parsed.path
    return netloc.removeprefix("www.")


def is_cloud_storage_url(
    url: str,
    allowed_domains: tuple[str, ...] = CLOUD_STORAGE_DOMAINS,
) -> bool:
    """Kiểm tra URL có thuộc danh sách domain lưu trữ đám mây được phép hay không.

    Hỗ trợ khớp chính xác domain hoặc subdomain (ví dụ: uetvnu-my.sharepoint.com,
    www.dropbox.com, drive.google.com).
    """
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False
    netloc = parsed.netloc.lower().split(":")[0]
    if not netloc:
        return False
    for domain in allowed_domains:
        if netloc == domain or netloc.endswith("." + domain):
            return True
    return False

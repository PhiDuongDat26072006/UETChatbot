"""Phát hiện lỗi trang, Soft-404 và kiểm tra sửa lỗi trailing slash cho URL."""
from __future__ import annotations

from urllib.parse import urlparse, urlunparse
import requests

from ..filters import clean_url


def get_alternative_slash_url(url: str) -> str:
    """Tạo URL đảo ngược trạng thái trailing slash (có / <-> không /)."""
    parsed = urlparse(url)
    path = parsed.path
    if not path or path == "/":
        return url
    if path.endswith("/"):
        alt_path = path.rstrip("/")
    else:
        alt_path = path + "/"
    return urlunparse((parsed.scheme, parsed.netloc, alt_path, parsed.params, parsed.query, parsed.fragment))


def is_page_not_found(r: requests.Response) -> bool:
    """
    Kiểm tra xem trang có phải là lỗi 404 hoặc Page Not Found hay không.
    Bao gồm cả trường hợp HTTP status là 404, hoặc status 200 nhưng server chuyển hướng sang trang 404.
    """
    if r.status_code == 404:
        return True
    if r.status_code != 200:
        return False

    # Kiểm tra URL cuối cùng sau redirect (ví dụ /page/404, /404)
    parsed_url = urlparse(r.url)
    clean_path = parsed_url.path.rstrip("/").lower()
    if clean_path in ("/page/404", "/404", "/error", "/not-found", "/error-404"):
        return True

    # Kiểm tra nội dung thẻ title hoặc tiêu đề thông báo lỗi
    text_lower = r.text[:6000].lower()
    error_patterns = (
        "<title>page not found",
        "<title>404 not found",
        "<title>404 - not found",
        "<title>không tìm thấy",
        "<title>lỗi 404",
    )
    for pattern in error_patterns:
        if pattern in text_lower:
            return True

    return False


def check_and_fix_endpoint(url: str, session: requests.Session) -> tuple[str, str, str]:
    """
    Kiểm tra 1 URL:
    - Chuẩn hóa HTTP sang HTTPS nếu server hỗ trợ.
    - Nếu 200 và nội dung chuẩn: giữ nguyên.
    - Nếu lỗi (404/Page not found): thử đảo trạng thái trailing slash (/ <-> không /).
    - Nếu đảo slash thành công: trả về URL đã sửa.
    - Nếu cả hai đều 404 thực tế: trả về None (loại bỏ).
    - Nếu server bị timeout/connection error: GIỮ NGUYÊN URL, không xóa nhầm.
    """
    orig_url = url.strip()
    if not orig_url:
        return orig_url, None, "EMPTY"

    # Loại bỏ API click tracking rác
    parsed_orig = urlparse(orig_url)
    if "/api/Content/Decl/" in parsed_orig.path:
        return orig_url, None, "DROPPED_TRACKING"

    # Ưu tiên kiểm tra giao thức HTTPS nếu là HTTP
    test_url = orig_url
    if test_url.startswith("http://"):
        test_url = "https://" + test_url[7:]

    # 1. Thử URL chính (HTTPS)
    try:
        r = session.get(test_url, verify=False, timeout=8, allow_redirects=True)
        if not is_page_not_found(r):
            # URL hoạt động tốt. Nếu có redirect nội bộ cùng domain chuẩn, giữ URL redirect hoặc test_url
            r_parsed = urlparse(r.url)
            orig_parsed = urlparse(test_url)
            if r_parsed.netloc.lower().removeprefix("www.") == orig_parsed.netloc.lower().removeprefix("www."):
                final_clean = clean_url(r.url)
                return orig_url, final_clean, "OK"
            return orig_url, test_url, "OK"

        # 2. Nếu URL chính bị 404 / Page Not Found -> Thử đảo trailing slash
        alt_url = get_alternative_slash_url(test_url)
        if alt_url != test_url:
            try:
                r_alt = session.get(alt_url, verify=False, timeout=8, allow_redirects=True)
                if not is_page_not_found(r_alt):
                    r_alt_parsed = urlparse(r_alt.url)
                    alt_parsed = urlparse(alt_url)
                    if r_alt_parsed.netloc.lower().removeprefix("www.") == alt_parsed.netloc.lower().removeprefix("www."):
                        final_clean = clean_url(r_alt.url)
                        return orig_url, final_clean, "CORRECTED_SLASH"
                    return orig_url, alt_url, "CORRECTED_SLASH"
            except Exception:
                pass

        # 3. Thử lại URL HTTP gốc nếu test_url HTTPS lỗi (phòng trường hợp site chỉ hỗ trợ HTTP)
        if test_url != orig_url:
            try:
                r_orig = session.get(orig_url, verify=False, timeout=8, allow_redirects=True)
                if not is_page_not_found(r_orig):
                    return orig_url, orig_url, "OK"
                alt_orig = get_alternative_slash_url(orig_url)
                if alt_orig != orig_url:
                    r_alt_orig = session.get(alt_orig, verify=False, timeout=8, allow_redirects=True)
                    if not is_page_not_found(r_alt_orig):
                        return orig_url, alt_orig, "CORRECTED_SLASH"
            except Exception:
                pass

        # Cả URL gốc và đảo slash đều trả về 404
        return orig_url, None, "DROPPED_DEAD"

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        # Tránh xóa nhầm URL khi server đối phương bị quá tải / timeout
        return orig_url, orig_url, "KEPT_TIMEOUT"
    except Exception:
        return orig_url, orig_url, "KEPT_ERROR"

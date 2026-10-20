"""Kiểm tra phản hồi HTTP và xử lý lỗi trailing slash / chuyển hướng cho endpoint."""
from __future__ import annotations

from urllib.parse import urlparse
import requests

from ..filters import clean_url
from .detector import get_alternative_slash_url, is_page_not_found


def check_and_fix_endpoint(url: str, session: requests.Session) -> tuple[str, str | None, str]:
    """Kiểm tra 1 URL: chuẩn hóa HTTPS, thử đảo slash nếu 404, giữ nguyên nếu timeout."""
    orig_url = url.strip()
    if not orig_url:
        return orig_url, None, "EMPTY"

    parsed_orig = urlparse(orig_url)
    if "/api/Content/Decl/" in parsed_orig.path:
        return orig_url, None, "DROPPED_TRACKING"

    test_url = "https://" + orig_url[7:] if orig_url.startswith("http://") else orig_url

    try:
        r = session.get(test_url, verify=False, timeout=8, allow_redirects=True)
        if not is_page_not_found(r):
            r_netloc = urlparse(r.url).netloc.lower().removeprefix("www.")
            t_netloc = urlparse(test_url).netloc.lower().removeprefix("www.")
            return orig_url, (clean_url(r.url) if r_netloc == t_netloc else test_url), "OK"

        alt_url = get_alternative_slash_url(test_url)
        if alt_url != test_url:
            try:
                r_alt = session.get(alt_url, verify=False, timeout=8, allow_redirects=True)
                if not is_page_not_found(r_alt):
                    ra_netloc = urlparse(r_alt.url).netloc.lower().removeprefix("www.")
                    a_netloc = urlparse(alt_url).netloc.lower().removeprefix("www.")
                    return orig_url, (clean_url(r_alt.url) if ra_netloc == a_netloc else alt_url), "CORRECTED_SLASH"
            except Exception:
                pass

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

        return orig_url, None, "DROPPED_DEAD"

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return orig_url, orig_url, "KEPT_TIMEOUT"
    except Exception:
        return orig_url, orig_url, "KEPT_ERROR"

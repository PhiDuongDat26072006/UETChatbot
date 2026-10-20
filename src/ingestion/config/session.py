"""HTTP Session factory tái sử dụng connection pool và quản lý cấu hình HTTP."""
from __future__ import annotations

from urllib.parse import urljoin

import requests
import urllib3

from .faculties import url_in_scope

# Tắt cảnh báo InsecureRequestWarning cho toàn bộ pipeline
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT: int = 10
DOWNLOAD_TIMEOUT: int = 20
VALIDATION_TIMEOUT: int = 8
TITLE_TIMEOUT: int = 5
MAX_RETRIES: int = 1
VALIDATION_RETRIES: int = 2


def create_http_session(
    pool_size: int = 15,
    max_retries: int = MAX_RETRIES,
    headers: dict[str, str] | None = None,
) -> requests.Session:
    """Tạo requests.Session với connection pool và retry policy."""
    session = requests.Session()
    session.headers.update(headers or DEFAULT_HEADERS)
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=max_retries,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_source_response(
    session: requests.Session,
    url: str,
    domain: str,
    allowed_path_prefixes: list[str] | tuple[str, ...] = (),
    timeout: int = REQUEST_TIMEOUT,
    *, stream: bool = False,
) -> requests.Response:
    """Fetch scoped HTML without following redirects outside the source.

    Unscoped sources retain requests' redirect behavior. Scoped responses are
    checked before each request; callers must close the returned response.
    """
    if not allowed_path_prefixes:
        return session.get(url, timeout=timeout, verify=False, allow_redirects=True, stream=stream)
    for _ in range(session.max_redirects + 1):
        if not url_in_scope(url, domain, allowed_path_prefixes):
            raise requests.RequestException(f"URL outside source scope: {url}")
        response = session.get(url, timeout=timeout, verify=False, allow_redirects=False, stream=stream)
        if not response.is_redirect:
            if not url_in_scope(response.url, domain, allowed_path_prefixes):
                response.close()
                raise requests.RequestException(f"Response outside source scope: {response.url}")
            return response
        url = urljoin(response.url, response.headers["Location"])
        response.close()
    raise requests.TooManyRedirects(f"Too many redirects: {url}")

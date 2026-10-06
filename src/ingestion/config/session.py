"""HTTP Session factory tái sử dụng connection pool."""
from __future__ import annotations

import requests
import urllib3

from .constants import HEADERS

# Tắt cảnh báo InsecureRequestWarning một lần duy nhất cho toàn bộ pipeline
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def create_http_session(
    pool_size: int = 15,
    max_retries: int = 1,
) -> requests.Session:
    """Tạo requests.Session tái sử dụng connection pool cho pipeline."""
    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=max_retries,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

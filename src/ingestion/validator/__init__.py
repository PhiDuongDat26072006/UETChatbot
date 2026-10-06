"""Package xác thực trailing slash và phát hiện Soft-404 cho endpoints."""
from __future__ import annotations

from .detector import (
    check_and_fix_endpoint,
    get_alternative_slash_url,
    is_page_not_found,
)
from .fixer import (
    verify_all_endpoints,
    verify_endpoint_file,
)

__all__ = [
    "get_alternative_slash_url",
    "is_page_not_found",
    "check_and_fix_endpoint",
    "verify_endpoint_file",
    "verify_all_endpoints",
]

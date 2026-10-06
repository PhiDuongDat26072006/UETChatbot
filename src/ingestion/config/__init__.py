"""Cấu hình toàn diện cho module Ingestion.

Re-export toàn bộ hằng số, đường dẫn và session factory để tương thích ngược 100%.
"""
from __future__ import annotations

from .constants import (
    A_HREF_REGEX,
    CLOUD_STORAGE_DOMAINS,
    DOCUMENT_EXTENSIONS,
    DOWNLOAD_TIMEOUT,
    EMBED_SRC_REGEX,
    HEADERS,
    IGNORED_EXTENSIONS,
    IGNORED_PATH_PATTERNS,
    IGNORED_PATH_REGEX,
    MAX_CRAWL_PAGES_PER_TARGET,
    REQUEST_TIMEOUT,
    WORKERS_API,
    WORKERS_DOWNLOAD,
    WORKERS_SCRAPE,
)
from .faculties import (
    DOMAIN_FOLDERS,
    DOMAIN_TO_UNIT_MAP,
    FACULTY_TARGETS,
    TARGET_DOMAINS,
    resolve_unit,
)
from .paths import (
    BASE_DIR,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    _load_repo_paths,
)
from .session import create_http_session

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "_load_repo_paths",
    "DOCUMENT_EXTENSIONS",
    "CLOUD_STORAGE_DOMAINS",
    "IGNORED_EXTENSIONS",
    "IGNORED_PATH_PATTERNS",
    "IGNORED_PATH_REGEX",
    "A_HREF_REGEX",
    "EMBED_SRC_REGEX",
    "HEADERS",
    "WORKERS_API",
    "WORKERS_SCRAPE",
    "WORKERS_DOWNLOAD",
    "REQUEST_TIMEOUT",
    "DOWNLOAD_TIMEOUT",
    "MAX_CRAWL_PAGES_PER_TARGET",
    "FACULTY_TARGETS",
    "TARGET_DOMAINS",
    "DOMAIN_FOLDERS",
    "DOMAIN_TO_UNIT_MAP",
    "resolve_unit",
    "create_http_session",
]

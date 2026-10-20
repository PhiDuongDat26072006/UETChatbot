"""Public ingestion configuration entry points."""
from .faculties import FACULTY_TARGETS, faculty_storage_name, find_faculty, resolve_unit
from .paths import BASE_DIR, DATA_DIR, PROCESSED_DATA_DIR
from .session import create_http_session

__all__ = [
    "FACULTY_TARGETS", "faculty_storage_name", "find_faculty", "resolve_unit",
    "BASE_DIR", "DATA_DIR", "PROCESSED_DATA_DIR", "create_http_session",
]

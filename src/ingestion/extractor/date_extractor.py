"""Module trích xuất và chuẩn hóa thời gian công bố (Published Date Extraction)."""
from __future__ import annotations

from pathlib import Path

from .date_filename import extract_date_from_filename
from .date_html import extract_html_date
from .date_patterns import (
    EVERGREEN_CATEGORIES,
    SEASONAL_CATEGORIES,
    _year_of,
    is_evergreen_page,
    normalize_date_string,
)
from .date_text import extract_date_from_text, extract_date_from_url

__all__ = [
    "extract_date_from_url",
    "extract_date_from_text",
    "extract_date_from_filename",
    "extract_html_date",
    "extract_doc_date",
    "is_recent_document",
    "is_evergreen_page",
    "normalize_date_string",
]


def extract_doc_date(file_path: Path | str, text: str | None = None) -> str | None:
    """Trích xuất thời gian ban hành/công bố của file tài liệu."""
    path = Path(file_path)
    if text:
        d = extract_date_from_text(text, max_chars=1000)
        if d:
            return d
    return extract_date_from_filename(path.name)


def is_recent_document(
    published_date: str | None,
    category: str | None = None,
    min_year: int = 2020,
) -> bool:
    """Kiểm tra tài liệu có nên được giữ lại theo độ mới (từ min_year trở lại đây)."""
    if not published_date:
        return True
    if category in EVERGREEN_CATEGORIES:
        return True
    if category not in SEASONAL_CATEGORIES:
        return True
    year = _year_of(published_date)
    return True if year is None else year >= min_year

"""Chuẩn hóa chuỗi ngày tháng và re-export các hằng số ngày tháng."""
from __future__ import annotations

import re

from .date_rules import (
    CMS_INITIAL_SEED_DATES,
    EVERGREEN_CATEGORIES,
    EVERGREEN_URL_MARKERS,
    META_DATE_PROPERTIES,
    SEASONAL_CATEGORIES,
    _year_of,
    is_evergreen_page,
)

__all__ = [
    "CMS_INITIAL_SEED_DATES",
    "EVERGREEN_URL_MARKERS",
    "EVERGREEN_CATEGORIES",
    "SEASONAL_CATEGORIES",
    "META_DATE_PROPERTIES",
    "is_evergreen_page",
    "_year_of",
    "normalize_date_string",
]


def normalize_date_string(raw_date: str | None) -> str | None:
    """Chuẩn hóa chuỗi ngày tháng về định dạng chuẩn YYYY-MM-DD, YYYY-MM hoặc YYYY."""
    if not raw_date or not isinstance(raw_date, str):
        return None
    s = raw_date.strip()
    if not s:
        return None

    # 1. ISO 8601: YYYY-MM-DD...
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2035 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # 2. Định dạng ngày Việt Nam: DD/MM/YYYY hoặc DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2035 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # 3. Định dạng YYYY/MM/DD
    m = re.match(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2035 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # 4. Định dạng YYYY-MM
    m = re.match(r"^(\d{4})-(\d{1,2})$", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1990 <= y <= 2035 and 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}"

    # 5. Định dạng chỉ có năm YYYY
    m = re.match(r"^(\d{4})$", s)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2035:
            return f"{y:04d}"

    return None

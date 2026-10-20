"""Bóc tách ngày tháng từ tên file tài liệu."""
from __future__ import annotations

import re
import urllib.parse


def extract_date_from_filename(filename: str) -> str | None:
    """Trích xuất ngày/năm từ tên file tài liệu."""
    if not filename:
        return None
    name = urllib.parse.unquote(filename)

    # 1. YYYY.MM.DD hoặc YYYY-MM-DD hoặc YYYY_MM_DD
    m = re.search(r"(?:^|[^\d])(20[12]\d)[\._\-](0?[1-9]|1[0-2])[\._\-](0?[1-9]|[12]\d|3[01])(?=[^\d]|$)", name)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # 2. DD.MM.YYYY hoặc DD-MM-YYYY hoặc DD_MM_YYYY
    m = re.search(r"(?:^|[^\d])(0?[1-9]|[12]\d|3[01])[\._\-](0?[1-9]|1[0-2])[\._\-](20[12]\d)(?=[^\d]|$)", name)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # 3. Năm học YYYY-YYYY
    m = re.search(r"(?:^|[^\d])(20[12]\d)[\-_]20[12]\d(?=[^\d]|$)", name)
    if m:
        return m.group(1)

    # 4. MM.YYYY hoặc MM_YYYY
    m = re.search(r"(?:^|[^\d])(0?[1-9]|1[0-2])[\._\-](20[12]\d)(?=[^\d]|$)", name)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        return f"{y:04d}-{mo:02d}"

    # 5. YYYY.MM hoặc YYYY_MM
    m = re.search(r"(?:^|[^\d])(20[12]\d)[\._\-](0?[1-9]|1[0-2])(?=[^\d]|$)", name)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return f"{y:04d}-{mo:02d}"

    # 6. Đứng sau từ hoặc ký tự
    m = re.search(r"(?:[A-Za-z_-]|^)(20[12]\d)(?:[A-Za-z_\.\-]|$)", name)
    return m.group(1) if m else None

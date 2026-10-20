"""Bóc tách ngày tháng từ đường dẫn URL và đoạn văn bản thô."""
from __future__ import annotations

import re
import urllib.parse


def extract_date_from_url(url: str) -> str | None:
    """Trích xuất ngày/tháng/năm từ đường dẫn URL."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    path = parsed.path

    m = re.search(r"/(20[12]\d)/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])(?:/|$)", path)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    m = re.search(r"/(20[12]\d)/(0?[1-9]|1[0-2])(?:/|$)", path)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"

    m = re.search(r"/(20[12]\d)(?:/|$)", path)
    return m.group(1) if m else None


def extract_date_from_text(text: str, max_chars: int = 500) -> str | None:
    """Trích xuất ngày tháng từ đoạn văn bản đầu."""
    if not text:
        return None
    snippet = text[:max_chars]

    # 1. "ngày DD tháng MM năm YYYY"
    m = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(20[12]\d)", snippet, re.IGNORECASE)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # 2. "DD/MM/YYYY" hoặc "DD-MM-YYYY"
    m = re.search(r"\b([0-3]?\d)[/\-]([01]?\d)/(20[12]\d)\b", snippet)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # 3. "tháng MM năm YYYY"
    m = re.search(r"tháng\s+(\d{1,2})\s+năm\s+(20[12]\d)", snippet, re.IGNORECASE)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}"

    # 4. "năm học YYYY-YYYY" hoặc "năm YYYY"
    m = re.search(r"năm\s+học\s+(20[12]\d)", snippet, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"\bnăm\s+(20[12]\d)\b", snippet, re.IGNORECASE)
    if m:
        return m.group(1)

    return None

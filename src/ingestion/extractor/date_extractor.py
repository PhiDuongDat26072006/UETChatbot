"""Module trích xuất và chuẩn hóa thời gian công bố (Published Date Extraction)."""
from __future__ import annotations

import logging
import re
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
import trafilatura

from .date_patterns import (
    CMS_INITIAL_SEED_DATES,
    EVERGREEN_CATEGORIES,
    META_DATE_PROPERTIES,
    SEASONAL_CATEGORIES,
    _year_of,
    is_evergreen_page,
    normalize_date_string,
)

logger = logging.getLogger(__name__)


def extract_date_from_url(url: str) -> str | None:
    """Trích xuất ngày/tháng/năm từ đường dẫn URL."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    path = parsed.path

    # Khớp YYYY/MM/DD
    m = re.search(r"/(20[12]\d)/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])(?:/|$)", path)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # Khớp YYYY/MM
    m = re.search(r"/(20[12]\d)/(0?[1-9]|1[0-2])(?:/|$)", path)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return f"{y:04d}-{mo:02d}"

    # Khớp /YYYY/
    m = re.search(r"/(20[12]\d)(?:/|$)", path)
    if m:
        return m.group(1)

    return None


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
    if m:
        return m.group(1)

    return None


def extract_html_date(
    html_text: str | None,
    url: str | None = None,
    text: str | None = None,
    soup: BeautifulSoup | None = None,
    category: str | None = None,
) -> str | None:
    """Trích xuất ngày đăng từ HTML qua Trafilatura, meta tags, URL slug và text regex."""
    if not html_text and not soup:
        if url:
            d = extract_date_from_url(url)
            if d:
                return d
        if text:
            d = extract_date_from_text(text, max_chars=500)
            if d:
                return d
        return None

    if not soup and html_text:
        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            soup = None

    trafilatura_date: str | None = None
    if html_text:
        try:
            meta = trafilatura.extract_metadata(html_text)
            if meta and meta.date:
                trafilatura_date = normalize_date_string(str(meta.date))
        except Exception as e:
            logger.debug("Trafilatura metadata extraction gặp lỗi: %s", e)

    meta_date: str | None = None
    if soup:
        for attr, val in META_DATE_PROPERTIES:
            tag = soup.find("meta", {attr: val})
            if tag and tag.get("content"):
                normalized = normalize_date_string(tag["content"])
                if normalized:
                    meta_date = normalized
                    break

        if not meta_date:
            for time_tag in soup.find_all("time"):
                dt_val = time_tag.get("datetime")
                if dt_val:
                    normalized = normalize_date_string(dt_val)
                    if normalized:
                        meta_date = normalized
                        break
                time_text = time_tag.get_text().strip()
                if time_text:
                    normalized = normalize_date_string(time_text) or extract_date_from_text(time_text, max_chars=100)
                    if normalized:
                        meta_date = normalized
                        break

    url_date = extract_date_from_url(url) if url else None
    text_date = extract_date_from_text(text, max_chars=500) if text else None

    meta_date = None if meta_date in CMS_INITIAL_SEED_DATES else meta_date
    trafilatura_date = None if trafilatura_date in CMS_INITIAL_SEED_DATES else trafilatura_date
    url_date = None if url_date in CMS_INITIAL_SEED_DATES else url_date
    text_date = None if text_date in CMS_INITIAL_SEED_DATES else text_date

    result = _select_best_date(meta_date, trafilatura_date, url_date, text_date)

    result_year = _year_of(result)
    text_year = _year_of(text_date)
    if result_year is not None and result_year < 2020 and text_year is not None and text_year >= 2020:
        return text_date

    return result


def _select_best_date(
    meta_date: str | None,
    trafilatura_date: str | None,
    url_date: str | None,
    text_date: str | None,
) -> str | None:
    """Chọn kết quả chính xác nhất từ các nguồn ngày theo thứ tự ưu tiên."""
    if meta_date:
        return meta_date
    if trafilatura_date and not trafilatura_date.endswith("-01-01"):
        return trafilatura_date
    if url_date and ("-" in url_date):
        return url_date
    if text_date:
        return text_date
    if url_date:
        return url_date
    if trafilatura_date:
        return trafilatura_date
    return None


def extract_doc_date(file_path: Path | str, text: str | None = None) -> str | None:
    """Trích xuất thời gian ban hành/công bố của file tài liệu."""
    path = Path(file_path)
    if text:
        d = extract_date_from_text(text, max_chars=1000)
        if d:
            return d
    d = extract_date_from_filename(path.name)
    if d:
        return d
    return None


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
    if year is None:
        return True
    return year >= min_year

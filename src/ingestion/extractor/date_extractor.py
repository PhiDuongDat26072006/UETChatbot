"""Module trích xuất và chuẩn hóa thời gian công bố (Published Date Extraction)."""
from __future__ import annotations

import logging
import re
import urllib.parse
from pathlib import Path

from bs4 import BeautifulSoup
import trafilatura

from ..classifier.taxonomy import Category

logger = logging.getLogger(__name__)

# Danh sách thuộc tính meta HTML chứa thời gian đăng bài
META_DATE_PROPERTIES: list[tuple[str, str]] = [
    ("property", "article:published_time"),
    ("name", "article:published_time"),
    ("property", "og:published_time"),
    ("name", "pubdate"),
    ("property", "pubdate"),
    ("name", "publishdate"),
    ("name", "date"),
    ("property", "rnews:datePublished"),
    ("property", "og:updated_time"),
    ("property", "article:modified_time"),
]

# Ngày khởi tạo database/theme rác đã biết của WordPress CMS
CMS_INITIAL_SEED_DATES: set[str] = {"2015-08-09"}

# Mảnh slug URL đặc trưng của trang tĩnh thường trực
EVERGREEN_URL_MARKERS: tuple[str, ...] = (
    "/bo-mon-",
    "/gioi-thieu/",
    "/chuong-trinh-dao-tao/",
    "/chuan-dau-ra/",
)

# Date retention policy belongs to extraction; category IDs come from taxonomy.
SEASONAL_CATEGORIES = frozenset({Category.TIN_TUC_SU_KIEN.value, Category.KHAC.value})
EVERGREEN_CATEGORIES = frozenset(category.value for category in Category if category.value not in SEASONAL_CATEGORIES)

__all__ = [
    "CMS_INITIAL_SEED_DATES",
    "EVERGREEN_URL_MARKERS",
    "EVERGREEN_CATEGORIES",
    "SEASONAL_CATEGORIES",
    "META_DATE_PROPERTIES",
    "normalize_date_string",
    "is_evergreen_page",
    "extract_date_from_url",
    "extract_date_from_text",
    "extract_date_from_filename",
    "extract_html_date",
    "extract_doc_date",
    "is_recent_document",
]


def _year_of(date_str: str | None) -> int | None:
    """Lấy năm (int) từ chuỗi ngày đã chuẩn hóa."""
    if not date_str:
        return None
    m = re.match(r"^(\d{4})", date_str)
    return int(m.group(1)) if m else None


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


def is_evergreen_page(url: str | None, category: str | None = None) -> bool:
    """Xác định trang tĩnh thường trực dựa trên category hoặc cấu trúc slug URL."""
    if category and category in EVERGREEN_CATEGORIES:
        return True
    if url:
        path = urllib.parse.urlparse(url).path.lower()
        if not path.endswith("/"):
            path += "/"
        return any(marker in path for marker in EVERGREEN_URL_MARKERS)
    return False


def select_best_date(
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
    return url_date or trafilatura_date


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


def extract_html_date(
    html_text: str | None = None,
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

    trafilatura_date = None
    if html_text:
        try:
            meta = trafilatura.extract_metadata(html_text)
            if meta and meta.date:
                trafilatura_date = normalize_date_string(str(meta.date))
        except Exception as e:
            logger.debug("Trafilatura metadata extraction gặp lỗi: %s", e)

    meta_date = None
    if soup:
        for attr, val in META_DATE_PROPERTIES:
            tag = soup.find("meta", {attr: val})
            if tag and tag.get("content") and (norm := normalize_date_string(tag["content"])):
                meta_date = norm
                break

        if not meta_date:
            for time_tag in soup.find_all("time"):
                if (dt := time_tag.get("datetime")) and (norm := normalize_date_string(dt)):
                    meta_date = norm
                    break
                t_txt = time_tag.get_text().strip()
                if t_txt and (norm := normalize_date_string(t_txt) or extract_date_from_text(t_txt, max_chars=100)):
                    meta_date = norm
                    break

    url_date = extract_date_from_url(url) if url else None
    text_date = extract_date_from_text(text, max_chars=500) if text else None

    # Lọc bỏ ngày seed CMS rác
    meta_date = None if meta_date in CMS_INITIAL_SEED_DATES else meta_date
    trafilatura_date = None if trafilatura_date in CMS_INITIAL_SEED_DATES else trafilatura_date
    url_date = None if url_date in CMS_INITIAL_SEED_DATES else url_date
    text_date = None if text_date in CMS_INITIAL_SEED_DATES else text_date

    result = select_best_date(meta_date, trafilatura_date, url_date, text_date)

    if is_evergreen_page(url=url, category=category):
        if result in CMS_INITIAL_SEED_DATES:
            return None

    result_year = _year_of(result)
    text_year = _year_of(text_date)
    if result_year is not None and result_year < 2020 and text_year is not None and text_year >= 2020:
        return text_date

    return result


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

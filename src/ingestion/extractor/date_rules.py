"""Hằng số danh mục và quy tắc evergreen/seasonal cho ngày tháng."""
from __future__ import annotations

import re
import urllib.parse

# Danh sách thuộc tính meta HTML chứa thời gian đăng bài
META_DATE_PROPERTIES = [
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

# Danh mục taxonomy chứa nội dung bất biến
EVERGREEN_CATEGORIES: set[str] = {
    "CAN_BO_GIANG_VIEN",
    "DAO_TAO",
    "QUY_CHE_BIEU_MAU",
    "TUYEN_SINH",
    "NGHIEN_CUU_HOP_TAC",
}

# Danh mục thời vụ: chỉ các danh mục này mới bị lọc theo min_year
SEASONAL_CATEGORIES: set[str] = {"TIN_TUC_SU_KIEN", "KHAC"}


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


def _year_of(date_str: str | None) -> int | None:
    """Lấy năm (int) từ chuỗi ngày đã chuẩn hóa."""
    if not date_str:
        return None
    m = re.match(r"^(\d{4})", date_str)
    return int(m.group(1)) if m else None


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

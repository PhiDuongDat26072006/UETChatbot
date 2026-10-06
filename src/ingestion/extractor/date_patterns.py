"""Các mẫu regex, thuộc tính thẻ meta và hằng số nhận diện ngày tháng."""
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

# Ngày khởi tạo database/theme rác đã biết của WordPress CMS (không phải ngày đăng thực tế)
CMS_INITIAL_SEED_DATES: set[str] = {"2015-08-09"}

# Mảnh slug URL đặc trưng của trang tĩnh thường trực (evergreen)
EVERGREEN_URL_MARKERS: tuple[str, ...] = (
    "/bo-mon-",
    "/gioi-thieu/",
    "/chuong-trinh-dao-tao/",
    "/chuan-dau-ra/",
)

# Danh mục taxonomy chứa nội dung bất biến, không bao giờ bị lọc theo thời gian
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

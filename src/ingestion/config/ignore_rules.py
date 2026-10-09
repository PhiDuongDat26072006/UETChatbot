"""Quy tắc lọc URL rác, phần mở rộng tệp và biểu thức chính quy (Ignore Rules)."""
from __future__ import annotations

import re

# Các định dạng tài liệu văn bản cần thu thập
DOCUMENT_EXTENSIONS: tuple[str, ...] = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".rtf", ".txt", ".csv"
)

# Danh sách tên miền lưu trữ đám mây được phép trích xuất liên kết
CLOUD_STORAGE_DOMAINS: tuple[str, ...] = (
    "drive.google.com",
    "docs.google.com",
    "forms.gle",
    "1drv.ms",
    "onedrive.live.com",
    "dropbox.com",
    "sharepoint.com",
)

# Các định dạng media, hình ảnh, assets web và file nhị phân cần bỏ qua
IGNORED_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".tiff", ".tif", ".avif", ".jfif", ".eps", ".psd", ".ai",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm",
    ".wav", ".ogg", ".m4a", ".m4v", ".3gp",
    ".css", ".js", ".mjs", ".map", ".json", ".xml",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".exe", ".dmg", ".apk", ".bin", ".iso"
)

# Các mẫu đường dẫn hệ thống WordPress, API, đăng nhập, giỏ hàng cần loại bỏ
IGNORED_PATH_PATTERNS: list[str] = [
    r'/api/Content/Decl/',
    r'/wp-json(/|$)',
    r'/wp-includes(/|$)',
    r'/wp-content(/|$)',
    r'/xmlrpc\.php',
    r'/feed(/|$)',
    r'/comments/feed(/|$)',
    r'/trackback(/|$)',
    r'/site/setLang(/|$)',
    r'/setLang(/|$)',
    r'/language(/|$)',
    r'/login(/|$)',
    r'/logout(/|$)',
    r'/lostpassword(/|$)',
    r'/forgot-password(/|$)',
    r'/reset-password(/|$)',
    r'/resetpass(/|$)',
    r'/user-login(/|$)',
    r'/user-register(/|$)',
    r'/register(/|$)',
    r'/tai-khoan(/|$)',
    r'/account(/|$)',
    r'/user-account(/|$)',
    r'/your-profile(/|$)',
    r'/lp-checkout(/|$)',
    r'/lp-profile(/|$)',
    r'/cart(/|$)',
    r'/checkout(/|$)',
    r'/woocommerce-shop(/|$)',
    r'/post-carousel(/|$)',
    r'/pricing-table(/|$)',
    r'/university-course-carousel(/|$)',
    r'/university-event-grid(/|$)',
    r'/university-revolution-slider(/|$)',
    r'/video-banner(/|$)',
    r'/video(/|$)',
    r'/Ly[0-9A-Za-z+/=]{10,}',
]

IGNORED_PATH_REGEX = re.compile('|'.join(IGNORED_PATH_PATTERNS), re.IGNORECASE)


IGNORED_QUERY_PARAMETERS = frozenset({"ver", "preview", "replytocom", "share", "ref", "fbclid"})

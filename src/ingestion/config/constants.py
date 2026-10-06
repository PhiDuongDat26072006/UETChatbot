"""Các hằng số cấu hình: định dạng file, regex bộ lọc, User-Agent, timeout và worker limits."""
from __future__ import annotations

import re

# Các định dạng tài liệu văn bản cần thu thập để tải về
DOCUMENT_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".rtf", ".txt", ".csv"
)

# Danh sách tên miền dịch vụ lưu trữ đám mây được phép trích xuất liên kết tài liệu
CLOUD_STORAGE_DOMAINS: tuple[str, ...] = (
    "drive.google.com",
    "docs.google.com",
    "forms.gle",
    "1drv.ms",
    "onedrive.live.com",
    "dropbox.com",
    "sharepoint.com",
)

# Các định dạng media, hình ảnh, web assets và file nhị phân rác cần loại bỏ triệt để
IGNORED_EXTENSIONS = (
    # Hình ảnh
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".tiff", ".tif", ".avif", ".jfif", ".eps", ".psd", ".ai",
    # Âm thanh / Video
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm",
    ".wav", ".ogg", ".m4a", ".m4v", ".3gp",
    # Assets web / styles / scripts / fonts
    ".css", ".js", ".mjs", ".map", ".json", ".xml",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # File nén / thực thi rác
    ".zip", ".rar", ".7z", ".tar", ".gz", ".exe", ".dmg", ".apk", ".bin", ".iso"
)

# Các mẫu đường dẫn hệ thống WordPress, API, trang chức năng, tài khoản, cache cần loại bỏ
IGNORED_PATH_PATTERNS = [
    # WordPress internals & assets
    r'/wp-json(/|$)',
    r'/wp-includes(/|$)',
    r'/wp-content(/|$)',
    r'/xmlrpc\.php',
    # Feeds / RSS / Trackback
    r'/feed(/|$)',
    r'/comments/feed(/|$)',
    r'/trackback(/|$)',
    # Đổi ngôn ngữ
    r'/site/setLang(/|$)',
    r'/setLang(/|$)',
    r'/language(/|$)',
    # Đăng nhập / Tài khoản / Phân quyền
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
    # Giỏ hàng / Thanh toán
    r'/lp-checkout(/|$)',
    r'/lp-profile(/|$)',
    r'/cart(/|$)',
    r'/checkout(/|$)',
    r'/woocommerce-shop(/|$)',
    # Các mẫu layout builder / template / carousel không chứa nội dung bài viết
    r'/post-carousel(/|$)',
    r'/pricing-table(/|$)',
    r'/university-course-carousel(/|$)',
    r'/university-event-grid(/|$)',
    r'/university-revolution-slider(/|$)',
    r'/video-banner(/|$)',
    r'/video(/|$)',
    # Chuỗi cache mã hóa Base64 (ví dụ: lazy-load image cache /Ly8... hoặc /Ly9...)
    r'/Ly[0-9A-Za-z+/=]{10,}',
]

IGNORED_PATH_REGEX = re.compile('|'.join(IGNORED_PATH_PATTERNS), re.IGNORECASE)

# Chỉ trích xuất href từ thẻ <a ...> (tránh lấy <img>, <script>, <link rel="stylesheet">, ...)
A_HREF_REGEX = re.compile(r'<a\b[^>]*?\bhref=["\']([^"\'#\s>]+)', re.IGNORECASE)

# Trích xuất file tài liệu nếu có nhúng trong iframe hoặc embed
EMBED_SRC_REGEX = re.compile(r'<(?:embed|iframe)\b[^>]*?\bsrc=["\']([^"\'#\s>]+)', re.IGNORECASE)

# User-Agent cho HTTP requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Cấu hình worker threads & giới hạn
WORKERS_API = 3
WORKERS_SCRAPE = 10
WORKERS_DOWNLOAD = 8
REQUEST_TIMEOUT = 10
DOWNLOAD_TIMEOUT = 20
MAX_CRAWL_PAGES_PER_TARGET = 150

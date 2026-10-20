"""Các hằng số cấu hình: định dạng file, User-Agent, timeout và worker limits."""
from __future__ import annotations

from .patterns import (
    A_HREF_REGEX,
    EMBED_SRC_REGEX,
    IGNORED_PATH_PATTERNS,
    IGNORED_PATH_REGEX,
)

# Các định dạng tài liệu văn bản cần thu thập để tải về
DOCUMENT_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".rtf", ".txt", ".csv"
)

# Danh sách tên miền dịch vụ lưu trữ đám mây được phép trích xuất liên kết
CLOUD_STORAGE_DOMAINS: tuple[str, ...] = (
    "drive.google.com",
    "docs.google.com",
    "forms.gle",
    "1drv.ms",
    "onedrive.live.com",
    "dropbox.com",
    "sharepoint.com",
)

# Các định dạng media, hình ảnh, web assets và file nhị phân rác
IGNORED_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".tiff", ".tif", ".avif", ".jfif", ".eps", ".psd", ".ai",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm",
    ".wav", ".ogg", ".m4a", ".m4v", ".3gp",
    ".css", ".js", ".mjs", ".map", ".json", ".xml",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".exe", ".dmg", ".apk", ".bin", ".iso"
)

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

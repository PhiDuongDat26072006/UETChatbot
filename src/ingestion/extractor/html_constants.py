"""Các hằng số selector, noise tags và regex mẫu cho HTML cleaner."""
from __future__ import annotations

import re

# Các selector phổ biến chứa nội dung bài viết WordPress trên các site UET
WP_CONTENT_SELECTORS = [
    "article",
    "div.content_single",
    "div.content_pages",
    "div.content_detail",
    "div.blog-content",
    "div.entry-content",
    "div.post-content",
    "div.content-area",
    "div.td-post-content",
    "div.single-post-content",
    "section#pages",
    "main#main",
    "main",
    "div.main-content",
    "div.site-content",
    "div#content",
    "div.page-content",
]

# Các tag rác cần loại bỏ triệt để trong fallback
NOISE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside",
    "noscript", "svg", "form", "iframe", "button",
]

# Các class và id liên quan đến nhân sự, giảng viên, CTĐT được bảo vệ
PROTECTED_CLASSES = {
    "team", "member", "profile", "faculty", "staff", "lecturer",
    "giang-vien", "can-bo", "curriculum", "program", "course", "subject",
    "mon-hoc", "dao-tao", "blog-content", "content_single", "content_pages",
    "content_detail",
}

# Regex nhận diện email cán bộ/giảng viên VNU và mã môn học UET
EMAIL_PATTERN = re.compile(r"[\w\.-]+@(?:[a-zA-Z0-9-]+\.)*vnu\.edu\.vn", re.IGNORECASE)
COURSE_CODE_PATTERN = re.compile(
    r"\b(?:INT|MAT|PHY|AIT|PES|ENG|BSA|EMA|CHE|BIO|ENV|GEO|MNS|ELT|TEL|EPN)\s*\d{4}\b",
    re.IGNORECASE,
)

# Danh sách từ khóa anchor text chung chung
GENERIC_LINK_TEXTS = {
    "tại đây", "tai day", "đây", "day",
    "tại link", "tai link", "link", "đường link", "duong link",
    "xem tại đây", "xem tai day", "xem chi tiết", "xem chi tiet",
    "tải về", "tai ve", "tải tại đây", "tai tai day", "tải file", "tai file",
    "tải xuống", "tai xuong",
    "download", "click here", "here", "chi tiết", "chi tiet",
}

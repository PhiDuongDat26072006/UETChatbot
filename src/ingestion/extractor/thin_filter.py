"""Lọc bỏ thin content, trang rác và menu dump."""
from __future__ import annotations

import re

from .html_constants import COURSE_CODE_PATTERN, EMAIL_PATTERN


def is_thin_content(
    content: str,
    title: str | None = None,
    min_length: int = 100,
    category: str | None = None,
) -> bool:
    """Kiểm tra nội dung có phải là stub rỗng hoặc thin content không có giá trị."""
    if not content or not isinstance(content, str):
        return True

    clean = content.strip()
    if len(clean) < 30:
        return True

    emails = EMAIL_PATTERN.findall(clean)
    course_codes = COURSE_CODE_PATTERN.findall(clean)
    has_markdown_table = "|" in clean and "---" in clean

    non_generic_emails = [e for e in emails if not e.lower().startswith("fit@")]
    if len(non_generic_emails) >= 1 or len(course_codes) >= 1 or has_markdown_table:
        if len(clean) >= 40:
            return False

    if len(clean) < min_length:
        return True

    if title:
        norm_title = re.escape(title.strip())
        substantive = re.sub(norm_title, "", clean, flags=re.IGNORECASE)
        boilerplate_patterns = [
            r"©\s*VNU-UET[^\n]*",
            r"All rights reserved[^\n]*",
            r"Bản quyền thuộc về[^\n]*",
            r"Khoa Công nghệ Thông tin[^\n]*",
            r"Trường Đại học Công nghệ[^\n]*",
            r"Đại học Quốc gia Hà Nội[^\n]*",
            r"fit@vnu\.edu\.vn",
            r"\(024\)\s*3\s*754\s*7064",
        ]
        for bp in boilerplate_patterns:
            substantive = re.sub(bp, "", substantive, flags=re.IGNORECASE)

        substantive = substantive.strip()
        words = substantive.split()
        if len(substantive) < 50 or len(words) < 10:
            return True

    all_words = clean.split()
    if len(all_words) < 12:
        return True

    if not non_generic_emails and not course_codes and not has_markdown_table:
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if len(lines) >= 8:
            avg_line_len = sum(len(line) for line in lines) / len(lines)
            if avg_line_len < 30:
                menu_keywords = {
                    "trang chủ", "giới thiệu", "quá trình phát triển", "tầm nhìn và sứ mệnh",
                    "tầm nhìn", "sứ mệnh", "ban chủ nhiệm khoa", "ban chủ nhiệm",
                    "các bộ môn và phòng thí nghiệm", "giảng viên", "liên hệ",
                    "đào tạo", "đại học", "sau đại học", "nghiên cứu",
                    "các hướng nghiên cứu", "các đề tài dự án", "các sản phẩm công nghệ",
                    "chuyên san cntt-tt", "hợp tác", "đối tác", "đối tác khối công nghệ",
                    "đối tác khối hàn lâm", "tin tức", "người học", "thành tích nổi bật",
                    "thông tin việc làm", "các câu hỏi thường gặp", "cựu sinh viên",
                    "ngôn ngữ", "sitemap", "menu",
                }
                menu_matches = sum(1 for line in lines if line.lower().strip() in menu_keywords)
                if (menu_matches / len(lines)) > 0.4:
                    return True

    return False

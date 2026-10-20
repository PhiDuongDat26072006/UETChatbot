"""Tiện ích dùng chung (shared utilities) cho toàn bộ pipeline UET Chatbot."""
from __future__ import annotations


def clean_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng và dòng trống trong văn bản.

    - Xóa khoảng trắng thừa ở đầu/cuối mỗi dòng.
    - Giới hạn tối đa 1 dòng trống liên tiếp.
    - Loại bỏ dòng trống dư ở đầu và cuối chuỗi.
    """
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            cleaned_lines.append(line)
            blank_count = 0
        elif blank_count < 1:
            cleaned_lines.append("")
            blank_count += 1
    return "\n".join(cleaned_lines).strip()

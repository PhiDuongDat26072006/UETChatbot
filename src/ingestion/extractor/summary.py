"""Module sinh tóm tắt ngữ cảnh súc tích (Document Context Summarization) cho RAG pipeline.

Trích xuất 1-2 câu đầu hoặc đoạn văn đại diện có ý nghĩa (khoảng 150-250 ký tự),
loại bỏ các tiêu đề hành chính, phân trang hoặc ký tự thừa, và đính kèm tiền tố ngữ cảnh [Domain | Title].
"""
from __future__ import annotations

import re


def generate_document_summary(
    domain: str,
    title: str,
    content: str,
    max_chars: int = 200,
) -> str:
    """Sinh chuỗi tóm tắt ngữ cảnh súc tích từ nội dung tài liệu.

    Args:
        domain: Tên miền đơn vị (ví dụ: 'fit.uet.vnu.edu.vn').
        title: Tiêu đề bài viết hoặc tên file tài liệu.
        content: Nội dung văn bản sạch của tài liệu.
        max_chars: Độ dài trích đoạn mục tiêu (mặc định: 200 ký tự).

    Returns:
        Chuỗi tóm tắt có tiền tố ngữ cảnh định dạng: '[Domain | Title]: <Tóm tắt đoạn đầu>'.
    """
    clean_title = title.strip() if title else ""
    prefix = f"[{domain} | {clean_title}]: " if clean_title else f"[{domain}]: "

    if not content or not content.strip():
        return f"[{domain} | {clean_title}]" if clean_title else f"[{domain}]"

    # 1. Tách dòng và lọc bỏ boilerplate hành chính hoặc đánh dấu phân trang
    skip_patterns = [
        r"^\s*cộng\s+hòa\s+xã\s+hội\s+chủ\s+nghĩa\s+việt\s+nam",
        r"^\s*độc\s+lập\s*-\s*tự\s+do\s*-\s*hạnh\s+phúc",
        r"^\s*---\s*trang\s+\d+\s*---",
        r"^\s*===\s*bảng:?[^=]+===",
        r"^\s*đại\s+học\s+quốc\s+gia\s+hà\s+nội",
        r"^\s*trường\s+đại\s+học\s+công\s+nghệ",
        r"^\s*khoa\s+công\s+nghệ\s+thông\s+tin",
        r"^\s*\[tài\s+liệu\s+&\s+biểu\s+mẫu\s+liên\s+kết\]",
    ]

    meaningful_lines: list[str] = []
    for line in content.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        # Bỏ qua nếu khớp các dòng hành chính/phân trang/tiêu đề liên kết
        if any(re.match(pat, line_clean, re.IGNORECASE) for pat in skip_patterns):
            continue
        # Bỏ qua các dòng chỉ chứa dấu bullet hoặc gạch ngang
        if re.match(r"^[\s\-\*•_=+~#]+$", line_clean):
            continue
        meaningful_lines.append(line_clean)

    full_text = " ".join(meaningful_lines)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    if not full_text:
        full_text = re.sub(r"\s+", " ", content).strip()

    # 2. Cắt đoạn trích có ý nghĩa quanh mốc max_chars (150-250 ký tự)
    if len(full_text) <= max_chars:
        excerpt = full_text
    else:
        # Cố gắng cắt ở ranh giới câu (. ! ?) trong khoảng [max_chars - 40, max_chars + 60]
        sentences = re.split(r"(?<=[\.\!\?])\s+", full_text)
        accum: list[str] = []
        curr_len = 0
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if curr_len + len(s_clean) <= max_chars + 60:
                accum.append(s_clean)
                curr_len += len(s_clean)
                if curr_len >= max_chars - 40:
                    break
            else:
                break

        if accum:
            excerpt = " ".join(accum)
        else:
            # Nếu câu đầu tiên quá dài, cắt ở ranh giới từ gần mốc max_chars
            cut = full_text[:max_chars]
            last_space = cut.rfind(" ")
            if last_space > int(max_chars * 0.7):
                excerpt = cut[:last_space] + "..."
            else:
                excerpt = cut + "..."

    return prefix + excerpt

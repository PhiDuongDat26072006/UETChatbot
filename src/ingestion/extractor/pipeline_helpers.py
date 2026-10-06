"""Hàm phụ trợ cho pipeline bóc tách: băm ID, kiểm tra trùng lặp stub và đọc cache."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .html_cleaner import COURSE_CODE_PATTERN, EMAIL_PATTERN

logger = logging.getLogger(__name__)

# Ngưỡng ký tự tối đa của một trang stub/template fallback cần loại bỏ khi trùng lặp nội dung 100%
DUPLICATE_STUB_MAX_LENGTH = 500


def canonicalize_url(url: str) -> str:
    """Chuẩn hóa URL về dạng canonical: bỏ www., bỏ trailing slash, chữ thường hostname."""
    if not url.startswith("http"):
        return url
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme}://{netloc}{path}{query}"


def is_duplicate_stub(content: str) -> bool:
    """Kiểm tra xem nội dung có phải là stub rỗng trùng lặp cần loại bỏ hay không."""
    emails = EMAIL_PATTERN.findall(content)
    course_codes = COURSE_CODE_PATTERN.findall(content)
    has_table = "|" in content and "---" in content

    if len(emails) >= 2 or len(course_codes) >= 2 or has_table:
        return False

    if len(content) < 150:
        return True

    if len(content) < DUPLICATE_STUB_MAX_LENGTH and len(emails) == 0 and len(course_codes) == 0:
        return True

    return False


def generate_doc_id(source: str) -> str:
    """Tạo mã định danh duy nhất (MD5 hash 16 ký tự) từ URL hoặc đường dẫn file."""
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:16]


def load_existing_processed_docs(output_path: Path) -> dict[str, dict[str, Any]]:
    """Đọc các bản ghi đã bóc tách từ trước để hỗ trợ xử lý tăng dần (incremental)."""
    existing_docs: dict[str, dict[str, Any]] = {}
    if not output_path.exists():
        return existing_docs

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    source = record.get("source_url_or_path")
                    if source:
                        existing_docs[source] = record
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning("Không thể đọc cache file %s: %s", output_path, e)
    return existing_docs

"""Module trích xuất nội dung văn bản sạch từ các tệp tài liệu."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import urllib.parse

from ..utils import clean_whitespace
from .date_extractor import extract_doc_date
from .doc_readers import (
    extract_doc_legacy,
    extract_docx,
    extract_pdf,
    extract_txt,
    extract_xlsx,
)

__all__ = ["extract_document_text"]


def extract_document_text(file_path: Path | str, min_length: int = 30) -> dict[str, Any] | None:
    """Nhận diện và trích xuất nội dung văn bản sạch từ một file tài liệu."""
    path = Path(file_path)
    if not path.is_file() or path.name.startswith(".") or path.name in ("files_list.txt", "endpoints.txt"):
        return None

    ext = path.suffix.lower()
    raw_text = ""

    if ext == ".pdf":
        raw_text = extract_pdf(path)
    elif ext == ".docx":
        raw_text = extract_docx(path)
    elif ext in (".xlsx", ".xlsm"):
        raw_text = extract_xlsx(path)
    elif ext == ".doc":
        raw_text = extract_doc_legacy(path)
    elif ext in (".txt", ".md", ".csv"):
        raw_text = extract_txt(path)
    else:
        return None

    cleaned_text = clean_whitespace(raw_text)
    if len(cleaned_text) < min_length:
        return None

    readable_filename = urllib.parse.unquote(path.name)
    published_date = extract_doc_date(file_path=path, text=cleaned_text)

    return {
        "filename": readable_filename,
        "file_path": str(path),
        "file_type": ext.lstrip("."),
        "published_date": published_date,
        "text": cleaned_text,
        "text_length": len(cleaned_text),
    }

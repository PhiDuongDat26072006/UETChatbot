"""Module trích xuất nội dung văn bản sạch từ các tệp tài liệu (.pdf, .docx, .xlsx, .doc, .txt)."""
from __future__ import annotations

import logging
import re
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import docx
import openpyxl
from pypdf import PdfReader

from ..utils import clean_whitespace
from .date_extractor import extract_doc_date

logger = logging.getLogger(__name__)

__all__ = [
    "extract_pdf",
    "extract_docx",
    "extract_xlsx",
    "extract_doc_legacy",
    "extract_txt",
    "extract_document_text",
]


def extract_pdf(file_path: Path) -> str:
    """Bóc tách text từ file PDF sử dụng pypdf."""
    try:
        with PdfReader(str(file_path), strict=False) as reader:
            pages: list[str] = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append(f"--- Trang {page_number} ---\n{text.strip()}")
                except Exception as error:
                    logger.warning("Cannot extract %s page %s: %s", file_path, page_number, error)
                    logger.debug("PDF page failure", exc_info=True)
            return "\n\n".join(pages)
    except Exception as error:
        logger.warning("Cannot extract document %s: %s", file_path, error)
        logger.debug("Document extraction failure", exc_info=True)
        return ""


def extract_docx(file_path: Path) -> str:
    """Bóc tách text và bảng biểu từ file DOCX sử dụng python-docx."""
    try:
        doc = docx.Document(str(file_path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                unique: list[str] = []
                for c in cells:
                    if not unique or c != unique[-1]:
                        unique.append(c)
                if unique:
                    parts.append(" | ".join(unique))
        return "\n".join(parts)
    except Exception as error:
        logger.warning("Cannot extract document %s: %s", file_path, error)
        logger.debug("Document extraction failure", exc_info=True)
        return ""


def extract_xlsx(file_path: Path) -> str:
    """Bóc tách text từ các sheet bảng tính Excel (.xlsx)."""
    try:
        with closing(openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)) as workbook:
            sheet_texts: list[str] = []
            for sheet in workbook.worksheets:
                rows: list[str] = []
                for row in sheet.iter_rows(values_only=True):
                    values = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                    if values:
                        rows.append(" | ".join(values))
                if rows:
                    sheet_texts.append(f"=== Bảng: {sheet.title} ===\n" + "\n".join(rows))
            return "\n\n".join(sheet_texts)
    except Exception as error:
        logger.warning("Cannot extract document %s: %s", file_path, error)
        logger.debug("Document extraction failure", exc_info=True)
        return ""


def extract_doc_legacy(file_path: Path) -> str:
    """Thử bóc tách text từ file .doc định dạng cũ hoặc RTF / nhị phân OLE."""
    try:
        raw_bytes = file_path.read_bytes()
        if raw_bytes.startswith(b"{\\rtf"):
            decoded = raw_bytes.decode("latin-1", errors="ignore")
            clean = re.sub(r"\\[a-z0-9]+ ?", " ", decoded)
            return re.sub(r"[{}]", "", clean)

        extracted: list[str] = []
        for match in re.finditer(rb"(?:[\x20-\x7e\xa0-\xff]\x00){4,}", raw_bytes):
            try:
                chunk = match.group().decode("utf-16le", errors="ignore").strip()
                if len(chunk) >= 4 and not all(c in "-=_*~# " for c in chunk):
                    extracted.append(chunk)
            except Exception:
                logger.debug("Cannot decode legacy DOC segment: %s", file_path, exc_info=True)
        return "\n".join(extracted)
    except Exception as error:
        logger.warning("Cannot extract document %s: %s", file_path, error)
        logger.debug("Document extraction failure", exc_info=True)
        return ""


def extract_txt(file_path: Path) -> str:
    """Đọc file text thô với nhiều bộ mã tiếng Việt dự phòng."""
    for enc in ["utf-8", "utf-16", "cp1258", "latin-1"]:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeError:
            logger.debug("Cannot decode %s with %s", file_path, enc)
        except OSError as error:
            logger.warning("Cannot read %s: %s", file_path, error)
            return ""
    return ""


def extract_document_text(file_path: Path | str, min_length: int = 30) -> dict[str, Any] | None:
    """
    Nhận diện định dạng và trích xuất nội dung văn bản sạch từ một file tài liệu.

    Hỗ trợ PDF, DOCX, XLSX, DOC, TXT/MD/CSV. Trích xuất cả ngày ban hành từ tên file
    hoặc nội dung.
    """
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

    readable_filename = unquote(path.name)
    published_date = extract_doc_date(file_path=path, text=cleaned_text)

    return {
        "filename": readable_filename,
        "file_path": str(path),
        "file_type": ext.lstrip("."),
        "published_date": published_date,
        "text": cleaned_text,
        "text_length": len(cleaned_text),
    }

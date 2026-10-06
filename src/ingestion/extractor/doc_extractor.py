"""Module trích xuất nội dung văn bản sạch từ các tệp tài liệu (.pdf, .docx, .doc, .xlsx, .txt).

Hỗ trợ các định dạng văn bản phổ biến trong dữ liệu các khoa/viện trực thuộc UET.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
import urllib.parse

from pypdf import PdfReader
import docx
import openpyxl

from ..utils import clean_whitespace
from .date_extractor import extract_doc_date

logger = logging.getLogger(__name__)


def _extract_pdf(file_path: Path) -> str:
    """Bóc tách text từ file PDF sử dụng pypdf."""
    try:
        reader = PdfReader(str(file_path), strict=False)
        pages_content: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_content.append(f"--- Trang {i + 1} ---\n{page_text.strip()}")
            except Exception as e:
                logger.debug("Lỗi đọc trang %d của PDF %s: %s", i + 1, file_path.name, e)
                continue
        return "\n\n".join(pages_content)
    except Exception as e:
        logger.warning("Không thể đọc file PDF %s: %s", file_path.name, e)
        return ""


def _extract_docx(file_path: Path) -> str:
    """Bóc tách text và bảng biểu từ file DOCX sử dụng python-docx."""
    try:
        doc = docx.Document(str(file_path))
        parts: list[str] = []

        # 1. Đọc từng đoạn văn bản
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                parts.append(text)

        # 2. Đọc bảng biểu dữ liệu
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                # Loại bỏ cell trùng lặp do merge cell
                unique_cells: list[str] = []
                for cell in cells:
                    if not unique_cells or cell != unique_cells[-1]:
                        unique_cells.append(cell)
                if unique_cells:
                    parts.append(" | ".join(unique_cells))

        return "\n".join(parts)
    except Exception as e:
        logger.warning("Không thể đọc file DOCX %s: %s", file_path.name, e)
        return ""


def _extract_xlsx(file_path: Path) -> str:
    """Bóc tách text từ các sheet bảng tính Excel (.xlsx)."""
    try:
        wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        sheet_texts: list[str] = []

        for sheet in wb.worksheets:
            rows_data: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                # Lọc bỏ các ô None và ghép thành chuỗi dòng
                valid_cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if valid_cells:
                    rows_data.append(" | ".join(valid_cells))

            if rows_data:
                sheet_header = f"=== Bảng: {sheet.title} ==="
                sheet_texts.append(sheet_header + "\n" + "\n".join(rows_data))

        wb.close()
        return "\n\n".join(sheet_texts)
    except Exception as e:
        logger.warning("Không thể đọc file XLSX %s: %s", file_path.name, e)
        return ""


def _extract_doc_legacy(file_path: Path) -> str:
    """Thử bóc tách text từ file .doc định dạng cũ hoặc RTF / nhị phân OLE."""
    try:
        raw_bytes = file_path.read_bytes()

        # Kiểm tra nếu là file RTF
        if raw_bytes.startswith(b"{\\rtf"):
            decoded = raw_bytes.decode("latin-1", errors="ignore")
            # Loại bỏ các mã điều khiển RTF cơ bản
            clean = re.sub(r"\\[a-z0-9]+ ?", " ", decoded)
            clean = re.sub(r"[{}]", "", clean)
            return clean

        # Bóc tách chuỗi UTF-16LE và chuỗi ký tự hiển thị được từ nhị phân OLE
        extracted_chunks: list[str] = []

        # Chuỗi UTF-16LE (tối thiểu 4 ký tự)
        for match in re.finditer(rb"(?:[\x20-\x7e\xa0-\xff]\x00){4,}", raw_bytes):
            try:
                chunk = match.group().decode("utf-16le", errors="ignore").strip()
                if len(chunk) >= 4 and not all(c in "-=_*~# " for c in chunk):
                    extracted_chunks.append(chunk)
            except Exception:
                pass

        if extracted_chunks:
            return "\n".join(extracted_chunks)

        return ""
    except Exception as e:
        logger.debug("Bỏ qua file .doc không đọc được %s: %s", file_path.name, e)
        return ""


def _extract_txt(file_path: Path) -> str:
    """Đọc file text thô với nhiều bộ mã tiếng Việt dự phòng."""
    encodings = ["utf-8", "utf-16", "cp1258", "latin-1"]
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return ""


def extract_document_text(file_path: Path | str, min_length: int = 30) -> dict[str, Any] | None:
    """Nhận diện và trích xuất nội dung văn bản sạch từ một file tài liệu.

    Args:
        file_path: Đường dẫn tới file tài liệu.
        min_length: Độ dài tối thiểu để chấp nhận tài liệu (mặc định: 30 ký tự).

    Returns:
        dict gồm {"filename": str, "file_path": str, "file_type": str, "published_date": str | None, "text": str, "text_length": int}
        hoặc None nếu không hỗ trợ / file rỗng / lỗi.
    """
    path = Path(file_path)
    if not path.is_file():
        return None

    # Bỏ qua các file rác hệ thống hoặc danh sách tạm
    if path.name.startswith(".") or path.name in ("files_list.txt", "endpoints.txt"):
        return None

    ext = path.suffix.lower()
    raw_text = ""

    if ext == ".pdf":
        raw_text = _extract_pdf(path)
    elif ext == ".docx":
        raw_text = _extract_docx(path)
    elif ext in (".xlsx", ".xlsm"):
        raw_text = _extract_xlsx(path)
    elif ext == ".doc":
        raw_text = _extract_doc_legacy(path)
    elif ext in (".txt", ".md", ".csv"):
        raw_text = _extract_txt(path)
    else:
        logger.debug("Định dạng không được hỗ trợ bóc tách: %s (%s)", ext, path.name)
        return None

    cleaned_text = clean_whitespace(raw_text)
    if len(cleaned_text) < min_length:
        return None

    # Tên file giải mã unquote (ví dụ %C4%90%C6%A1n -> Đơn)
    readable_filename = urllib.parse.unquote(path.name)

    # Trích xuất thời gian ban hành/công bố của tài liệu
    published_date = extract_doc_date(file_path=path, text=cleaned_text)

    return {
        "filename": readable_filename,
        "file_path": str(path),
        "file_type": ext.lstrip("."),
        "published_date": published_date,
        "text": cleaned_text,
        "text_length": len(cleaned_text),
    }

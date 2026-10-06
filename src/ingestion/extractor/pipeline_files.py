"""Bóc tách các file tài liệu trong thư mục files/ cho pipeline."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import urllib.parse

from ..classifier import classify_endpoint
from ..config import resolve_unit
from .doc_extractor import extract_document_text
from .pipeline_helpers import generate_doc_id, is_duplicate_stub
from .summary import generate_document_summary


def extract_files_for_domain(
    files_dir: Path,
    domain: str,
    records: dict[str, dict[str, Any]],
    seen_hashes: dict[str, str],
    force: bool,
) -> list[Path]:
    """Bóc tách toàn bộ tài liệu đính kèm (.pdf, .docx, .xlsx...) trong files/."""
    doc_files = []
    if files_dir.exists():
        exts = (".pdf", ".docx", ".xlsx", ".doc", ".txt", ".csv")
        doc_files = [p for p in files_dir.rglob("*") if p.is_file() and p.name != "files_list.txt" and p.suffix.lower() in exts]

    for fp in [p for p in doc_files if force or str(p) not in records]:
        ext = extract_document_text(fp)
        if not ext:
            continue
        chash = hashlib.md5(ext["text"].strip().encode("utf-8")).hexdigest()
        if chash in seen_hashes and is_duplicate_stub(ext["text"]):
            continue
        seen_hashes[chash] = str(fp)
        cat = classify_endpoint(str(fp), urllib.parse.unquote(fp.stem))
        records[str(fp)] = {
            "id": generate_doc_id(str(fp)),
            "source_type": "file",
            "source_url_or_path": str(fp),
            "title": ext["filename"],
            "domain": domain,
            "unit": resolve_unit(domain),
            "category": cat.value,
            "published_date": ext.get("published_date"),
            "summary": generate_document_summary(domain, ext["filename"], ext["text"]),
            "content": ext["text"],
            "content_length": ext["text_length"],
            "cloud_links": [],
        }

    return doc_files

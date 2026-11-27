"""Shared helpers for ingestion, preprocessing, and backward-compatible pipeline resolution."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.base import BasePreprocessor, ProcessedData, RawData
from src.indexing.chunks import load_chunks_from_dir
from src.ingestion import clean_whitespace
from src.utils.helpers import get_logger

logger = get_logger("pipeline")


def get_pipeline_attr(name: str, default: Any) -> Any:
    """Resolve an attribute through ``src.pipeline`` when loaded so test monkeypatches apply."""
    pipeline_mod = sys.modules.get("src.pipeline")
    if pipeline_mod is not None and hasattr(pipeline_mod, name):
        return getattr(pipeline_mod, name)
    return default


class UETPreprocessor(BasePreprocessor):
    """
    Quy trình tiền xử lý và làm sạch dữ liệu văn bản UET.
    Chuyển đổi RawData thành ProcessedData: loại bỏ khoảng trắng thừa, chuẩn hóa tiêu đề và metadata.
    """

    def process(self, raw: RawData) -> ProcessedData:
        clean_text = clean_whitespace(raw.content)
        title = (raw.title or "").strip()
        if not title:
            first_line = clean_text.split("\n")[0].strip() if clean_text else ""
            title = first_line[:120] if first_line else "Tài liệu đào tạo UET"

        metadata = dict(raw.metadata or {})
        metadata.setdefault("source_uri", raw.source_uri)
        metadata.setdefault("title", title)

        return ProcessedData(
            raw_data_id=raw.id,
            title=title,
            content=clean_text,
            metadata=metadata,
        )


def _extract_subdomain_before_uet(parts: List[str]) -> Optional[str]:
    """Return the subdomain preceding 'uet' in a dot-separated host or stem."""
    if "uet" not in parts:
        return None
    idx = parts.index("uet")
    if idx > 0:
        candidate = parts[idx - 1]
        if candidate == "www" and idx > 1:
            return parts[idx - 2]
        return candidate
    return "uet"


def extract_faculty_code(
    data: Dict[str, Any],
    meta: Dict[str, Any],
    file_path: Optional[Path] = None,
) -> str:
    """
    Trích xuất mã khoa chuẩn hóa (fit, fat, fce, fema, fepn, fet, iai, sae, uet...)
    từ unit, domain, URL nguồn hoặc tên file.
    """
    raw_unit = str(data.get("unit") or meta.get("unit") or meta.get("faculty") or "").strip().lower()
    if raw_unit:
        clean_unit = "".join(c for c in raw_unit if c.isalnum() or c in "_-")
        if clean_unit:
            return clean_unit

    raw_domain = str(data.get("domain") or meta.get("domain") or "").strip().lower()
    if raw_domain:
        parts = raw_domain.split(".")
        sub = _extract_subdomain_before_uet(parts)
        if sub:
            return sub
        if parts and parts[0]:
            return parts[0]

    source = str(data.get("source_url_or_path") or meta.get("source") or "").strip()
    if source:
        if "://" in source:
            try:
                parts = urlparse(source).netloc.lower().split(".")
                sub = _extract_subdomain_before_uet(parts)
                if sub:
                    return sub
                if parts and parts[0]:
                    return parts[0]
            except Exception:
                pass
        for part in source.replace("\\", "/").split("/"):
            if "uet." in part.lower():
                sub = _extract_subdomain_before_uet(part.lower().split("."))
                if sub:
                    return sub

    if file_path is not None:
        stem = file_path.stem.lower()
        if "uet." in stem:
            sub = _extract_subdomain_before_uet(stem.split("."))
            if sub:
                return sub
        elif stem and stem != "ingested_processed":
            return stem

    return "general"


def load_processed_docs(base_dir: Path) -> List[ProcessedData]:
    """
    Nạp danh sách tài liệu sạch từ data/processed_data/*.jsonl.
    Đảm bảo nguyên vẹn cấu trúc văn bản (không bị cắt vụn 800 ký tự)
    để AdaptiveSemanticExtractor phân tách thành các Semantic Units hoàn chỉnh.
    """
    processed_dir = base_dir / "data" / "processed_data"
    docs: List[ProcessedData] = []
    if processed_dir.exists():
        for file_path in sorted(processed_dir.glob("*.jsonl")):
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if line.strip():
                            docs.append(ProcessedData(**json.loads(line)))
            except Exception as err:
                logger.warning(f"Lỗi khi đọc file {file_path}: {err}")

    if not docs:
        chunked_dir = base_dir / "data" / "chunked_data"
        if chunked_dir.exists():
            chunks = load_chunks_from_dir(chunked_dir)
            if chunks:
                logger.info(f"Tái tạo ProcessedData từ {len(chunks)} chunks trong {chunked_dir}...")
                doc_map: Dict[Optional[str], List[Any]] = defaultdict(list)
                for chunk in chunks:
                    doc_map[chunk.document_id].append(chunk)

                max_doc_chars = 60_000
                for doc_id, doc_chunks in doc_map.items():
                    doc_chunks.sort(key=lambda x: getattr(x, "chunk_index", 0))
                    merged_text = "\n\n".join(c.text for c in doc_chunks)[:max_doc_chars]
                    first_chunk = doc_chunks[0]
                    meta = dict(first_chunk.metadata or {})
                    title = meta.get("title") or meta.get("heading") or "Tài liệu đào tạo UET"
                    docs.append(
                        ProcessedData(
                            id=doc_id,
                            raw_data_id=doc_id,
                            title=title,
                            content=merged_text,
                            metadata=meta,
                        )
                    )
    return docs

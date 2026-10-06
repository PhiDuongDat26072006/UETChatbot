"""Hàm phụ trợ cho UETDataLoader: đọc JSONL, chuẩn hóa đường dẫn và chuyển đổi RawData."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from src.base import DataSource, RawData
from src.utils.helpers import load_yaml_config

logger = logging.getLogger("ingestion")

METADATA_FIELDS = (
    "title",
    "domain",
    "unit",
    "category",
    "published_date",
    "summary",
    "cloud_links",
    "source_type",
    "content_length",
)


def default_processed_dir(repo_root: Path) -> Path:
    """Lấy thư mục processed_data từ config.yaml (paths.processed_dir)."""
    cfg = load_yaml_config() or {}
    rel = (cfg.get("paths") or {}).get("processed_dir", "data/processed_data")
    return repo_root / rel


def resolve_uri(uri: str, repo_root: Path) -> Path:
    """Chuyển đổi URI tương đối thành đường dẫn tuyệt đối."""
    p = Path(uri)
    return p if p.is_absolute() else (repo_root / p)


def iter_jsonl(path: Path) -> Iterator[Optional[Dict[str, Any]]]:
    """Đọc từng dòng JSONL; trả về None cho dòng lỗi để bên gọi thống kê."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"{path.name}:{line_no} JSON không hợp lệ ({e})")
                yield None


def to_raw_data(
    record: Dict[str, Any], content: str, jsonl_file: Path, source: DataSource
) -> RawData:
    """Chuyển bản ghi JSONL thành đối tượng RawData theo giao ước base.py."""
    url = record.get("source_url_or_path") or ""
    metadata: Dict[str, Any] = {k: record.get(k) for k in METADATA_FIELDS}
    metadata["url"] = url
    metadata["cloud_links"] = metadata.get("cloud_links") or []
    metadata["processed_file"] = jsonl_file.name
    metadata["source_id"] = source.source_id

    kwargs: Dict[str, Any] = dict(
        content=content,
        source_uri=url or str(jsonl_file),
        title=record.get("title"),
        raw_metadata=metadata,
    )
    if record.get("id"):
        kwargs["id"] = str(record["id"])
    return RawData(**kwargs)

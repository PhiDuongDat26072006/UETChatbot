"""src/ingestion/loader.py - UETDataLoader kế thừa BaseDataCrawler theo giao ước base.py."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# Direct script invocation needs the repository root before application imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.base import BaseDataCrawler, DataSource, ProcessedData, RawData
from src.ingestion.config.paths import BASE_DIR, DATA_SOURCE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)

METADATA_FIELDS: tuple[str, ...] = (
    "title",
    "domain",
    "unit",
    "category",
    "published_date",
    "summary",
    "cloud_links",
    "source_type",
    "content_length",
    "source_id",
    "raw_data_id",
)


def iter_jsonl(path: Path) -> Iterator[Optional[Dict[str, Any]]]:
    """Đọc từng dòng JSONL; trả về None cho dòng lỗi để bên gọi thống kê."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSONL record must be an object")
                yield record
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"{path.name}:{line_no} JSON không hợp lệ ({e})")
                yield None


def to_raw_data(
    record: Dict[str, Any], content: str, jsonl_file: Path, source: DataSource
) -> RawData:
    """Chuyển bản ghi JSONL thành đối tượng RawData theo giao ước base.py."""
    url = record.get("source_url_or_path") or ""
    metadata: Dict[str, Any] = {k: record.get(k) for k in METADATA_FIELDS if record.get(k) is not None}
    metadata["url"] = url
    metadata["cloud_links"] = metadata.get("cloud_links") or []
    metadata["processed_file"] = jsonl_file.name
    metadata["source_id"] = record.get("source_id") or source.source_id
    if record.get("raw_data_id"):
        metadata["raw_data_id"] = record["raw_data_id"]

    kwargs: Dict[str, Any] = dict(
        content=content,
        source_uri=url or str(jsonl_file),
        title=record.get("title"),
        raw_metadata=metadata,
    )
    if record.get("id"):
        kwargs["id"] = str(record["id"])
    return RawData(**kwargs)


class UETDataLoader(BaseDataCrawler):
    """Bộ nạp dữ liệu cho UET Chatbot, kế thừa BaseDataCrawler (src/base.py)."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(PROCESSED_DATA_DIR)

    def crawl(self, source: DataSource) -> List[RawData]:
        """Đọc dữ liệu đã bóc tách từ một nguồn (thư mục hoặc file JSONL)."""
        path = Path(source.uri)
        if not path.is_absolute():
            path = BASE_DIR / path
        started = time.monotonic()
        logger.info("[START] Loading records from %s", path)

        if source.source_type == "directory" or path.is_dir():
            files = sorted(path.glob("*.jsonl"))
        elif source.source_type == "file" or path.is_file():
            files = [path]
        else:
            raise ValueError(f"source_type '{source.source_type}' không được hỗ trợ.")

        if not files:
            logger.warning(f"Không tìm thấy file .jsonl nào tại {path}")
            return []

        results: List[RawData] = []
        seen_ids: set[str] = set()
        skipped_empty = skipped_dup = skipped_bad = 0

        for jsonl_file in files:
            file_count = 0
            for record in iter_jsonl(jsonl_file):
                if record is None:
                    skipped_bad += 1
                    continue
                content = (record.get("content") or "").strip()
                doc_id = str(record.get("id") or "")
                if not content:
                    skipped_empty += 1
                elif doc_id and doc_id in seen_ids:
                    skipped_dup += 1
                else:
                    raw = to_raw_data(record, content, jsonl_file, source)
                    seen_ids.add(raw.id)
                    results.append(raw)
                    file_count += 1
            logger.debug("%s: %s records", jsonl_file.name, file_count)

        logger.debug("Skipped %s empty, %s duplicate, and %s malformed records", skipped_empty, skipped_dup, skipped_bad)
        logger.info("[DONE] Loaded %s records in %.1fs from %s", len(results), time.monotonic() - started, path)
        return results

    def load_all(self) -> List[RawData]:
        """Nạp toàn bộ dữ liệu trong thư mục processed_data mặc định."""
        return self.crawl(DataSource(source_type="directory", uri=str(self.data_dir)))

    def load_sources(
        self,
        data_source_dir: Optional[Path | str] = None,
        domain: Optional[str] = None,
    ) -> List[DataSource]:
        """Nạp danh sách DataSource từ data/data_source/<domain>.jsonl."""
        base_path = Path(data_source_dir or DATA_SOURCE_DIR)
        if not base_path.is_absolute():
            base_path = BASE_DIR / base_path

        files = sorted(base_path.glob("*.jsonl"))
        if not files:
            files = sorted(base_path.glob("*/sources.jsonl"))

        results: List[DataSource] = []
        for file_path in files:
            source_domain = file_path.stem if file_path.name != "sources.jsonl" else file_path.parent.name
            if domain and domain.lower() not in source_domain.lower():
                continue
            for record in iter_jsonl(file_path):
                if not record:
                    continue
                results.append(
                    DataSource(
                        source_id=record.get("source_id", ""),
                        source_type=record.get("source_type", "web"),
                        uri=record.get("uri", ""),
                        metadata=record.get("metadata", {}),
                    )
                )
        return results

    def load_raw_manifests(
        self,
        raw_dir: Optional[Path | str] = None,
        domain: Optional[str] = None,
    ) -> List[RawData]:
        """Nạp danh sách RawData manifests từ data/raw_data/<domain>/raw_records.jsonl."""
        base_path = Path(raw_dir or RAW_DATA_DIR)
        if not base_path.is_absolute():
            base_path = BASE_DIR / base_path

        files = sorted(base_path.glob("*/raw_records.jsonl"))
        results: List[RawData] = []
        for file_path in files:
            folder_domain = file_path.parent.name
            if domain and domain.lower() not in folder_domain.lower():
                continue
            for record in iter_jsonl(file_path):
                if not record:
                    continue
                raw_meta = dict(record.get("raw_metadata") or {})
                if record.get("source_id"):
                    raw_meta["source_id"] = record["source_id"]
                if record.get("raw_file_path"):
                    raw_meta["raw_file_path"] = record["raw_file_path"]

                results.append(
                    RawData(
                        id=record.get("id", ""),
                        source_uri=record.get("source_uri", ""),
                        content=record.get("content") or "",
                        title=record.get("title"),
                        raw_metadata=raw_meta,
                    )
                )
        return results

    def load_processed(
        self,
        processed_dir: Optional[Path | str] = None,
        domain: Optional[str] = None,
    ) -> List[ProcessedData]:
        """Nạp danh sách ProcessedData từ data/processed_data/*.jsonl."""
        base_path = Path(processed_dir or self.data_dir)
        if not base_path.is_absolute():
            base_path = BASE_DIR / base_path

        files = sorted(base_path.glob("*.jsonl"))
        results: List[ProcessedData] = []
        for file_path in files:
            if domain and domain.lower() not in file_path.name.lower():
                continue
            for record in iter_jsonl(file_path):
                if not record:
                    continue
                meta = {k: v for k, v in record.items() if k not in ("id", "title", "content", "raw_data_id")}
                results.append(
                    ProcessedData(
                        id=record.get("id", ""),
                        raw_data_id=record.get("raw_data_id"),
                        title=record.get("title", ""),
                        content=record.get("content", ""),
                        metadata=meta,
                    )
                )
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    loader = UETDataLoader()
    loader.load_all()


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

from src.base import BaseDataCrawler, DataSource, RawData
from src.ingestion.config.paths import BASE_DIR, PROCESSED_DATA_DIR

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    loader = UETDataLoader()
    loader.load_all()

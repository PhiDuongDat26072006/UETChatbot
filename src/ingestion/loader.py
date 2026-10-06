"""src/ingestion/loader.py - UETDataLoader kế thừa BaseDataCrawler theo giao ước base.py."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.base import BaseDataCrawler, DataSource, RawData  # noqa: E402
from src.utils.helpers import get_logger  # noqa: E402

try:
    from .loader_helpers import (  # noqa: E402
        METADATA_FIELDS,
        default_processed_dir,
        iter_jsonl,
        resolve_uri,
        to_raw_data,
    )
except ImportError:
    from src.ingestion.loader_helpers import (  # noqa: E402
        METADATA_FIELDS,
        default_processed_dir,
        iter_jsonl,
        resolve_uri,
        to_raw_data,
    )

logger = get_logger("ingestion")


class UETDataLoader(BaseDataCrawler):
    """Bộ nạp dữ liệu cho UET Chatbot, kế thừa BaseDataCrawler (src/base.py)."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else default_processed_dir(_REPO_ROOT)

    def crawl(self, source: DataSource) -> List[RawData]:
        """Đọc dữ liệu đã bóc tách từ một nguồn (thư mục hoặc file JSONL)."""
        path = resolve_uri(source.uri, _REPO_ROOT)
        logger.info(f"Bắt đầu nạp dữ liệu từ: {path} (loại: {source.source_type})")

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
            logger.info(f"  - {jsonl_file.name}: {file_count} tài liệu")

        logger.info(
            f"Hoàn tất: {len(results)} tài liệu từ {len(files)} file "
            f"(bỏ qua: {skipped_empty} rỗng, {skipped_dup} trùng id, {skipped_bad} lỗi JSON)"
        )
        return results

    def load_all(self) -> List[RawData]:
        """Nạp toàn bộ dữ liệu trong thư mục processed_data mặc định."""
        return self.crawl(DataSource(source_type="directory", uri=str(self.data_dir)))


if __name__ == "__main__":
    loader = UETDataLoader()
    docs = loader.load_all()
    print(f"\nĐã nạp thành công: {len(docs)} tài liệu từ {loader.data_dir}")
    by_unit = Counter(d.raw_metadata.get("unit") for d in docs)
    print("Phân bố theo đơn vị:", dict(by_unit.most_common()))

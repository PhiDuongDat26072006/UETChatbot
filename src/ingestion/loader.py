"""
src/ingestion/loader.py - Phụ trách quy trình Data Ingestion & Crawling.
Thành viên phụ trách: Người thu thập & đọc dữ liệu.
Nhiệm vụ: Đọc dữ liệu đã được bóc tách sạch (data/processed_data/*.jsonl) do bộ crawler
`src/ingestion/chatbot` sinh ra và chuyển thành các đối tượng RawData theo giao ước base.py.

Bố cục dữ liệu (theo config.yaml -> paths):
    data/raw_data/<domain>/endpoints/   # danh sách URL + metadata phân loại
    data/raw_data/<domain>/files/       # tài liệu gốc (.pdf, .docx, .xlsx, .doc, ...)
    data/processed_data/<domain>.jsonl  # văn bản sạch, mỗi dòng 1 tài liệu
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Cho phép chạy trực tiếp `python src/ingestion/loader.py` từ thư mục gốc repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.base import BaseDataCrawler, DataSource, RawData  # noqa: E402
from src.utils.helpers import get_logger, load_yaml_config  # noqa: E402

logger = get_logger("ingestion")

# Các trường ngữ cảnh được chép từ bản ghi JSONL sang RawData.raw_metadata
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


def _default_processed_dir() -> Path:
    """Lấy thư mục processed_data từ config.yaml (paths.processed_dir)."""
    cfg = load_yaml_config() or {}
    rel = (cfg.get("paths") or {}).get("processed_dir", "data/processed_data")
    return _REPO_ROOT / rel


class UETDataLoader(BaseDataCrawler):
    """
    Bộ nạp dữ liệu cho UET Chatbot, kế thừa BaseDataCrawler (src/base.py).

    `crawl(source)` hỗ trợ:
      - source_type "directory": quét toàn bộ `*.jsonl` trong thư mục `source.uri`
      - source_type "file":      đọc một file `.jsonl` cụ thể
    Đường dẫn tương đối trong `source.uri` được tính từ thư mục gốc repo.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else _default_processed_dir()

    # ------------------------------------------------------------------ #
    # Giao ước bắt buộc từ BaseDataCrawler
    # ------------------------------------------------------------------ #
    def crawl(self, source: DataSource) -> List[RawData]:
        """
        Đọc dữ liệu đã bóc tách từ một nguồn (thư mục hoặc file JSONL).
        :param source: DataSource với source_type "directory" | "file" và uri là đường dẫn
        :return: Danh sách RawData (đã loại bỏ bản ghi rỗng và trùng id)
        """
        path = self._resolve(source.uri)
        logger.info(f"Bắt đầu nạp dữ liệu từ: {path} (loại: {source.source_type})")

        if source.source_type == "directory" or path.is_dir():
            files = sorted(path.glob("*.jsonl"))
        elif source.source_type == "file" or path.is_file():
            files = [path]
        else:
            raise ValueError(
                f"source_type '{source.source_type}' không được hỗ trợ (chỉ 'directory' hoặc 'file')."
            )

        if not files:
            logger.warning(f"Không tìm thấy file .jsonl nào tại {path}")
            return []

        results: List[RawData] = []
        seen_ids: set[str] = set()
        skipped_empty = skipped_dup = skipped_bad = 0

        for jsonl_file in files:
            file_count = 0
            for record in self._iter_jsonl(jsonl_file):
                if record is None:
                    skipped_bad += 1
                    continue
                content = (record.get("content") or "").strip()
                if not content:
                    skipped_empty += 1
                    continue
                doc_id = str(record.get("id") or "")
                if doc_id and doc_id in seen_ids:
                    skipped_dup += 1
                    continue

                raw = self._to_raw_data(record, content, jsonl_file, source)
                seen_ids.add(raw.id)
                results.append(raw)
                file_count += 1
            logger.info(f"  - {jsonl_file.name}: {file_count} tài liệu")

        logger.info(
            f"Hoàn tất: {len(results)} tài liệu từ {len(files)} file "
            f"(bỏ qua: {skipped_empty} rỗng, {skipped_dup} trùng id, {skipped_bad} lỗi JSON)"
        )
        return results

    # ------------------------------------------------------------------ #
    # Tiện ích
    # ------------------------------------------------------------------ #
    def load_all(self) -> List[RawData]:
        """Nạp toàn bộ dữ liệu trong thư mục processed_data mặc định."""
        return self.crawl(DataSource(source_type="directory", uri=str(self.data_dir)))

    @staticmethod
    def _resolve(uri: str) -> Path:
        p = Path(uri)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[Optional[Dict[str, Any]]]:
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

    @staticmethod
    def _to_raw_data(
        record: Dict[str, Any], content: str, jsonl_file: Path, source: DataSource
    ) -> RawData:
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


if __name__ == "__main__":
    # Kiểm thử độc lập: python src/ingestion/loader.py
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from collections import Counter

    loader = UETDataLoader()
    docs = loader.load_all()

    print("\n" + "=" * 60)
    print(f"Đã nạp thành công: {len(docs)} tài liệu từ {loader.data_dir}")
    by_unit = Counter(d.raw_metadata.get("unit") for d in docs)
    print("Phân bố theo đơn vị:", dict(by_unit.most_common()))
    if docs:
        d = docs[0]
        print("\nVí dụ tài liệu đầu tiên:")
        print(f"  id       : {d.id}")
        print(f"  title    : {d.title}")
        print(f"  url      : {d.raw_metadata.get('url')}")
        print(f"  category : {d.raw_metadata.get('category')}")
        print(f"  content  : {d.content[:150]!r}...")
    print("=" * 60)

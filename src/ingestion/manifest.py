"""
src/ingestion/manifest.py - Quản lý trạng thái và chống xử lý trùng lặp dữ liệu (Manifest Tracker & Deduplication).
Sử dụng mã băm nội dung (SHA-256 Content Hashing) và lưu trữ trạng thái tại data/.ingest_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DataManifestTracker:
    """
    Theo dõi vòng đời dữ liệu qua các chặng: raw -> processed -> chunked -> vectordb.
    Ngăn chặn việc xử lý, cắt chunk hoặc tính embedding lặp lại trên cùng một tài liệu.
    """

    DEFAULT_MANIFEST_FILE = Path("data/.ingest_manifest.json")

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else self.DEFAULT_MANIFEST_FILE
        self._data: Dict[str, Any] = {
            "version": "1.0",
            "last_updated": None,
            "documents": {},
        }
        self.load()

    @staticmethod
    def compute_hash(content: str, title: Optional[str] = None) -> str:
        """
        Tính mã băm SHA-256 bất biến của nội dung và tiêu đề tài liệu.
        """
        raw_str = f"title:{title or ''}\ncontent:{content or ''}"
        return hashlib.sha256(raw_str.encode("utf-8", errors="replace")).hexdigest()

    def load(self) -> None:
        """Nạp sổ cái trạng thái từ file JSON trên ổ đĩa."""
        if not self.manifest_path.exists():
            logger.debug(f"Manifest file '{self.manifest_path}' chưa tồn tại, bắt đầu sổ cái mới.")
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "documents" in loaded:
                    self._data = loaded
            logger.debug(f"Đã nạp manifest với {len(self._data['documents'])} tài liệu.")
        except Exception as e:
            logger.warning(f"Lỗi khi đọc manifest file ({e}), khởi tạo lại sổ cái mới.")
            self._data = {"version": "1.0", "last_updated": None, "documents": {}}

    def save(self) -> None:
        """Lưu sổ cái trạng thái xuống file JSON trên ổ đĩa."""
        try:
            self._data["last_updated"] = datetime.now().isoformat()
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Đã lưu manifest ({len(self._data['documents'])} tài liệu).")
        except Exception as e:
            logger.error(f"Lỗi khi lưu manifest xuống file '{self.manifest_path}': {e}")

    def is_processed(self, doc_id: str, stage: str, current_hash: Optional[str] = None) -> bool:
        """
        Kiểm tra xem một tài liệu đã hoàn thành giai đoạn `stage` hay chưa.
        Nếu truyền `current_hash`, kiểm tra thêm xem nội dung có bị thay đổi không.
        """
        docs = self._data.get("documents", {})
        if doc_id not in docs:
            return False

        doc_record = docs[doc_id]
        if current_hash and doc_record.get("hash") != current_hash:
            # Nội dung đã bị thay đổi -> Cần xử lý lại
            return False

        stage_record = doc_record.get("stages", {}).get(stage)
        return stage_record is not None and stage_record.get("status") == "completed"

    def mark_processed(
        self,
        doc_id: str,
        stage: str,
        content_hash: str,
        title: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Đánh dấu một tài liệu đã hoàn thành một giai đoạn xử lý cụ thể.
        """
        docs = self._data.setdefault("documents", {})
        if doc_id not in docs:
            docs[doc_id] = {
                "hash": content_hash,
                "title": title or "",
                "created_at": datetime.now().isoformat(),
                "stages": {},
            }

        doc_record = docs[doc_id]
        doc_record["hash"] = content_hash
        if title:
            doc_record["title"] = title

        stage_meta = {
            "status": "completed",
            "updated_at": datetime.now().isoformat(),
        }
        if extra_meta:
            stage_meta.update(extra_meta)

        doc_record.setdefault("stages", {})[stage] = stage_meta

    def get_stats(self) -> Dict[str, Any]:
        """
        Thống kê tổng số lượng tài liệu đã qua từng chặng trong manifest.
        """
        docs = self._data.get("documents", {})
        total_docs = len(docs)
        stages_count = {"processed": 0, "chunked": 0, "vectordb": 0}

        for d in docs.values():
            stages = d.get("stages", {})
            for s in stages_count:
                if stages.get(s, {}).get("status") == "completed":
                    stages_count[s] += 1

        return {
            "total_tracked_documents": total_docs,
            "stages_completed": stages_count,
            "last_updated": self._data.get("last_updated"),
        }

    def clear(self) -> None:
        """Xóa sạch sổ cái trạng thái theo dõi."""
        self._data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "documents": {},
        }
        if self.manifest_path.exists():
            try:
                self.manifest_path.unlink()
                logger.info(f"Đã xóa file manifest '{self.manifest_path}'.")
            except Exception as e:
                logger.warning(f"Không thể xóa file manifest ({e}).")

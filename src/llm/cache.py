"""Persistent JSON disk cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.utils.helpers import get_logger

logger = get_logger("llm")


class LLMDiskCache:
    """Thread-safe JSON disk cache keyed by SHA-256 hash of prompt text."""

    def __init__(self, cache_file: Path, enabled: bool = False):
        self.cache_file = Path(cache_file)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        if self.enabled:
            self.init_file()

    def init_file(self) -> None:
        """Create the cache file and its parent directory if missing."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.cache_file.exists():
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Không thể khởi tạo file cache {self.cache_file}: {exc}")

    @staticmethod
    def compute_prompt_hash(prompt_text: str) -> str:
        """Compute deterministic SHA-256 hash of a stripped UTF-8 prompt."""
        return hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()

    def get(self, prompt_text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Look up a cached response tuple ``(answer, model_name, metadata)``."""
        if not self.enabled or not self.cache_file.exists():
            return None

        prompt_hash = self.compute_prompt_hash(prompt_text)
        with self._lock:
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if prompt_hash in cache_data:
                    entry = cache_data[prompt_hash]
                    return (
                        entry.get("answer", ""),
                        entry.get("model_name", "disk-cache"),
                        entry.get("metadata", {}),
                    )
            except Exception as exc:
                logger.debug(f"Lỗi đọc disk cache: {exc}")
        return None

    def save(
        self,
        prompt_text: str,
        answer: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a successful LLM response to the JSON cache file."""
        if not self.enabled or not answer or "⚠️" in answer:
            return

        prompt_hash = self.compute_prompt_hash(prompt_text)
        with self._lock:
            try:
                cache_data: Dict[str, Any] = {}
                if self.cache_file.exists():
                    try:
                        with open(self.cache_file, "r", encoding="utf-8") as f:
                            cache_data = json.load(f)
                    except Exception:
                        cache_data = {}

                cache_data[prompt_hash] = {
                    "answer": answer,
                    "model_name": model_name,
                    "metadata": metadata or {},
                    "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.debug(f"Lỗi ghi disk cache: {exc}")

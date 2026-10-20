"""Quản lý các đường dẫn thư mục gốc và thư mục dữ liệu."""
from __future__ import annotations

import os
from pathlib import Path

# Thư mục gốc repo UETChatbot (paths.py -> config -> ingestion -> src -> root)
BASE_DIR = Path(__file__).resolve().parents[3]


def _load_repo_paths() -> dict:
    """Đọc mục `paths` trong config.yaml ở thư mục gốc repo (nếu có)."""
    cfg_file = BASE_DIR / "config.yaml"
    if not cfg_file.exists():
        return {}
    try:
        import yaml

        with open(cfg_file, "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("paths", {}) or {}
    except Exception:
        return {}


_REPO_PATHS = _load_repo_paths()

# Dữ liệu thô theo domain: data/raw_data/<domain>/{endpoints,files}/
DATA_DIR = os.getenv(
    "CHATBOT_DATA_DIR",
    str(BASE_DIR / _REPO_PATHS.get("data_dir", "data/raw_data")),
)
# Dữ liệu đã bóc tách: data/processed_data/<domain>.jsonl
PROCESSED_DATA_DIR = os.getenv(
    "CHATBOT_PROCESSED_DIR",
    str(BASE_DIR / _REPO_PATHS.get("processed_dir", "data/processed_data")),
)

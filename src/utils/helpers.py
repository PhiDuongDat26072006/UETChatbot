"""
src/utils/helpers.py - Các hàm tiện ích dùng chung trong toàn hệ thống.
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def get_logger(name: str = "uet_chatbot") -> logging.Logger:
    """Khởi tạo logger chuẩn ghi ra console và file logs/app.log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Định dạng log
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Đảm bảo UTF-8 trên Windows console
        if sys.platform.startswith("win"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        # Handler ghi ra console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler ghi ra file logs/app.log
        file_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Đọc file cấu hình config.yaml."""
    path = config_path or (BASE_DIR / "config.yaml")
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_project_path(path_value: str | Path, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir or BASE_DIR) / path


def get_ascii_safe_chroma_path(persist_dir: str | Path) -> str:
    """
    Return an ASCII-safe path for ChromaDB on Windows where hnswlib (C++) fails to open
    paths containing non-ASCII characters (e.g. Vietnamese folder names).
    Prefers an ASCII relative path from CWD, then Windows 8.3 short path, before returning
    the resolved string.
    """
    resolved = Path(persist_dir).resolve()
    path_str = str(resolved)
    try:
        path_str.encode("ascii")
        return path_str
    except UnicodeEncodeError:
        pass

    if sys.platform == "win32":
        try:
            rel_str = os.path.relpath(path_str, os.getcwd())
            rel_str.encode("ascii")
            return rel_str
        except (ValueError, UnicodeEncodeError):
            pass

        try:
            import ctypes

            buf_len = ctypes.windll.kernel32.GetShortPathNameW(path_str, None, 0)
            if buf_len > 0:
                buf = ctypes.create_unicode_buffer(buf_len)
                if ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, buf_len):
                    short_str = buf.value
                    short_str.encode("ascii")
                    return short_str
        except Exception:
            pass

    return path_str


def count_nonempty_lines(file_path: Path) -> int:
    """Count non-whitespace lines in a UTF-8 file, returning 0 on read error."""
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            return sum(1 for line in fp if line.strip())
    except Exception:
        return 0


class Timer:
    """Context manager đo thời gian thực thi (latency)."""

    def __init__(self):
        self.start = time.time()
        self.end: Optional[float] = None

    def __enter__(self):
        self.start = time.time()
        self.end = None
        return self

    def __exit__(self, *args):
        self.end = time.time()

    @property
    def elapsed(self) -> float:
        if self.end is not None:
            return self.end - self.start
        return time.time() - self.start

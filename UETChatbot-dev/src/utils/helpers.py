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


class Timer:
    """Context manager đo thời gian thực thi (latency)."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

"""Cho phép chạy module trực tiếp từ thư mục gốc repo: python -m src.ingestion"""
import sys

from src.ingestion.main import main

if __name__ == "__main__":
    sys.exit(main())

"""Điều phối chạy trích xuất toàn bộ các Khoa và Viện trực thuộc UET."""
from __future__ import annotations

import logging
from typing import Any

from ..config import DOMAIN_FOLDERS
from .pipeline_domain import extract_domain_documents
from .pipeline_reporter import print_extraction_summary

logger = logging.getLogger(__name__)


def extract_all_domains(
    domains: list[str] | None = None,
    max_workers: int = 10,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Chạy quy trình bóc tách văn bản cho toàn bộ hoặc danh sách domain được chỉ định."""
    targets = domains if domains else [d["folder"] for d in DOMAIN_FOLDERS]
    all_stats: dict[str, dict[str, Any]] = {}

    print("\n" + "=" * 95)
    print("[*] TIẾN HÀNH BÓC TÁCH NỘI DUNG VĂN BẢN (TEXT EXTRACTION & CLEANING) CHO RAG PIPELINE")
    print("=" * 95 + "\n")

    for domain in targets:
        try:
            stats = extract_domain_documents(domain, max_workers=max_workers, force=force)
            all_stats[domain] = stats
        except Exception as e:
            logger.error("Lỗi bóc tách cho domain %s: %s", domain, e, exc_info=True)

    print_extraction_summary(all_stats)
    return all_stats

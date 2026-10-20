"""Điều phối phân loại toàn bộ endpoints theo domain và xuất dữ liệu nhãn danh mục."""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config.faculties import FACULTY_TARGETS, faculty_storage_name, find_faculty, resolve_unit
from ..config.paths import DATA_DIR
from ..config.session import create_http_session
from ..filters import is_valid_html_endpoint
from .scoring import classify_endpoint, fetch_title_fast
from .taxonomy import Category

logger = logging.getLogger(__name__)

__all__ = [
    "load_existing_metadata",
    "classify_domain",
    "classify_all_domains",
]


def load_existing_metadata(metadata_file: str) -> dict[str, str]:
    """Tải tiêu đề đã lưu từ file metadata jsonl trước đó để tránh cào lại qua HTTP."""
    cached_titles: dict[str, str] = {}
    if not os.path.exists(metadata_file):
        return cached_titles
    try:
        with open(metadata_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Metadata record must be an object")
                    if record.get("url") and record.get("title"):
                        cached_titles[record["url"]] = record["title"]
                except (ValueError, TypeError) as error:
                    logger.warning("Invalid metadata %s:%s (%s)", metadata_file, line_number, error)
    except OSError as error:
        logger.warning("Cannot read metadata cache %s: %s", metadata_file, error)
    return cached_titles


def classify_domain(folder_name: str, max_workers: int = 12) -> dict:
    """Phân loại toàn bộ endpoints của một domain và lưu kết quả theo từng danh mục."""
    target = find_faculty(folder_name)
    if target:
        folder_name = faculty_storage_name(target["faculty_id"])
    domain_dir = os.path.join(DATA_DIR, folder_name)
    endpoints_dir = os.path.join(domain_dir, "endpoints")
    ep_file = os.path.join(endpoints_dir, "endpoints.txt")

    if not os.path.exists(ep_file):
        logger.warning("Missing endpoint file: %s", ep_file)
        return {"folder": folder_name, "total": 0, "categories": {}, "failed": True}

    with open(ep_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    if target and target["allowed_path_prefixes"]:
        urls = [url for url in urls if is_valid_html_endpoint(url, target["domain"], allowed_path_prefixes=target["allowed_path_prefixes"])]
    failed_requests = 0
    metadata_file = os.path.join(endpoints_dir, "endpoints_metadata.jsonl")
    cached_titles = load_existing_metadata(metadata_file)

    urls_to_fetch = [u for u in urls if u not in cached_titles]
    with create_http_session(pool_size=15) as session:
        if urls_to_fetch:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_title_fast, u, session, target): u for u in urls_to_fetch}
                for future in as_completed(futures):
                    try:
                        cached_titles[futures[future]] = future.result()
                    except Exception:
                        failed_requests += 1
                        cached_titles[futures[future]] = ""

    categorized_urls: dict[Category, list[str]] = {cat: [] for cat in Category}
    metadata_records: list[dict] = []

    for u in urls:
        title = cached_titles.get(u, "")
        cat = classify_endpoint(u, title)
        categorized_urls[cat].append(u)
        metadata_records.append({
            "url": u, "domain": target["domain"] if target else folder_name,
            "faculty_id": target["faculty_id"] if target else resolve_unit(folder_name, u),
            "category": cat.value, "title": title
        })

    with open(metadata_file, "w", encoding="utf-8") as f:
        for rec in metadata_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_cat_dir = os.path.join(endpoints_dir, "by_category")
    os.makedirs(by_cat_dir, exist_ok=True)
    for cat in Category:
        with open(os.path.join(by_cat_dir, f"{cat.value}.txt"), "w", encoding="utf-8") as f:
            for u in sorted(categorized_urls[cat]):
                f.write(u + "\n")

    logger.info("[SAVING] %s | %s records", metadata_file, len(metadata_records))
    cat_counts = {cat.value: len(categorized_urls[cat]) for cat in Category}
    return {
        "folder": folder_name,
        "total": len(urls),
        "categories": cat_counts,
        "metadata_file": metadata_file,
        "failed": bool(failed_requests),
    }


def classify_all_domains(max_workers: int = 12) -> list[dict]:
    """Classify stored endpoint lists, retaining successful targets on failure."""
    started = time.monotonic()
    logger.info("[START] Classification started")
    results: list[dict] = []
    failures = 0
    for target in FACULTY_TARGETS:
        folder = faculty_storage_name(target["faculty_id"])
        logger.info("[CLASSIFYING] %s (%s)", target["domain"], target["faculty_id"])
        try:
            result = classify_domain(folder, max_workers=max_workers)
            results.append(result)
            failures += bool(result.get("failed"))
        except Exception as error:
            failures += 1
            logger.error("[ERROR] %s: %s", folder, error)
            logger.debug("Classification failure", exc_info=True)
    if failures:
        raise RuntimeError(f"Classification incomplete: {failures} targets failed")
    logger.info("[DONE] Completed in %.1fs | %s records saved", time.monotonic() - started, sum(result["total"] for result in results))
    return results

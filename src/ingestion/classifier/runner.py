"""Điều phối phân loại toàn bộ endpoints theo domain và xuất dữ liệu."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import DATA_DIR, DOMAIN_FOLDERS, create_http_session
from .reporter import print_classification_summary
from .scoring import classify_endpoint, fetch_title_fast
from .taxonomy import CATEGORY_LABELS, Category


def load_existing_metadata(metadata_file: str) -> dict[str, str]:
    """Tải tiêu đề đã lưu từ file metadata jsonl trước đó (nếu có)."""
    cached_titles = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if obj.get("url") and obj.get("title"):
                            cached_titles[obj["url"]] = obj["title"]
        except Exception:
            pass
    return cached_titles


def classify_domain(folder_name: str, max_workers: int = 12) -> dict:
    """Phân loại toàn bộ endpoints của một domain."""
    domain_dir = os.path.join(DATA_DIR, folder_name)
    endpoints_dir = os.path.join(domain_dir, "endpoints")
    ep_file = os.path.join(endpoints_dir, "endpoints.txt")

    if not os.path.exists(ep_file):
        return {"folder": folder_name, "total": 0, "categories": {}}

    with open(ep_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    metadata_file = os.path.join(endpoints_dir, "endpoints_metadata.jsonl")
    cached_titles = load_existing_metadata(metadata_file)

    urls_to_fetch = [u for u in urls if u not in cached_titles]
    session = create_http_session(pool_size=15)

    if urls_to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_title_fast, u, session): u for u in urls_to_fetch}
            for future in as_completed(futures):
                cached_titles[futures[future]] = future.result()

    categorized_urls = {cat: [] for cat in Category}
    metadata_records = []

    for u in urls:
        title = cached_titles.get(u, "")
        cat = classify_endpoint(u, title)
        categorized_urls[cat].append(u)
        metadata_records.append({
            "url": u, "domain": folder_name, "category": cat.value, "title": title
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

    cat_counts = {cat.value: len(categorized_urls[cat]) for cat in Category}
    return {
        "folder": folder_name,
        "total": len(urls),
        "categories": cat_counts,
        "metadata_file": metadata_file,
    }


def classify_all_domains(max_workers: int = 12) -> list[dict]:
    """Phân loại toàn bộ các Khoa và Viện trực thuộc UET."""
    t0 = time.time()
    domains = [d["folder"] for d in DOMAIN_FOLDERS]
    all_stats = []
    for d in domains:
        stat = classify_domain(d, max_workers=max_workers)
        all_stats.append(stat)

    print_classification_summary(all_stats, round(time.time() - t0, 2))
    return all_stats

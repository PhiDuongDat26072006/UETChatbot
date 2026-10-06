"""Điều phối phân loại toàn bộ endpoints theo domain và xuất dữ liệu."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import DATA_DIR, DOMAIN_FOLDERS, create_http_session
from .scoring import classify_endpoint, fetch_title_fast
from .taxonomy import CATEGORY_LABELS, Category


def load_existing_metadata(metadata_file: str) -> dict[str, str]:
    """Tải tiêu đề đã lưu từ file metadata jsonl trước đó (nếu có) để tái sử dụng nhanh."""
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
    """
    Phân loại toàn bộ endpoints của một domain:
    1. Đọc data/raw_data/<domain>/endpoints/endpoints.txt.
    2. Lấy title (từ cache hoặc tải streaming nhanh).
    3. Phân loại theo Category Enum.
    4. Xuất file metadata JSONL và các file txt theo danh mục.
    """
    domain_dir = os.path.join(DATA_DIR, folder_name)
    endpoints_dir = os.path.join(domain_dir, "endpoints")
    ep_file = os.path.join(endpoints_dir, "endpoints.txt")

    if not os.path.exists(ep_file):
        return {"folder": folder_name, "total": 0, "categories": {}}

    with open(ep_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    metadata_file = os.path.join(endpoints_dir, "endpoints_metadata.jsonl")
    cached_titles = load_existing_metadata(metadata_file)

    # 1. Thu thập tiêu đề cho các URL chưa có trong cache
    urls_to_fetch = [u for u in urls if u not in cached_titles]
    session = create_http_session(pool_size=15)

    if urls_to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_title_fast, u, session): u for u in urls_to_fetch}
            for future in as_completed(futures):
                u = futures[future]
                title = future.result()
                cached_titles[u] = title

    # 2. Phân loại từng URL
    categorized_urls = {cat: [] for cat in Category}
    metadata_records = []

    for u in urls:
        title = cached_titles.get(u, "")
        cat = classify_endpoint(u, title)
        categorized_urls[cat].append(u)
        metadata_records.append({
            "url": u,
            "domain": folder_name,
            "category": cat.value,
            "title": title
        })

    # 3. Ghi file tổng hợp endpoints_metadata.jsonl
    with open(metadata_file, "w", encoding="utf-8") as f:
        for rec in metadata_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4. Ghi các file txt chia theo danh mục vào data/raw_data/<domain>/endpoints/by_category/<category>.txt
    by_cat_dir = os.path.join(endpoints_dir, "by_category")
    os.makedirs(by_cat_dir, exist_ok=True)

    for cat in Category:
        cat_file = os.path.join(by_cat_dir, f"{cat.value}.txt")
        url_list = sorted(categorized_urls[cat])
        with open(cat_file, "w", encoding="utf-8") as f:
            for u in url_list:
                f.write(u + "\n")

    cat_counts = {cat.value: len(categorized_urls[cat]) for cat in Category}
    return {
        "folder": folder_name,
        "total": len(urls),
        "categories": cat_counts,
        "metadata_file": metadata_file,
        "by_category_dir": by_cat_dir
    }


def classify_all_domains(max_workers: int = 12) -> list[dict]:
    """Phân loại toàn bộ các Khoa và Viện trực thuộc UET."""
    t0 = time.time()
    print("\n" + "=" * 80)
    print("[*] TIẾN HÀNH PHÂN LOẠI URL (URL CLASSIFICATION) CHO TOÀN BỘ ENDPOINTS")
    print("=" * 80)

    domains = [d["folder"] for d in DOMAIN_FOLDERS]

    all_stats = []
    for d in domains:
        print(f"\n[*] Đang phân loại [{d}]...")
        stat = classify_domain(d, max_workers=max_workers)
        all_stats.append(stat)

        # In tóm tắt nhanh cho domain
        print(f"    -> Tổng: {stat['total']} URLs")
        for cat in Category:
            count = stat["categories"].get(cat.value, 0)
            if count > 0:
                print(f"       + {cat.value:<20}: {count:>4} ({CATEGORY_LABELS[cat]})")

        # Đồng bộ iai.uet.vnu.edu.vn sang uet.vnu.edu.vn nếu thư mục đó tồn tại
        if d == "iai.uet.vnu.edu.vn":
            alt_dir = os.path.join(DATA_DIR, "uet.vnu.edu.vn", "endpoints")
            if os.path.exists(alt_dir):
                meta_src = stat["metadata_file"]
                meta_dst = os.path.join(alt_dir, "endpoints_metadata.jsonl")
                if os.path.exists(meta_src):
                    with open(meta_src, "r", encoding="utf-8") as f1, open(meta_dst, "w", encoding="utf-8") as f2:
                        f2.write(f1.read())

    elapsed = round(time.time() - t0, 2)
    print_classification_summary(all_stats, elapsed)
    return all_stats


def print_classification_summary(stats: list[dict], elapsed_time: float = None):
    """In bảng tổng hợp phân loại chuyên nghiệp theo chuẩn dự án."""
    print("\n" + "=" * 90)
    print("[✓] BẢNG TỔNG HỢP PHÂN LOẠI ENDPOINTS TOÀN TRƯỜNG ĐẠI HỌC CÔNG NGHỆ (UET)")
    if elapsed_time:
        print(f"    Thời gian hoàn thành: {elapsed_time} giây")
    print("=" * 90)

    header = f"{'Domain':<22} | {'Tổng':<5} | {'DAO_TAO':<7} | {'TUYEN_SINH':<10} | {'QUY_CHE':<7} | {'NC_HOPTAC':<9} | {'CAN_BO':<6} | {'TIN_TUC':<7} | {'KHAC':<5}"
    print(header)
    print("-" * 90)

    totals = {cat.value: 0 for cat in Category}
    grand_total = 0

    for s in stats:
        folder = s["folder"]
        tot = s["total"]
        grand_total += tot
        c = s["categories"]
        for cat in Category:
            totals[cat.value] += c.get(cat.value, 0)

        line = (
            f"{folder:<22} | {tot:<5} | "
            f"{c.get(Category.DAO_TAO.value, 0):<7} | "
            f"{c.get(Category.TUYEN_SINH.value, 0):<10} | "
            f"{c.get(Category.QUY_CHE_BIEU_MAU.value, 0):<7} | "
            f"{c.get(Category.NGHIEN_CUU_HOP_TAC.value, 0):<9} | "
            f"{c.get(Category.CAN_BO_GIANG_VIEN.value, 0):<6} | "
            f"{c.get(Category.TIN_TUC_SU_KIEN.value, 0):<7} | "
            f"{c.get(Category.KHAC.value, 0):<5}"
        )
        print(line)

    print("-" * 90)
    tot_line = (
        f"{'TỔNG CỘNG':<22} | {grand_total:<5} | "
        f"{totals[Category.DAO_TAO.value]:<7} | "
        f"{totals[Category.TUYEN_SINH.value]:<10} | "
        f"{totals[Category.QUY_CHE_BIEU_MAU.value]:<7} | "
        f"{totals[Category.NGHIEN_CUU_HOP_TAC.value]:<9} | "
        f"{totals[Category.CAN_BO_GIANG_VIEN.value]:<6} | "
        f"{totals[Category.TIN_TUC_SU_KIEN.value]:<7} | "
        f"{totals[Category.KHAC.value]:<5}"
    )
    print(tot_line)
    print("=" * 90)

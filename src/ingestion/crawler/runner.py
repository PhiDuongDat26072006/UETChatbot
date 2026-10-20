"""Điều phối chạy crawler cho toàn bộ các Khoa và Viện trực thuộc UET."""
from __future__ import annotations

import time

from ..config import DATA_DIR, FACULTY_TARGETS
from ..filters import get_domain_folder_name


def crawl_domain(target_url: str, base_data_dir: str = DATA_DIR):
    """Cào dữ liệu cho 1 domain đơn lẻ (hỗ trợ tương thích ngược)."""
    from .engine import crawl_target

    target_info = {
        "url": target_url,
        "domain": get_domain_folder_name(target_url),
        "code": get_domain_folder_name(target_url).split(".")[0].upper(),
        "name": target_url,
    }
    return crawl_target(target_info, base_data_dir=base_data_dir)


def crawl_all_faculties(targets: list[dict] = FACULTY_TARGETS, download_files: bool = True):
    """Chạy thu thập toàn bộ các Khoa và Viện trực thuộc UET."""
    from .engine import crawl_target

    start_time = time.time()
    summary = []

    print("\n" + "=" * 60)
    print(f"[*] BẮT ĐẦU CÀO DỮ LIỆU {len(targets)} KHOA & VIỆN TRỰC THUỘC UET")
    print("=" * 60)

    for target in targets:
        code = target.get("code", "")
        name = target.get("name", "")
        t0 = time.time()
        ep_count, fl_count = crawl_target(target, download_files=download_files)
        summary.append({
            "code": code,
            "name": name,
            "endpoints": ep_count,
            "files": fl_count,
            "time": round(time.time() - t0, 2),
        })

    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 60)
    print(f"[✓] TỔNG KẾT HOÀN THÀNH TRONG {total_time} GIÂY")
    print("=" * 60)
    print(f"{'Mã':<8} | {'Tên Khoa/Viện':<36} | {'Endpoints':<10} | {'Files':<6} | {'Thời gian'}")
    print("-" * 75)
    total_eps = sum(s["endpoints"] for s in summary)
    total_fls = sum(s["files"] for s in summary)
    for s in summary:
        print(f"{s['code']:<8} | {s['name']:<36} | {s['endpoints']:<10} | {s['files']:<6} | {s['time']}s")
    print("-" * 75)
    print(f"{'TỔNG':<8} | {len(targets)} Khoa/Viện{'':<24} | {total_eps:<10} | {total_fls:<6} | {total_time}s")


def main():
    """Hàm khởi chạy chính."""
    crawl_all_faculties(FACULTY_TARGETS, download_files=True)

"""Điều phối kiểm tra và sửa trailing slash cho toàn bộ các Khoa và Viện."""
from __future__ import annotations

import os
import time

from ..config import DATA_DIR, DOMAIN_FOLDERS


def verify_all_endpoints():
    """Kiểm tra và sửa lỗi trailing slash cho toàn bộ các Khoa và Viện."""
    from .fixer import verify_endpoint_file

    t0 = time.time()
    print("\n" + "=" * 80)
    print("[*] TIẾN HÀNH KIỂM TRA & TỰ ĐỘNG SỬA TRAILING SLASH CHO TOÀN BỘ ENDPOINTS")
    print("=" * 80)

    stats = []
    for item in DOMAIN_FOLDERS:
        code = item["code"]
        name = item["name"]
        folder = item["folder"]
        ep_file = os.path.join(DATA_DIR, folder, "endpoints", "endpoints.txt")

        if folder == "iai.uet.vnu.edu.vn" and not os.path.exists(ep_file):
            alt_ep = os.path.join(DATA_DIR, "uet.vnu.edu.vn", "endpoints", "endpoints.txt")
            if os.path.exists(alt_ep):
                os.makedirs(os.path.dirname(ep_file), exist_ok=True)
                with open(alt_ep, "r", encoding="utf-8") as f_in, open(ep_file, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())

        if not os.path.exists(ep_file):
            continue

        res = verify_endpoint_file(ep_file, max_workers=10)
        if folder == "iai.uet.vnu.edu.vn":
            uet_vnu_ep = os.path.join(DATA_DIR, "uet.vnu.edu.vn", "endpoints", "endpoints.txt")
            if os.path.exists(os.path.dirname(uet_vnu_ep)):
                with open(ep_file, "r", encoding="utf-8") as f_src, open(uet_vnu_ep, "w", encoding="utf-8") as f_dst:
                    f_dst.write(f_src.read())

        stats.append({"code": code, "name": name, "folder": folder, **res})

    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT KIỂM TRA TOÀN BỘ ENDPOINTS TRONG {elapsed} GIÂY")
    print("=" * 80)
    print(f"{'Mã':<6} | {'Thư mục':<25} | {'Tổng':<6} | {'Đã sửa /':<9} | {'Bỏ 404':<8} | {'Hợp lệ'}")
    print("-" * 80)
    for s in stats:
        print(f"{s['code']:<6} | {s['folder']:<25} | {s['total']:<6} | {s['corrected']:<9} | {s['dropped']:<8} | {s['final']}")
    print("-" * 80)
    tot_before = sum(s["total"] for s in stats)
    tot_corr = sum(s["corrected"] for s in stats)
    tot_drop = sum(s["dropped"] for s in stats)
    tot_final = sum(s["final"] for s in stats)
    print(f"{'TỔNG':<6} | {'Toàn bộ đơn vị':<25} | {tot_before:<6} | {tot_corr:<9} | {tot_drop:<8} | {tot_final}")
    print("=" * 90)

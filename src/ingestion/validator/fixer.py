"""Xác thực hàng loạt và cập nhật endpoints.txt cho các đơn vị."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import DATA_DIR, DOMAIN_FOLDERS, create_http_session
from .detector import check_and_fix_endpoint


def verify_endpoint_file(file_path: str, max_workers: int = 10) -> dict:
    """Xác thực toàn bộ URL trong file data/raw_data/<domain>/endpoints/endpoints.txt."""
    if not os.path.exists(file_path):
        return {"total": 0, "valid": 0, "corrected": 0, "dropped": 0, "final": 0}

    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    session = create_http_session(pool_size=15, max_retries=2)
    final_urls = set()
    valid_count = 0
    corrected_count = 0
    dropped_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_and_fix_endpoint, u, session): u for u in urls}
        for future in as_completed(futures):
            orig_url, result_url, status = future.result()
            if result_url:
                final_urls.add(result_url)
                if status == "CORRECTED_SLASH":
                    corrected_count += 1
                else:
                    valid_count += 1
            else:
                dropped_count += 1

    sorted_urls = sorted(final_urls)

    # Ghi đè file sạch chỉ tại endpoints/endpoints.txt
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for u in sorted_urls:
            f.write(u + "\n")

    return {
        "total": len(urls),
        "valid": valid_count,
        "corrected": corrected_count,
        "dropped": dropped_count,
        "final": len(sorted_urls)
    }


def verify_all_endpoints():
    """Kiểm tra và sửa lỗi trailing slash cho toàn bộ các Khoa và Viện trong thư mục data/."""
    t0 = time.time()

    print("\n" + "=" * 80)
    print("[*] TIẾN HÀNH KIỂM TRA & TỰ ĐỘNG SỬA TRAILING SLASH CHO TOÀN BỘ ENDPOINTS")
    print("    Lưu trữ chuẩn tại: data/raw_data/<domain>/endpoints/endpoints.txt")
    print("=" * 80)

    stats = []

    for item in DOMAIN_FOLDERS:
        code = item["code"]
        name = item["name"]
        folder = item["folder"]
        ep_file = os.path.join(DATA_DIR, folder, "endpoints", "endpoints.txt")

        # Đồng bộ nếu IAI nằm trong uet.vnu.edu.vn
        if folder == "iai.uet.vnu.edu.vn" and not os.path.exists(ep_file):
            alt_ep = os.path.join(DATA_DIR, "uet.vnu.edu.vn", "endpoints", "endpoints.txt")
            if os.path.exists(alt_ep):
                os.makedirs(os.path.dirname(ep_file), exist_ok=True)
                with open(alt_ep, "r", encoding="utf-8") as f_in, open(ep_file, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())

        if not os.path.exists(ep_file):
            print(f"\n[-] Bỏ qua [{code}] (không tìm thấy file {ep_file})")
            continue

        print(f"\n[*] Đang xác thực [{code}]: {name} ({ep_file})...")
        res = verify_endpoint_file(ep_file, max_workers=10)

        # Nếu là IAI, cập nhật đồng thời cho cả uet.vnu.edu.vn nếu thư mục đó tồn tại
        if folder == "iai.uet.vnu.edu.vn":
            uet_vnu_ep = os.path.join(DATA_DIR, "uet.vnu.edu.vn", "endpoints", "endpoints.txt")
            if os.path.exists(os.path.dirname(uet_vnu_ep)):
                with open(ep_file, "r", encoding="utf-8") as f_src, open(uet_vnu_ep, "w", encoding="utf-8") as f_dst:
                    f_dst.write(f_src.read())

        stats.append({
            "code": code,
            "name": name,
            "folder": folder,
            **res
        })
        print(f"    -> Trước: {res['total']} | Chuẩn: {res['valid']} | Đã sửa /: {res['corrected']} | Bỏ 404: {res['dropped']} -> Còn lại: {res['final']}")

    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT KIỂM TRA TOÀN BỘ ENDPOINTS TRONG {elapsed} GIÂY")
    print("=" * 80)
    print(f"{'Mã':<6} | {'Thư mục':<25} | {'Tổng':<6} | {'Đã sửa /':<9} | {'Bỏ 404':<8} | {'Hợp lệ'}")
    print("-" * 80)
    tot_before = sum(s["total"] for s in stats)
    tot_corr = sum(s["corrected"] for s in stats)
    tot_drop = sum(s["dropped"] for s in stats)
    tot_final = sum(s["final"] for s in stats)

    for s in stats:
        print(f"{s['code']:<6} | {s['folder']:<25} | {s['total']:<6} | {s['corrected']:<9} | {s['dropped']:<8} | {s['final']}")
    print("-" * 80)
    print(f"{'TỔNG':<6} | {'Toàn bộ đơn vị':<25} | {tot_before:<6} | {tot_corr:<9} | {tot_drop:<8} | {tot_final}")
    print("=" * 90)

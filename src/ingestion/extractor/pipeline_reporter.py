"""Báo cáo và in bảng tổng kết bóc tách văn bản toàn trường UET."""
from __future__ import annotations

from typing import Any


def print_extraction_summary(all_stats: dict[str, dict[str, Any]]):
    """In bảng tổng hợp kết quả bóc tách văn bản ra màn hình console."""
    print("\n" + "=" * 95)
    print("[✓] BẢNG TỔNG HỢP KẾT QUẢ BÓC TÁCH NỘI DUNG VĂN BẢN TOÀN TRƯỜNG ĐẠI HỌC CÔNG NGHỆ (UET)")
    print("=" * 95)
    header = (
        f"{'Đơn vị (Domain)':<24} | {'HTMLs':<11} | {'Files':<11} | "
        f"{'Tổng Docs':<10} | {'Dung lượng Text':<18} | {'Ước tính Tokens':<15}"
    )
    print(header)
    print("-" * 95)

    tot_ep = tot_html = tot_f_all = tot_f_ok = tot_docs = tot_chars = 0

    for domain, s in all_stats.items():
        tot_ep += s["total_endpoints"]
        tot_html += s["html_success"]
        tot_f_all += s["total_files"]
        tot_f_ok += s["files_success"]
        tot_docs += s["total_docs"]
        tot_chars += s["total_chars"]

        html_str = f"{s['html_success']}/{s['total_endpoints']}"
        file_str = f"{s['files_success']}/{s['total_files']}"
        chars_str = f"{s['total_chars']:,} chars"
        est_tokens = f"~{s['total_chars'] // 3:,} tokens"

        print(f"{domain:<24} | {html_str:<11} | {file_str:<11} | {s['total_docs']:<10} | {chars_str:<18} | {est_tokens:<15}")

    print("-" * 95)
    print(
        f"{'TỔNG CỘNG':<24} | {tot_html}/{tot_ep:<10} | {tot_f_ok}/{tot_f_all:<10} | "
        f"{tot_docs:<10} | {tot_chars:,} chars{'':<6} | ~{tot_chars // 3:,} tokens"
    )
    print("=" * 95 + "\n")

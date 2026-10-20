#!/usr/bin/env python3
"""Entry point chính để khởi chạy hệ thống crawler, phân loại và bóc tách dữ liệu UET Chatbot.

Cách chạy (từ thư mục gốc repo):
    python -m src.ingestion.main --crawl
    python -m src.ingestion.main --classify [--domain fit.uet.vnu.edu.vn]
    python -m src.ingestion.main --extract  [--domain fit.uet.vnu.edu.vn] [--force]
"""
import sys
import argparse
from pathlib import Path

# Đảm bảo thư mục gốc repo nằm trong sys.path khi chạy trực tiếp `python src/ingestion/main.py`
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ingestion.crawler import crawl_all_faculties
from src.ingestion.validator import verify_all_endpoints
from src.ingestion.classifier import classify_all_domains, classify_domain, print_classification_summary
from src.ingestion.extractor import extract_all_domains, extract_domain_documents


def main():
    parser = argparse.ArgumentParser(
        description="Hệ thống Crawler, Phân loại & Bóc tách Dữ liệu UET Chatbot"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Chạy bóc tách nội dung văn bản sạch (Text Extraction) từ HTML endpoints và files tài liệu (.pdf, .docx...)."
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Chạy phân loại URL (URL classification) cho các endpoints và xuất metadata JSONL, chia theo danh mục."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Chỉ định domain cụ thể để xử lý (ví dụ: fema.uet.vnu.edu.vn, fit.uet.vnu.edu.vn, fepn.uet.vnu.edu.vn)."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Chạy kiểm tra và sửa lỗi trailing slash cho các endpoints."
    )
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Chạy cào toàn bộ endpoints và tài liệu cho các Khoa/Viện."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Không tải file tài liệu khi cào dữ liệu."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bỏ qua cache và thực hiện xử lý lại toàn bộ."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Số luồng xử lý đồng thời khi tải/bóc tách dữ liệu (mặc định: 10)."
    )

    args = parser.parse_args()

    if args.extract:
        if args.domain:
            extract_domain_documents(args.domain, max_workers=args.workers, force=args.force)
        else:
            extract_all_domains(max_workers=args.workers, force=args.force)
    elif args.classify:
        if args.domain:
            stat = classify_domain(args.domain)
            print_classification_summary([stat])
        else:
            classify_all_domains()
    elif args.verify:
        verify_all_endpoints()
    else:
        # Mặc định chạy crawler nếu gọi --crawl hoặc không có flag
        crawl_all_faculties(download_files=not args.no_download)


if __name__ == "__main__":
    main()

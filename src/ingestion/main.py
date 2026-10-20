#!/usr/bin/env python3
"""Entry point chính để khởi chạy hệ thống crawler, phân loại và bóc tách dữ liệu UET Chatbot.

Cách chạy (từ thư mục gốc repo):
    python -m src.ingestion.main --crawl
    python -m src.ingestion.main --classify [--domain fit.uet.vnu.edu.vn]
    python -m src.ingestion.main --extract  [--domain fit.uet.vnu.edu.vn] [--force]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Đảm bảo thư mục gốc repo nằm trong sys.path khi chạy trực tiếp `python src/ingestion/main.py`
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ingestion.crawler import crawl_all_faculties
from src.ingestion.validator import verify_all_endpoints
from src.ingestion.classifier import classify_all_domains, classify_domain
from src.ingestion.extractor import extract_all_domains, extract_domain_documents


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the selected stage; return nonzero for incomplete or failed work."""
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
        help="Faculty ID, storage folder, or domain to classify/extract (use IAI for the AI Institute)."
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
        "--migrate",
        action="store_true",
        help="Thực thi di trú dữ liệu an toàn sang cấu trúc DataSource -> RawData -> ProcessedData.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Số luồng bóc tách nội dung (mặc định: 10)."
    )

    parser.add_argument("--debug", action="store_true", help="Enable detailed diagnostics.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    if args.debug:
        logging.getLogger("src.ingestion").setLevel(logging.DEBUG)
    started = time.monotonic()
    single_target = args.domain and (args.extract or args.classify)
    if single_target:
        logger.info("[START] %s started", "Extraction" if args.extract else "Classification")
        logger.info("[%s] %s", "EXTRACTING" if args.extract else "CLASSIFYING", args.domain)
    try:
        if args.migrate:
            from src.ingestion.migration import migrate_all
            migrate_all()
        elif args.extract:
            if args.domain:
                result = extract_domain_documents(args.domain, max_workers=args.workers, force=args.force)
                if result.get("failed"):
                    raise RuntimeError("Extraction incomplete; successful records were saved")
            else:
                extract_all_domains(max_workers=args.workers, force=args.force)
        elif args.classify:
            if args.domain:
                result = classify_domain(args.domain)
                if result.get("failed"):
                    raise RuntimeError("Classification incomplete; available results were saved")
            else:
                classify_all_domains()
        elif args.verify:
            verify_all_endpoints()
        else:
            crawl_all_faculties(download_files=not args.no_download)
    except Exception as error:
        logger.error("[FAILED] Ingestion failed after %.1fs | %s", time.monotonic() - started, error)
        logger.debug("Ingestion failure", exc_info=True)
        return 1
    if single_target:
        count = result["total_docs"] if args.extract else result["total"]
        logger.info("[DONE] Completed in %.1fs | %s records saved", time.monotonic() - started, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())

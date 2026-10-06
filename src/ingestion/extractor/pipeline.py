"""Module điều phối quy trình bóc tách văn bản (Extraction Pipeline) cho các đơn vị UET.

Tải song song nội dung bài viết HTML và bóc tách các file tài liệu đính kèm,
gắn nhãn metadata taxonomy và xuất ra data/processed_data/<domain>.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..classifier import Category, classify_endpoint
from ..config import DATA_DIR, DOMAIN_FOLDERS, PROCESSED_DATA_DIR, resolve_unit
from .doc_extractor import extract_document_text
from .html_extractor import extract_html_content
from .pipeline_helpers import (
    DUPLICATE_STUB_MAX_LENGTH,
    canonicalize_url,
    generate_doc_id,
    is_duplicate_stub,
    load_existing_processed_docs,
)
from .summary import generate_document_summary

logger = logging.getLogger(__name__)


def extract_domain_documents(
    domain: str,
    max_workers: int = 10,
    force: bool = False,
) -> dict[str, Any]:
    """Bóc tách toàn bộ tài liệu HTML và files tài liệu cho một domain.

    Args:
        domain: Tên miền đơn vị (ví dụ: 'fit.uet.vnu.edu.vn').
        max_workers: Số luồng xử lý đồng thời khi tải HTML (mặc định: 10).
        force: Nếu True, bỏ qua cache và bóc tách lại từ đầu.

    Returns:
        dict thống kê kết quả bóc tách.
    """
    domain_dir = Path(DATA_DIR) / domain
    endpoints_dir = domain_dir / "endpoints"
    files_dir = domain_dir / "files"
    output_file = Path(PROCESSED_DATA_DIR) / f"{domain}.jsonl"

    domain_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Đọc danh sách endpoints và metadata đã phân loại
    metadata_file = endpoints_dir / "endpoints_metadata.jsonl"
    endpoints_file = endpoints_dir / "endpoints.txt"

    endpoints_info: list[dict[str, Any]] = []
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        endpoints_info.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    elif endpoints_file.exists():
        with open(endpoints_file, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url:
                    endpoints_info.append({
                        "url": url,
                        "domain": domain,
                        "category": Category.KHAC.value,
                        "title": "",
                    })

    # 2. Đọc tài liệu đã bóc tách từ trước (nếu không dùng cờ force)
    raw_existing = {} if force else load_existing_processed_docs(output_file)
    seen_content_hashes: dict[str, str] = {}
    processed_records: dict[str, dict[str, Any]] = {}

    if not force:
        for source, rec in raw_existing.items():
            content = rec.get("content", "")
            chash = hashlib.md5(content.strip().encode("utf-8")).hexdigest()
            if chash in seen_content_hashes and is_duplicate_stub(content):
                logger.info("Skipping cached duplicate stub: %s matches %s", source, seen_content_hashes[chash])
                continue
            seen_content_hashes[chash] = source
            # Đảm bảo record có trường summary nếu trước đó chưa có
            if "summary" not in rec:
                rec["summary"] = generate_document_summary(
                    domain=rec.get("domain", domain),
                    title=rec.get("title", ""),
                    content=content,
                )
            # Đảm bảo record có trường unit nếu trước đó chưa có
            if "unit" not in rec:
                rec["unit"] = resolve_unit(rec.get("domain", domain))
            processed_records[source] = rec

    # Bảng tra cứu canonical URL để tránh lệch www. vs không www. hoặc trailing slash gây tải sót/trùng
    canonical_processed: dict[str, str] = {
        canonicalize_url(source): source
        for source in processed_records
    }

    # 3. Lọc danh sách URL cần bóc tách
    urls_to_process: list[dict[str, Any]] = [
        item for item in endpoints_info
        if force or (
            item["url"] not in processed_records and
            canonicalize_url(item["url"]) not in canonical_processed
        )
    ]

    print(f"[*] [{domain}] Bóc tách HTML ({len(urls_to_process)}/{len(endpoints_info)} URLs cần tải)...")

    # 4. Tải và bóc tách HTML đa luồng
    def process_url(item: dict[str, Any]) -> dict[str, Any] | None:
        url = item["url"]
        meta_cat = item.get("category", Category.KHAC.value)
        meta_title = item.get("title", "")

        extracted = extract_html_content(url=url, timeout=10, category=meta_cat)
        if not extracted:
            return None

        # Sử dụng tiêu đề từ trích xuất hoặc từ metadata phân loại
        title = extracted.get("title") or meta_title or url
        content = extracted["text"]

        summary = generate_document_summary(
            domain=domain,
            title=title,
            content=content,
        )

        return {
            "id": generate_doc_id(url),
            "source_type": "html",
            "source_url_or_path": url,
            "title": title,
            "domain": domain,
            "unit": resolve_unit(domain),
            "category": meta_cat,
            "published_date": extracted.get("published_date"),
            "summary": summary,
            "content": content,
            "content_length": extracted["text_length"],
            "cloud_links": extracted.get("cloud_links", []),
        }

    html_success_new = 0
    if urls_to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(process_url, item): item["url"]
                for item in urls_to_process
            }
            for future in as_completed(future_to_url):
                res = future.result()
                if res:
                    content = res["content"]
                    chash = hashlib.md5(content.strip().encode("utf-8")).hexdigest()
                    if chash in seen_content_hashes:
                        seen_url = seen_content_hashes[chash]
                        if is_duplicate_stub(content):
                            logger.info("Skipping duplicate stub content: %s matches %s", res["source_url_or_path"], seen_url)
                            continue
                    seen_content_hashes[chash] = res["source_url_or_path"]
                    processed_records[res["source_url_or_path"]] = res
                    canonical_processed[canonicalize_url(res["source_url_or_path"])] = res["source_url_or_path"]
                    html_success_new += 1

    # 5. Quét và bóc tách các file tài liệu trong files/
    doc_files: list[Path] = []
    if files_dir.exists():
        for p in files_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".xlsx", ".doc", ".txt", ".csv"):
                # Bỏ qua files_list.txt
                if p.name != "files_list.txt":
                    doc_files.append(p)

    files_to_process = [
        p for p in doc_files
        if force or str(p) not in processed_records
    ]

    print(f"[*] [{domain}] Bóc tách Files ({len(files_to_process)}/{len(doc_files)} tệp cần đọc)...")

    files_success_new = 0
    for file_path in files_to_process:
        extracted = extract_document_text(file_path)
        if not extracted:
            continue

        clean_name = extracted["filename"]
        content = extracted["text"]
        chash = hashlib.md5(content.strip().encode("utf-8")).hexdigest()
        if chash in seen_content_hashes:
            seen_url = seen_content_hashes[chash]
            if is_duplicate_stub(content):
                logger.info("Skipping duplicate stub content: %s matches %s", str(file_path), seen_url)
                continue
        seen_content_hashes[chash] = str(file_path)

        # Phân loại danh mục cho file dựa trên tên file đã giải mã
        file_category = classify_endpoint(
            url=str(file_path),
            title=urllib.parse.unquote(file_path.stem),
        )

        summary = generate_document_summary(
            domain=domain,
            title=clean_name,
            content=content,
        )

        record = {
            "id": generate_doc_id(str(file_path)),
            "source_type": "file",
            "source_url_or_path": str(file_path),
            "title": clean_name,
            "domain": domain,
            "unit": resolve_unit(domain),
            "category": file_category.value,
            "published_date": extracted.get("published_date"),
            "summary": summary,
            "content": content,
            "content_length": extracted["text_length"],
            "cloud_links": [],
        }
        processed_records[str(file_path)] = record
        files_success_new += 1

    # 6. Ghi toàn bộ bản ghi ra file data/processed_data/<domain>.jsonl (ghi nguyên khối an toàn)
    temp_output_file = output_file.with_suffix(".tmp")
    with open(temp_output_file, "w", encoding="utf-8") as f:
        for record in processed_records.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_output_file.replace(output_file)

    # 7. Thống kê số liệu
    html_records = [r for r in processed_records.values() if r.get("source_type") == "html"]
    file_records = [r for r in processed_records.values() if r.get("source_type") == "file"]

    html_chars = sum(r.get("content_length", 0) for r in html_records)
    file_chars = sum(r.get("content_length", 0) for r in file_records)

    stats = {
        "domain": domain,
        "total_endpoints": len(endpoints_info),
        "html_success": len(html_records),
        "html_chars": html_chars,
        "total_files": len(doc_files),
        "files_success": len(file_records),
        "files_chars": file_chars,
        "total_docs": len(processed_records),
        "total_chars": html_chars + file_chars,
    }

    print(
        f"    -> Xong: {stats['html_success']}/{stats['total_endpoints']} HTMLs | "
        f"{stats['files_success']}/{stats['total_files']} Files | "
        f"Tổng: {stats['total_docs']} docs ({stats['total_chars']:,} ký tự)"
    )

    return stats


def extract_all_domains(
    domains: list[str] | None = None,
    max_workers: int = 10,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Chạy quy trình bóc tách văn bản cho toàn bộ hoặc danh sách domain được chỉ định.

    Args:
        domains: Danh sách domain cần bóc tách. Nếu None, lấy tất cả domain trong DOMAIN_FOLDERS.
        max_workers: Số luồng xử lý đồng thời.
        force: Nếu True, bóc tách lại toàn bộ.

    Returns:
        dict thống kê kết quả từng domain.
    """
    if domains:
        targets = domains
    else:
        targets = [d["folder"] for d in DOMAIN_FOLDERS]
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

    # In bảng tổng hợp
    print("\n" + "=" * 95)
    print("[✓] BẢNG TỔNG HỢP KẾT QUẢ BÓC TÁCH NỘI DUNG VĂN BẢN TOÀN TRƯỜNG ĐẠI HỌC CÔNG NGHỆ (UET)")
    print("=" * 95)
    header = (
        f"{'Đơn vị (Domain)':<24} | {'HTMLs':<11} | {'Files':<11} | "
        f"{'Tổng Docs':<10} | {'Dung lượng Text':<18} | {'Ước tính Tokens':<15}"
    )
    print(header)
    print("-" * 95)

    tot_ep = 0
    tot_html = 0
    tot_f_all = 0
    tot_f_ok = 0
    tot_docs = 0
    tot_chars = 0

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

        print(
            f"{domain:<24} | {html_str:<11} | {file_str:<11} | "
            f"{s['total_docs']:<10} | {chars_str:<18} | {est_tokens:<15}"
        )

    print("-" * 95)
    tot_html_str = f"{tot_html}/{tot_ep}"
    tot_file_str = f"{tot_f_ok}/{tot_f_all}"
    tot_chars_str = f"{tot_chars:,} chars"
    tot_tokens = f"~{tot_chars // 3:,} tokens"
    print(
        f"{'TỔNG CỘNG':<24} | {tot_html_str:<11} | {tot_file_str:<11} | "
        f"{tot_docs:<10} | {tot_chars_str:<18} | {tot_tokens:<15}"
    )
    print("=" * 95 + "\n")

    return all_stats

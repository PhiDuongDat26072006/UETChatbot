"""Module điều phối quy trình bóc tách văn bản toàn diện (Extraction Pipeline)."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..classifier import Category, classify_endpoint
from ..config.faculties import FACULTY_TARGETS, faculty_storage_name, find_faculty, resolve_unit
from ..config.paths import DATA_DIR, PROCESSED_DATA_DIR
from ..config.session import create_http_session
from ..filters import is_valid_html_endpoint
from .doc_extractor import extract_document_text
from .html_cleaner import COURSE_CODE_PATTERN, EMAIL_PATTERN
from .html_extractor import extract_html_content, fetch_html_text
from .summary import generate_document_summary

logger = logging.getLogger(__name__)

# Ngưỡng ký tự tối đa của một trang stub/template fallback cần loại bỏ khi trùng lặp nội dung 100%
DUPLICATE_STUB_MAX_LENGTH: int = 500

__all__ = [
    "DUPLICATE_STUB_MAX_LENGTH",
    "canonicalize_url",
    "is_duplicate_stub",
    "generate_doc_id",
    "load_existing_processed_docs",
    "extract_files_for_domain",
    "extract_domain_documents",
    "extract_all_domains",
]


def canonicalize_url(url: str) -> str:
    """Chuẩn hóa URL về dạng canonical: bỏ www., bỏ trailing slash, chữ thường hostname."""
    if not url.startswith("http"):
        return url
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme}://{netloc}{path}{query}"


def is_duplicate_stub(content: str) -> bool:
    """Kiểm tra xem nội dung có phải là stub rỗng trùng lặp cần loại bỏ hay không."""
    emails = EMAIL_PATTERN.findall(content)
    course_codes = COURSE_CODE_PATTERN.findall(content)
    has_table = "|" in content and "---" in content

    if len(emails) >= 2 or len(course_codes) >= 2 or has_table:
        return False

    if len(content) < 150:
        return True

    if len(content) < DUPLICATE_STUB_MAX_LENGTH and len(emails) == 0 and len(course_codes) == 0:
        return True

    return False


def generate_doc_id(source: str) -> str:
    """Tạo mã định danh duy nhất (MD5 hash 16 ký tự) từ URL hoặc đường dẫn file."""
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:16]


def load_existing_processed_docs(output_path: Path) -> dict[str, dict[str, Any]]:
    """Đọc các bản ghi đã bóc tách từ trước để hỗ trợ xử lý tăng dần (incremental)."""
    existing_docs: dict[str, dict[str, Any]] = {}
    if not output_path.exists():
        return existing_docs

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Processed record must be an object")
                    source = record.get("source_url_or_path")
                    if source:
                        existing_docs[source] = record
                except (ValueError, TypeError) as error:
                    logger.warning("Invalid cached record in %s: %s", output_path, error)
                    continue
    except Exception as e:
        logger.warning("Không thể đọc cache file %s: %s", output_path, e)
    return existing_docs


def extract_files_for_domain(
    files_dir: Path,
    domain: str,
    records: dict[str, dict[str, Any]],
    seen_hashes: dict[str, str],
    force: bool,
    faculty_id: str | None = None,
) -> tuple[list[Path], int]:
    """Add attachment records; return discovered files and unextractable-file count."""
    doc_files: list[Path] = []
    if files_dir.exists():
        exts = (".pdf", ".docx", ".xlsx", ".doc", ".txt", ".csv")
        doc_files = [p for p in files_dir.rglob("*") if p.is_file() and p.name != "files_list.txt" and p.suffix.lower() in exts]

    failed_files = 0
    for file_path in doc_files:
        if not force and str(file_path) in records:
            continue
        extracted = extract_document_text(file_path)
        if not extracted:
            failed_files += 1
            logger.warning("No extractable content in %s", file_path)
            continue
        content_hash = hashlib.md5(extracted["text"].strip().encode("utf-8")).hexdigest()
        if content_hash in seen_hashes and is_duplicate_stub(extracted["text"]):
            continue
        seen_hashes[content_hash] = str(file_path)
        category = classify_endpoint(str(file_path), unquote(file_path.stem))
        records[str(file_path)] = {
            "id": generate_doc_id(str(file_path)),
            "source_type": "file",
            "source_url_or_path": str(file_path),
            "title": extracted["filename"],
            "domain": domain,
            "unit": faculty_id or resolve_unit(domain),
            "category": category.value,
            "published_date": extracted.get("published_date"),
            "summary": generate_document_summary(domain, extracted["filename"], extracted["text"]),
            "content": extracted["text"],
            "content_length": extracted["text_length"],
            "cloud_links": [],
        }

    return doc_files, failed_files


def extract_domain_documents(domain: str, max_workers: int = 10, force: bool = False) -> dict[str, Any]:
    """Extract a faculty ID/storage folder/host to its established JSONL file.

    Reuse cached records unless force is set, refresh source attribution, and
    remove duplicate stubs. Save atomically; report request failures separately
    from intentionally rejected thin content. Shared hosts use separate files.
    """
    target = find_faculty(domain)
    folder = faculty_storage_name(target["faculty_id"]) if target else domain
    faculty_id = target["faculty_id"] if target else resolve_unit(domain)
    domain = target["domain"] if target else domain
    domain_dir = Path(DATA_DIR) / folder
    endpoints_dir, files_dir = domain_dir / "endpoints", domain_dir / "files"
    output_file = Path(PROCESSED_DATA_DIR) / f"{folder}.jsonl"
    domain_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metadata_file = endpoints_dir / "endpoints_metadata.jsonl"
    ep_file = endpoints_dir / "endpoints.txt"
    ep_info: list[dict[str, Any]] = []

    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            ep_info = [json.loads(line) for line in f if line.strip()]
    elif ep_file.exists():
        with open(ep_file, "r", encoding="utf-8") as f:
            ep_info = [{"url": line.strip(), "domain": domain, "category": Category.KHAC.value, "title": ""} for line in f if line.strip()]

    if target and target["allowed_path_prefixes"]:
        ep_info = [item for item in ep_info if is_valid_html_endpoint(item["url"], domain, allowed_path_prefixes=target["allowed_path_prefixes"])]

    raw_existing = {} if force else load_existing_processed_docs(output_file)
    seen_hashes: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}

    if not force:
        for src, rec in raw_existing.items():
            if target and target["allowed_path_prefixes"] and rec.get("source_type") == "html" and not is_valid_html_endpoint(src, domain, allowed_path_prefixes=target["allowed_path_prefixes"]):
                continue
            content = rec.get("content", "")
            content_hash = hashlib.md5(content.strip().encode("utf-8")).hexdigest()
            if content_hash in seen_hashes and is_duplicate_stub(content):
                continue
            seen_hashes[content_hash] = src
            if rec.get("domain") != domain or not rec.get("summary"):
                rec["summary"] = generate_document_summary(domain, rec.get("title", ""), content)
            rec["unit"] = faculty_id
            rec["domain"] = domain
            records[src] = rec

    canonical_processed = {canonicalize_url(source) for source in records}
    urls_to_process: list[dict[str, Any]] = []
    scheduled_urls: set[str] = set()
    for item in ep_info:
        canonical_url = canonicalize_url(item["url"])
        if canonical_url in scheduled_urls or (not force and canonical_url in canonical_processed):
            continue
        scheduled_urls.add(canonical_url)
        urls_to_process.append(item)

    def extract_endpoint(item: dict[str, Any]) -> dict[str, Any] | None:
        """Fetch and extract one endpoint, attaching this source's stable identity."""
        source_url = item["url"]
        html_text = fetch_html_text(source_url, session=session, target=target)
        if not html_text:
            return None
        extracted = extract_html_content(url=source_url, html_text=html_text, category=item.get("category", Category.KHAC.value))
        if not extracted:
            return None
        title = extracted.get("title") or item.get("title", "") or source_url
        return {
            "id": generate_doc_id(source_url),
            "source_type": "html",
            "source_url_or_path": source_url,
            "title": title,
            "domain": domain,
            "unit": faculty_id,
            "category": item.get("category", Category.KHAC.value),
            "published_date": extracted.get("published_date"),
            "summary": generate_document_summary(domain, title, extracted["text"]),
            "content": extracted["text"],
            "content_length": extracted["text_length"],
            "cloud_links": extracted.get("cloud_links", []),
        }

    failed_requests = 0
    with create_http_session(pool_size=max_workers) as session:
        if urls_to_process:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(extract_endpoint, item): item["url"] for item in urls_to_process}
                for future in as_completed(futures):
                    try:
                        extracted_record = future.result()
                    except Exception as error:
                        failed_requests += 1
                        logger.warning("Extraction request failed: %s (%s)", futures[future], error)
                        logger.debug("Extraction request failure", exc_info=True)
                        continue
                    if extracted_record:
                        content_hash = hashlib.md5(extracted_record["content"].strip().encode("utf-8")).hexdigest()
                        if content_hash in seen_hashes and is_duplicate_stub(extracted_record["content"]):
                            continue
                        seen_hashes[content_hash] = extracted_record["source_url_or_path"]
                        records[extracted_record["source_url_or_path"]] = extracted_record

    doc_files, failed_files = extract_files_for_domain(files_dir, domain, records, seen_hashes, force, faculty_id)

    logger.info("[SAVING] %s | %s records", output_file, len(records))
    temp_out = output_file.with_suffix(".tmp")
    with open(temp_out, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    temp_out.replace(output_file)

    html_recs = [r for r in records.values() if r.get("source_type") == "html"]
    file_recs = [r for r in records.values() if r.get("source_type") == "file"]
    return {
        "domain": domain,
        "failed": bool(failed_requests or failed_files) or not (metadata_file.exists() or ep_file.exists() or files_dir.exists()),
        "failed_requests": failed_requests,
        "failed_files": failed_files,
        "total_endpoints": len(ep_info),
        "html_success": len(html_recs),
        "total_files": len(doc_files),
        "files_success": len(file_recs),
        "total_docs": len(records),
        "total_chars": sum(r.get("content_length", 0) for r in records.values()),
    }


def extract_all_domains(
    domains: list[str] | None = None, max_workers: int = 10, force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Extract each source independently, reporting incomplete work as failure."""
    started = time.monotonic()
    targets = domains if domains is not None else [faculty_storage_name(target["faculty_id"]) for target in FACULTY_TARGETS]
    results: dict[str, dict[str, Any]] = {}
    failures = 0
    logger.info("[START] Extraction started")
    for source in targets:
        logger.info("[EXTRACTING] %s", source)
        try:
            result = extract_domain_documents(source, max_workers=max_workers, force=force)
            results[source] = result
            failures += bool(result.get("failed"))
        except Exception as error:
            failures += 1
            logger.error("[ERROR] %s: %s", source, error)
            logger.debug("Extraction failure", exc_info=True)
    if failures:
        raise RuntimeError(f"Extraction incomplete: {failures} targets failed")
    logger.info("[DONE] Completed in %.1fs | %s records saved", time.monotonic() - started, sum(result["total_docs"] for result in results.values()))
    return results

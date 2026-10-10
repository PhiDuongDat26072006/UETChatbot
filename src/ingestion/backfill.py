"""src/ingestion/backfill.py - Recrawl và bảo tồn nguyên vẹn RawData HTML cho UETChatbot.

Thực hiện:
1. Đọc danh sách DataSource và RawData hiện có cho từng phân hệ (domain).
2. Fetch lại nội dung HTML nguyên bản bằng HTTP session tái sử dụng connection pool.
3. Bảo tồn toàn bộ response bytes nguyên bản (không qua BeautifulSoup/cleaning) vào:
   data/raw_data/<domain>/html/<raw_id>.html
4. Tính toán mã băm SHA-256 thực tế, kích thước file, mã trạng thái HTTP, URL chuyển hướng.
5. Cập nhật bản ghi RawData trong data/raw_data/<domain>/raw_records.jsonl với đường dẫn
   tương đối so với project root (POSIX format), status: 'preserved'.
6. Xử lý triệt để các trường hợp lỗi (404, 500, timeout, SSL, non-HTML):
   ghi nhận status: 'failed', mã lỗi HTTP, failure_reason mà không đánh dấu là preserved.
7. Chuẩn hóa mọi đường dẫn cục bộ tuyệt đối trong hệ thống sang project-relative POSIX paths.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.ingestion.config.faculties import FACULTY_TARGETS, faculty_storage_name, find_faculty
from src.ingestion.config.paths import (
    BASE_DIR,
    CHUNK_DATA_DIR,
    DATA_DIR,
    DATA_SOURCE_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    resolve_project_path,
    to_project_relative_path,
)
from src.ingestion.config.session import (
    REQUEST_TIMEOUT,
    create_http_session,
    get_source_response,
)
from src.ingestion.models import (
    compute_bytes_sha256,
    format_html_raw_record,
    generate_raw_id,
    generate_source_id,
)

logger = logging.getLogger(__name__)


def backfill_single_html(
    session: requests.Session,
    source_url: str,
    domain: str,
    prefixes: list[str] | tuple[str, ...],
    raw_id: str,
    source_id: str,
    html_dir: Path,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[bool, Dict[str, Any]]:
    """Fetch và bảo tồn 1 trang HTML nguyên bản dạng bytes.
    
    Trả về (thành công: bool, bản ghi RawData manifest).
    """
    html_target_path = html_dir / f"{raw_id}.html"

    try:
        response = get_source_response(
            session=session,
            url=source_url,
            domain=domain,
            allowed_path_prefixes=(),
            timeout=timeout,
        )
        status_code = response.status_code

        final_url = response.url

        if status_code != 200:
            return False, format_html_raw_record(
                source_id=source_id,
                source_uri=source_url,
                domain=domain,
                raw_id=raw_id,
                http_status=status_code,
                final_url=final_url,
                error_message=f"HTTP {status_code}",
            )

        content_type = response.headers.get("Content-Type", "")
        # Kiểm tra nội dung phải là HTML
        if content_type and not any(ct in content_type.lower() for ct in ("text/html", "application/xhtml", "text/plain")):
            return False, format_html_raw_record(
                source_id=source_id,
                source_uri=source_url,
                domain=domain,
                raw_id=raw_id,
                http_status=status_code,
                final_url=final_url,
                error_message=f"Non-HTML Content-Type: {content_type}",
            )

        # Lưu nguyên bản response bytes (không làm sạch, không parse HTML)
        raw_bytes = response.content
        if not raw_bytes:
            return False, format_html_raw_record(
                source_id=source_id,
                source_uri=source_url,
                domain=domain,
                raw_id=raw_id,
                http_status=status_code,
                final_url=final_url,
                error_message="Empty response body",
            )

        # Ghi nguyên tử ra đĩa
        temp_file = html_target_path.with_suffix(".tmp")
        temp_file.write_bytes(raw_bytes)
        temp_file.replace(html_target_path)

        rec = format_html_raw_record(
            source_id=source_id,
            source_uri=source_url,
            domain=domain,
            raw_id=raw_id,
            html_bytes=raw_bytes,
            http_status=status_code,
            final_url=final_url,
            created_at=datetime.now(timezone.utc),
        )
        return True, rec

    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {str(exc)[:150]}"
        return False, format_html_raw_record(
            source_id=source_id,
            source_uri=source_url,
            domain=domain,
            raw_id=raw_id,
            error_message=err_msg,
        )


def backfill_domain(
    domain_folder: str,
    raw_base_dir: Path | str = RAW_DATA_DIR,
    data_source_base_dir: Path | str = DATA_SOURCE_DIR,
    max_workers: int = 15,
    timeout: int = REQUEST_TIMEOUT,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Cào lại và bảo tồn toàn bộ HTML còn thiếu của một phân hệ (domain)."""
    raw_base = Path(raw_base_dir)
    domain_raw_dir = raw_base / domain_folder
    html_dir = domain_raw_dir / "html"
    manifest_path = domain_raw_dir / "raw_records.jsonl"
    source_file = Path(data_source_base_dir) / f"{domain_folder}.jsonl"

    html_dir.mkdir(parents=True, exist_ok=True)

    target_cfg = find_faculty(domain_folder)
    domain_name = target_cfg["domain"] if target_cfg else domain_folder
    prefixes = target_cfg["allowed_path_prefixes"] if target_cfg else ()

    # 1. Đọc các bản ghi raw_records.jsonl hiện tại
    existing_raw_by_id: Dict[str, Dict[str, Any]] = {}
    existing_raw_by_uri: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                # Chuẩn hóa đường dẫn nếu có
                if r.get("source_uri") and r["source_uri"].startswith("/"):
                    r["source_uri"] = to_project_relative_path(r["source_uri"])
                if r.get("raw_file_path") and r["raw_file_path"].startswith("/"):
                    r["raw_file_path"] = to_project_relative_path(r["raw_file_path"])
                existing_raw_by_id[r["id"]] = r
                existing_raw_by_uri[r["source_uri"]] = r
            except Exception:
                pass

    # 2. Đọc DataSource để đảm bảo không sót URL web nào
    web_sources: List[Dict[str, Any]] = []
    if source_file.exists():
        for line in source_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                if s.get("source_type") == "web" and s.get("uri", "").startswith("http"):
                    web_sources.append(s)
            except Exception:
                pass

    # 3. Lọc danh sách cần fetch
    to_fetch: List[Tuple[str, str, str]] = []  # (source_url, source_id, raw_id)
    already_preserved = 0

    for s in web_sources:
        url = s["uri"]
        sid = s["source_id"]
        # Tìm bản ghi raw hiện có
        raw_rec = existing_raw_by_uri.get(url)
        raw_id = raw_rec["id"] if raw_rec else generate_raw_id(sid, None)

        if not force and raw_rec and raw_rec.get("raw_metadata", {}).get("status") == "preserved":
            rel_path = raw_rec.get("raw_file_path")
            if rel_path:
                try:
                    full_p = resolve_project_path(rel_path)
                    if full_p.exists() and full_p.stat().st_size > 0:
                        already_preserved += 1
                        continue
                except Exception:
                    pass

        to_fetch.append((url, sid, raw_id))

    started = time.monotonic()
    logger.info("[START] HTML raw data backfill: %s (%d URLs to fetch, %d already preserved)", domain_folder, len(to_fetch), already_preserved)
    print(f"[START] HTML raw data backfill\n[FETCHING] {domain_folder} ({len(to_fetch)} URLs)")

    saved_count = 0
    failed_count = 0
    close_session_at_end = False

    if session is None:
        session = create_http_session(pool_size=max_workers)
        close_session_at_end = True

    try:
        if to_fetch:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        backfill_single_html,
                        session,
                        url,
                        domain_folder,
                        prefixes,
                        raw_id,
                        sid,
                        html_dir,
                        timeout,
                    ): (url, raw_id)
                    for url, sid, raw_id in to_fetch
                }

                for future in as_completed(futures):
                    url, raw_id = futures[future]
                    try:
                        success, rec = future.result()
                    except Exception as exc:
                        success = False
                        rec = format_html_raw_record(
                            source_id=existing_raw_by_id.get(raw_id, {}).get("source_id", ""),
                            source_uri=url,
                            domain=domain_folder,
                            raw_id=raw_id,
                            error_message=f"Executor exception: {str(exc)[:100]}",
                        )

                    existing_raw_by_id[raw_id] = rec
                    existing_raw_by_uri[url] = rec
                    if success:
                        saved_count += 1
                    else:
                        failed_count += 1

        # Ghi lại raw_records.jsonl nguyên tử
        temp_manifest = manifest_path.with_suffix(".tmp")
        with open(temp_manifest, "w", encoding="utf-8") as f:
            for r in existing_raw_by_id.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        temp_manifest.replace(manifest_path)

    finally:
        if close_session_at_end:
            session.close()

    elapsed = time.monotonic() - started
    print(f"[SAVED] {saved_count} HTML documents\n[FAILED] {failed_count} URLs\n[DONE] Completed in {elapsed:.1f}s\n[OUTPUT] {domain_raw_dir.relative_to(BASE_DIR)}/html/")
    logger.info("[DONE] %s: saved %d, failed %d in %.1fs", domain_folder, saved_count, failed_count, elapsed)

    return {
        "domain": domain_folder,
        "total_examined": len(web_sources),
        "already_preserved": already_preserved,
        "saved": saved_count,
        "failed": failed_count,
        "elapsed_seconds": elapsed,
        "output_dir": f"data/raw_data/{domain_folder}/html/",
    }


def normalize_all_persisted_paths() -> Dict[str, int]:
    """Chuẩn hóa mọi đường dẫn file tuyệt đối sang project-relative POSIX paths trong toàn bộ datasets."""
    stats = {"raw_records": 0, "processed_data": 0, "chunk_data": 0}

    # 1. Chuẩn hóa raw_records.jsonl
    for d in Path(RAW_DATA_DIR).iterdir():
        if not d.is_dir():
            continue
        rec_file = d / "raw_records.jsonl"
        if not rec_file.exists():
            continue
        modified = False
        recs = []
        for line in rec_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            src_uri = rec.get("source_uri")
            raw_path = rec.get("raw_file_path")
            if src_uri and src_uri.startswith("/") and not src_uri.startswith("http"):
                rec["source_uri"] = to_project_relative_path(src_uri)
                modified = True
                stats["raw_records"] += 1
            if raw_path and raw_path.startswith("/") and not raw_path.startswith("http"):
                rec["raw_file_path"] = to_project_relative_path(raw_path)
                modified = True
                stats["raw_records"] += 1
            recs.append(rec)
        if modified:
            tmp = rec_file.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
            tmp.replace(rec_file)

    # 2. Chuẩn hóa processed_data/*.jsonl
    for proc_file in Path(PROCESSED_DATA_DIR).glob("*.jsonl"):
        modified = False
        recs = []
        for line in proc_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get("source_url_or_path")
            if src and src.startswith("/") and not src.startswith("http"):
                rec["source_url_or_path"] = to_project_relative_path(src)
                modified = True
                stats["processed_data"] += 1
            recs.append(rec)
        if modified:
            tmp = proc_file.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
            tmp.replace(proc_file)

    # 3. Chuẩn hóa chunk_data/*.jsonl
    for chunk_file in Path(CHUNK_DATA_DIR).glob("*.jsonl"):
        modified = False
        recs = []
        for line in chunk_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("metadata") or {}
            src = meta.get("source")
            if src and src.startswith("/") and not src.startswith("http"):
                meta["source"] = to_project_relative_path(src)
                rec["metadata"] = meta
                modified = True
                stats["chunk_data"] += 1
            recs.append(rec)
        if modified:
            tmp = chunk_file.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
            tmp.replace(chunk_file)

    return stats


def verify_all_raw_data() -> Dict[str, Any]:
    """Kiểm tra toàn diện tính toàn vẹn dữ liệu cho toàn bộ file nhị phân và HTML đã bảo tồn."""
    results = {
        "html_examined": 0,
        "html_preserved_valid": 0,
        "html_failed": 0,
        "html_missing": 0,
        "binary_preserved_valid": 0,
        "binary_failed": 0,
        "path_errors": [],
        "checksum_errors": [],
    }

    raw_dir = Path(RAW_DATA_DIR)
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        rec_file = d / "raw_records.jsonl"
        if not rec_file.exists():
            continue

        for line in rec_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("raw_metadata") or {}
            status = meta.get("status")
            ctype = meta.get("content_type", "")
            is_html = "html" in ctype

            if is_html:
                results["html_examined"] += 1
                if status == "preserved":
                    rel_path = rec.get("raw_file_path")
                    if not rel_path:
                        results["path_errors"].append(f"Missing raw_file_path for {rec.get('id')}")
                        continue
                    if rel_path.startswith("/"):
                        results["path_errors"].append(f"Absolute path in {rec.get('id')}: {rel_path}")
                        continue
                    try:
                        p = resolve_project_path(rel_path)
                    except Exception as err:
                        results["path_errors"].append(f"Resolution error for {rel_path}: {err}")
                        continue
                    if not p.exists():
                        results["path_errors"].append(f"File missing on disk: {rel_path}")
                        continue
                    actual_bytes = p.read_bytes()
                    actual_sha = compute_bytes_sha256(actual_bytes)
                    if actual_sha != meta.get("sha256") or len(actual_bytes) != meta.get("file_size"):
                        results["checksum_errors"].append(f"Checksum mismatch for HTML {rel_path}")
                        continue
                    results["html_preserved_valid"] += 1
                elif status == "failed":
                    results["html_failed"] += 1
                else:
                    results["html_missing"] += 1
            else:
                if status == "preserved":
                    rel_path = rec.get("raw_file_path")
                    if not rel_path or rel_path.startswith("/"):
                        results["path_errors"].append(f"Invalid binary path {rel_path}")
                        results["binary_failed"] += 1
                        continue
                    try:
                        p = resolve_project_path(rel_path)
                    except Exception as err:
                        results["path_errors"].append(f"Resolution error for {rel_path}: {err}")
                        results["binary_failed"] += 1
                        continue
                    if not p.exists():
                        results["path_errors"].append(f"Binary missing: {rel_path}")
                        results["binary_failed"] += 1
                        continue
                    actual_bytes = p.read_bytes()
                    actual_sha = compute_bytes_sha256(actual_bytes)
                    if actual_sha != meta.get("sha256") or len(actual_bytes) != meta.get("file_size"):
                        results["checksum_errors"].append(f"Checksum mismatch for binary {rel_path}")
                        results["binary_failed"] += 1
                        continue
                    results["binary_preserved_valid"] += 1

    return results


def backfill_all_domains(
    domains: Optional[List[str]] = None,
    max_workers: int = 15,
    timeout: int = REQUEST_TIMEOUT,
    force: bool = False,
) -> Dict[str, Any]:
    """Thực thi backfill HTML cho toàn bộ các domain của UETChatbot."""
    raw_dir = Path(RAW_DATA_DIR)
    target_domains = domains or sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])

    overall = {
        "domains": {},
        "total_examined": 0,
        "total_preserved": 0,
        "total_failed": 0,
    }

    # Đầu tiên chuẩn hóa toàn bộ đường dẫn tuyệt đối cũ
    norm_stats = normalize_all_persisted_paths()
    logger.info("Đã chuẩn hóa đường dẫn tương đối: %s", norm_stats)

    with create_http_session(pool_size=max_workers) as shared_session:
        for domain in target_domains:
            res = backfill_domain(
                domain_folder=domain,
                max_workers=max_workers,
                timeout=timeout,
                force=force,
                session=shared_session,
            )
            overall["domains"][domain] = res
            overall["total_examined"] += res["total_examined"]
            overall["total_preserved"] += res["already_preserved"] + res["saved"]
            overall["total_failed"] += res["failed"]

    return overall


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    backfill_all_domains()

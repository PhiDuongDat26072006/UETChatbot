"""src/ingestion/migration.py - Di trú và đồng bộ dữ liệu Ingestion an toàn, không phá hủy.

Chuyển đổi dữ liệu từ cấu trúc cũ:
  data/raw_data/<domain>/endpoints/ (endpoints.txt, endpoints_metadata.jsonl)
  data/raw_data/<domain>/files/ (files_list.txt, *.pdf, *.docx...)
sang cấu trúc phân lớp vòng đời dữ liệu mới:
  data/data_source/<domain>/sources.jsonl (DataSource: định danh & xuất xứ)
  data/raw_data/<domain>/raw_records.jsonl (RawData: manifest liên kết payload gốc)
  data/raw_data/<domain>/html/ (Raw HTML payloads)
  data/raw_data/<domain>/files/ (Raw binary payloads - bảo tồn nguyên vẹn)
  data/processed_data/<domain>.jsonl (ProcessedData: bổ sung source_id & raw_data_id)
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import posixpath
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from src.ingestion.config.faculties import FACULTY_TARGETS, faculty_storage_name, find_faculty
from src.ingestion.config.paths import (
    BASE_DIR,
    DATA_DIR,
    DATA_SOURCE_DIR,
    PROCESSED_DATA_DIR,
    to_project_relative_path,
)
from src.ingestion.models import (
    compute_bytes_sha256,
    generate_raw_id,
    generate_source_id,
)


logger = logging.getLogger(__name__)


def guess_content_type(file_path: Path) -> str:
    """Xác định MIME type từ đuôi file."""
    ctype, _ = mimetypes.guess_type(str(file_path))
    if ctype:
        return ctype
    suffix = file_path.suffix.lower()
    custom_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".txt": "text/plain",
        ".html": "text/html",
    }
    return custom_types.get(suffix, "application/octet-stream")


def migrate_domain_discovery(
    domain_folder: str,
    raw_base_dir: Path,
    data_source_base_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Di trú endpoints và files_list sang DataSource (sources.jsonl).
    
    Trả về:
      - Danh sách bản ghi DataSource
      - Bảng ánh xạ: uri_or_path -> source_id
    """
    domain_raw_dir = raw_base_dir / domain_folder
    endpoints_dir = domain_raw_dir / "endpoints"
    files_dir = domain_raw_dir / "files"

    target = find_faculty(domain_folder)
    domain_name = target["domain"] if target else domain_folder
    unit_id = target["faculty_id"] if target else domain_folder

    sources: List[Dict[str, Any]] = []
    seen_uris: set[str] = set()
    uri_to_source_id: Dict[str, str] = {}

    # 1. Đọc endpoints metadata / endpoints.txt (Nguồn web)
    ep_meta_path = endpoints_dir / "endpoints_metadata.jsonl"
    ep_txt_path = endpoints_dir / "endpoints.txt"

    ep_items: List[Dict[str, Any]] = []
    if ep_meta_path.exists():
        with open(ep_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ep_items.append(json.loads(line))
                    except Exception:
                        pass
    elif ep_txt_path.exists():
        with open(ep_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    ep_items.append({"url": u, "domain": domain_name, "category": "KHAC", "title": ""})

    for item in ep_items:
        url = item.get("url") or ""
        if not url or url in seen_uris:
            continue
        seen_uris.add(url)
        sid = generate_source_id("web", url)
        uri_to_source_id[url] = sid

        source_meta = {
            "domain": item.get("domain") or domain_name,
            "unit": item.get("faculty_id") or unit_id,
            "category": item.get("category", "KHAC"),
            "title": item.get("title", ""),
            "discovered_at": datetime.now(timezone.utc).isoformat(),
        }
        sources.append({
            "source_id": sid,
            "source_type": "web",
            "uri": url,
            "metadata": source_meta,
        })

    # 2. Đọc files_list.txt (Nguồn file URL đính kèm)
    files_list_path = files_dir / "files_list.txt"
    if files_list_path.exists():
        with open(files_list_path, "r", encoding="utf-8") as f:
            for line in f:
                furl = line.strip()
                if not furl or furl in seen_uris:
                    continue
                seen_uris.add(furl)
                sid = generate_source_id("file", furl)
                uri_to_source_id[furl] = sid
                fname = posixpath.basename(urlparse(furl).path)
                sources.append({
                    "source_id": sid,
                    "source_type": "file",
                    "uri": furl,
                    "metadata": {
                        "domain": domain_name,
                        "unit": unit_id,
                        "file_name": unquote(fname),
                        "discovered_at": datetime.now(timezone.utc).isoformat(),
                    },
                })

    # 3. Ghi ra data_source/<domain>.jsonl (nguyên tử)
    data_source_base_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_source_base_dir / f"{domain_folder}.jsonl"
    temp_file = out_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        for s in sources:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    temp_file.replace(out_file)

    # Dọn dẹp thư mục con cũ data_source/<domain>/sources.jsonl nếu còn tồn tại
    old_subdir = data_source_base_dir / domain_folder
    old_subfile = old_subdir / "sources.jsonl"
    if old_subfile.exists():
        try:
            old_subfile.unlink()
            old_subdir.rmdir()
        except Exception:
            pass

    return sources, uri_to_source_id


def migrate_domain_raw_payloads(
    domain_folder: str,
    raw_base_dir: Path,
    uri_to_source_id: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Tạo raw_records.jsonl từ các file gốc bảo tồn và đánh dấu payload còn thiếu.
    
    Trả về:
      - Danh sách bản ghi RawData manifest
      - Bảng ánh xạ: uri_or_path -> raw_data_id
    """
    domain_raw_dir = raw_base_dir / domain_folder
    files_dir = domain_raw_dir / "files"
    html_dir = domain_raw_dir / "html"
    raw_records: List[Dict[str, Any]] = []
    uri_to_raw_id: Dict[str, str] = {}
    seen_raw_ids: set[str] = set()

    # 1. Quét các file nhị phân nguyên bản trong files/ (*.pdf, *.docx, *.doc, *.xlsx...)
    if files_dir.exists():
        for file_path in sorted(files_dir.iterdir()):
            if not file_path.is_file() or file_path.name == "files_list.txt":
                continue

            try:
                content_bytes = file_path.read_bytes()
                sha256_hash = compute_bytes_sha256(content_bytes)
                file_size = len(content_bytes)
            except Exception as e:
                logger.warning("Không thể đọc file nguyên bản %s: %s", file_path, e)
                continue

            # Ghép với source_id nếu có URL tương ứng, hoặc tạo source_id dựa trên tên file
            fname = file_path.name
            matched_sid = None
            matched_uri = to_project_relative_path(file_path)

            for uri, sid in uri_to_source_id.items():
                if fname in uri or unquote(fname) in uri:
                    matched_sid = sid
                    matched_uri = uri
                    break

            if not matched_sid:
                matched_sid = generate_source_id("file", matched_uri)
                uri_to_source_id[matched_uri] = matched_sid

            raw_id = generate_raw_id(matched_sid, sha256_hash)
            uri_to_raw_id[str(file_path)] = raw_id
            uri_to_raw_id[matched_uri] = raw_id

            rel_path = to_project_relative_path(file_path)

            rec = {
                "id": raw_id,
                "source_id": matched_sid,
                "source_uri": matched_uri,
                "raw_file_path": rel_path,
                "content": None,
                "title": file_path.name,
                "raw_metadata": {
                    "content_type": guess_content_type(file_path),
                    "sha256": sha256_hash,
                    "file_size": file_size,
                    "status": "preserved",
                },
                "created_at": datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
            if raw_id not in seen_raw_ids:
                seen_raw_ids.add(raw_id)
                raw_records.append(rec)

    # 2. Quét các file HTML nguyên bản nếu đã có trong html/
    if html_dir.exists():
        for html_path in sorted(html_dir.iterdir()):
            if not html_path.is_file() or html_path.name.endswith(".tmp"):
                continue
            try:
                h_bytes = html_path.read_bytes()
                sha256_hash = compute_bytes_sha256(h_bytes)
                file_size = len(h_bytes)
            except Exception:
                continue

            rel_path = to_project_relative_path(html_path)
            raw_id = html_path.stem

            # Tìm xem có source_id tương ứng hay không
            matched_sid = None
            matched_uri = None
            for u, s in uri_to_source_id.items():
                if generate_raw_id(s, None) == raw_id or generate_raw_id(s, sha256_hash) == raw_id:
                    matched_sid = s
                    matched_uri = u
                    break

            if not matched_sid:
                matched_sid = generate_source_id("web", raw_id)
                matched_uri = raw_id

            rec = {
                "id": raw_id,
                "source_id": matched_sid,
                "source_uri": matched_uri,
                "raw_file_path": rel_path,
                "content": None,
                "title": None,
                "raw_metadata": {
                    "content_type": "text/html",
                    "sha256": sha256_hash,
                    "file_size": file_size,
                    "status": "preserved",
                    "http_status": 200,
                },
                "created_at": datetime.fromtimestamp(html_path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
            uri_to_raw_id[matched_uri] = raw_id
            if raw_id not in seen_raw_ids:
                seen_raw_ids.add(raw_id)
                raw_records.append(rec)


    # 3. Với các URL web chưa có raw HTML payload, lưu bản ghi đánh dấu "missing"
    for uri, sid in uri_to_source_id.items():
        if uri.startswith("http") and uri not in uri_to_raw_id:
            raw_id = generate_raw_id(sid, None)
            uri_to_raw_id[uri] = raw_id
            rec = {
                "id": raw_id,
                "source_id": sid,
                "source_uri": uri,
                "raw_file_path": None,
                "content": None,
                "title": None,
                "raw_metadata": {
                    "content_type": "text/html",
                    "status": "missing",
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if raw_id not in seen_raw_ids:
                seen_raw_ids.add(raw_id)
                raw_records.append(rec)

    # 4. Ghi ra raw_records.jsonl (nguyên tử)
    raw_manifest_path = domain_raw_dir / "raw_records.jsonl"
    temp_file = raw_manifest_path.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    temp_file.replace(raw_manifest_path)

    return raw_records, uri_to_raw_id


def enrich_processed_data(
    processed_file: Path,
    uri_to_source_id: Dict[str, str],
    uri_to_raw_id: Dict[str, str],
) -> int:
    """Bổ sung source_id và raw_data_id vào file processed jsonl mà không làm hỏng schema cũ."""
    if not processed_file.exists():
        return 0

    updated_records: List[Dict[str, Any]] = []
    with open(processed_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                src = rec.get("source_url_or_path") or ""
                if src and src.startswith("/") and not src.startswith("http"):
                    src = to_project_relative_path(src)
                    rec["source_url_or_path"] = src

                # Ghép nối source_id & raw_data_id
                sid = uri_to_source_id.get(src)
                rid = uri_to_raw_id.get(src)

                # Nếu chưa tìm thấy chính xác, suy ra theo định danh chuẩn
                if not sid and src:
                    stype = rec.get("source_type") or ("web" if src.startswith("http") else "file")
                    sid = generate_source_id(stype, src)
                if not rid and sid:
                    rid = generate_raw_id(sid, None)

                if sid and not rec.get("source_id"):
                    rec["source_id"] = sid
                if rid and not rec.get("raw_data_id"):
                    rec["raw_data_id"] = rid

                updated_records.append(rec)

            except Exception as e:
                logger.warning("Bỏ qua bản ghi lỗi trong %s: %s", processed_file, e)

    temp_file = processed_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        for r in updated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    temp_file.replace(processed_file)

    return len(updated_records)


def migrate_all(
    base_data_dir: Optional[Path | str] = None,
    data_source_dir: Optional[Path | str] = None,
    processed_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Thực thi di trú toàn bộ kho dữ liệu UETChatbot theo kiến trúc mới."""
    raw_base = Path(base_data_dir or DATA_DIR)
    source_base = Path(data_source_dir or DATA_SOURCE_DIR)
    proc_base = Path(processed_dir or PROCESSED_DATA_DIR)

    source_base.mkdir(parents=True, exist_ok=True)
    raw_base.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "domains": {},
        "total_sources": 0,
        "total_raw_records": 0,
        "total_preserved_files": 0,
        "total_processed_enriched": 0,
    }

    domain_dirs = sorted([d for d in raw_base.iterdir() if d.is_dir()])
    for d_path in domain_dirs:
        domain_name = d_path.name
        # 1. Di trú DataSource
        sources, uri_to_sid = migrate_domain_discovery(domain_name, raw_base, source_base)
        # 2. Di trú RawData manifest & payload
        raw_recs, uri_to_rid = migrate_domain_raw_payloads(domain_name, raw_base, uri_to_sid)
        # 3. Bổ sung liên kết vào ProcessedData
        proc_file = proc_base / f"{domain_name}.jsonl"
        enriched_count = enrich_processed_data(proc_file, uri_to_sid, uri_to_rid)

        preserved_count = sum(1 for r in raw_recs if r.get("raw_metadata", {}).get("status") == "preserved")

        summary["domains"][domain_name] = {
            "sources": len(sources),
            "raw_records": len(raw_recs),
            "preserved_files": preserved_count,
            "processed_docs": enriched_count,
        }
        summary["total_sources"] += len(sources)
        summary["total_raw_records"] += len(raw_recs)
        summary["total_preserved_files"] += preserved_count
        summary["total_processed_enriched"] += enriched_count

        logger.info(
            "[MIGRATED] %s: %d sources | %d raw records (%d preserved files) | %d processed docs",
            domain_name, len(sources), len(raw_recs), preserved_count, enriched_count,
        )

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = migrate_all()
    print("\n--- KẾT QUẢ DI TRÚ DỮ LIỆU ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))

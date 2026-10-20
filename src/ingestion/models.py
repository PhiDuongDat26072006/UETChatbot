"""src/ingestion/models.py - Định nghĩa các quy chuẩn dữ liệu và định danh nguồn cho Ingestion.

Cung cấp các hàm tạo ID tất định (stable deterministic source_id / raw_data_id),
tính mã băm SHA-256 từ bytes nguyên bản, chuẩn hóa cấu trúc lưu trữ và chuyển đổi
giữa các bản ghi JSONL và mô hình src.base (DataSource, RawData, ProcessedData).
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import unquote, urlparse

from src.base import DataSource, ProcessedData, RawData
from src.ingestion.config.paths import (
    BASE_DIR,
    resolve_project_path,
    to_project_relative_path,
)


def generate_source_id(source_type: str, uri: str) -> str:
    """Tạo mã định danh nguồn tất định (stable ID) từ source_type và URI.
    
    Đảm bảo tính bất biến qua các lần cào lại (không bị UUID churn).
    """
    clean_type = (source_type or "web").strip().lower()
    clean_uri = (uri or "").strip()
    if clean_type == "web" and clean_uri.startswith("http"):
        parsed = urlparse(clean_uri)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        canonical_uri = f"{parsed.scheme}://{netloc}{path}{query}"
    else:
        canonical_uri = clean_uri.lower()

    key = f"{clean_type}:{canonical_uri}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def generate_raw_id(source_id: str, checksum: Optional[str] = None) -> str:
    """Tạo mã định danh dữ liệu thô (raw_data_id) gắn kết với DataSource."""
    key = f"raw:{source_id}:{checksum or 'missing'}"
    return f"raw_{hashlib.md5(key.encode('utf-8')).hexdigest()[:16]}"


def compute_bytes_sha256(data: Union[bytes, bytearray, memoryview, str]) -> str:
    """Tính mã băm SHA-256 từ bytes nguyên bản của payload."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def safe_payload_filename(uri: str, default_ext: str = ".html") -> str:
    """Tạo tên file lưu trữ payload an toàn, chống va chạm (collision-resistant)."""
    parsed = urlparse(uri)
    raw_name = posixpath.basename(unquote(parsed.path or ""))
    name, ext = posixpath.splitext(raw_name)
    if not ext:
        ext = default_ext

    # Lấy hash của URI làm tiền tố để chống trùng lặp tên file giữa các path khác nhau
    uri_hash = hashlib.md5(uri.encode("utf-8")).hexdigest()[:10]
    safe_name = re.sub(r"[^\w\.\-]", "_", name)[:60].strip("_")
    if not safe_name:
        safe_name = "payload"
    return f"{uri_hash}_{safe_name}{ext}"


def create_data_source(
    source_type: str,
    uri: str,
    metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
) -> DataSource:
    """Khởi tạo đối tượng DataSource với source_id ổn định và đường dẫn tương đối."""
    rel_uri = to_project_relative_path(uri) if not uri.startswith("http") else uri
    sid = source_id or generate_source_id(source_type, rel_uri)
    return DataSource(
        source_id=sid,
        source_type=source_type,
        uri=rel_uri,
        metadata=dict(metadata or {}),
    )


def create_raw_record(
    source_id: str,
    source_uri: str,
    raw_file_path: Optional[str] = None,
    content: Optional[str] = None,
    title: Optional[str] = None,
    raw_metadata: Optional[Dict[str, Any]] = None,
    raw_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> RawData:
    """Khởi tạo đối tượng RawData với đầy đủ liên kết tới DataSource và đường dẫn tương đối."""
    meta = dict(raw_metadata or {})
    checksum = meta.get("sha256")
    rid = raw_id or generate_raw_id(source_id, checksum)
    ts = created_at or datetime.now(timezone.utc)

    # Chuẩn hóa đường dẫn tương đối so với project root
    rel_source_uri = to_project_relative_path(source_uri) if not source_uri.startswith("http") else source_uri
    rel_file_path = to_project_relative_path(raw_file_path) if raw_file_path else None

    # Đảm bảo raw_metadata chứa source_id và raw_file_path để truy vết
    meta["source_id"] = source_id
    if rel_file_path:
        meta["raw_file_path"] = rel_file_path

    return RawData(
        id=rid,
        source_uri=rel_source_uri,
        content=content if content is not None else "",
        title=title,
        raw_metadata=meta,
        created_at=ts,
    )


def format_html_raw_record(
    source_id: str,
    source_uri: str,
    domain: str,
    raw_id: Optional[str] = None,
    html_bytes: Optional[bytes] = None,
    http_status: Optional[int] = None,
    final_url: Optional[str] = None,
    error_message: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Tạo dict bản ghi RawData manifest cho tài liệu HTML tuân thủ schema chuẩn.
    
    Khi html_bytes được cung cấp và thành công, bản ghi được đánh dấu status: 'preserved',
    lưu đường dẫn file tương đối data/raw_data/<domain>/html/<raw_id>.html.
    Khi thất bại, bản ghi được đánh dấu status: 'failed', ghi nhận lý do và mã lỗi HTTP.
    """
    rid = raw_id or generate_raw_id(source_id, None)
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    rel_html_path = f"data/raw_data/{domain}/html/{rid}.html"

    if html_bytes is not None:
        sha256_hash = compute_bytes_sha256(html_bytes)
        file_size = len(html_bytes)
        meta = {
            "content_type": "text/html",
            "sha256": sha256_hash,
            "file_size": file_size,
            "status": "preserved",
            "http_status": http_status or 200,
        }
        if final_url and final_url != source_uri:
            meta["final_url"] = final_url
        return {
            "id": rid,
            "source_id": source_id,
            "source_uri": source_uri,
            "raw_file_path": rel_html_path,
            "content": None,
            "title": None,
            "raw_metadata": meta,
            "created_at": ts,
        }
    else:
        meta = {
            "content_type": "text/html",
            "status": "failed",
            "failure_reason": error_message or "Unknown fetch error",
        }
        if http_status is not None:
            meta["http_status"] = http_status
        return {
            "id": rid,
            "source_id": source_id,
            "source_uri": source_uri,
            "raw_file_path": None,
            "content": None,
            "title": None,
            "raw_metadata": meta,
            "created_at": ts,
        }

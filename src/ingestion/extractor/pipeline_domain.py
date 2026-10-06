"""Bóc tách tài liệu HTML và files cho một domain cụ thể."""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..classifier import Category
from ..config import DATA_DIR, PROCESSED_DATA_DIR, resolve_unit
from .html_extractor import extract_html_content
from .pipeline_files import extract_files_for_domain
from .pipeline_helpers import (
    canonicalize_url,
    generate_doc_id,
    is_duplicate_stub,
    load_existing_processed_docs,
)
from .summary import generate_document_summary

logger = logging.getLogger(__name__)


def extract_domain_documents(domain: str, max_workers: int = 10, force: bool = False) -> dict[str, Any]:
    """Bóc tách toàn bộ tài liệu HTML và files tài liệu cho một domain."""
    domain_dir = Path(DATA_DIR) / domain
    endpoints_dir, files_dir = domain_dir / "endpoints", domain_dir / "files"
    output_file = Path(PROCESSED_DATA_DIR) / f"{domain}.jsonl"
    domain_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metadata_file, ep_file = endpoints_dir / "endpoints_metadata.jsonl", endpoints_dir / "endpoints.txt"
    ep_info: list[dict[str, Any]] = []
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            ep_info = [json.loads(line) for line in f if line.strip()]
    elif ep_file.exists():
        with open(ep_file, "r", encoding="utf-8") as f:
            ep_info = [{"url": line.strip(), "domain": domain, "category": Category.KHAC.value, "title": ""} for line in f if line.strip()]

    raw_existing = {} if force else load_existing_processed_docs(output_file)
    seen_hashes, records = {}, {}
    if not force:
        for src, rec in raw_existing.items():
            content = rec.get("content", "")
            chash = hashlib.md5(content.strip().encode("utf-8")).hexdigest()
            if chash in seen_hashes and is_duplicate_stub(content):
                continue
            seen_hashes[chash] = src
            rec.setdefault("summary", generate_document_summary(rec.get("domain", domain), rec.get("title", ""), content))
            rec.setdefault("unit", resolve_unit(rec.get("domain", domain)))
            records[src] = rec

    canonical_processed = {canonicalize_url(src): src for src in records}
    urls_to_proc = [it for it in ep_info if force or (it["url"] not in records and canonicalize_url(it["url"]) not in canonical_processed)]

    def _proc_url(item: dict[str, Any]) -> dict[str, Any] | None:
        u = item["url"]
        extracted = extract_html_content(url=u, timeout=10, category=item.get("category", Category.KHAC.value))
        if not extracted:
            return None
        t = extracted.get("title") or item.get("title", "") or u
        return {
            "id": generate_doc_id(u), "source_type": "html", "source_url_or_path": u,
            "title": t, "domain": domain, "unit": resolve_unit(domain), "category": item.get("category", Category.KHAC.value),
            "published_date": extracted.get("published_date"), "summary": generate_document_summary(domain, t, extracted["text"]),
            "content": extracted["text"], "content_length": extracted["text_length"], "cloud_links": extracted.get("cloud_links", []),
        }

    if urls_to_proc:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_proc_url, it): it["url"] for it in urls_to_proc}
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    chash = hashlib.md5(res["content"].strip().encode("utf-8")).hexdigest()
                    if chash in seen_hashes and is_duplicate_stub(res["content"]):
                        continue
                    seen_hashes[chash] = res["source_url_or_path"]
                    records[res["source_url_or_path"]] = res
                    canonical_processed[canonicalize_url(res["source_url_or_path"])] = res["source_url_or_path"]

    doc_files = extract_files_for_domain(files_dir, domain, records, seen_hashes, force)

    temp_out = output_file.with_suffix(".tmp")
    with open(temp_out, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    temp_out.replace(output_file)

    html_recs = [r for r in records.values() if r.get("source_type") == "html"]
    file_recs = [r for r in records.values() if r.get("source_type") == "file"]
    return {
        "domain": domain, "total_endpoints": len(ep_info), "html_success": len(html_recs),
        "total_files": len(doc_files), "files_success": len(file_recs), "total_docs": len(records),
        "total_chars": sum(r.get("content_length", 0) for r in records.values()),
    }

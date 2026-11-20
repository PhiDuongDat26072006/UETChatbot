"""Offline document scanning, safe upload, and local read helpers."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import unicodedata
from urllib.parse import quote, unquote, urlsplit
from src.base import DataSource
from .config.profiles import load_profiles
from .curation_yaml import safe_normalize_url, sync_overrides_yaml, sync_site_seeds_yaml
from .identity import source_id
from .storage import Storage, atomic_bytes, now, writer_lock

OFFLINE_SUFFIXES = {".pdf", ".html", ".htm"}


def sanitize_offline_filename(filename: str) -> str:
    name = Path(str(filename or "")).name.strip()
    if not name or name in {".", ".."} or "\x00" in name or "/" in name or "\\" in name:
        raise ValueError("Invalid filename")
    parsed = PurePosixPath(name)
    suffix = parsed.suffix.lower()
    if suffix not in OFFLINE_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {suffix or 'none'}; allowed: .pdf, .html, .htm"
        )
    ascii_stem = unicodedata.normalize(
        "NFKD", parsed.stem.replace("đ", "d").replace("Đ", "D")
    )
    ascii_stem = ascii_stem.encode("ascii", "ignore").decode("ascii")
    clean_stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", ascii_stem).strip("-_.")
    if not clean_stem:
        clean_stem = "document"
    return f"{clean_stem}{suffix}"


def count_offline_docs(data_dir) -> int:
    offline_dir = Path(data_dir) / "offline_docs"
    if not offline_dir.is_dir():
        return 0
    return sum(
        1
        for p in offline_dir.iterdir()
        if p.is_file() and p.suffix.lower() in OFFLINE_SUFFIXES
    )


def fetch_offline_or_http(fetcher, url: str, allowed_fn, data_dir) -> dict:
    parts = urlsplit(url)
    if parts.path.startswith("/offline/"):
        rel_name = unquote(parts.path.split("/offline/", 1)[1]).strip("/")
        offline_root = (Path(data_dir) / "offline_docs").resolve()
        local_path = (offline_root / rel_name).resolve()
        if (
            not rel_name
            or not local_path.is_relative_to(offline_root)
            or not local_path.is_file()
        ):
            raise FileNotFoundError(f"Offline document not found: {rel_name}")
        body = local_path.read_bytes()
        is_pdf = local_path.suffix.lower() == ".pdf" or body.startswith(b"%PDF-")
        return {
            "requested_url": url,
            "final_url": url,
            "status_code": 200,
            "headers": {
                "content-type": (
                    "application/pdf" if is_pdf else "text/html; charset=utf-8"
                )
            },
            "body": body,
            "encoding": "binary" if is_pdf else "utf-8",
        }
    return fetcher.fetch(url, allowed_fn)


def scan_offline_docs(data_dir, site="OTHER", profile_dir=None) -> dict:
    """Discover new .pdf/.html files in data/offline_docs into pending without overwriting."""
    site = site.upper()
    profiles = load_profiles(profile_dir)
    if site not in profiles:
        raise ValueError(f"Unknown site: {site}")
    profile = profiles[site]
    offline_dir = Path(data_dir) / "offline_docs"
    if not offline_dir.is_dir():
        return {"scanned": 0, "new_pending": 0, "existing": 0}
    files = sorted(
        p
        for p in offline_dir.iterdir()
        if p.is_file() and p.suffix.lower() in OFFLINE_SUFFIXES
    )
    storage = Storage(data_dir)
    scanned = len(files)
    new_pending = existing = 0
    with writer_lock(storage.root):
        sources = storage.records("sources", site)
        for path in files:
            uri = safe_normalize_url(
                f"https://uet.vnu.edu.vn/offline/{quote(path.name, safe='._-~')}",
                profile,
            )
            sid = source_id(uri)
            if sid in sources:
                existing += 1
                continue
            kind = "pdf" if path.suffix.lower() == ".pdf" else "html"
            sources[sid] = DataSource(
                source_id=sid,
                source_type="web",
                uri=uri,
                metadata={
                    "site": site,
                    "faculty": site,
                    "resource_type": kind,
                    "title": path.stem.replace("-", " ").replace("_", " ").strip(),
                    "discovered_from": None,
                    "crawl_depth": 0,
                    "decision": "pending",
                    "decision_reason": "offline_discovered",
                },
            ).model_dump(mode="json")
            new_pending += 1
        if new_pending:
            storage.flush("sources", site)
    return {"scanned": scanned, "new_pending": new_pending, "existing": existing}


def upload_offline_doc(
    data_dir,
    filename,
    content_bytes,
    decision="pending",
    site="OTHER",
    profile_dir=None,
    overrides_path=None,
) -> dict:
    """Save an uploaded PDF/HTML atomically into offline_docs and register its DataSource."""
    site = site.upper()
    if decision not in {"pending", "accepted"}:
        raise ValueError("Offline upload decision must be 'pending' or 'accepted'")
    if not isinstance(content_bytes, (bytes, bytearray)) or not content_bytes:
        raise ValueError("Uploaded file content must be non-empty bytes")
    profiles = load_profiles(profile_dir)
    if site not in profiles:
        raise ValueError(f"Unknown site: {site}")
    profile = profiles[site]
    safe_name = sanitize_offline_filename(filename)
    offline_path = Path(data_dir) / "offline_docs" / safe_name
    atomic_bytes(offline_path, bytes(content_bytes))

    uri = safe_normalize_url(f"https://uet.vnu.edu.vn/offline/{safe_name}", profile)
    sid = source_id(uri)
    kind = "pdf" if PurePosixPath(safe_name).suffix.lower() == ".pdf" else "html"
    storage = Storage(data_dir)
    with writer_lock(storage.root):
        sources = storage.records("sources", site)
        meta = dict(sources.get(sid, {}).get("metadata", {}))
        meta.update(
            {
                "site": site,
                "faculty": site,
                "resource_type": kind,
                "title": meta.get("title")
                or PurePosixPath(safe_name)
                .stem.replace("-", " ")
                .replace("_", " ")
                .strip(),
                "discovered_from": None,
                "crawl_depth": 0,
                "decision": decision,
                "decision_reason": (
                    "manual_curation" if decision == "accepted" else "offline_discovered"
                ),
            }
        )
        if decision == "accepted":
            meta["reviewed_at"] = now()
        sources[sid] = DataSource(
            source_id=sid, source_type="web", uri=uri, metadata=meta
        ).model_dump(mode="json")
        storage.flush("sources", site)
        if decision == "accepted":
            sync_overrides_yaml(
                site,
                profile,
                updates={uri: {"decision": "accepted", "note": ""}},
                overrides_path=overrides_path,
            )
            sync_site_seeds_yaml(
                site, profile, add_urls=[uri], profile_dir=profile_dir
            )
    return {
        "source_id": sid,
        "uri": uri,
        "filename": safe_name,
        "site": site,
        "decision": decision,
        "resource_type": kind,
        "bytes": len(content_bytes),
    }

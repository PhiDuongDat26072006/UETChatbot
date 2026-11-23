"""FastAPI application for UET ingestion source curation and incremental runs."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path, PurePosixPath
import threading
from typing import Any, Literal, Optional
from urllib.parse import quote, unquote, urlsplit
import webbrowser

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests
import uvicorn

from ..config.profiles import load_profiles
from ..config.settings import DEFAULT_DATA_DIR
from ..crawler.parser import metadata as extract_page_metadata
from ..crawler.policy import resource_type
from ..curation import (
    add_manual_urls,
    deduplicate_all_sites,
    deduplicate_site,
    get_all_sites_summary,
    get_processed_document,
    list_site_sources,
    remove_sources,
    scan_offline_docs,
    set_sources_decision,
    upload_offline_doc,
)
from ..curation_offline import OFFLINE_SUFFIXES, fetch_offline_or_http
from ..curation_yaml import safe_normalize_url
from ..downloader.http import HTMLFetcher
from ..extractor.document.docx_native import extract_docx
from ..extractor.html.extractor import extract_html
from ..identity import canonicalize_url, source_id
from ..storage import Storage
from .jobs import IngestJobManager, JobConflictError

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
PREVIEW_EXCERPT_CHARS = 800


class DecisionRequest(BaseModel):
    source_ids: list[str]
    decision: Literal["accepted", "listing", "pending", "rejected"]
    note: str = ""


class AddUrlsRequest(BaseModel):
    urls: list[str]
    decision: Literal["accepted", "listing", "pending", "rejected"] = "accepted"
    note: str = ""


class RemoveSourcesRequest(BaseModel):
    source_ids: list[str]
    purge_data: bool = False


class OfflineUploadRequest(BaseModel):
    filename: str
    content_base64: str
    decision: Literal["pending", "accepted"] = "pending"


class IncrementalIngestRequest(BaseModel):
    sites: Optional[list[str]] = Field(default=None)


class DedupAllRequest(BaseModel):
    sites: Optional[list[str]] = Field(default=None)
    cross_site: bool = True


def _curation_http_error(exc: Exception, *, unknown_site_is_404: bool = True) -> HTTPException:
    msg = str(exc) or type(exc).__name__
    if isinstance(exc, RuntimeError) and "writer lock" in msg.lower():
        return HTTPException(status_code=409, detail=msg)
    if isinstance(exc, ValueError):
        status = 404 if unknown_site_is_404 and msg.startswith("Unknown site:") else 400
        return HTTPException(status_code=status, detail=msg)
    return HTTPException(status_code=500, detail=msg)


def _decode_base64_upload(content_base64: str) -> bytes:
    raw_b64 = (content_base64 or "").strip()
    if raw_b64.startswith("data:") and "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    raw_b64 = "".join(raw_b64.split())
    if not raw_b64:
        raise HTTPException(status_code=400, detail="Empty base64 file content")
    if len(raw_b64) > (MAX_UPLOAD_BYTES * 4 // 3) + 4096:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds 30MB limit")
    try:
        content_bytes = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds 30MB limit")
    return content_bytes


def _preview_error(uri: str, status_code: int | None, error: str, final_url: str | None = None) -> dict[str, Any]:
    return {
        "uri": uri,
        "final_url": final_url or uri,
        "status_code": status_code,
        "is_pdf": False,
        "content_length": 0,
        "title": None,
        "published_at": None,
        "excerpt": "",
        "pdf_links": [],
        "error": error,
    }


def _collect_attachment_links(soup: BeautifulSoup, final_url: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    links: list[dict[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        if not isinstance(href, str) or not href.strip():
            continue
        try:
            target = canonicalize_url(href, final_url)
        except ValueError:
            continue
        if target in seen or resource_type(target) not in {"pdf", "office"}:
            continue
        seen.add(target)
        fallback_name = PurePosixPath(unquote(urlsplit(target).path)).name
        links.append({"url": target, "link_text": anchor.get_text(" ", strip=True) or fallback_name})
    return links


def _inspect_live_source(data_dir: Path, profile_dir: Path | None, site: str, source_id: str) -> dict[str, Any]:
    """Fetch a source live (10s timeout) without mutating raw_data or sources."""
    norm_site = site.upper()
    profiles = load_profiles(profile_dir)
    if norm_site not in profiles:
        raise HTTPException(status_code=404, detail=f"Unknown site: {norm_site}")
    profile = profiles[norm_site]
    source = Storage(data_dir).records("sources", norm_site).get(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    uri = source["uri"]
    allowed_hosts = {h.lower() for h in profile.get("scope", {}).get("hosts", [])}
    if host := urlsplit(uri).netloc.lower():
        allowed_hosts.add(host)

    def allowed(target: str) -> bool:
        parsed = urlsplit(target)
        return parsed.scheme.lower() in {"http", "https"} and parsed.netloc.lower() in allowed_hosts

    fetcher = HTMLFetcher(timeout=10, retries=1, delay=0.0)
    try:
        page = fetch_offline_or_http(fetcher, uri, allowed, data_dir)
    except requests.HTTPError as exc:
        resp = exc.response
        code = resp.status_code if resp is not None else 502
        final = str(resp.url) if resp is not None and getattr(resp, "url", None) else uri
        return _preview_error(uri, code, str(exc), final)
    except FileNotFoundError as exc:
        return _preview_error(uri, 404, str(exc), uri)
    except requests.Timeout as exc:
        return _preview_error(uri, 504, str(exc), uri)
    except (requests.RequestException, ValueError, OSError) as exc:
        return _preview_error(uri, None, str(exc), uri)
    finally:
        fetcher.close()

    body = page.get("body") or b""
    final_url = page.get("final_url") or uri
    status_code = page.get("status_code", 200)
    mime = (page.get("headers") or {}).get("content-type", "").split(";")[0].strip().lower()
    is_docx = (
        mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or final_url.lower().endswith(".docx")
    )
    if is_docx:
        try:
            extracted_docx = extract_docx(body)
        except Exception:
            extracted_docx = {}
        clean_docx = (extracted_docx.get("content") or "").strip()
        excerpt = clean_docx[:PREVIEW_EXCERPT_CHARS].rstrip()
        if len(clean_docx) > PREVIEW_EXCERPT_CHARS:
            excerpt += "..."
        stem = PurePosixPath(unquote(urlsplit(final_url).path)).stem
        title = (
            extracted_docx.get("title")
            or source.get("metadata", {}).get("title")
            or stem.replace("-", " ").replace("_", " ").strip()
            or "DOCX Document"
        )
        return {
            "uri": uri, "final_url": final_url, "status_code": status_code,
            "is_pdf": False, "is_docx": True, "content_length": len(body), "title": title,
            "published_at": None, "excerpt": excerpt, "pdf_links": [], "error": None,
        }

    is_pdf = (
        page.get("encoding") == "binary"
        or mime == "application/pdf"
        or final_url.lower().endswith(".pdf")
        or body.startswith(b"%PDF-")
    )
    if is_pdf:
        stem = PurePosixPath(unquote(urlsplit(final_url).path)).stem
        title = (
            source.get("metadata", {}).get("title")
            or source.get("metadata", {}).get("link_text")
            or stem.replace("-", " ").replace("_", " ").strip()
            or "PDF Document"
        )
        return {
            "uri": uri, "final_url": final_url, "status_code": status_code,
            "is_pdf": True, "is_docx": False, "content_length": len(body), "title": title,
            "published_at": None, "excerpt": "", "pdf_links": [], "error": None,
        }

    html_text = body.decode(page.get("encoding") or "utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "html.parser")
    meta = extract_page_metadata(soup, profile)
    try:
        extracted = extract_html(html_text, final_url, profile) or {}
    except Exception:
        extracted = {}
    title = (
        extracted.get("title") or meta.get("title")
        or (soup.title.get_text(" ", strip=True) if soup.title else "")
        or source.get("metadata", {}).get("title") or ""
    )
    published_at = extracted.get("metadata", {}).get("published_at") or meta.get("published_at")
    clean_text = (extracted.get("content") or "").strip() or soup.get_text(" ", strip=True)
    excerpt = clean_text[:PREVIEW_EXCERPT_CHARS].rstrip()
    if len(clean_text) > PREVIEW_EXCERPT_CHARS:
        excerpt += "..."
    return {
        "uri": uri, "final_url": final_url, "status_code": status_code,
        "is_pdf": False, "content_length": len(body), "title": title,
        "published_at": published_at, "excerpt": excerpt,
        "pdf_links": _collect_attachment_links(soup, final_url), "error": None,
    }


def _auto_scan_offline_if_needed(data_dir: Path, profile_dir: Path | None) -> None:
    offline_dir = data_dir / "offline_docs"
    if not offline_dir.is_dir():
        return
    files = [p for p in offline_dir.iterdir() if p.is_file() and p.suffix.lower() in OFFLINE_SUFFIXES]
    if not files:
        return
    other_prof = load_profiles(profile_dir).get("OTHER")
    if not other_prof:
        return
    existing = Storage(data_dir).records("sources", "OTHER")
    if any(
        source_id(safe_normalize_url(f"https://uet.vnu.edu.vn/offline/{quote(p.name, safe='._-~')}", other_prof)) not in existing
        for p in files
    ):
        try:
            scan_offline_docs(data_dir, profile_dir=profile_dir)
        except RuntimeError:
            pass


def create_review_app(
    data_dir: str | Path | None = None,
    profile_dir: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI app for local source curation and incremental ingestion."""
    resolved_data = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    resolved_profiles = Path(profile_dir) if profile_dir is not None else None
    resolved_overrides = Path(overrides_path) if overrides_path is not None else None
    _auto_scan_offline_if_needed(resolved_data, resolved_profiles)
    job_manager = IngestJobManager(resolved_data, resolved_profiles, resolved_overrides)

    app = FastAPI(title="UET Ingestion Review UI", version="1.0.0")
    app.state.job_manager = job_manager
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")

    @app.middleware("http")
    async def no_cache_static_middleware(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def serve_index():
        index_file = STATIC_DIR / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_file)

    @app.get("/api/sites")
    def api_sites():
        _auto_scan_offline_if_needed(resolved_data, resolved_profiles)
        return get_all_sites_summary(resolved_data, resolved_profiles)

    @app.get("/api/sites/{site}/sources")
    def api_site_sources(
        site: str,
        status: Optional[str] = None,
        resource_type: Optional[str] = None,
        search: Optional[str] = None,
        discovered_from: Optional[str] = None,
        sort: str = "priority",
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
    ):
        try:
            return list_site_sources(
                resolved_data, site, status=status, resource_type=resource_type,
                search=search, discovered_from=discovered_from, sort=sort,
                page=page, page_size=page_size, profile_dir=resolved_profiles,
            )
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc) from exc

    @app.post("/api/sites/{site}/decision")
    def api_site_decision(site: str, body: DecisionRequest):
        try:
            return set_sources_decision(
                resolved_data, site, body.source_ids, body.decision, note=body.note,
                profile_dir=resolved_profiles, overrides_path=resolved_overrides,
            )
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc) from exc

    @app.post("/api/sites/{site}/add")
    def api_site_add(site: str, body: AddUrlsRequest):
        try:
            return add_manual_urls(
                resolved_data, site, body.urls, decision=body.decision, note=body.note,
                profile_dir=resolved_profiles, overrides_path=resolved_overrides,
            )
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc) from exc

    @app.post("/api/sites/{site}/remove")
    def api_site_remove(site: str, body: RemoveSourcesRequest):
        try:
            return remove_sources(
                resolved_data, site, body.source_ids, purge_data=body.purge_data,
                profile_dir=resolved_profiles, overrides_path=resolved_overrides,
            )
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc) from exc

    @app.post("/api/sites/{site}/dedup")
    def api_site_dedup(site: str):
        try:
            return deduplicate_site(
                resolved_data,
                site,
                profile_dir=resolved_profiles,
                overrides_path=resolved_overrides,
            )
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc) from exc

    @app.post("/api/dedup")
    def api_dedup_all(body: Optional[DedupAllRequest] = None):
        req = body or DedupAllRequest()
        try:
            return deduplicate_all_sites(
                resolved_data,
                sites=req.sites,
                cross_site=req.cross_site,
                profile_dir=resolved_profiles,
                overrides_path=resolved_overrides,
            )
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc, unknown_site_is_404=False) from exc

    @app.get("/api/sites/{site}/preview/{source_id}")
    def api_site_preview(site: str, source_id: str):
        return _inspect_live_source(resolved_data, resolved_profiles, site, source_id)

    @app.get("/api/sites/{site}/processed/{source_id}")
    def api_site_processed(site: str, source_id: str):
        norm_site = site.upper()
        if norm_site not in load_profiles(resolved_profiles):
            raise HTTPException(status_code=404, detail=f"Unknown site: {norm_site}")
        doc = get_processed_document(resolved_data, norm_site, source_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Processed document not found")
        return doc

    @app.post("/api/offline/scan")
    def api_offline_scan():
        try:
            return scan_offline_docs(resolved_data, profile_dir=resolved_profiles)
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc, unknown_site_is_404=False) from exc

    @app.post("/api/offline/upload")
    def api_offline_upload(body: OfflineUploadRequest):
        content_bytes = _decode_base64_upload(body.content_base64)
        try:
            return upload_offline_doc(
                resolved_data, body.filename, content_bytes, decision=body.decision,
                profile_dir=resolved_profiles, overrides_path=resolved_overrides,
            )
        except (ValueError, RuntimeError) as exc:
            raise _curation_http_error(exc, unknown_site_is_404=False) from exc

    @app.post("/api/ingest/incremental", status_code=202)
    def api_ingest_incremental(body: Optional[IncrementalIngestRequest] = None):
        sites = [s.upper() for s in body.sites if s.strip()] if body and body.sites else None
        if sites:
            unknown = sorted(set(sites) - set(load_profiles(resolved_profiles)))
            if unknown:
                raise HTTPException(status_code=400, detail=f"Unknown sites: {unknown}")
        try:
            job = job_manager.start_job(sites=sites or None)
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(status_code=202, content={"job_id": job["job_id"], "status": job["status"]})

    @app.post("/api/sites/{site}/reingest/{source_id}", status_code=202)
    def api_site_reingest(site: str, source_id: str):
        norm_site = site.upper()
        if norm_site not in load_profiles(resolved_profiles):
            raise HTTPException(status_code=404, detail=f"Unknown site: {norm_site}")
        rec = Storage(resolved_data).records("sources", norm_site).get(source_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
        dec = rec.get("metadata", {}).get("decision")
        if dec not in {"accepted", "listing"}:
            raise HTTPException(
                status_code=400,
                detail=f"Source must be accepted or listing to re-ingest (got {dec})",
            )
        try:
            job = job_manager.start_job(sites=[norm_site], force_source_id=source_id)
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(status_code=202, content={"job_id": job["job_id"], "status": job["status"]})

    @app.get("/api/ingest/status")
    def api_ingest_status():
        return job_manager.get_status()

    @app.get("/api/ingest/jobs/{job_id}")
    def api_ingest_job(job_id: str):
        job = job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job

    return app


def run_review_web(
    host: str = "127.0.0.1",
    port: int = 8001,
    data_dir: str | Path | None = None,
    open_browser: bool = True,
    profile_dir: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> None:
    """Launch the local Uvicorn server for the ingestion curation web UI."""
    app = create_review_app(data_dir=data_dir, profile_dir=profile_dir, overrides_path=overrides_path)
    if open_browser:
        browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        url = f"http://{browser_host}:{port}"

        def _open() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        timer = threading.Timer(0.8, _open)
        timer.daemon = True
        timer.start()

    uvicorn.run(app, host=host, port=port)

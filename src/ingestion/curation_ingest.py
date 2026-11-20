"""Incremental ingestion runner for curated accepted and listing sources."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import PurePosixPath
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit
from uuid import uuid4
from src.base import DataSource
from .config.profiles import load_profiles
from .config.settings import load_runtime
from .crawler.parser import parse_page
from .crawler.policy import discovery_verdict, in_discovery_scope, resource_type
from .curation_offline import fetch_offline_or_http
from .curation_yaml import ordered_site_names
from .identity import source_id
from .overrides import attach as attach_overrides, lookup
from .storage import Storage, now, write_json, writer_lock
from .streaming import StreamingExtractor
from .utils import completion_map


def _emit_progress(on_progress, progress):
    if on_progress is not None:
        on_progress(dict(progress))


def _record_discovered_links(
    links, parent_url, depth, site, profile, storage, seen_ids, site_stats, progress
):
    existing = storage.records("sources", site)
    for link in links:
        target = link["url"]
        kind = resource_type(target)
        if kind == "asset":
            continue
        child_sid = source_id(target)
        if child_sid in seen_ids or child_sid in existing:
            prev = existing.get(child_sid, {})
            if prev.get("metadata", {}).get("decision") == "rejected":
                site_stats["rejected_preserved"] += 1
                progress["rejected_preserved"] += 1
            seen_ids.add(child_sid)
            continue
        seen_ids.add(child_sid)
        verdict, why = discovery_verdict(target, profile)
        if verdict == "pending":
            site_stats["new_pending"] += 1
            progress["new_pending"] += 1
        storage.save_source(
            DataSource(
                source_id=child_sid,
                source_type="web",
                uri=target,
                metadata={
                    "site": site,
                    "faculty": site,
                    "resource_type": kind,
                    "discovered_from": parent_url,
                    "link_text": link.get("link_text", ""),
                    "crawl_depth": depth + 1,
                    "decision": verdict,
                    "decision_reason": why,
                },
            )
        )


def _prepare_site_jobs(storage, site, sources, force_source_id, site_stats, progress):
    processed = storage.records("processed", site)
    states = storage.records("state", site)
    latest_raw = {}
    for raw_rec in storage.records("raw", site).values():
        sid = raw_rec["metadata"]["source_id"]
        if sid not in latest_raw or raw_rec["created_at"] > latest_raw[sid]["created_at"]:
            latest_raw[sid] = raw_rec

    download_jobs, extract_only = [], []
    for sid, rec in sorted(sources.items()):
        if force_source_id is not None and sid != force_source_id:
            continue
        meta = rec.get("metadata", {})
        dec = meta.get("decision")
        if dec not in {"accepted", "listing"}:
            continue
        forced = force_source_id == sid
        if dec == "listing":
            crawl_state = states.get(f"crawl:{sid}", {})
            if forced or (
                meta.get("final_url") is None and crawl_state.get("status") != "completed"
            ):
                download_jobs.append((rec, "listing"))
            else:
                site_stats["skipped"] += 1
                progress["skipped"] += 1
            continue

        raw_rec = latest_raw.get(sid)
        raw_valid = False
        if raw_rec is not None:
            try:
                storage.load_raw(raw_rec)
                raw_valid = True
            except (OSError, ValueError):
                raw_valid = False
        raw_state = states.get(f"raw:{sid}", {})
        proc_state = states.get(f"processed:{sid}", {})
        proc_rec = processed.get(sid)
        proc_valid = (
            raw_valid
            and proc_rec is not None
            and proc_rec.get("raw_data_id") == raw_rec["id"]
            and raw_state.get("status") != "failed"
            and proc_state.get("status") != "failed"
        )
        if raw_valid and proc_valid and not forced:
            site_stats["skipped"] += 1
            progress["skipped"] += 1
        elif raw_valid and raw_state.get("status") != "failed" and not forced:
            extract_only.append(raw_rec)
        else:
            download_jobs.append((rec, "accepted"))
    return download_jobs, extract_only


def _apply_download_result(
    rec, role, page, err, site, profile, storage, seen_ids, site_stats, progress, errors, stream
):
    source = DataSource.model_validate(rec)
    sid = source.source_id
    if err:
        stage_name = "raw" if role == "accepted" else "crawl"
        storage.state(site, sid, stage_name, status="failed", error=err)
        site_stats["errors"] += 1
        progress["errors"] += 1
        errors.append({"site": site, "source_id": sid, "stage": stage_name, "error": err})
        return

    is_pdf = (
        source.metadata.get("resource_type") == "pdf"
        or page.get("encoding") == "binary"
        or page["final_url"].lower().endswith(".pdf")
    )
    if is_pdf:
        clean_name = (
            PurePosixPath(unquote(urlsplit(page["final_url"]).path))
            .stem.replace("-", " ")
            .replace(".", " ")
            .strip()
        )
        parsed_meta = {
            "title": clean_name or source.metadata.get("title"),
            "resource_type": "pdf",
        }
        links = []
    else:
        html = page["body"].decode(page["encoding"], errors="replace")
        parsed_meta, links = parse_page(html, page["final_url"], profile)

    source.metadata.update({k: v for k, v in parsed_meta.items() if v is not None})
    source.metadata["final_url"] = page["final_url"]
    source.metadata["decision"] = role
    source = storage.resolve_redirect(source, page, profile)
    redirect_existing = storage.records("sources", site).get(source.source_id)
    if (
        redirect_existing
        and redirect_existing.get("metadata", {}).get("decision") == "rejected"
    ):
        site_stats["rejected_preserved"] += 1
        progress["rejected_preserved"] += 1
        return
    storage.save_source(source)
    seen_ids.add(source.source_id)
    _record_discovered_links(
        links,
        source.uri,
        source.metadata.get("crawl_depth", 0),
        site,
        profile,
        storage,
        seen_ids,
        site_stats,
        progress,
    )
    if role == "listing":
        storage.state(site, source.source_id, "crawl", status="completed", url=source.uri)
        return
    raw_rec = storage.save_raw(source, page)
    site_stats["downloaded"] += 1
    progress["downloaded"] += 1
    old_proc = storage.records("processed", site).get(source.source_id)
    if old_proc and old_proc.get("raw_data_id") != raw_rec["id"]:
        storage.remove_processed(site, source.source_id)
    if stream is not None:
        stream.stage(raw_rec)


def _ingest_site(
    site, profile, storage, fetcher, runtime_cfg, force_source_id, progress, errors, on_progress
):
    site_stats = {
        "skipped": 0,
        "downloaded": 0,
        "processed_saved": 0,
        "new_pending": 0,
        "rejected_preserved": 0,
        "errors": 0,
    }
    sources = storage.records("sources", site)
    if force_source_id is not None:
        if force_source_id not in sources:
            return site_stats
        cur_dec = sources[force_source_id].get("metadata", {}).get("decision")
        if cur_dec not in {"accepted", "listing"}:
            raise ValueError(
                f"Source {force_source_id} must be accepted or listing (got {cur_dec})"
            )

    seen_ids = set(sources.keys())
    download_jobs, extract_only = _prepare_site_jobs(
        storage, site, sources, force_source_id, site_stats, progress
    )
    if not download_jobs and not extract_only:
        _emit_progress(on_progress, progress)
        return site_stats

    rejected_sids = {
        sid
        for sid, r in sources.items()
        if r.get("metadata", {}).get("decision") == "rejected"
    }
    extract_counts = Counter()
    stream_opts = SimpleNamespace(
        retry_failed=False,
        reprocess=bool(force_source_id),
        flush_every=runtime_cfg["flush_every"],
    )
    need_extract = bool(extract_only) or any(role == "accepted" for _, role in download_jobs)
    extract_workers = max(1, min(runtime_cfg["extract_workers"], 4))
    pool = (
        ProcessPoolExecutor(max_workers=extract_workers, mp_context=get_context("spawn"))
        if need_extract
        else None
    )
    stream = None
    try:
        if pool is not None:
            allowed_backlog_sids = {r["metadata"]["source_id"] for r in extract_only}
            stream = StreamingExtractor(
                storage, site, profile, pool, extract_workers, stream_opts, extract_counts, errors
            )
            stream.backlog = deque(
                r for r in stream.backlog if r["metadata"]["source_id"] in allowed_backlog_sids
            )
            with storage.batched():
                storage.flush_pending()
                stream.on_commit()

        def run_download(job):
            rec, role = job
            url = rec["uri"]

            def allowed(target):
                if source_id(target) in rejected_sids:
                    return False
                rule = lookup(target, profile)
                if rule:
                    return rule["decision"] in {"accepted", "listing"}
                if target == url:
                    return True
                return site == "OTHER" or in_discovery_scope(target, profile)

            try:
                return rec, role, fetch_offline_or_http(fetcher, url, allowed, storage.root), None
            except Exception as exc:
                return rec, role, None, str(exc)

        progress["phase"] = "downloading"
        _emit_progress(on_progress, progress)
        workers = max(1, min(runtime_cfg["download_workers"], runtime_cfg["per_host"]))
        with storage.batched():
            for idx, (_, (rec, role, page, err)) in enumerate(
                completion_map(run_download, download_jobs, workers, on_tick=stream.poll if stream else None),
                1,
            ):
                _apply_download_result(
                    rec, role, page, err, site, profile, storage, seen_ids, site_stats, progress, errors, stream
                )
                if idx % runtime_cfg["flush_every"] == 0:
                    storage.flush_pending()
                    if stream is not None:
                        stream.on_commit()
                        storage.flush_pending()
                _emit_progress(on_progress, progress)
        if stream is not None:
            progress["phase"] = "extracting"
            _emit_progress(on_progress, progress)
            stream.finish()
            stream = None
    finally:
        if stream is not None:
            stream.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    site_stats["processed_saved"] += extract_counts["processed_saved"]
    site_stats["errors"] += extract_counts["processed_failed"]
    progress["processed_saved"] += extract_counts["processed_saved"]
    progress["errors"] += extract_counts["processed_failed"]
    _emit_progress(on_progress, progress)
    return site_stats


def run_incremental_ingest(
    data_dir,
    sites=None,
    *,
    force_source_id=None,
    profile_dir=None,
    overrides_path=None,
    on_progress=None,
) -> dict:
    """Incrementally download/extract accepted links and discover new pending links."""
    from .downloader.http import HTMLFetcher

    target_sites = [s.upper() for s in sites] if sites else None
    all_profiles = load_profiles(profile_dir)
    if target_sites:
        missing = set(target_sites) - set(all_profiles)
        if missing:
            raise ValueError(f"Unknown sites: {sorted(missing)}")
    attach_overrides(all_profiles, overrides_path)
    profiles = {s: all_profiles[s] for s in target_sites} if target_sites else all_profiles
    for profile in profiles.values():
        profile["_corpus_mode"] = "off"

    runtime_cfg = load_runtime()["ingestion"]
    storage = Storage(data_dir)
    progress = {
        "current_site": None,
        "phase": "starting",
        "skipped": 0,
        "downloaded": 0,
        "processed_saved": 0,
        "new_pending": 0,
        "rejected_preserved": 0,
        "errors": 0,
    }
    _emit_progress(on_progress, progress)

    site_results, errors, run_id, started_at = {}, [], str(uuid4()), now()
    fetcher = HTMLFetcher(
        runtime_cfg["timeout"], runtime_cfg["max_html_bytes"], runtime_cfg["retries"], runtime_cfg["delay"]
    )
    try:
        with writer_lock(storage.root):
            for site in ordered_site_names(profiles):
                progress["current_site"] = site
                progress["phase"] = "preparing"
                _emit_progress(on_progress, progress)
                site_results[site] = _ingest_site(
                    site, profiles[site], storage, fetcher, runtime_cfg, force_source_id, progress, errors, on_progress
                )
            if force_source_id is not None and len(profiles) == 1:
                only_site = next(iter(profiles))
                if force_source_id not in storage.records("sources", only_site):
                    raise ValueError(f"Source not found in {only_site}: {force_source_id}")
    finally:
        fetcher.close()
        progress["current_site"] = None
        progress["phase"] = "completed"
        _emit_progress(on_progress, progress)

    status = "failed" if errors else "success"
    write_json(
        storage.root / "ingestion_reports" / f"{run_id}.json",
        {
            "run_id": run_id,
            "mode": "incremental",
            "started_at": started_at,
            "finished_at": now(),
            "status": status,
            "sites": site_results,
            "errors": errors,
        },
    )
    return {"status": status, "sites": site_results, "errors": errors}

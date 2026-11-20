"""Source curation API: migrate, summarize, list, decide, add, remove, offline, ingest."""

from __future__ import annotations

from urllib.parse import urlsplit
from src.base import DataSource
from .config.profiles import load_profiles
from .crawler.policy import resource_type
from .curation_helpers import compute_site_state, load_corpus_policies, topic_info
from .curation_ingest import run_incremental_ingest
from .curation_offline import (
    count_offline_docs,
    scan_offline_docs,
    upload_offline_doc,
)
from .curation_yaml import (
    ordered_site_names,
    sync_overrides_yaml,
    sync_site_seeds_yaml,
)
from .identity import normalize_site_url, source_id
from .storage import Storage, now, writer_lock

VALID_DECISIONS = {"accepted", "listing", "pending", "rejected"}
__all__ = [
    "migrate_non_accepted_to_pending",
    "get_all_sites_summary",
    "list_site_sources",
    "set_sources_decision",
    "add_manual_urls",
    "remove_sources",
    "scan_offline_docs",
    "upload_offline_doc",
    "get_processed_document",
    "run_incremental_ingest",
]


def _load_site_profile(site: str, profile_dir=None) -> tuple[str, dict]:
    norm_site = site.upper()
    profiles = load_profiles(profile_dir)
    if norm_site not in profiles:
        raise ValueError(f"Unknown site: {norm_site}")
    return norm_site, profiles[norm_site]


def migrate_non_accepted_to_pending(data_dir, profile_dir=None) -> dict:
    """Convert non-accepted/non-alias sources to pending without losing accepted/alias."""
    profiles = load_profiles(profile_dir)
    storage = Storage(data_dir)
    summary = {}
    with writer_lock(storage.root):
        for site in ordered_site_names(profiles):
            if not storage.path("sources", site).is_file():
                continue
            rows = storage.records("sources", site)
            kept_accepted = kept_alias = moved_to_pending = 0
            for record in rows.values():
                meta = record.setdefault("metadata", {})
                decision = meta.get("decision")
                if decision == "accepted":
                    kept_accepted += 1
                elif decision == "alias":
                    kept_alias += 1
                else:
                    meta["decision"] = "pending"
                    meta["decision_reason"] = "migrated_to_pending"
                    meta.pop("automatic_decision", None)
                    meta.pop("override_note", None)
                    moved_to_pending += 1
            accepted_export = storage.root / "data_source" / f"{site}-accepted.jsonl"
            if moved_to_pending or not accepted_export.exists():
                storage.flush("sources", site)
            summary[site] = {
                "kept_accepted": kept_accepted,
                "kept_alias": kept_alias,
                "moved_to_pending": moved_to_pending,
            }
    return summary


def get_all_sites_summary(data_dir, profile_dir=None) -> dict:
    """Aggregate source, ingestion, and failure counters across all configured sites."""
    profiles = load_profiles(profile_dir)
    storage = Storage(data_dir)
    sites_list = []
    totals = {
        "total": 0,
        "pending": 0,
        "accepted_total": 0,
        "accepted_ingested": 0,
        "accepted_queued": 0,
        "listing": 0,
        "rejected_total": 0,
        "rejected_manual": 0,
        "rejected_auto": 0,
        "failed": 0,
    }
    for site in ordered_site_names(profiles):
        stats, _ = compute_site_state(storage, site, profiles[site])
        sites_list.append(stats)
        for key in totals:
            totals[key] += stats[key]
    return {
        "sites": sites_list,
        "totals": totals,
        "offline_docs_count": count_offline_docs(data_dir),
    }


def _matches_status(dec: str | None, st: dict, status_filter: str | None) -> bool:
    if status_filter is None:
        return True
    if status_filter == "pending":
        return dec not in {"accepted", "listing", "rejected"}
    if status_filter == "accepted":
        return dec == "accepted"
    if status_filter == "accepted_ingested":
        return dec == "accepted" and st["ingested"]
    if status_filter == "accepted_queued":
        return dec == "accepted" and not st["ingested"]
    if status_filter == "listing":
        return dec == "listing"
    if status_filter == "rejected":
        return dec == "rejected"
    if status_filter == "failed":
        return st["is_failed"]
    return True


def list_site_sources(
    data_dir,
    site,
    *,
    status=None,
    resource_type=None,
    search=None,
    discovered_from=None,
    sort="priority",
    page=1,
    page_size=50,
    profile_dir=None,
) -> dict:
    """List, filter, score, sort, and paginate curatable sources for one site."""
    site = site.upper()
    profiles = load_profiles(profile_dir)
    if site not in profiles:
        raise ValueError(f"Unknown site: {site}")
    storage = Storage(data_dir)
    stats, item_status = compute_site_state(storage, site, profiles[site])
    corpus_policy = load_corpus_policies(profiles).get(site)
    sources = storage.records("sources", site)

    discovered_options = sorted(
        {
            rec.get("metadata", {}).get("discovered_from")
            for rec in sources.values()
            if rec.get("metadata", {}).get("decision") != "alias"
            and rec.get("metadata", {}).get("discovered_from")
        }
    )
    status_filter = None if status in (None, "", "all") else status
    type_filter = None if resource_type in (None, "", "all") else resource_type
    parent_filter = None if discovered_from in (None, "", "all") else discovered_from
    query = (search or "").strip().casefold()

    items = []
    for sid, rec in sources.items():
        meta = rec.get("metadata", {})
        dec = meta.get("decision")
        if dec == "alias":
            continue
        uri, st = rec["uri"], item_status[sid]
        if not _matches_status(dec, st, status_filter):
            continue
        is_local = "/offline/" in urlsplit(uri).path
        kind = meta.get("resource_type") or globals()["resource_type"](uri)
        if type_filter == "local" and not is_local:
            continue
        if type_filter in {"html", "pdf", "office"} and kind != type_filter:
            continue
        parent = meta.get("discovered_from")
        if parent_filter is not None and parent != parent_filter:
            continue

        hint, prio = topic_info(rec, corpus_policy)
        title = meta.get("title") or ""
        link_text = meta.get("link_text") or ""
        override_note = meta.get("override_note") or ""
        if query and query not in " ".join((uri, title, link_text, parent or "", hint, override_note)).casefold():
            continue

        items.append(
            {
                "source_id": sid,
                "uri": uri,
                "decision": dec or "pending",
                "decision_reason": meta.get("decision_reason") or "",
                "resource_type": kind,
                "is_local": is_local,
                "title": title,
                "link_text": link_text,
                "discovered_from": parent,
                "crawl_depth": meta.get("crawl_depth", 0),
                "ingested": st["ingested"],
                "raw_exists": st["raw_exists"],
                "fail_error": st["fail_error"],
                "reviewed_at": meta.get("reviewed_at"),
                "override_note": override_note,
                "topic_hint": hint,
                "topic_priority": prio,
                "char_count": st["char_count"],
                "_sort_ts": meta.get("reviewed_at") or st["raw_created_at"] or meta.get("published_at") or "",
            }
        )

    if sort == "newest":
        items.sort(key=lambda x: (x["_sort_ts"], x["uri"]), reverse=True)
    elif sort in {"url", "uri", "az"}:
        items.sort(key=lambda x: (x["uri"].casefold(), x["source_id"]))
    elif sort == "depth":
        items.sort(key=lambda x: (x["crawl_depth"], -x["topic_priority"], x["uri"]))
    else:
        items.sort(key=lambda x: (-x["topic_priority"], x["crawl_depth"], x["uri"]))

    total_filtered = len(items)
    page_num = max(1, int(page))
    size = max(1, min(200, int(page_size)))
    start = (page_num - 1) * size
    paged = []
    for row in items[start : start + size]:
        row.pop("_sort_ts", None)
        paged.append(row)

    return {
        "site": site,
        "stats": stats,
        "items": paged,
        "total_filtered": total_filtered,
        "page": page_num,
        "page_size": size,
        "discovered_from_options": discovered_options,
    }


def set_sources_decision(
    data_dir,
    site,
    source_ids,
    decision,
    note="",
    profile_dir=None,
    overrides_path=None,
) -> dict:
    """Update decisions for selected sources and sync overrides.yaml + site seeds."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision: {decision}")
    site, profile = _load_site_profile(site, profile_dir)
    storage = Storage(data_dir)
    updated, override_updates, override_removals = 0, {}, set()
    seed_adds, seed_removals = [], []

    with writer_lock(storage.root):
        sources = storage.records("sources", site)
        timestamp = now()
        for sid in source_ids:
            rec = sources.get(sid)
            if rec is None or rec.get("metadata", {}).get("decision") == "alias":
                continue
            uri = rec["uri"]
            kind = resource_type(uri)
            if decision == "listing" and kind != "html":
                raise ValueError(f"Only HTML URLs can be listing: {uri}")
            if decision == "accepted" and kind not in {"html", "pdf"}:
                raise ValueError(f"Only HTML and PDF URLs can be accepted: {uri}")
            meta = rec.setdefault("metadata", {})
            meta.update(
                decision=decision,
                decision_reason="manual_curation",
                reviewed_at=timestamp,
                override_note=note or "",
            )
            updated += 1
            if decision != "accepted":
                storage.remove_processed(site, sid)
                seed_removals.append(uri)
            else:
                seed_adds.append(uri)
            if decision == "pending":
                override_removals.add(uri)
            else:
                override_updates[uri] = {"decision": decision, "note": note or ""}

        if updated:
            storage.flush("sources", site)
            sync_overrides_yaml(site, profile, override_updates, override_removals, overrides_path)
            sync_site_seeds_yaml(site, profile, seed_adds, seed_removals, profile_dir)
    return {"updated": updated, "site": site, "decision": decision}


def add_manual_urls(
    data_dir,
    site,
    urls,
    decision="accepted",
    note="",
    profile_dir=None,
    overrides_path=None,
) -> dict:
    """Validate, normalize, add manual URLs to a site, and sync configs."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision: {decision}")
    site, profile = _load_site_profile(site, profile_dir)
    storage = Storage(data_dir)
    added, duplicates, invalid = [], [], []
    override_updates, seed_adds = {}, []

    with writer_lock(storage.root):
        sources, batch_seen, timestamp = storage.records("sources", site), {}, now()
        for raw_url in urls:
            raw_str = str(raw_url or "").strip()
            if not raw_str:
                continue
            try:
                norm = normalize_site_url(raw_str, profile)
            except ValueError as exc:
                invalid.append({"uri": raw_str, "error": str(exc)})
                continue
            parts = urlsplit(norm)
            if site != "OTHER" and not parts.path.startswith("/offline/") and parts.netloc.lower() not in profile["scope"]["hosts"]:
                invalid.append({"uri": norm, "error": f"Host {parts.netloc} does not belong to {site}"})
                continue
            kind = resource_type(norm)
            if kind == "asset" or (decision == "listing" and kind != "html") or (decision == "accepted" and kind not in {"html", "pdf"}):
                invalid.append({"uri": norm, "error": f"Resource type '{kind}' is incompatible with decision '{decision}'"})
                continue
            sid = source_id(norm)
            if sid in sources or sid in batch_seen:
                existing_dec = sources[sid].get("metadata", {}).get("decision") if sid in sources else batch_seen[sid]
                duplicates.append({"uri": norm, "source_id": sid, "existing_decision": existing_dec})
                continue
            batch_seen[sid] = decision
            sources[sid] = DataSource(
                source_id=sid,
                source_type="web",
                uri=norm,
                metadata={
                    "site": site,
                    "faculty": site,
                    "resource_type": kind,
                    "discovered_from": None,
                    "crawl_depth": 0,
                    "decision": decision,
                    "decision_reason": "manual_curation",
                    "reviewed_at": timestamp,
                    "override_note": note or "",
                },
            ).model_dump(mode="json")
            added.append({"uri": norm, "source_id": sid, "decision": decision})
            if decision != "pending":
                override_updates[norm] = {"decision": decision, "note": note or ""}
            if decision == "accepted":
                seed_adds.append(norm)

        if added:
            storage.flush("sources", site)
            if override_updates:
                sync_overrides_yaml(site, profile, updates=override_updates, overrides_path=overrides_path)
            if seed_adds:
                sync_site_seeds_yaml(site, profile, add_urls=seed_adds, profile_dir=profile_dir)
    return {"added": added, "duplicates": duplicates, "invalid": invalid}


def remove_sources(
    data_dir,
    site,
    source_ids,
    purge_data=False,
    profile_dir=None,
    overrides_path=None,
) -> dict:
    """Remove source records, clean overrides/seeds, and optionally purge raw/processed."""
    site, profile = _load_site_profile(site, profile_dir)
    storage = Storage(data_dir)
    sid_set, removed_uris, removed = set(source_ids), [], 0

    with writer_lock(storage.root):
        sources = storage.records("sources", site)
        for sid in sid_set:
            rec = sources.pop(sid, None)
            if rec is not None:
                removed += 1
                removed_uris.append(rec["uri"])
        if removed:
            storage.flush("sources", site)
            sync_overrides_yaml(site, profile, removals=set(removed_uris), overrides_path=overrides_path)
            sync_site_seeds_yaml(site, profile, remove_urls=removed_uris, profile_dir=profile_dir)
        if purge_data and sid_set:
            processed = storage.records("processed", site)
            proc_changed = False
            for sid in sid_set:
                if processed.pop(sid, None) is not None:
                    proc_changed = True
            if proc_changed:
                storage.flush("processed", site)
            raw_rows = storage.records("raw", site)
            raw_site_root = (storage.root / "raw_data" / site).resolve()
            raw_changed = False
            for raw_key in list(raw_rows):
                row = raw_rows[raw_key]
                if row.get("metadata", {}).get("source_id") in sid_set:
                    raw_path = row.get("metadata", {}).get("raw_path")
                    if raw_path:
                        target = (storage.root / raw_path).resolve()
                        if target.is_relative_to(raw_site_root):
                            target.unlink(missing_ok=True)
                    raw_rows.pop(raw_key)
                    raw_changed = True
            if raw_changed:
                storage.flush("raw", site)
            states = storage.records("state", site)
            state_changed = False
            for sid in sid_set:
                for stage in ("crawl", "raw", "processed"):
                    if states.pop(f"{stage}:{sid}", None) is not None:
                        state_changed = True
            if state_changed:
                storage.flush("state", site)
    return {"removed": removed, "purged": bool(purge_data)}


def get_processed_document(data_dir, site, source_id) -> dict | None:
    """Return a single processed document with char_count, or None if missing."""
    site = site.upper()
    doc = Storage(data_dir).records("processed", site).get(source_id)
    if doc is None:
        return None
    return {**doc, "char_count": len(doc.get("content") or "")}

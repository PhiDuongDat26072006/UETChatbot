"""Two-way sync client and CLI handlers for the Cloudflare Review Worker + D1."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import requests
from .config.profiles import load_profiles
from .crawler.policy import is_accepted_type, resource_type
from .curation import VALID_DECISIONS, set_sources_decision
from .curation_helpers import compute_site_state, load_corpus_policies, topic_info
from .curation_yaml import ordered_site_names
from .storage import Storage, now, write_json
from .telegram_client import _read_dotenv


def _config_path(data_dir) -> Path:
    return Path(data_dir).resolve() / "ingestion_state" / "cloudflare_config.json"


def load_cloudflare_config(data_dir, url=None, secret=None) -> dict[str, str]:
    path = _config_path(data_dir)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    dotenv = _read_dotenv()
    raw_url = (url or "").strip() or str(saved.get("url") or "").strip() or os.environ.get("CLOUDFLARE_REVIEW_URL", "").strip() or dotenv.get("CLOUDFLARE_REVIEW_URL", "").strip()
    resolved_secret = (secret or "").strip() or str(saved.get("sync_secret") or "").strip() or os.environ.get("CLOUDFLARE_SYNC_SECRET", "").strip() or dotenv.get("CLOUDFLARE_SYNC_SECRET", "").strip()
    return {"url": raw_url.rstrip("/"), "sync_secret": resolved_secret}


def save_cloudflare_config(data_dir, url: str, secret: str) -> Path:
    path = _config_path(data_dir)
    write_json(path, {"url": url.strip().rstrip("/"), "sync_secret": secret.strip(), "updated_at": now()})
    return path


class CloudflareSyncClient:
    """HTTP sync client for the deployed Cloudflare Review Worker."""

    def __init__(self, base_url: str, sync_secret: str, timeout: float = 30.0):
        clean_url = (base_url or "").strip().rstrip("/")
        clean_secret = (sync_secret or "").strip()
        if not clean_url.startswith(("http://", "https://")):
            raise ValueError("Missing or invalid Cloudflare Worker URL (--url or CLOUDFLARE_REVIEW_URL)")
        if not clean_secret:
            raise ValueError("Missing SYNC_SECRET (--secret or CLOUDFLARE_SYNC_SECRET)")
        self.base_url = clean_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-Sync-Secret": clean_secret})

    def _request(self, method: str, path: str, json_body=None) -> dict:
        resp = self.session.request(
            method, f"{self.base_url}{path}", json=json_body, timeout=self.timeout
        )
        try:
            data = resp.json()
        except ValueError as exc:
            resp.raise_for_status()
            raise RuntimeError(f"Invalid JSON response from {path}") from exc
        if not resp.ok:
            raise RuntimeError(f"Cloudflare API {resp.status_code} ({path}): {data.get('error') or resp.text[:200]}")
        return data

    def push_batch(self, items: list[dict], force: bool = False) -> dict:
        return self._request("POST", "/api/sync/push", {"items": items, "force": bool(force)})

    def pull_dirty(self) -> dict:
        return self._request("GET", "/api/sync/pull")

    def sync_stats(self, sites: list[dict]) -> dict:
        return self._request("POST", "/api/sync/stats", {"sites": sites})

    def ack_items(self, items: list[dict], batch_size: int = 200) -> dict:
        chunk_size = max(1, min(200, int(batch_size)))
        acked = 0
        for i in range(0, len(items), chunk_size):
            res = self._request("POST", "/api/sync/ack", {"items": items[i : i + chunk_size]})
            acked += int(res.get("acked") or 0)
        return {"requested": len(items), "acked": acked}


def _build_client(data_dir, url=None, secret=None) -> tuple[CloudflareSyncClient, str]:
    cfg = load_cloudflare_config(data_dir, url=url, secret=secret)
    return CloudflareSyncClient(cfg["url"], cfg["sync_secret"]), cfg["url"]


def setup_cloudflare_sync(data_dir, *, url=None, secret=None) -> dict:
    """Verify connection & SYNC_SECRET against the Worker and save local config."""
    client, resolved_url = _build_client(data_dir, url=url, secret=secret)
    cfg = load_cloudflare_config(data_dir, url=url, secret=secret)
    pulled = client.pull_dirty()
    cfg_path = save_cloudflare_config(data_dir, resolved_url, cfg["sync_secret"])
    dirty_cnt = int(pulled.get("total") or len(pulled.get("items") or []))
    return {"status": "ok", "url": resolved_url, "config_path": str(cfg_path), "dirty_on_cloud": dirty_cnt}


def _collect_push_items(
    data_dir, *, sites=None, status_filter=None, source_ids=None, profile_dir=None
) -> tuple[list[dict], list[str]]:
    profiles = load_profiles(profile_dir)
    if sites:
        wanted = {s.upper() for s in sites}
        for s in wanted:
            if s not in profiles:
                raise ValueError(f"Unknown site: {s}")
        target_sites = [s for s in ordered_site_names(profiles) if s in wanted]
    else:
        target_sites = ordered_site_names(profiles)

    if status_filter and status_filter not in VALID_DECISIONS:
        raise ValueError(f"Invalid status filter: {status_filter}")

    storage = Storage(data_dir)
    corpus_policies = load_corpus_policies(profiles)
    wanted_ids = set(source_ids) if source_ids is not None else None
    items: list[dict] = []
    sites_used: list[str] = []

    for site in target_sites:
        _, item_status = compute_site_state(storage, site, profiles[site])
        records = storage.records("sources", site)
        policy = corpus_policies.get(site)
        site_count = 0
        for sid, rec in records.items():
            if wanted_ids is not None and sid not in wanted_ids:
                continue
            meta = rec.get("metadata", {})
            raw_dec = meta.get("decision")
            if raw_dec == "alias":
                continue
            dec = raw_dec if raw_dec in VALID_DECISIONS else "pending"
            if status_filter and dec != status_filter:
                continue
            uri = rec["uri"]
            st = item_status[sid]
            hint, prio = topic_info(rec, policy)
            kind = meta.get("resource_type") or resource_type(uri)
            items.append({
                "source_id": sid, "site": site, "uri": uri,
                "title": str(meta.get("title") or ""),
                "link_text": str(meta.get("link_text") or ""),
                "discovered_from": str(meta.get("discovered_from") or ""),
                "resource_type": kind, "crawl_depth": int(meta.get("crawl_depth") or 0),
                "topic_hint": hint, "topic_priority": int(prio),
                "decision": dec, "decision_reason": str(meta.get("decision_reason") or ""),
                "ingested": bool(st["ingested"]),
                "override_note": str(meta.get("override_note") or ""),
                "updated_at": str(meta.get("reviewed_at") or st.get("raw_created_at") or meta.get("published_at") or ""),
            })
            site_count += 1
        if site_count:
            sites_used.append(site)
    return items, sites_used


def sync_site_stats_to_cloudflare(data_dir, *, url=None, secret=None, profile_dir=None) -> dict:
    """Compute local site counts adjusted by cloud dirty decisions and upsert site_stats."""
    client, _ = _build_client(data_dir, url=url, secret=secret)
    profiles, storage = load_profiles(profile_dir), Storage(data_dir)
    stats_by_site: dict[str, dict] = {}
    for site in ordered_site_names(profiles):
        counts, _ = compute_site_state(storage, site, profiles[site])
        stats_by_site[site] = {
            "site": site, "total": counts["total"], "pending": counts["pending"],
            "accepted_total": counts["accepted_total"], "accepted_ingested": counts["accepted_ingested"],
            "listing": counts["listing"], "rejected_total": counts["rejected_total"], "dirty_count": 0,
        }
    key_map = {"pending": "pending", "accepted": "accepted_total", "listing": "listing", "rejected": "rejected_total"}
    cached: dict[str, dict] = {}
    for row in client.pull_dirty().get("items") or []:
        site = str(row.get("site") or "").strip().upper()
        if site not in stats_by_site:
            continue
        cached.setdefault(site, storage.records("sources", site))
        rec = cached[site].get(str(row.get("source_id") or "").strip())
        if rec is None or rec.get("metadata", {}).get("decision") == "alias":
            continue
        stats_by_site[site]["dirty_count"] += 1
        old_d = rec.get("metadata", {}).get("decision")
        new_d = str(row.get("decision") or "pending").strip().lower()
        old_k, new_k = key_map.get(old_d, "pending"), key_map.get(new_d, "pending")
        if old_k != new_k:
            stats_by_site[site][old_k] = max(0, stats_by_site[site][old_k] - 1)
            stats_by_site[site][new_k] += 1
    non_empty = [s for s in stats_by_site.values() if s["total"] > 0]
    return client.sync_stats(non_empty)


def push_to_cloudflare(
    data_dir, *, sites=None, status_filter=None, source_ids=None, batch_size=100, force=False, url=None, secret=None, profile_dir=None
) -> dict:
    """Push non-alias sources from local data_source to Cloudflare D1 in batches."""
    client, resolved_url = _build_client(data_dir, url=url, secret=secret)
    items, sites_used = _collect_push_items(data_dir, sites=sites, status_filter=status_filter, source_ids=source_ids, profile_dir=profile_dir)
    chunk_size = max(1, min(200, int(batch_size)))
    batches = changed = 0
    for i in range(0, len(items), chunk_size):
        res = client.push_batch(items[i : i + chunk_size], force=force)
        batches += 1
        changed += int(res.get("changed") or 0)
    sync_site_stats_to_cloudflare(data_dir, url=url, secret=secret, profile_dir=profile_dir)
    return {
        "status": "ok", "url": resolved_url, "sites_pushed": sites_used,
        "total_sent": len(items), "batches": batches, "changed": changed, "force": bool(force),
    }


def pull_from_cloudflare(
    data_dir, *, sites=None, status_filter=None, dry_run=False, details=False,
    url=None, secret=None, profile_dir=None, overrides_path=None,
) -> dict:
    """Pull dirty mobile curation decisions from Cloudflare D1 into local JSONL + YAML."""
    if status_filter and status_filter not in VALID_DECISIONS:
        raise ValueError(f"Invalid status filter: {status_filter}")
    client, resolved_url = _build_client(data_dir, url=url, secret=secret)
    pulled_rows = client.pull_dirty().get("items") or []
    wanted = {s.upper() for s in sites} if sites else None
    profiles, storage = load_profiles(profile_dir), Storage(data_dir)
    cached_sources: dict[str, dict] = {}
    grouped: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    by_decision: dict[str, int] = defaultdict(int)
    changes_by_decision: dict[str, list[dict]] = defaultdict(list)
    skipped: list[dict] = []
    errors: list[dict] = []

    for row in pulled_rows:
        site = str(row.get("site") or "").strip().upper()
        if wanted is not None and site not in wanted:
            continue
        sid = str(row.get("source_id") or "").strip()
        decision = str(row.get("decision") or "").strip().lower()
        if status_filter and decision != status_filter:
            continue
        note, updated_at = str(row.get("override_note") or ""), str(row.get("updated_at") or "")
        if site not in profiles or decision not in VALID_DECISIONS:
            errors.append({"site": site, "source_id": sid, "error": f"Invalid site/decision: {site}/{decision}"})
            continue
        if site not in cached_sources:
            cached_sources[site] = storage.records("sources", site)
        rec = cached_sources[site].get(sid)
        if rec is None or rec.get("metadata", {}).get("decision") == "alias":
            skipped.append({"site": site, "source_id": sid, "reason": "missing_or_alias_locally"})
            continue
        uri, meta = rec["uri"], rec.get("metadata", {})
        kind = resource_type(uri)
        if decision == "listing" and kind != "html":
            errors.append({"site": site, "source_id": sid, "error": f"Non-HTML cannot be listing: {uri}"})
            continue
        if decision == "accepted" and not is_accepted_type(uri):
            errors.append({"site": site, "source_id": sid, "error": f"Unsupported type for accepted: {uri}"})
            continue
        grouped[(site, decision, note)].append((sid, updated_at))
        by_decision[decision] += 1
        if details:
            changes_by_decision[decision].append({
                "site": site, "source_id": sid, "old_decision": str(meta.get("decision") or "pending"),
                "decision": decision, "uri": uri, "title": str(meta.get("title") or meta.get("link_text") or ""),
                "note": note, "updated_at": updated_at,
            })

    by_site: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    to_ack: list[dict] = []
    applied = 0
    for (site, decision, note), pairs in grouped.items():
        if not dry_run:
            set_sources_decision(
                data_dir, site, [sid for sid, _ in pairs], decision,
                note=note, profile_dir=profile_dir, overrides_path=overrides_path,
            )
            to_ack.extend({"source_id": sid, "updated_at": ts} for sid, ts in pairs)
        applied += len(pairs)
        by_site[site][decision] += len(pairs)

    acked = client.ack_items(to_ack)["acked"] if (to_ack and not dry_run) else 0
    result = {
        "status": "ok", "url": resolved_url, "dry_run": bool(dry_run),
        "pulled_total": len(pulled_rows), "applied": applied, "acked": acked,
        "skipped": skipped, "errors": errors,
        "by_decision": dict(sorted(by_decision.items())),
        "by_site": {s: dict(counts) for s, counts in sorted(by_site.items())},
    }
    if details:
        result["changes"] = {dec: rows for dec, rows in sorted(changes_by_decision.items())}
    return result

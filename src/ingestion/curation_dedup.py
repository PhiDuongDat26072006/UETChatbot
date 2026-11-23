"""Per-site and cross-site source deduplication backed by immutable alias records."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit
import yaml
from .config.profiles import load_profiles
from .crawler.policy import in_discovery_scope
from .curation_yaml import (
    DEFAULT_OVERRIDES_PATH, find_site_yaml_path, ordered_site_names,
    safe_normalize_url, sync_overrides_yaml, sync_site_seeds_yaml,
)
from .storage import Storage, now, writer_lock

_TRACKING_PARAMS = {"fbclid", "gclid"}
_DECISION_ORDER = {"accepted": 4, "listing": 3, "rejected": 2, "pending": 1}


def dedup_url_key(uri: str, profile: dict | None = None) -> str:
    """Canonical key collapsing http/https, www, uet aliases, slashes, query order."""
    raw = str(uri or "").strip()
    if not raw:
        return ""
    p = urlsplit(raw)
    if not p.hostname:
        return raw
    scheme = "https" if p.scheme.lower() in {"http", "https"} else p.scheme.lower()
    host = p.hostname.lower().removeprefix("www.")
    norm = (profile or {}).get("url_normalization") or {}
    if norm.get("canonical_origin"):
        c_host = (urlsplit(norm["canonical_origin"]).hostname or "").lower().removeprefix("www.")
        if host in {h.lower().removeprefix("www.") for h in norm.get("alias_hosts", [])} and c_host:
            host = c_host
    if host == "uet.vnu.edu.vn":
        host = "uet.edu.vn"
    try:
        port = p.port
    except ValueError:
        port = None
    if port and (scheme, port) not in {("https", 443), ("http", 80)}:
        host = f"{host}:{port}"
    path = " ".join(unquote(p.path or "/").split()).strip()
    while "//" in path:
        path = path.replace("//", "/")
    q_pairs = sorted(
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    )
    return urlunsplit((scheme, host, path.rstrip("/") or "/", urlencode(q_pairs), ""))


def _decision_tier(rec: dict, ingested_sids: set[str], raw_sids: set[str]) -> tuple[int, int]:
    meta, sid = rec.get("metadata", {}), rec["source_id"]
    dec = meta.get("decision") or "pending"
    manual = int(meta.get("decision_reason") in {"manual_curation", "manual_override", "manual_seed"})
    if dec == "accepted":
        return (5 if sid in ingested_sids else (4 if sid in raw_sids else 3), manual)
    return (2, manual) if dec == "listing" else ((1, manual) if dec == "rejected" else (0, 0))


def _host_clean_score(uri: str, site: str) -> tuple[int, int, int]:
    p = urlsplit(uri)
    clean_path = int(not any(ch.isspace() for ch in unquote(p.path or "/")))
    is_https = int(p.scheme.lower() == "https")
    host = (p.hostname or "").lower()
    host_pref = (2 if host == "uet.edu.vn" else int(host == "uet.vnu.edu.vn")) if site == "UET" else int(not host.startswith("www."))
    return (clean_path, is_https, host_pref)


def _merge_metadata(winner: dict, losers: list[dict], target_dec: str, cross_site: bool = False) -> bool:
    meta = winner.setdefault("metadata", {})
    old_dec = meta.get("decision") or "pending"
    promoted = target_dec != old_dec
    if promoted:
        meta["decision"] = target_dec
        if not meta.get("decision_reason") or old_dec == "pending":
            reasons = [r.get("metadata", {}).get("decision_reason") for r in losers if r.get("metadata", {}).get("decision") == target_dec and r.get("metadata", {}).get("decision_reason")]
            meta["decision_reason"] = reasons[0] if reasons else ("cross_site_dedup_merge" if cross_site else "dedup_merge")
    for field in ("title", "link_text", "published_at", "override_note", "reviewed_at"):
        if not meta.get(field):
            for rec in losers:
                if val := rec.get("metadata", {}).get(field):
                    meta[field] = val
                    break
    return promoted


def _apply_losers_alias(
    storage: Storage, win_site: str, win_rec: dict, losers: list[tuple[str, dict]],
    reason: str, ts: str, proc_by_site: dict[str, dict], yaml_ops: dict[str, dict],
) -> tuple[int, bool]:
    proc_deduped, any_non_pending = 0, False
    for lsite, lrec in losers:
        lmeta = lrec.setdefault("metadata", {})
        old_dec = lmeta.get("decision") or "pending"
        lmeta.update(
            decision="alias", decision_reason=reason, canonical_source_id=win_rec["source_id"],
            canonical_site=win_site, canonical_uri=win_rec["uri"], dedup_from_decision=old_dec, dedup_at=ts,
        )
        if lrec["source_id"] in proc_by_site[lsite]:
            storage.remove_processed(lsite, lrec["source_id"])
            proc_deduped += 1
        if old_dec in {"accepted", "listing", "rejected"}:
            yaml_ops[lsite]["removals"].add(lrec["uri"])
            any_non_pending = True
        if old_dec == "accepted":
            yaml_ops[lsite]["seed_removals"].append(lrec["uri"])
    return proc_deduped, any_non_pending


def _sync_yaml_for_site(site: str, profile: dict, op: dict, profile_dir, overrides_path) -> None:
    ovr_file = Path(overrides_path or DEFAULT_OVERRIDES_PATH)
    existing_ovr = set()
    if ovr_file.is_file():
        ovr_data = yaml.safe_load(ovr_file.read_text(encoding="utf-8")) or {}
        existing_ovr = {safe_normalize_url(u, profile) for u in ((ovr_data.get("sites") or {}).get(site) or {}).get("urls", {})}
    if op["updates"] or ({safe_normalize_url(u, profile) for u in op["removals"]} & existing_ovr):
        sync_overrides_yaml(site, profile, updates=op["updates"], removals=op["removals"], overrides_path=overrides_path)

    site_doc = yaml.safe_load(find_site_yaml_path(site, profile_dir).read_text(encoding="utf-8")) or {}
    existing_seeds = {safe_normalize_url(str(u), profile) for u in (site_doc.get("seeds") or [])}
    if ({safe_normalize_url(u, profile) for u in op["seed_adds"]} - existing_seeds) or ({safe_normalize_url(u, profile) for u in op["seed_removals"]} & existing_seeds):
        sync_site_seeds_yaml(site, profile, add_urls=op["seed_adds"], remove_urls=op["seed_removals"], profile_dir=profile_dir)


def _restore_redirect_aliases(sources: dict[str, dict], profile: dict, yaml_op: dict) -> int:
    restored = 0
    for rec in sources.values():
        meta = rec.get("metadata", {})
        if meta.get("decision") != "alias" or meta.get("decision_reason") not in {"dedup_alias", "cross_site_dedup_alias"}:
            continue
        canon_uri = meta.get("canonical_uri", "")
        if dedup_url_key(rec["uri"], profile) == dedup_url_key(canon_uri, profile):
            continue
        prev_dec = meta.pop("dedup_from_decision", "pending") or "pending"
        meta["decision"] = prev_dec
        meta["decision_reason"] = "manual_curation" if prev_dec in {"accepted", "listing", "rejected"} else "discovered_pending"
        for k in ("canonical_source_id", "canonical_site", "canonical_uri", "dedup_at"):
            meta.pop(k, None)
        if meta.get("reviewed_at") and prev_dec in {"accepted", "listing", "rejected"}:
            yaml_op["updates"][rec["uri"]] = {"decision": prev_dec, "note": meta.get("override_note", "")}
            if prev_dec == "accepted":
                yaml_op["seed_adds"].append(rec["uri"])
        restored += 1
    return restored


def _dedup_site_locked(storage: Storage, site: str, profile: dict, *, profile_dir=None, overrides_path=None) -> dict:
    sources, processed = storage.records("sources", site), storage.records("processed", site)
    raw_sids = {r["metadata"]["source_id"] for r in storage.records("raw", site).values()}
    proc_sids = set(processed.keys())
    yaml_ops = {site: {"updates": {}, "removals": set(), "seed_adds": [], "seed_removals": []}}
    restored = _restore_redirect_aliases(sources, profile, yaml_ops[site])

    active = [r for r in sources.values() if r.get("metadata", {}).get("decision") != "alias"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in active:
        groups[dedup_url_key(rec["uri"], profile)].append(rec)

    ts, merged_total, promoted, proc_deduped = now(), 0, 0, 0
    samples = []
    for _, items in groups.items():
        if len(items) < 2:
            continue
        def score(rec: dict):
            tier, manual = _decision_tier(rec, proc_sids, raw_sids)
            cp, hts, hp = _host_clean_score(rec["uri"], site)
            has_title = int(bool((rec.get("metadata", {}).get("title") or "").strip()))
            depth = int(rec.get("metadata", {}).get("crawl_depth", 0) or 0)
            return (-tier, -manual, -cp, -hts, -hp, -has_title, depth, rec["uri"], rec["source_id"])

        ranked = sorted(items, key=score)
        winner, losers = ranked[0], ranked[1:]
        best_dec = max((r.get("metadata", {}).get("decision") or "pending" for r in items), key=lambda d: _DECISION_ORDER.get(d, 0))
        was_promoted = _merge_metadata(winner, losers, best_dec)
        promoted += int(was_promoted)
        merged_total += len(losers)
        pd, any_non_pending = _apply_losers_alias(storage, site, winner, [(site, l) for l in losers], "dedup_alias", ts, {site: processed}, yaml_ops)
        proc_deduped += pd
        win_dec = winner["metadata"].get("decision") or "pending"
        if (was_promoted or any_non_pending) and win_dec in {"accepted", "listing", "rejected"}:
            yaml_ops[site]["updates"][winner["uri"]] = {"decision": win_dec, "note": winner["metadata"].get("override_note", "")}
            if win_dec == "accepted":
                yaml_ops[site]["seed_adds"].append(winner["uri"])
        if len(samples) < 10:
            samples.append({"canonical_uri": winner["uri"], "winner_decision": win_dec, "merged_uris": [l["uri"] for l in losers]})

    if merged_total or restored:
        storage.flush("sources", site)
        _sync_yaml_for_site(site, profile, yaml_ops[site], profile_dir, overrides_path)
    return {
        "site": site, "scanned_active": len(active), "duplicate_groups": sum(1 for v in groups.values() if len(v) > 1),
        "duplicates_merged": merged_total, "url_norm_merged": merged_total, "redirect_merged": 0,
        "restored_redirects": restored, "promoted_decisions": promoted, "processed_deduped": proc_deduped,
        "remaining_active": len(active) - merged_total, "samples": samples,
    }


def deduplicate_site(data_dir, site: str, *, profile_dir=None, overrides_path=None) -> dict:
    """Deduplicate sources inside one site while preserving alias source_ids."""
    norm_site, profiles = site.upper(), load_profiles(profile_dir)
    if norm_site not in profiles:
        raise ValueError(f"Unknown site: {norm_site}")
    storage = Storage(data_dir)
    with writer_lock(storage.root):
        return _dedup_site_locked(storage, norm_site, profiles[norm_site], profile_dir=profile_dir, overrides_path=overrides_path)


def deduplicate_all_sites(
    data_dir, *, sites=None, cross_site: bool = True, profile_dir=None, overrides_path=None
) -> dict:
    """Run per-site deduplication on selected sites, then cross-site deduplication."""
    profiles = load_profiles(profile_dir)
    req = [s.upper() for s in sites] if sites else ordered_site_names(profiles)
    if unknown := (set(req) - set(profiles)):
        raise ValueError(f"Unknown sites: {sorted(unknown)}")
    selected = [s for s in ordered_site_names(profiles) if s in set(req)]
    site_order = {s: idx for idx, s in enumerate(selected)}
    storage, per_site, per_site_merged, proc_deduped, cross_merged, cross_groups, cross_samples = Storage(data_dir), {}, 0, 0, 0, 0, []

    with writer_lock(storage.root):
        for site in selected:
            res = _dedup_site_locked(storage, site, profiles[site], profile_dir=profile_dir, overrides_path=overrides_path)
            per_site[site], per_site_merged, proc_deduped = res, per_site_merged + res["duplicates_merged"], proc_deduped + res["processed_deduped"]

        if cross_site and len(selected) > 1:
            proc_by_site = {s: storage.records("processed", s) for s in selected}
            raw_by_site = {s: {r["metadata"]["source_id"] for r in storage.records("raw", s).values()} for s in selected}
            all_rows = [(s, r, profiles[s]) for s in selected for r in storage.records("sources", s).values() if r.get("metadata", {}).get("decision") != "alias"]
            global_groups: dict[str, list[tuple[str, dict, bool]]] = defaultdict(list)
            for s, rec, prof in all_rows:
                global_groups[dedup_url_key(rec["uri"], prof)].append((s, rec, in_discovery_scope(rec["uri"], prof)))

            dirty_sites, ts = set(), now()
            yaml_ops = {s: {"updates": {}, "removals": set(), "seed_adds": [], "seed_removals": []} for s in selected}
            for _, entries in global_groups.items():
                if len({s for s, _, _ in entries}) < 2:
                    continue
                cross_groups += 1
                def gscore(entry: tuple[str, dict, bool]):
                    s, rec, in_sc = entry
                    tier, manual = _decision_tier(rec, set(proc_by_site[s].keys()), raw_by_site[s])
                    cp, hts, hp = _host_clean_score(rec["uri"], s)
                    return (int(not in_sc), -tier, -manual, -hts, -cp, -hp, site_order[s], rec["uri"])

                ranked = sorted(entries, key=gscore)
                (win_site, win_rec, _), loser_entries = ranked[0], ranked[1:]
                loser_recs = [r for _, r, _ in loser_entries]
                win_dec = target_dec = win_rec.get("metadata", {}).get("decision") or "pending"
                if win_dec == "pending":
                    pos_decs = [r.get("metadata", {}).get("decision") for r in loser_recs if r.get("metadata", {}).get("decision") in {"accepted", "listing"}]
                    if pos_decs:
                        target_dec = max(pos_decs, key=lambda d: _DECISION_ORDER[d])
                    elif not any(in_sc for _, _, in_sc in entries) and any(r.get("metadata", {}).get("decision") == "rejected" for r in loser_recs):
                        target_dec = "rejected"
                if _merge_metadata(win_rec, loser_recs, target_dec, cross_site=True):
                    dirty_sites.add(win_site)
                    yaml_ops[win_site]["updates"][win_rec["uri"]] = {"decision": target_dec, "note": win_rec["metadata"].get("override_note", "")}
                    if target_dec == "accepted":
                        yaml_ops[win_site]["seed_adds"].append(win_rec["uri"])

                pd, _ = _apply_losers_alias(storage, win_site, win_rec, [(ls, lr) for ls, lr, _ in loser_entries], "cross_site_dedup_alias", ts, proc_by_site, yaml_ops)
                proc_deduped += pd
                for ls, _, _ in loser_entries:
                    dirty_sites.add(ls)
                    cross_merged += 1
                    per_site[ls]["remaining_active"] -= 1
                if len(cross_samples) < 10:
                    cross_samples.append({"canonical_site": win_site, "canonical_uri": win_rec["uri"], "merged": [{"site": ls, "uri": lr["uri"]} for ls, lr, _ in loser_entries]})
            for s in sorted(dirty_sites):
                storage.flush("sources", s)
                _sync_yaml_for_site(s, profiles[s], yaml_ops[s], profile_dir, overrides_path)

    return {
        "status": "success", "per_site_merged": per_site_merged, "cross_site_merged": cross_merged,
        "total_merged": per_site_merged + cross_merged,
        "duplicate_groups": sum(r["duplicate_groups"] for r in per_site.values()) + cross_groups,
        "processed_deduped": proc_deduped, "remaining_active": sum(r["remaining_active"] for r in per_site.values()),
        "sites": per_site, "cross_site_samples": cross_samples,
    }

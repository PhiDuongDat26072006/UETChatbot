"""Exact-URL manual policy; immutable per-run snapshot shared by all workers."""

from pathlib import Path
from urllib.parse import urlsplit
import json
import yaml
from src.base import DataSource
from .identity import normalize_site_url, source_id
from .storage import write_json

DEFAULT_PATH = Path(__file__).parent / "config" / "overrides.yaml"


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def attach(profiles, path=None):
    path = Path(path or DEFAULT_PATH)
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    if (
        not isinstance(config, dict)
        or set(config) != {"version", "sites"}
        or config["version"] != 1
        or not isinstance(config["sites"], dict)
    ):
        raise ValueError("Override YAML requires version: 1 and sites: {}")
    unknown = set(config["sites"]) - set(profiles)
    if unknown:
        raise ValueError(f"Unknown override sites: {sorted(unknown)}")
    for site, spec in config["sites"].items():
        if (
            not isinstance(spec, dict)
            or set(spec) != {"urls"}
            or not isinstance(spec["urls"], dict)
        ):
            raise ValueError(f"{site}: expected urls mapping")
        rules = {}
        for url, rule in spec["urls"].items():
            if (
                not isinstance(url, str)
                or not isinstance(rule, dict)
                or set(rule) - {"decision", "note"}
            ):
                raise ValueError(f"Invalid override: {site} {url}")
            if rule.get("decision") not in {
                "accepted",
                "listing",
                "rejected",
                "deferred",
            } or not isinstance(rule.get("note", ""), str):
                raise ValueError(f"Invalid decision/note: {url}")
            url = normalize_site_url(url, profiles[site])
            if url in rules:
                raise ValueError(f"Duplicate normalized override URL: {url}")
            if rule["decision"] in {"accepted", "listing"}:
                from .crawler.policy import resource_type

                if urlsplit(url).netloc.lower() not in profiles[site]["scope"]["hosts"]:
                    raise ValueError(
                        f"{url}: add host to site profile or route to another site first"
                    )
                if resource_type(url) != "html":
                    raise ValueError(
                        f"{url}: unsupported for HTML ingestion; use deferred"
                    )
            rules[url] = dict(rule)
        profiles[site]["_overrides"] = rules
    return profiles


def lookup(url, profile):
    return profile.get("_overrides", {}).get(normalize_site_url(url, profile))


def reconcile(storage, site, profile):
    """Seed explicit URLs and requeue changed eligibility even after complete crawl."""
    from .crawler.policy import evaluate

    rows = storage.records("sources", site)
    for url in profile.get("_overrides", {}):
        sid = source_id(url)
        if sid not in rows:
            storage.save_source(
                DataSource(
                    source_id=sid,
                    source_type="web",
                    uri=url,
                    metadata={
                        "site": site,
                        "faculty": site,
                        "decision": "deferred",
                        "decision_reason": "manual_seed",
                        "resource_type": "html",
                    },
                )
            )
    checkpoint = storage.root / "ingestion_state" / (site + "-crawl.json")
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    queue = []
    for row in list(rows.values()):
        meta = row["metadata"]
        if meta.get("decision") == "alias":
            continue
        old = meta.get("decision")
        new, _ = evaluate(row["uri"], profile, dict(meta))
        if new in {"accepted", "listing"} and old != new:
            queue.append(row["uri"])
    # Commit replay intent before reconcile_sources changes decisions.
    if queue:
        if saved is not None:
            seen = set(saved.get("seen", [])) - set(queue)
            frontier = saved.get("frontier", [])
            queued = {x[0] for x in frontier}
            frontier.extend([u, 0, None] for u in queue if u not in queued)
            saved.update(seen=sorted(seen), frontier=frontier, complete=False)
            write_json(checkpoint, saved)
        profile["_manual_seeds"] = queue

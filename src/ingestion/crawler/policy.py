"""Traversal and retention are separate decisions."""

from datetime import date
from pathlib import PurePosixPath
import re
from urllib.parse import urlsplit, parse_qs, unquote

ATTACHMENTS = {
    ".pdf": "pdf",
    ".doc": "office",
    ".docx": "office",
    ".xls": "office",
    ".xlsx": "office",
    ".ppt": "office",
    ".pptx": "office",
}
ASSETS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".zip",
    ".rar",
    ".mp4",
    ".mp3",
    ".woff",
    ".woff2",
}


def resource_type(url):
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    return ATTACHMENTS.get(suffix, "asset" if suffix in ASSETS else "html")


def has_blocked_fragment(url, profile):
    """Check before canonicalization, which deliberately removes all fragments."""
    fragment = unquote(urlsplit(url).fragment).casefold()
    return fragment in {
        str(value).casefold() for value in profile["scope"].get("exclude_fragments", [])
    }


def automatic_in_scope(url, profile):
    p = urlsplit(url)
    scope = profile["scope"]
    path = unquote(p.path).rstrip("/") or "/"
    if p.netloc.lower() not in scope["hosts"]:
        return False
    seeds = {
        s.rstrip("/")
        for s in profile.get("_manual_seeds", []) + profile.get("seeds", [])
    }
    if resource_type(url) == "pdf" and url.rstrip("/") not in seeds:
        return False
    if scope.get("exact_paths") and path not in [
        s.rstrip("/") or "/" for s in scope["exact_paths"]
    ]:
        return False
    if not any(
        path == x.rstrip("/") or path.startswith(x.rstrip("/") + "/")
        for x in scope.get("path_prefixes", ["/"])
    ):
        return False
    if any(
        path == x.rstrip("/") or path.startswith(x.rstrip("/") + "/")
        for x in scope.get("exclude_prefixes", [])
    ):
        return False
    if any(re.search(pattern, path) for pattern in scope.get("exclude_patterns", [])):
        return False
    if scope.get("include_patterns") and not any(
        re.search(pattern, path) for pattern in scope["include_patterns"]
    ):
        return False
    query = {
        k.casefold(): values
        for k, values in parse_qs(p.query, keep_blank_values=True).items()
    }
    blocked_keys = scope.get("exclude_query_keys", ["s", "replytocom", "preview"])
    if any(key.casefold() in query for key in blocked_keys):
        return False
    for key, values in scope.get("exclude_query_values", {}).items():
        blocked = {str(v).casefold() for v in values}
        if any(value.casefold() in blocked for value in query.get(key.casefold(), [])):
            return False
    return not has_blocked_fragment(url, profile)


def in_scope(url, profile):
    from ..overrides import lookup

    rule = lookup(url, profile)
    if rule:
        return (
            rule["decision"] in {"accepted", "listing"}
            and urlsplit(url).netloc.lower() in profile["scope"]["hosts"]
        )
    return automatic_in_scope(url, profile)


def is_listing(url, profile):
    p = urlsplit(url)
    path = unquote(p.path).rstrip("/") or "/"
    scope = profile["scope"]
    if path in [x.rstrip("/") or "/" for x in scope.get("listing_exact", [])]:
        return True
    if any(re.search(pattern, path) for pattern in scope.get("listing_patterns", [])):
        return True
    return any(k in parse_qs(p.query) for k in ["page", "paged"])


def automatic_evaluate(url, profile, metadata=None):
    kind = resource_type(url)
    if kind == "pdf":
        if not automatic_in_scope(url, profile):
            return "rejected", "outside_scope"
        return "accepted", "pdf_document"
    if kind != "html":
        return "deferred" if kind != "asset" else "rejected", kind
    if not automatic_in_scope(url, profile):
        return "rejected", "outside_scope"
    if is_listing(url, profile):
        return "listing", "navigation_page"
    policy = profile.get("policy", {})
    metadata = metadata or {}
    cutoff = policy.get("published_after")
    # Date filtering is opt-in and only for explicitly time-sensitive paths.
    if cutoff and any(
        re.search(p, urlsplit(url).path)
        for p in policy.get("time_sensitive_patterns", [])
    ):
        published = metadata.get("published_at")
        if not published:
            return policy.get("unknown_date", "review"), "unknown_date"
        if date.fromisoformat(published[:10]) < date.fromisoformat(str(cutoff)):
            return "rejected", "older_than_cutoff"
    return "accepted", "html_document"


def evaluate(url, profile, metadata=None):
    from ..overrides import lookup

    auto, reason = automatic_evaluate(url, profile, metadata)
    rule = lookup(url, profile)
    if metadata is not None:
        metadata.pop("automatic_decision", None)
        metadata.pop("override_note", None)
        if rule:
            metadata.update(automatic_decision=auto, override_note=rule.get("note", ""))
    return (rule["decision"], "manual_override") if rule else (auto, reason)


def reconcile_sources(storage, site, profile):
    """Re-evaluate saved sources offline after policy edits, before selecting jobs."""
    from ..identity import normalize_site_url, source_id
    from copy import deepcopy

    from ..overrides import reconcile

    reconcile(storage, site, profile)
    changed = 0
    rows = storage.records("sources", site)
    # Keep old identities as aliases for audit. Raw bytes/history remain immutable;
    # the canonical source is downloaded again before creating new processed data.
    for source in list(rows.values()):
        url = normalize_site_url(source["uri"], profile)
        sid = source_id(url)
        if sid == source["source_id"]:
            continue
        if sid not in rows:
            canonical = deepcopy(source)
            canonical.update(source_id=sid, uri=url)
            for key in ("canonical_source_id", "decision", "decision_reason"):
                canonical["metadata"].pop(key, None)
            rows[sid] = canonical
            changed += 1
        meta = source["metadata"]
        if meta.get("decision") != "alias" or meta.get("canonical_source_id") != sid:
            meta.update(
                decision="alias",
                decision_reason="normalized_url",
                canonical_source_id=sid,
            )
            changed += 1
        storage.remove_processed(site, source["source_id"])
    for source in storage.records("sources", site).values():
        metadata = source["metadata"]
        if metadata.get("decision") == "alias":
            continue
        before = dict(metadata)
        decision, reason = evaluate(source["uri"], profile, metadata)
        if before != metadata:
            changed += 1
        if (metadata.get("decision"), metadata.get("decision_reason")) != (
            decision,
            reason,
        ):
            metadata.update(decision=decision, decision_reason=reason)
            changed += 1
        if decision != "accepted":
            storage.remove_processed(site, source["source_id"])
    if changed:
        storage.flush("sources", site)
    return changed

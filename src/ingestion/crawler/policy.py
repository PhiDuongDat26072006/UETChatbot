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


DEFAULT_TECH_EXCLUDE_PREFIXES = (
    "/wp-admin",
    "/wp-login.php",
    "/feed",
    "/wp-json",
    "/xmlrpc.php",
    "/wp-trackback.php",
    "/wp-includes",
    "/wp-content/themes",
    "/wp-content/plugins",
    "/tag",
    "/author",
)
DEFAULT_TECH_EXCLUDE_QUERY_KEYS = ("s", "replytocom", "preview", "share", "format", "print", "amp")
DEFAULT_TECH_EXCLUDE_FRAGMENTS = ("respond", "comments")
_TECH_PATH_RE = re.compile(
    r"(?:^|/)(?:wp-admin|wp-login\.php|wp-json|xmlrpc\.php|wp-trackback\.php|wp-includes)(?:/|$)"
    r"|(?:^|/)(?:en/)?(?:tag|author)(?:/|$)"
    r"|(?:^|/)feed(?:/|$)"
)


def resource_type(url):
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    return ATTACHMENTS.get(suffix, "asset" if suffix in ASSETS else "html")


def is_docx(url):
    return PurePosixPath(urlsplit(url).path).suffix.lower() == ".docx"


def is_accepted_type(url):
    return resource_type(url) in {"html", "pdf"} or is_docx(url)


def has_blocked_fragment(url, profile):
    """Check before canonicalization, which deliberately removes all fragments."""
    fragment = unquote(urlsplit(url).fragment).casefold()
    blocked = set(DEFAULT_TECH_EXCLUDE_FRAGMENTS)
    blocked.update(
        str(value).casefold() for value in profile["scope"].get("exclude_fragments", [])
    )
    return fragment in blocked


def _normalized_seeds(profile):
    from ..identity import normalize_site_url

    seeds = set()
    for raw in profile.get("_manual_seeds", []) + profile.get("seeds", []):
        try:
            seeds.add(normalize_site_url(raw, profile))
        except ValueError:
            seeds.add(raw.rstrip("/"))
    return seeds


def in_discovery_scope(url, profile):
    """Technical discovery eligibility: valid host, non-asset, and non-junk."""
    p = urlsplit(url)
    scope = profile["scope"]
    path = unquote(p.path).rstrip("/") or "/"
    path_lower = path.casefold()
    if p.path.startswith("/offline/"):
        return is_accepted_type(url)
    if p.netloc.lower() not in scope["hosts"]:
        return False
    if resource_type(url) == "asset":
        return False
    if not any(
        path == x.rstrip("/") or path.startswith(x.rstrip("/") + "/")
        for x in scope.get("path_prefixes", ["/"])
    ):
        return False
    excluded_prefixes = list(DEFAULT_TECH_EXCLUDE_PREFIXES) + list(
        scope.get("exclude_prefixes", [])
    )
    if any(
        path_lower == x.rstrip("/").casefold()
        or path_lower.startswith(x.rstrip("/").casefold() + "/")
        for x in excluded_prefixes
    ):
        return False
    if _TECH_PATH_RE.search(path_lower):
        return False
    if any(re.search(pattern, path) for pattern in scope.get("exclude_patterns", [])):
        return False
    query = {
        k.casefold(): values
        for k, values in parse_qs(p.query, keep_blank_values=True).items()
    }
    blocked_keys = set(DEFAULT_TECH_EXCLUDE_QUERY_KEYS) | {
        str(key).casefold() for key in scope.get("exclude_query_keys", [])
    }
    if any(key in query for key in blocked_keys):
        return False
    for key, values in scope.get("exclude_query_values", {}).items():
        blocked = {str(v).casefold() for v in values}
        if any(value.casefold() in blocked for value in query.get(key.casefold(), [])):
            return False
    return not has_blocked_fragment(url, profile)


def discovery_verdict(url, profile):
    """Return initial (decision, reason) for a newly discovered child link."""
    from ..overrides import lookup

    rule = lookup(url, profile)
    if rule:
        return rule["decision"], "manual_override"
    if in_discovery_scope(url, profile):
        return "pending", "discovered_pending"
    p = urlsplit(url)
    if p.netloc.lower() not in profile["scope"]["hosts"] and not p.path.startswith(
        "/offline/"
    ):
        return "rejected", "outside_host"
    if resource_type(url) == "asset":
        return "rejected", "asset"
    return "rejected", "technical_junk"


def automatic_in_scope(url, profile):
    from ..identity import normalize_site_url

    p = urlsplit(url)
    scope = profile["scope"]
    path = unquote(p.path).rstrip("/") or "/"
    try:
        norm = normalize_site_url(url, profile)
    except ValueError:
        norm = url
    seeds = _normalized_seeds(profile)
    is_seed = norm in seeds or url.rstrip("/") in {s.rstrip("/") for s in seeds}
    if is_seed and resource_type(url) != "asset":
        return True
    if not in_discovery_scope(url, profile):
        return False
    if resource_type(url) == "pdf":
        return False
    if scope.get("exact_paths") and path not in [
        s.rstrip("/") or "/" for s in scope["exact_paths"]
    ]:
        return False
    if scope.get("include_patterns") and not any(
        re.search(pattern, path) for pattern in scope["include_patterns"]
    ):
        return False
    return True


def in_scope(url, profile):
    from ..overrides import lookup

    rule = lookup(url, profile)
    if rule:
        if rule["decision"] not in {"accepted", "listing"}:
            return False
        p = urlsplit(url)
        return p.scheme.lower() in {"http", "https"} and bool(p.netloc)
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
    if kind == "pdf" or is_docx(url):
        if not automatic_in_scope(url, profile):
            return "rejected", "outside_scope"
        return "accepted", "docx_document" if is_docx(url) else "pdf_document"
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

    from ..overrides import lookup, reconcile

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
        current_decision = metadata.get("decision")
        if current_decision == "alias":
            continue
        rule = lookup(source["uri"], profile)
        if rule is None:
            if current_decision in {"rejected", "pending"}:
                storage.remove_processed(site, source["source_id"])
                continue
            if (
                current_decision in {"accepted", "listing"}
                and metadata.get("decision_reason") == "manual_curation"
            ):
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
    accepted_export = storage.root / "data_source" / f"{site}-accepted.jsonl"
    if changed or (storage.path("sources", site).exists() and not accepted_export.exists()):
        storage.flush("sources", site)
    return changed

"""Stable source identity and immutable raw-version identity."""

import hashlib
import re
import unicodedata
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode
from uuid import UUID, uuid5

NAMESPACE = UUID("a21f62b5-395e-5c46-aa02-61feac17b2ce")


def canonicalize_url(url, base_url=None):
    parts = urlsplit(urljoin(base_url or "", url.strip()))
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only absolute HTTP(S) URLs are supported")
    if parts.username or parts.password:
        raise ValueError("Credentials in source URL are not supported")
    host = parts.hostname.lower().encode("idna").decode()
    if ":" in host:
        host = "[" + host + "]"
    if parts.port and (parts.scheme.lower(), parts.port) not in {
        ("http", 80),
        ("https", 443),
    }:
        host += f":{parts.port}"
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}
    ]
    # Preserve slash and query order: either may carry application semantics.
    return urlunsplit(
        (parts.scheme.lower(), host, parts.path or "/", urlencode(query), "")
    )


def source_id(url):
    return str(uuid5(NAMESPACE, canonicalize_url(url)))


def digest(content):
    return hashlib.sha256(content).hexdigest()


def raw_id(source, content_hash):
    return str(uuid5(NAMESPACE, source + ":" + content_hash))


def slug(url):
    path = urlsplit(url).path.strip("/").split("/")[-1] or "index"
    value = unicodedata.normalize("NFKD", path).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:70] or "page"


def normalize_site_url(url, profile, base_url=None):
    """Apply an explicit site's origin aliases before computing IDs or queuing."""
    normalized = canonicalize_url(url, base_url)
    parts = urlsplit(normalized)
    rule = profile.get("url_normalization", {})
    if parts.netloc in rule.get("alias_hosts", []):
        origin = urlsplit(rule["canonical_origin"])
        normalized = urlunsplit(
            (origin.scheme, origin.netloc, parts.path, parts.query, "")
        )
    return normalized

"""Load one authoritative site profile for crawling and extraction."""

from pathlib import Path
import re
from urllib.parse import urlsplit
import yaml
from soupsieve import compile as compile_selector
from .settings import DEFAULT_PROFILE_DIR


def load_profiles(directory=None, sites=None):
    profiles = {}
    for path in sorted(Path(directory or DEFAULT_PROFILE_DIR).glob("*.yaml")):
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        site = profile["site"]
        if not re.fullmatch(r"[A-Z0-9_-]+", site):
            raise ValueError(f"Invalid site name: {site}")
        if site in profiles:
            raise ValueError(f"Duplicate site: {site}")
        rule = profile.get("url_normalization")
        if rule:
            origin = urlsplit(rule["canonical_origin"])
            hosts = profile["scope"]["hosts"]
            if (
                origin.scheme != "https"
                or origin.netloc not in hosts
                or origin.path not in ("", "/")
                or origin.query
                or origin.fragment
                or origin.username
                or origin.password
            ):
                raise ValueError(f"Invalid canonical_origin: {site}")
            aliases = rule.get("alias_hosts", [])
            if not aliases or any(host not in hosts for host in aliases):
                raise ValueError(f"Invalid alias_hosts: {site}")
        for group in (profile["discovery"], profile["extraction"]):
            for key, selectors in group.items():
                if key.endswith("selectors"):
                    for selector in selectors:
                        compile_selector(selector)
        for field in ("listing_patterns", "exclude_patterns"):
            for pattern in profile["scope"].get(field, []):
                re.compile(pattern)
        for selector in profile["extraction"].get("title_attributes", {}):
            compile_selector(selector)
        profiles[site] = profile
    if sites:
        missing = set(sites) - profiles.keys()
        if missing:
            raise ValueError(f"Unknown sites: {sorted(missing)}")
        profiles = {s: profiles[s] for s in sites}
    if not profiles:
        raise ValueError("No site profiles found")
    return profiles


def resolve_profile(url, profiles):
    from ..crawler.policy import in_scope

    matches = [p for p in profiles.values() if in_scope(url, p)]
    if not matches:
        raise ValueError(f"No profile for {url}; pass --site explicitly for local HTML")
    return max(
        matches, key=lambda p: max(map(len, p["scope"].get("path_prefixes", ["/"])))
    )

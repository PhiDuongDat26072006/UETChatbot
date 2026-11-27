"""Comment-preserving YAML synchronization for overrides.yaml and site seeds."""

from __future__ import annotations

from pathlib import Path
import re
import yaml
from .config.settings import DEFAULT_PROFILE_DIR
from .identity import normalize_site_url
from .overrides import DEFAULT_PATH as DEFAULT_OVERRIDES_PATH
from .storage import atomic_bytes

_TOP_LEVEL_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+\s*:")
_LIST_ITEM_RE = re.compile(r"^(\s*)-\s+(.*)$")


def ordered_site_names(profiles: dict) -> list[str]:
    middle = sorted(s for s in profiles if s not in {"UET", "OTHER"})
    names = []
    if "UET" in profiles:
        names.append("UET")
    names.extend(middle)
    if "OTHER" in profiles:
        names.append("OTHER")
    return names


def find_site_yaml_path(site: str, profile_dir=None) -> Path:
    directory = Path(profile_dir or DEFAULT_PROFILE_DIR)
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("site") == site:
            return path
    raise ValueError(f"Site profile YAML not found for {site} in {directory}")


def safe_normalize_url(url: str, profile: dict) -> str:
    try:
        return normalize_site_url(url, profile)
    except ValueError:
        return url


def _format_yaml_scalar(value: str) -> str:
    dumped = yaml.safe_dump([value], allow_unicode=True, default_flow_style=False).strip()
    if dumped.startswith("- "):
        return dumped[2:].strip()
    return dumped


def sync_overrides_yaml(
    site: str,
    profile: dict,
    updates: dict[str, dict] | None = None,
    removals: set[str] | None = None,
    overrides_path=None,
) -> None:
    path = Path(overrides_path or DEFAULT_OVERRIDES_PATH)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        data = {}
    data["version"] = 1
    sites_map = data.setdefault("sites", {})
    if not isinstance(sites_map, dict):
        sites_map = {}
        data["sites"] = sites_map
    site_spec = sites_map.get(site)
    urls_map = dict(site_spec.get("urls", {})) if isinstance(site_spec, dict) else {}

    normalized_removals = {safe_normalize_url(u, profile) for u in (removals or set())}
    normalized_updates = {
        safe_normalize_url(u, profile): dict(rule) for u, rule in (updates or {}).items()
    }
    drop_set = normalized_removals | set(normalized_updates)
    cleaned = {}
    for existing_url, rule in urls_map.items():
        norm_existing = safe_normalize_url(existing_url, profile)
        if norm_existing not in drop_set:
            cleaned[norm_existing] = rule
    for norm_url, rule in normalized_updates.items():
        entry = {"decision": rule["decision"]}
        if rule.get("note"):
            entry["note"] = str(rule["note"])
        cleaned[norm_url] = entry
    if cleaned:
        sites_map[site] = {"urls": cleaned}
    else:
        sites_map.pop(site, None)
    rendered = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000)
    atomic_bytes(path, rendered.encode("utf-8"))


def sync_site_seeds_yaml(
    site: str,
    profile: dict,
    add_urls: list[str] | None = None,
    remove_urls: list[str] | None = None,
    profile_dir=None,
) -> None:
    path = find_site_yaml_path(site, profile_dir)
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    seeds_idx = next(
        (i for i, line in enumerate(lines) if re.match(r"^seeds\s*:", line)), None
    )
    if seeds_idx is None:
        raise ValueError(f"Missing seeds key in {path}")
    end_idx = len(lines)
    for idx in range(seeds_idx + 1, len(lines)):
        line = lines[idx]
        if (
            line
            and not line[0].isspace()
            and not line.startswith("#")
            and _TOP_LEVEL_KEY_RE.match(line)
        ):
            end_idx = idx
            break

    remove_set = {safe_normalize_url(u, profile) for u in (remove_urls or [])}
    add_list = [safe_normalize_url(u, profile) for u in (add_urls or [])]
    add_list = list(dict.fromkeys(u for u in add_list if u not in remove_set))

    kept_block = []
    existing_norm = set()
    item_indent = None
    last_item_pos = -1
    for line in lines[seeds_idx + 1 : end_idx]:
        match = _LIST_ITEM_RE.match(line)
        if match:
            indent, raw_scalar = match.groups()
            item_indent = item_indent if item_indent is not None else indent
            parsed_scalar = yaml.safe_load(raw_scalar)
            norm_scalar = (
                safe_normalize_url(str(parsed_scalar), profile)
                if isinstance(parsed_scalar, str)
                else None
            )
            if norm_scalar and norm_scalar in remove_set:
                continue
            if norm_scalar:
                existing_norm.add(norm_scalar)
            last_item_pos = len(kept_block)
        kept_block.append(line)

    indent = item_indent if item_indent is not None else "  "
    to_insert = [
        f"{indent}- {_format_yaml_scalar(u)}" for u in add_list if u not in existing_norm
    ]
    if to_insert:
        insert_at = last_item_pos + 1 if last_item_pos >= 0 else 0
        kept_block = kept_block[:insert_at] + to_insert + kept_block[insert_at:]

    has_active = any(_LIST_ITEM_RE.match(line) for line in kept_block)
    header_line = "seeds:" if has_active else "seeds: []"
    updated_lines = lines[:seeds_idx] + [header_line] + kept_block + lines[end_idx:]
    updated_text = "\n".join(updated_lines) + ("\n" if raw_text.endswith("\n") else "")
    validated = yaml.safe_load(updated_text)
    if not isinstance(validated, dict) or not isinstance(validated.get("seeds"), list):
        raise ValueError(f"YAML validation failed after updating seeds in {path}")
    atomic_bytes(path, updated_text.encode("utf-8"))

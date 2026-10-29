"""BS4 discovery; selectors are supplied by site profiles."""

import logging
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from ..identity import normalize_site_url
from .policy import has_blocked_fragment


log = logging.getLogger(__name__)


def metadata(soup, profile):
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, "html.parser")
    result = {}
    for selector in profile["extraction"]["title_selectors"]:
        node = soup.select_one(selector)
        if node is None:
            continue
        attribute = profile["extraction"].get("title_attributes", {}).get(selector)
        value = str(
            node.get(attribute, "") if attribute else node.get_text(" ", strip=True)
        ).strip()
        ignored = {
            t.casefold() for t in profile["extraction"].get("ignored_titles", [])
        }
        if value and value.casefold() not in ignored:
            result.update(title=value, title_source=selector)
            break
    for selector in profile["extraction"]["date_selectors"]:
        node = soup.select_one(selector)
        if node:
            value = (
                node.get("datetime")
                or node.get("content")
                or node.get_text(" ", strip=True)
            )
            from dateutil.parser import parse

            try:
                import re
                from datetime import datetime

                parsed = (
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if re.match(r"^\d{4}-\d{2}-\d{2}", value)
                    else parse(value, dayfirst=True)
                )
                result.update(
                    published_at=parsed.isoformat(), published_at_source=selector
                )
                break
            except (ValueError, OverflowError):
                continue
    return result


def parse_page(html, url, profile):
    soup = BeautifulSoup(html, "html.parser")
    found = {}
    malformed = set()
    for selector in profile["discovery"]["link_selectors"]:
        for node in soup.select(selector):
            href = node.get("href", "")
            if not isinstance(href, str):
                continue
            href = href.strip()
            if not href or href.startswith("#"):
                continue
            try:
                if urlsplit(href).scheme.lower() not in {"", "http", "https"}:
                    continue
                # Fragment parsing can raise before URL canonicalization does.
                if has_blocked_fragment(href, profile):
                    continue
                target = normalize_site_url(href, profile, url)
            except ValueError:
                malformed.add(href)
                continue
            found.setdefault(
                target, {"url": target, "link_text": node.get_text(" ", strip=True)}
            )
    if malformed:
        log.warning("[%s] skipped %s malformed href(s) on %r; samples=%r",
                    profile.get("site", "?"), len(malformed), url,
                    [value[:160] for value in sorted(malformed)[:3]])
    return metadata(soup, profile), list(found.values())

"""BS4 discovery; selectors are supplied by site profiles."""

from bs4 import BeautifulSoup
from ..identity import normalize_site_url
from .policy import has_blocked_fragment


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
    for selector in profile["discovery"]["link_selectors"]:
        for node in soup.select(selector):
            href = node.get("href", "").strip()
            if (
                not href
                or href.startswith("#")
                or href.lower().startswith(("mailto:", "javascript:", "tel:"))
                or ("@" in href and "://" not in href)
                or has_blocked_fragment(href, profile)
            ):
                continue
            try:
                target = normalize_site_url(href, profile, url)
            except ValueError:
                continue
            found.setdefault(
                target, {"url": target, "link_text": node.get_text(" ", strip=True)}
            )
    return metadata(soup, profile), list(found.values())

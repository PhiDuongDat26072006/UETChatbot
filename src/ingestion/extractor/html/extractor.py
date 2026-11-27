"""Site-specific BS4 extraction, then fallback on untouched original HTML."""

import re
from bs4 import BeautifulSoup
from ...crawler.parser import metadata as page_metadata
from ...crawler.policy import resource_type
from ...identity import canonicalize_url
from .cleaner import clean_content
from .icons import clean_html_icons
from ..quality.validation import validate_html


def extract_html(html, url, profile):
    soup = BeautifulSoup(html, "html.parser")
    if clean_html_icons(soup):
        html = str(soup)  # Both selector extraction and trafilatura see cleaned text.
    meta = page_metadata(soup, profile)
    title = meta.pop("title", "")
    attempts = []
    candidates = []
    attachments = {}
    for selector in profile["extraction"]["content_selectors"]:
        nodes = soup.select(selector)
        if not nodes:
            attempts.append({"selector": selector, "reason": "no_match"})
            continue
        if len(nodes) != 1:
            attempts.append(
                {
                    "selector": selector,
                    "reason": "multiple_matches",
                    "count": len(nodes),
                }
            )
            continue
        content, node = clean_content(
            nodes[0], url, profile["extraction"].get("remove_selectors", [])
        )
        for a in node.select("a[href]"):
            try:
                target = canonicalize_url(a["href"], url)
            except ValueError:
                continue
            if resource_type(target) in {"pdf", "office"}:
                attachments[target] = {
                    "url": target,
                    "text": a.get_text(" ", strip=True),
                    "status": "deferred",
                }
        quality = validate_html(content, profile, node)
        result = {
            "title": title,
            "content": content,
            "metadata": {
                **meta,
                "extraction_method": "html_bs4",
                "content_selector": selector,
                "quality": quality,
                "attachments": list(attachments.values()),
            },
        }
        candidates.append(result)
        attempts.append({"selector": selector, "quality": quality})
        if quality["status"] == "accepted":
            result["metadata"]["attempts"] = attempts
            return result
    # Do not let fallback silently replace a selected table with flattened text.
    table_candidates = [r for r in candidates if "unresolved_table_structure" in r["metadata"]["quality"]["reasons"]]
    if table_candidates:
        result = table_candidates[0]
        result["metadata"]["attempts"] = attempts
        return result
    from trafilatura import extract

    content = (
        extract(
            html,
            url=url,
            output_format="markdown",
            include_tables=True,
            include_links=True,
            include_formatting=True,
            include_comments=False,
            deduplicate=False,
        )
        or ""
    )
    # Markdown fallback is not trusted to preserve source tables or raw HTML.
    residual = bool(re.search(r"</?[a-zA-Z][^>]*>", content))
    if residual:
        fragment = BeautifulSoup(content, "html.parser")
        content, _ = clean_content(fragment, url, [])
    quality = validate_html(content, profile)
    if residual:
        quality.update(status="review", reasons=[*quality["reasons"], "fallback_html_repaired"])
    if soup.find("table") is not None:
        quality.update(status="review", reasons=[*quality["reasons"], "fallback_table_coverage_unverified"])
    if (
        attachments
        and len(BeautifulSoup(html, "html.parser").get_text(" ", strip=True)) < 200
    ):
        quality.update(
            status="review", reasons=[*quality["reasons"], "attachment_only"]
        )
    fallback = {
        "title": title,
        "content": content,
        "metadata": {
            **meta,
            "extraction_method": "trafilatura",
            "quality": quality,
            "attachments": list(attachments.values()),
            "attempts": attempts,
        },
    }
    if quality["status"] == "accepted":
        return fallback
    candidates.append(fallback)
    result = max(candidates, key=lambda r: len(r["content"]))
    result["metadata"]["attempts"] = attempts
    return result

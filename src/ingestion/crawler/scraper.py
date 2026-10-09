"""Discover HTML endpoints and attachments from accepted source pages."""
from __future__ import annotations

import re
from urllib.parse import urljoin

import requests

from ..config.session import get_source_response
from ..filters import (
    clean_url, is_allowed_file_domain, is_document_file,
    is_media_or_asset_file, is_valid_html_endpoint, normalize_endpoint_url,
)

A_HREF_REGEX = re.compile(r"<a\b[^>]*?\bhref=[\"']([^\"'#\s>]+)", re.IGNORECASE)
EMBED_SRC_REGEX = re.compile(r"<(?:embed|iframe)\b[^>]*?\bsrc=[\"']([^\"'#\s>]+)", re.IGNORECASE)


def scrape_single_page(
    session: requests.Session, target_domain: str, url: str,
    *, allowed_path_prefixes: list[str] | tuple[str, ...] = (),
) -> tuple[set[str], set[str]]:
    """Collect links from one page; relative links use the final response URL.

    Scope is checked before requesting or following redirects. Request failures
    propagate to the crawler, which records partial failure without discarding
    successful pages. Non-HTML responses produce no links.
    """
    endpoints: set[str] = set()
    files: set[str] = set()
    if not is_valid_html_endpoint(url, target_domain, allowed_path_prefixes=allowed_path_prefixes):
        return endpoints, files
    with get_source_response(session, url, target_domain, allowed_path_prefixes) as response:
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", ""):
            return endpoints, files
        for raw_link in A_HREF_REGEX.findall(response.text):
            raw_link = raw_link.strip()
            if not raw_link or raw_link.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
                continue
            source_url = clean_url(urljoin(response.url, raw_link))
            if is_document_file(source_url):
                if is_allowed_file_domain(source_url, target_domain):
                    files.add(source_url)
            elif not is_media_or_asset_file(source_url):
                normalized_url = normalize_endpoint_url(source_url)
                if is_valid_html_endpoint(normalized_url, target_domain, allowed_path_prefixes=allowed_path_prefixes):
                    endpoints.add(normalized_url)
        for embedded_url in EMBED_SRC_REGEX.findall(response.text):
            source_url = clean_url(urljoin(response.url, embedded_url.strip()))
            if is_document_file(source_url) and is_allowed_file_domain(source_url, target_domain):
                files.add(source_url)
    return endpoints, files

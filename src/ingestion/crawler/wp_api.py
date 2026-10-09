"""WordPress discovery with shared pagination and graceful API fallback."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from urllib.parse import urlparse

import requests

from ..config.session import REQUEST_TIMEOUT
from ..filters import clean_url, is_document_file, is_valid_html_endpoint, normalize_endpoint_url

logger = logging.getLogger(__name__)


def _iter_wp_items(
    session: requests.Session, base_url: str, endpoint: str, max_pages: int,
) -> Iterator[dict]:
    """Yield API items while closing every response; unavailable APIs end discovery.

    WordPress is optional: missing endpoints, invalid payloads, and transport
    failures leave HTML discovery available. Details are logged at DEBUG.
    """
    parsed = urlparse(base_url)
    api_base = f"{parsed.scheme}://{parsed.netloc}/wp-json/wp/v2/{endpoint}"
    for page in range(1, max_pages + 1):
        api_url = f"{api_base}?per_page=100&page={page}"
        try:
            with session.get(api_url, timeout=REQUEST_TIMEOUT, verify=False) as response:
                if response.status_code != 200 or "application/json" not in response.headers.get("Content-Type", ""):
                    logger.debug("WordPress API unavailable: %s (HTTP %s)", api_url, response.status_code)
                    return
                items = response.json()
                if not isinstance(items, list) or not items:
                    return
                yield from (item for item in items if isinstance(item, dict))
                if page >= int(response.headers.get("X-WP-TotalPages", 1)):
                    return
        except (requests.RequestException, ValueError, TypeError) as error:
            logger.debug("WordPress API unavailable: %s (%s)", api_url, error, exc_info=True)
            return


def fetch_wp_media_documents(session: requests.Session, base_url: str, max_pages: int = 5) -> set[str]:
    """Collect document URLs from the site's global media API, excluding assets."""
    return {
        clean_url(item["source_url"])
        for item in _iter_wp_items(session, base_url, "media", max_pages)
        if isinstance(item.get("source_url"), str) and is_document_file(item["source_url"])
    }


def fetch_wp_posts_and_pages(
    session: requests.Session, base_url: str, endpoint: str, target_domain: str,
    max_pages: int = 5, *, allowed_path_prefixes: list[str] | tuple[str, ...] = (),
) -> set[str]:
    """Collect normalized post/page links passing the shared host and path rules."""
    endpoints: set[str] = set()
    for item in _iter_wp_items(session, base_url, endpoint, max_pages):
        if isinstance(item.get("link"), str):
            normalized_url = normalize_endpoint_url(item["link"])
            if is_valid_html_endpoint(normalized_url, target_domain, allowed_path_prefixes=allowed_path_prefixes):
                endpoints.add(normalized_url)
    return endpoints

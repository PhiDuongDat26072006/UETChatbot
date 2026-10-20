"""Bóc tách thẻ liên kết HTML và tài liệu nhúng trên một trang đơn lẻ."""
from __future__ import annotations

from urllib.parse import urljoin
import requests

from ..config import A_HREF_REGEX, EMBED_SRC_REGEX, REQUEST_TIMEOUT
from ..filters import (
    clean_url,
    is_allowed_file_domain,
    is_document_file,
    is_media_or_asset_file,
    is_valid_html_endpoint,
    normalize_endpoint_url,
)


def scrape_single_page(
    session: requests.Session,
    base_url: str,
    target_domain: str,
    url: str,
    subpath: str = None,
) -> tuple[set[str], set[str]]:
    """Cào nội dung trang: bóc tách <a href="..."> và embed/iframe tài liệu."""
    endpoints = set()
    files = set()
    try:
        res = session.get(url, timeout=REQUEST_TIMEOUT, verify=False)
        if res.status_code == 200 and "text/html" in res.headers.get("Content-Type", ""):
            for raw_link in A_HREF_REGEX.findall(res.text):
                raw_link = raw_link.strip()
                if not raw_link or raw_link.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
                    continue

                full_url = clean_url(urljoin(base_url, raw_link))
                if is_document_file(full_url):
                    if is_allowed_file_domain(full_url, target_domain):
                        files.add(full_url)
                elif is_media_or_asset_file(full_url):
                    continue
                else:
                    norm_url = normalize_endpoint_url(full_url)
                    if is_valid_html_endpoint(norm_url, target_domain, subpath):
                        endpoints.add(norm_url)

            for emb in EMBED_SRC_REGEX.findall(res.text):
                full_emb = clean_url(urljoin(base_url, emb.strip()))
                if is_document_file(full_emb) and is_allowed_file_domain(full_emb, target_domain):
                    files.add(full_emb)
    except Exception:
        pass
    return endpoints, files

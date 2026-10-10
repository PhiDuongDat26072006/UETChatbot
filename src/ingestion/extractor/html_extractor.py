"""Module bóc tách nội dung bài viết HTML sạch cho RAG pipeline."""
from __future__ import annotations

import logging
from typing import Any

import requests
import trafilatura
from bs4 import BeautifulSoup

from ..classifier import clean_page_title
from ..config.faculties import FacultyTarget
from ..config.session import REQUEST_TIMEOUT, create_http_session, get_source_response
from ..utils import clean_whitespace
from .cloud_links import extract_cloud_links, format_cloud_links_markdown
from .date_extractor import extract_html_date
from .html_cleaner import (
    NOISE_TAGS,
    PROTECTED_CLASSES,
    WP_CONTENT_SELECTORS,
    is_thin_content,
    table_to_markdown,
)

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_html_text",
    "extract_html_fallback",
    "extract_html_content",
    "extract_cloud_links",
    "format_cloud_links_markdown",
    "is_thin_content",
    "table_to_markdown",
]


def fetch_html_text(
    url: str, timeout: int = REQUEST_TIMEOUT, session: requests.Session | None = None,
    target: FacultyTarget | None = None,
) -> str | None:
    """Fetch HTML using configured transport; reject redirects outside scoped sources.

    Caller-owned sessions are reused and remain open. HTTP/transport failures
    propagate so the pipeline can distinguish failures from rejected content.
    """
    if session is None:
        with create_http_session() as owned_session:
            return fetch_html_text(url, timeout, owned_session, target)
    domain = target["domain"] if target else ""
    prefixes = target["allowed_path_prefixes"] if target else ()
    with get_source_response(session, url, domain, prefixes, timeout) as response:
        response.raise_for_status()
        if response.encoding is None or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
        return response.text


def extract_html_fallback(html_text: str) -> tuple[str, str]:
    """Fallback trích xuất bài viết qua BeautifulSoup khi Trafilatura thất bại hoặc thin."""
    soup = BeautifulSoup(html_text, "lxml")

    title = ""
    for sel in ["h1.entry-title", "h1.post-title", "h1.title", "h1"]:
        h1 = soup.select_one(sel)
        if h1 and h1.get_text().strip():
            title = clean_page_title(h1.get_text().strip())
            break
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = clean_page_title(title_tag.get_text().strip())

    for noise_name in NOISE_TAGS:
        for t in soup.find_all(noise_name):
            t.decompose()

    main_el = None
    for sel in WP_CONTENT_SELECTORS:
        found = soup.select_one(sel)
        if found:
            main_el = found
            break
    if not main_el:
        main_el = soup.body or soup

    for table_tag in main_el.find_all("table"):
        md_table = table_to_markdown(table_tag)
        if md_table:
            table_tag.replace_with(soup.new_string(md_table))
        else:
            table_tag.decompose()

    for el in main_el.find_all(["div", "section", "aside"]):
        attrs = getattr(el, "attrs", None) or {}
        classes = attrs.get("class", [])
        c_str = " ".join(classes).lower() if isinstance(classes, list) else str(classes).lower()
        if any(p in c_str for p in PROTECTED_CLASSES):
            continue
        if any(bad in c_str for bad in ("sidebar", "comment", "share", "widget", "related", "banner", "popup")):
            el.decompose()

    text = clean_whitespace(main_el.get_text(separator="\n", strip=True))
    return text, title


def extract_html_content(
    url: str,
    html_text: str | None = None,
    timeout: int = REQUEST_TIMEOUT,
    min_length: int = 100,
    category: str | None = None,
) -> dict[str, Any] | None:
    """Trích xuất nội dung văn bản chính và tiêu đề từ một URL hoặc chuỗi HTML."""
    if not html_text:
        try:
            html_text = fetch_html_text(url, timeout=timeout)
        except requests.RequestException as error:
            logger.warning("HTML request failed: %s (%s)", url, error)
            logger.debug("HTML request failure", exc_info=True)
            return None

    if not html_text or len(html_text.strip()) < 50:
        return None

    raw_title, cloud_links, title_soup = "", [], None
    try:
        title_soup = BeautifulSoup(html_text, "lxml")
        title_tag = title_soup.find("title")
        if title_tag and title_tag.string:
            raw_title = clean_page_title(title_tag.get_text().strip())
        cloud_links = extract_cloud_links(title_soup)
    except Exception:
        pass

    extracted_text, extracted_title = None, raw_title

    # 1. Trích xuất chính qua Trafilatura
    try:
        extracted_text = trafilatura.extract(
            html_text, url=url, include_comments=False, include_tables=True,
            include_links=True, output_format="txt", with_metadata=False, deduplicate=True,
        )
    except Exception:
        extracted_text = None

    # 2. Fallback sang BeautifulSoup nếu Trafilatura không đạt
    use_fallback = (
        not extracted_text
        or len(extracted_text.strip()) < min_length
        or is_thin_content(extracted_text, title=extracted_title, min_length=min_length, category=category)
    )

    if use_fallback:
        fb_text, fb_title = extract_html_fallback(html_text)
        if fb_text and len(fb_text) >= min_length:
            extracted_text = fb_text
            extracted_title = fb_title or extracted_title

    if not extracted_text:
        return None

    clean_content = clean_whitespace(extracted_text)
    if cloud_links:
        clean_content += format_cloud_links_markdown(cloud_links)

    if is_thin_content(clean_content, title=extracted_title, min_length=min_length, category=category):
        return None

    published_date = extract_html_date(
        html_text=html_text, url=url, text=clean_content, soup=title_soup, category=category,
    )

    return {
        "url": url,
        "title": extracted_title,
        "published_date": published_date,
        "text": clean_content,
        "text_length": len(clean_content),
        "cloud_links": cloud_links,
    }

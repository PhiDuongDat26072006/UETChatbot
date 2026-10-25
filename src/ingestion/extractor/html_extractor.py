"""Module bóc tách nội dung bài viết HTML sạch cho RAG pipeline."""
from __future__ import annotations

import logging
import re
from typing import Any

import requests
import trafilatura
from bs4 import BeautifulSoup, NavigableString, Tag

from ..classifier import clean_page_title
from ..config.faculties import FacultyTarget
from ..config.session import REQUEST_TIMEOUT, create_http_session, get_source_response
from ..utils import clean_whitespace
from .cloud_links import extract_cloud_links, format_cloud_links_markdown
from .date_extractor import extract_html_date
from .html_cleaner import (
    NOISE_TAGS,
    NOISE_CLASS_TOKENS,
    find_main_content,
    is_thin_content,
    strip_boilerplate_lines,
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
    """Structure-aware BS4 fallback; preserve headings, lists, tables, inline phrases."""
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

    main_el = find_main_content(soup)

    def render(node: Tag) -> list[str]:
        if not isinstance(node, Tag):
            return []
        classes = {part.lower() for part in node.get("class", [])}
        if node.name in NOISE_TAGS or (classes & NOISE_CLASS_TOKENS):
            return []
        if node.name == "table":
            return [table_to_markdown(node)]
        if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = node.get_text(" ", strip=True)
            return [f"{'#' * int(node.name[1])} {heading}"] if heading else []
        if node.name == "li":
            # A single list item is one semantic unit, even with inline links.
            line = node.get_text(" ", strip=True)
            return [f"- {line}"] if line else []
        if node.name == "p":
            line = node.get_text(" ", strip=True)
            return [line] if line else []
        result: list[str] = []
        for child in node.children:
            if isinstance(child, Tag):
                result.extend(render(child))
            elif isinstance(child, NavigableString) and child.strip():
                result.append(str(child).strip())
        return result

    text = clean_whitespace("\n\n".join(render(main_el)))
    return text, title


def _key_content_coverage(html_soup: BeautifulSoup, content: str) -> float:
    """Measure recall of high-value headings, list items, and table headers."""
    main = find_main_content(html_soup)
    phrases = []
    for node in main.find_all(["h2", "h3", "li", "th"]):
        if node.find_parent(["nav", "aside", "footer"]):
            continue
        phrase = node.get_text(" ", strip=True)
        if len(phrase) >= 4 and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= 80:
            break
    if not phrases:
        return 1.0
    normalized = re.sub(r"\s+", " ", content).casefold()
    return sum(re.sub(r"\s+", " ", phrase).casefold() in normalized for phrase in phrases) / len(phrases)


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
    method = "trafilatura"

    # 1. Trích xuất chính qua Trafilatura
    try:
        extracted_text = trafilatura.extract(
            html_text, url=url, include_comments=False, include_tables=True,
            include_links=True, output_format="markdown", with_metadata=False,
            favor_recall=True, deduplicate=False,
        )
    except Exception:
        extracted_text = None

    # 2. Fallback sang BeautifulSoup nếu Trafilatura không đạt
    use_fallback = (
        not extracted_text
        or len(extracted_text.strip()) < min_length
        or is_thin_content(extracted_text, title=extracted_title, min_length=min_length, category=category)
    )
    primary_coverage = _key_content_coverage(title_soup, extracted_text or "") if title_soup else 1.0
    if use_fallback or primary_coverage < 1.0:
        fb_text, fb_title = extract_html_fallback(html_text)
        fallback_coverage = _key_content_coverage(title_soup, fb_text) if title_soup else 1.0
        if fb_text and (
            (use_fallback and (len(fb_text) >= min_length or bool(cloud_links)))
            or fallback_coverage > primary_coverage + 0.05
        ):
            extracted_text = fb_text
            extracted_title = fb_title or extracted_title
            method = "beautifulsoup"

    if not extracted_text:
        return None

    clean_content = clean_whitespace(strip_boilerplate_lines(extracted_text))
    if cloud_links:
        # Avoid duplicating links already emitted inline by Trafilatura Markdown.
        missing_links = [link for link in cloud_links if link["url"] not in clean_content]
        if missing_links:
            clean_content += format_cloud_links_markdown(missing_links)

    if is_thin_content(clean_content, title=extracted_title, min_length=min_length, category=category) and not (
        cloud_links and len(clean_content.split()) >= 5
    ):
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
        "extraction_method": method,
    }

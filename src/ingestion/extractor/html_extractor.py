"""Module bóc tách nội dung bài viết HTML sạch cho RAG pipeline."""
from __future__ import annotations

import logging
from typing import Any
from bs4 import BeautifulSoup
import trafilatura

from ..classifier import clean_page_title
from ..utils import clean_whitespace
from .date_extractor import extract_html_date
from .html_cleaner import (
    extract_cloud_links,
    format_cloud_links_markdown,
    is_thin_content,
    table_to_markdown,
)
from .html_fallback import extract_html_fallback, fetch_html_text

logger = logging.getLogger(__name__)

__all__ = ["extract_html_content", "is_thin_content", "table_to_markdown"]


def extract_html_content(
    url: str,
    html_text: str | None = None,
    timeout: int = 10,
    min_length: int = 100,
    category: str | None = None,
) -> dict[str, Any] | None:
    """Trích xuất nội dung văn bản chính và tiêu đề từ một URL hoặc chuỗi HTML."""
    if not html_text:
        html_text = fetch_html_text(url, timeout=timeout)

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

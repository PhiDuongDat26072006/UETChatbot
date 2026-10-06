"""Module bóc tách nội dung bài viết HTML (HTML Content Extraction) sạch cho RAG pipeline."""
from __future__ import annotations

import logging
from typing import Any
import requests
from bs4 import BeautifulSoup
import trafilatura

from ..config import HEADERS
from ..classifier import clean_page_title
from ..utils import clean_whitespace
from .date_extractor import extract_html_date
from .html_cleaner import (
    NOISE_TAGS,
    PROTECTED_CLASSES,
    WP_CONTENT_SELECTORS,
    extract_cloud_links,
    format_cloud_links_markdown,
    is_thin_content,
    table_to_markdown,
)

logger = logging.getLogger(__name__)


def extract_html_content(
    url: str,
    html_text: str | None = None,
    timeout: int = 10,
    min_length: int = 100,
    category: str | None = None,
) -> dict[str, Any] | None:
    """Trích xuất nội dung văn bản chính và tiêu đề từ một URL hoặc chuỗi HTML."""
    if not html_text:
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
                verify=False,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                logger.debug("Bỏ qua URL %s vì HTTP status %d", url, resp.status_code)
                return None
            if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
                resp.encoding = resp.apparent_encoding or "utf-8"
            html_text = resp.text
        except requests.RequestException as e:
            logger.debug("Lỗi kết nối khi tải %s: %s", url, e)
            return None
        except Exception as e:
            logger.warning("Ngoại lệ không xác định khi tải %s: %s", url, e)
            return None

    if not html_text or len(html_text.strip()) < 50:
        return None

    raw_title = ""
    cloud_links: list[dict[str, str]] = []
    title_soup = None
    try:
        title_soup = BeautifulSoup(html_text, "lxml")
        title_tag = title_soup.find("title")
        if title_tag and title_tag.string:
            raw_title = clean_page_title(title_tag.get_text().strip())
        cloud_links = extract_cloud_links(title_soup)
    except Exception as e:
        logger.debug("Lỗi phân tích DOM ban đầu với URL %s: %s", url, e)

    extracted_text: str | None = None
    extracted_title: str = raw_title

    # 1. Trích xuất chính qua Trafilatura
    try:
        extracted_text = trafilatura.extract(
            html_text,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=True,
            output_format="txt",
            with_metadata=False,
            deduplicate=True,
        )
    except Exception as e:
        logger.debug("Trafilatura gặp lỗi với URL %s: %s", url, e)
        extracted_text = None

    # 2. Fallback sang BeautifulSoup
    use_fallback = (
        not extracted_text
        or len(extracted_text.strip()) < min_length
        or is_thin_content(extracted_text, title=extracted_title, min_length=min_length, category=category)
    )

    if use_fallback:
        try:
            soup = BeautifulSoup(html_text, "lxml")

            for tag_name in NOISE_TAGS:
                for element in soup.find_all(tag_name):
                    element.decompose()

            for el in soup.find_all(lambda t: t.name == "div" and (
                any(c in " ".join(t.get("class", [])).lower() for c in ["menu", "footer", "sidebar", "widget", "header_right", "topmenu", "mainmenu", "sub-menu"]) or
                any(i in str(t.get("id", "")).lower() for i in ["menu", "footer", "sidebar", "widget", "header-2", "topmenu", "mainmenu"])
            )):
                el_classes = " ".join(el.get("class", [])).lower()
                el_id = str(el.get("id", "")).lower()
                if not any(p in el_classes or p in el_id for p in PROTECTED_CLASSES):
                    el.decompose()

            if not extracted_title:
                h1_tag = soup.find("h1")
                if h1_tag:
                    extracted_title = clean_page_title(h1_tag.get_text().strip())

            content_node = None
            for selector in WP_CONTENT_SELECTORS:
                candidate = soup.select_one(selector)
                if candidate and len(candidate.get_text(strip=True)) > 20:
                    content_node = candidate
                    break

            target_node = content_node or (soup.find(id="wrapper") or (soup.body or soup))

            for table in target_node.find_all("table"):
                md_table = table_to_markdown(table)
                if md_table:
                    table.replace_with(soup.new_string(md_table))
                else:
                    table.decompose()

            for li in target_node.find_all("li"):
                li_text = li.get_text(separator=" ", strip=True)
                if li_text and not li_text.startswith(("-", "*", "•")):
                    li.string = f"\n- {li_text}\n"

            for level in range(1, 7):
                for h in target_node.find_all(f"h{level}"):
                    h_text = h.get_text(strip=True)
                    if h_text and not h_text.startswith("#"):
                        h.string = f"\n\n{'#' * level} {h_text}\n\n"

            fallback_text = target_node.get_text(separator="\n", strip=True)
            fallback_cleaned = clean_whitespace(fallback_text)

            if fallback_cleaned and (
                len(fallback_cleaned) >= min_length
                or not is_thin_content(fallback_cleaned, title=extracted_title, min_length=min_length, category=category)
                or (category in {"DAO_TAO", "CAN_BO_GIANG_VIEN"} and len(fallback_cleaned) >= 40)
            ):
                extracted_text = fallback_cleaned
        except Exception as e:
            logger.debug("BeautifulSoup fallback thất bại với %s: %s", url, e)

    if not extracted_text:
        return None

    clean_content = clean_whitespace(extracted_text)
    if cloud_links:
        clean_content += format_cloud_links_markdown(cloud_links)

    if is_thin_content(clean_content, title=extracted_title, min_length=min_length, category=category):
        return None

    published_date = extract_html_date(
        html_text=html_text,
        url=url,
        text=clean_content,
        soup=title_soup,
        category=category,
    )

    return {
        "url": url,
        "title": extracted_title or "Không có tiêu đề",
        "published_date": published_date,
        "text": clean_content,
        "text_length": len(clean_content),
        "cloud_links": cloud_links,
    }

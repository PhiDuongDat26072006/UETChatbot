"""Thuật toán tải HTML và fallback trích xuất bài viết qua BeautifulSoup."""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from ..classifier import clean_page_title
from ..config import HEADERS
from ..utils import clean_whitespace
from .html_constants import NOISE_TAGS, PROTECTED_CLASSES, WP_CONTENT_SELECTORS
from .table import table_to_markdown


def fetch_html_text(url: str, timeout: int = 10) -> str | None:
    """Tải chuỗi HTML từ URL từ xa với headers chuẩn."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, verify=False, allow_redirects=True)
        if resp.status_code != 200:
            return None
        if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except Exception:
        return None


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
        classes = el.get("class", [])
        c_str = " ".join(classes).lower() if isinstance(classes, list) else str(classes).lower()
        if any(p in c_str for p in PROTECTED_CLASSES):
            continue
        if any(bad in c_str for bad in ("sidebar", "comment", "share", "widget", "related", "banner", "popup")):
            el.decompose()

    text = clean_whitespace(main_el.get_text(separator="\n", strip=True))
    return text, title

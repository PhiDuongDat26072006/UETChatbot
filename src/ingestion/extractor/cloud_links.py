"""Trích xuất và định dạng liên kết tài liệu lưu trữ đám mây."""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from ..config.ignore_rules import CLOUD_STORAGE_DOMAINS
from ..filters import clean_url, is_cloud_storage_url
from ..utils import clean_whitespace
from .html_cleaner import GENERIC_LINK_TEXTS


def extract_cloud_links(
    html_text_or_soup: str | BeautifulSoup,
    allowed_domains: tuple[str, ...] = CLOUD_STORAGE_DOMAINS,
) -> list[dict[str, str]]:
    """Trích xuất và làm sạch danh sách liên kết tới các dịch vụ đám mây từ HTML."""
    if isinstance(html_text_or_soup, str):
        if not html_text_or_soup.strip():
            return []
        soup = BeautifulSoup(html_text_or_soup, "lxml")
    elif isinstance(html_text_or_soup, BeautifulSoup):
        soup = html_text_or_soup
    else:
        return []

    cloud_links: dict[str, dict[str, str]] = {}
    for a_tag in soup.find_all("a", href=True):
        if a_tag.find_parent(["nav", "header", "footer", "aside"]):
            continue

        raw_href = a_tag["href"].strip()
        if not raw_href or not is_cloud_storage_url(raw_href, allowed_domains):
            continue

        cleaned_url = clean_url(raw_href)
        if not cleaned_url:
            continue

        anchor_text = clean_whitespace(a_tag.get_text(separator=" ", strip=True))
        is_generic = (
            not anchor_text
            or anchor_text.lower() in GENERIC_LINK_TEXTS
            or anchor_text.startswith("http://")
            or anchor_text.startswith("https://")
        )

        if is_generic:
            context_text = ""
            if a_tag.previous_sibling:
                prev_str = a_tag.previous_sibling.get_text(strip=True) if hasattr(a_tag.previous_sibling, "get_text") else str(a_tag.previous_sibling).strip()
                if prev_str:
                    context_text = prev_str

            if not context_text and a_tag.parent and a_tag.parent.name in ("p", "li", "div", "td", "span", "h1", "h2", "h3", "h4", "h5", "h6"):
                parent_full = a_tag.parent.get_text(separator=" ", strip=True)
                if anchor_text and anchor_text in parent_full:
                    context_text = parent_full.split(anchor_text)[0].strip()
                elif not anchor_text:
                    context_text = parent_full.strip()

            if context_text:
                cleaned_ctx = re.sub(r"^[\s\-\*•\d+\.]+\s*", "", context_text).strip()
                cleaned_ctx = re.sub(r"[:：\s-]+$", "", cleaned_ctx).strip()
                if 3 <= len(cleaned_ctx) <= 100:
                    anchor_text = cleaned_ctx

        if not anchor_text or anchor_text.lower() in GENERIC_LINK_TEXTS:
            anchor_text = "Biểu mẫu đăng ký" if ("forms.gle" in cleaned_url or "/forms" in cleaned_url) else "Tài liệu đính kèm"
        else:
            anchor_text = re.sub(r"[:：\s-]+$", "", anchor_text).strip()

        if cleaned_url not in cloud_links or (cloud_links[cleaned_url]["text"] in ("Tài liệu đính kèm", "Biểu mẫu đăng ký") and anchor_text not in ("Tài liệu đính kèm", "Biểu mẫu đăng ký")):
            cloud_links[cleaned_url] = {"text": anchor_text, "url": cleaned_url}

    return list(cloud_links.values())


def format_cloud_links_markdown(cloud_links: list[dict[str, str]]) -> str:
    """Định dạng danh sách cloud links thành khối Markdown đính kèm cuối bài viết."""
    if not cloud_links:
        return ""
    lines = ["[Tài liệu & Biểu mẫu liên kết]:"]
    for item in cloud_links:
        text = item.get("text", "").strip() or "Tài liệu liên kết"
        clean_item_text = re.sub(r"[:：\s-]+$", "", text)
        lines.append(f"- {clean_item_text}: {item['url']}")
    return "\n\n" + "\n".join(lines)

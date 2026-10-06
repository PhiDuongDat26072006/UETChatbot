"""Trích xuất ngày đăng từ HTML qua Trafilatura, meta tags, URL slug và text regex."""
from __future__ import annotations

import logging
from bs4 import BeautifulSoup
import trafilatura

from .date_patterns import (
    CMS_INITIAL_SEED_DATES,
    META_DATE_PROPERTIES,
    _year_of,
    is_evergreen_page,
    normalize_date_string,
)
from .date_rules import select_best_date
from .date_text import extract_date_from_text, extract_date_from_url

logger = logging.getLogger(__name__)


def extract_html_date(
    html_text: str | None = None,
    url: str | None = None,
    text: str | None = None,
    soup: BeautifulSoup | None = None,
    category: str | None = None,
) -> str | None:
    """Trích xuất ngày đăng từ HTML qua Trafilatura, meta tags, URL slug và text regex."""
    if not html_text and not soup:
        if url:
            d = extract_date_from_url(url)
            if d:
                return d
        if text:
            d = extract_date_from_text(text, max_chars=500)
            if d:
                return d
        return None

    if not soup and html_text:
        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            soup = None

    trafilatura_date = None
    if html_text:
        try:
            meta = trafilatura.extract_metadata(html_text)
            if meta and meta.date:
                trafilatura_date = normalize_date_string(str(meta.date))
        except Exception as e:
            logger.debug("Trafilatura metadata extraction gặp lỗi: %s", e)

    meta_date = None
    if soup:
        for attr, val in META_DATE_PROPERTIES:
            tag = soup.find("meta", {attr: val})
            if tag and tag.get("content") and (norm := normalize_date_string(tag["content"])):
                meta_date = norm
                break

        if not meta_date:
            for time_tag in soup.find_all("time"):
                if (dt := time_tag.get("datetime")) and (norm := normalize_date_string(dt)):
                    meta_date = norm
                    break
                t_txt = time_tag.get_text().strip()
                if t_txt and (norm := normalize_date_string(t_txt) or extract_date_from_text(t_txt, max_chars=100)):
                    meta_date = norm
                    break

    url_date = extract_date_from_url(url) if url else None
    text_date = extract_date_from_text(text, max_chars=500) if text else None

    # Lọc bỏ ngày seed CMS rác
    meta_date = None if meta_date in CMS_INITIAL_SEED_DATES else meta_date
    trafilatura_date = None if trafilatura_date in CMS_INITIAL_SEED_DATES else trafilatura_date
    url_date = None if url_date in CMS_INITIAL_SEED_DATES else url_date
    text_date = None if text_date in CMS_INITIAL_SEED_DATES else text_date

    result = select_best_date(meta_date, trafilatura_date, url_date, text_date)

    if is_evergreen_page(url=url, category=category):
        if result in CMS_INITIAL_SEED_DATES:
            return None

    result_year = _year_of(result)
    text_year = _year_of(text_date)
    if result_year is not None and result_year < 2020 and text_year is not None and text_year >= 2020:
        return text_date

    return result

"""Thuật toán tính điểm phân loại endpoints theo Taxonomy."""
from __future__ import annotations

from urllib.parse import urlparse

from .taxonomy import (
    COMPILED_PATTERNS,
    Category,
    PRIORITY_ORDER,
    normalize_text,
)
from .title import clean_page_title, fetch_title_fast

__all__ = ["clean_page_title", "fetch_title_fast", "classify_endpoint"]


def classify_endpoint(url: str, title: str = "") -> Category:
    """Phân loại một endpoint dựa trên kết hợp URL path/slug và tiêu đề trang."""
    parsed = urlparse(url)
    path = parsed.path

    if path in ("", "/"):
        return Category.KHAC

    slug_text = path.replace('-', ' ').replace('_', ' ').replace('/', ' ')
    if parsed.query:
        slug_text += " " + parsed.query.replace('=', ' ').replace('&', ' ')

    norm_slug = normalize_text(slug_text)
    norm_title = normalize_text(title)

    scores = {cat: 0 for cat in Category}

    for cat, regexes in COMPILED_PATTERNS.items():
        for rx in regexes:
            if rx.search(norm_slug):
                scores[cat] += 4
            if title and rx.search(norm_title):
                scores[cat] += 3

    best_cat = Category.KHAC
    best_score = 0
    for cat in PRIORITY_ORDER:
        if scores[cat] > best_score:
            best_score = scores[cat]
            best_cat = cat

    return best_cat if best_score > 0 else Category.KHAC

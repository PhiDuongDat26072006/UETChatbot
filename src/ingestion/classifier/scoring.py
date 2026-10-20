"""Thuật toán trích xuất tiêu đề nhanh và tính điểm phân loại endpoints theo Taxonomy."""
from __future__ import annotations

import html
import logging
import re
from urllib.parse import urlparse

import requests

from ..config.faculties import FacultyTarget
from ..config.session import TITLE_TIMEOUT, get_source_response
from .taxonomy import (
    COMPILED_PATTERNS,
    PRIORITY_ORDER,
    TITLE_MATCH_WEIGHT,
    URL_MATCH_WEIGHT,
    Category,
    normalize_text,
)

logger = logging.getLogger(__name__)

__all__ = ["clean_page_title", "fetch_title_fast", "classify_endpoint"]


TITLE_SUFFIX_PATTERNS = (
    r'\s*[-|–—]\s*(Trường\s+)?Đại\s+học\s+Công\s+nghệ.*$',
    r'\s*[-|–—]\s*ĐHQGHN.*$',
    r'\s*[-|–—]\s*UET.*$',
    r'\s*[-|–—]\s*Khoa\s+.*$',
    r'\s*[-|–—]\s*Viện\s+.*$',
)


def clean_page_title(title: str) -> str:
    """
    Làm sạch tiêu đề trang HTML.

    Giải mã HTML entities, chuẩn hóa khoảng trắng và loại bỏ các hậu tố
    nhận diện trường học/khoa/viện để trích xuất nội dung cốt lõi của bài viết.
    """
    if not title:
        return ""
    clean_title = html.unescape(title).strip()
    clean_title = re.sub(r'\s+', ' ', clean_title)
    for pattern in TITLE_SUFFIX_PATTERNS:
        clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE)
    return clean_title.strip()


def fetch_title_fast(url: str, session: requests.Session, target: FacultyTarget | None = None) -> str:
    """Stream the HTML head and return a cleaned title, or empty if absent.

    Stop after a title, the head boundary, or 16 KiB. Transport failures raise
    so classification can save slug-based results while reporting partial failure.
    """
    domain = target["domain"] if target else ""
    prefixes = target["allowed_path_prefixes"] if target else ()
    try:
        with get_source_response(session, url, domain, prefixes, TITLE_TIMEOUT, stream=True) as response:
            response.raise_for_status()
            html_head = ""
            for chunk in response.iter_content(chunk_size=2048, decode_unicode=True):
                if not chunk:
                    continue
                html_head += chunk
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_head, re.IGNORECASE | re.DOTALL)
                if title_match:
                    return clean_page_title(title_match.group(1))
                if len(html_head) > 16384 or '</head>' in html_head.lower():
                    break
            return ""
    except Exception as error:
        logger.warning("Title request failed: %s (%s)", url, error)
        logger.debug("Title request failure", exc_info=True)
        raise


def classify_endpoint(url: str, title: str = "") -> Category:
    """
    Phân loại một endpoint vào một trong 7 danh mục Taxonomy.

    Tính điểm dựa trên kết hợp từ khóa trong URL slug và tiêu đề trang web,
    áp dụng thứ tự ưu tiên các danh mục khi có xung đột điểm.
    """
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
                scores[cat] += URL_MATCH_WEIGHT
            if title and rx.search(norm_title):
                scores[cat] += TITLE_MATCH_WEIGHT

    best_cat = Category.KHAC
    best_score = 0
    for cat in PRIORITY_ORDER:
        if scores[cat] > best_score:
            best_score = scores[cat]
            best_cat = cat

    return best_cat if best_score > 0 else Category.KHAC

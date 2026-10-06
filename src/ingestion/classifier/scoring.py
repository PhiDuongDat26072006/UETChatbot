"""Làm sạch tiêu đề và thuật toán tính điểm phân loại endpoints."""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse
import requests

from .taxonomy import (
    COMPILED_PATTERNS,
    Category,
    PRIORITY_ORDER,
    normalize_text,
)


def clean_page_title(title: str) -> str:
    """Làm sạch tiêu đề trang: giải mã HTML entity và loại bỏ hậu tố tên trường/khoa."""
    if not title:
        return ""
    t = html.unescape(title).strip()
    t = re.sub(r'\s+', ' ', t)
    suffixes = [
        r'\s*[-|–—]\s*(Trường\s+)?Đại\s+học\s+Công\s+nghệ.*$',
        r'\s*[-|–—]\s*ĐHQGHN.*$',
        r'\s*[-|–—]\s*UET.*$',
        r'\s*[-|–—]\s*Khoa\s+.*$',
        r'\s*[-|–—]\s*Viện\s+.*$',
    ]
    for suf in suffixes:
        t = re.sub(suf, '', t, flags=re.IGNORECASE)
    return t.strip()


def fetch_title_fast(url: str, session: requests.Session) -> str:
    """
    Đọc nhanh thẻ <title> của URL bằng stream chunks nhỏ (2KB-4KB),
    dừng ngay khi tìm thấy </title> hoặc </head> để tiết kiệm tối đa thời gian và băng thông.
    """
    try:
        with session.get(url, verify=False, timeout=5, stream=True, allow_redirects=True) as resp:
            if resp.status_code != 200:
                return ""
            chunks = []
            for chunk in resp.iter_content(chunk_size=2048, decode_unicode=True):
                if chunk:
                    chunks.append(chunk)
                    combined = ''.join(chunks)
                    m = re.search(r'<title[^>]*>(.*?)</title>', combined, re.IGNORECASE | re.DOTALL)
                    if m:
                        raw_title = re.sub(r'\s+', ' ', m.group(1)).strip()
                        return clean_page_title(raw_title)
                    if len(combined) > 16384 or '</head>' in combined.lower():
                        break
            combined = ''.join(chunks)
            m = re.search(r'<title[^>]*>(.*?)</title>', combined, re.IGNORECASE | re.DOTALL)
            return clean_page_title(m.group(1)) if m else ""
    except Exception:
        return ""


def classify_endpoint(url: str, title: str = "") -> Category:
    """
    Phân loại một endpoint dựa trên kết hợp:
    1. URL path / slug (trọng số cao).
    2. Nội dung tiêu đề trang (<title>).
    """
    parsed = urlparse(url)
    path = parsed.path

    # Trang chủ hoặc đường dẫn rỗng -> KHAC
    if path in ("", "/"):
        return Category.KHAC

    # Chuẩn hóa slug và title
    slug_text = path.replace('-', ' ').replace('_', ' ').replace('/', ' ')
    if parsed.query:
        slug_text += " " + parsed.query.replace('=', ' ').replace('&', ' ')

    norm_slug = normalize_text(slug_text)
    norm_title = normalize_text(title)

    scores = {cat: 0 for cat in Category}

    for cat, regexes in COMPILED_PATTERNS.items():
        for rx in regexes:
            if rx.search(norm_slug):
                scores[cat] += 4   # Điểm cao cho slug khớp trực tiếp
            if title and rx.search(norm_title):
                scores[cat] += 3   # Điểm cho title khớp

    # Chọn danh mục có điểm cao nhất theo thứ tự ưu tiên
    best_cat = Category.KHAC
    best_score = 0
    for cat in PRIORITY_ORDER:
        if scores[cat] > best_score:
            best_score = scores[cat]
            best_cat = cat

    # Nếu không có điểm nào vượt ngưỡng -> KHAC
    return best_cat if best_score > 0 else Category.KHAC

"""Trích xuất và làm sạch tiêu đề trang web phục vụ phân loại."""
from __future__ import annotations

import html
import re
import requests


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
    """Đọc nhanh thẻ <title> của URL bằng stream chunks nhỏ (2KB-4KB)."""
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

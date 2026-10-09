"""Package phân loại URL và gắn nhãn Taxonomy cho UET Chatbot."""
from __future__ import annotations

from .classification import (
    classify_all_domains,
    classify_domain,
)
from .scoring import clean_page_title, classify_endpoint, fetch_title_fast
from .taxonomy import (
    CATEGORY_PATTERNS,
    COMPILED_PATTERNS,
    PRIORITY_ORDER,
    Category,
    normalize_text,
)

__all__ = [
    "Category",
    "CATEGORY_PATTERNS",
    "COMPILED_PATTERNS",
    "PRIORITY_ORDER",
    "clean_page_title",
    "fetch_title_fast",
    "normalize_text",
    "classify_endpoint",
    "classify_domain",
    "classify_all_domains",
]

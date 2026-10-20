"""Module điều phối quy trình bóc tách văn bản (Extraction Pipeline)."""
from __future__ import annotations

from .pipeline_domain import extract_domain_documents
from .pipeline_helpers import (
    DUPLICATE_STUB_MAX_LENGTH,
    canonicalize_url,
    generate_doc_id,
    is_duplicate_stub,
    load_existing_processed_docs,
)
from .pipeline_runner import extract_all_domains

__all__ = [
    "extract_domain_documents",
    "extract_all_domains",
    "DUPLICATE_STUB_MAX_LENGTH",
    "canonicalize_url",
    "is_duplicate_stub",
    "generate_doc_id",
    "load_existing_processed_docs",
]

"""Module 1: Ingestion & Data Loader cho hệ thống trợ lý ảo UETChatbot."""

from .loader import UETDataLoader
from .crawler import crawl_domain, crawl_target, crawl_all_faculties
from .validator import verify_all_endpoints
from .classifier import Category, classify_endpoint, classify_domain, classify_all_domains
from .config import resolve_unit
from .filters import is_cloud_storage_url
from .extractor import (
    COURSE_CODE_PATTERN,
    DUPLICATE_STUB_MAX_LENGTH,
    EMAIL_PATTERN,
    canonicalize_url,
    extract_all_domains,
    extract_cloud_links,
    extract_doc_date,
    extract_document_text,
    extract_domain_documents,
    extract_html_content,
    extract_html_date,
    format_cloud_links_markdown,
    generate_document_summary,
    is_duplicate_stub,
    is_recent_document,
    is_thin_content,
    table_to_markdown,
)
from .utils import clean_whitespace
from .manifest import DataManifestTracker

__version__ = "0.8.0"
__all__ = [
    "UETDataLoader",
    "DataManifestTracker",
    "crawl_domain",
    "crawl_target",
    "crawl_all_faculties",
    "verify_all_endpoints",
    "Category",
    "classify_endpoint",
    "classify_domain",
    "classify_all_domains",
    "resolve_unit",
    "is_cloud_storage_url",
    "extract_html_content",
    "extract_cloud_links",
    "format_cloud_links_markdown",
    "is_thin_content",
    "table_to_markdown",
    "canonicalize_url",
    "is_duplicate_stub",
    "EMAIL_PATTERN",
    "COURSE_CODE_PATTERN",
    "extract_html_date",
    "extract_doc_date",
    "is_recent_document",
    "extract_document_text",
    "generate_document_summary",
    "DUPLICATE_STUB_MAX_LENGTH",
    "extract_domain_documents",
    "extract_all_domains",
    "clean_whitespace",
    "__version__",
]

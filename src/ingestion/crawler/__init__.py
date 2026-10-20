"""Package cào dữ liệu và tải tài liệu tự động cho UET Chatbot."""
from __future__ import annotations

from .downloader import download_file, download_files_parallel
from .crawl import (
    crawl_all_faculties,
    crawl_domain,
    crawl_target,
)
from .scraper import scrape_single_page
from .wp_api import (
    fetch_wp_media_documents,
    fetch_wp_posts_and_pages,
)

__all__ = [
    "download_file",
    "download_files_parallel",
    "fetch_wp_media_documents",
    "fetch_wp_posts_and_pages",
    "scrape_single_page",
    "crawl_target",
    "crawl_domain",
    "crawl_all_faculties",
]

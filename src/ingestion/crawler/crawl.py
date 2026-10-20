"""Crawl faculty sources, persist discovered URLs, and download attachments."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from ..config.faculties import FACULTY_TARGETS, FacultyTarget, faculty_storage_name, find_faculty
from ..config.paths import DATA_DIR, DATA_SOURCE_DIR
from ..config.session import create_http_session
from ..filters import is_valid_html_endpoint, normalize_endpoint_url
from ..models import generate_source_id
from .downloader import download_files_parallel
from .scraper import scrape_single_page
from .wp_api import fetch_wp_media_documents, fetch_wp_posts_and_pages

logger = logging.getLogger(__name__)
WORKERS_API = 3
WORKERS_SCRAPE = 10
MAX_CRAWL_PAGES_PER_TARGET = 150


def crawl_target(
    target_info: FacultyTarget, base_data_dir: str = DATA_DIR, download_files: bool = True,
) -> tuple[int, int]:
    """Persist a target's endpoints and attachment list in its established folder.

    HTML scope applies to seeds, API links, discovery, and redirects. Scoped
    sources collect attachments only from accepted pages, never global media.
    Successful results are saved even when individual requests fail; incomplete
    crawling or downloading then raises RuntimeError for accurate final status.
    """
    domain = target_info["domain"]
    prefixes = target_info["allowed_path_prefixes"]
    seeds = {normalize_endpoint_url(url) for url in target_info["start_urls"]}
    if not seeds or any(not is_valid_html_endpoint(url, domain, allowed_path_prefixes=prefixes) for url in seeds):
        raise ValueError(f"Invalid crawl seeds for {target_info['faculty_id']}")
    configured = find_faculty(target_info["faculty_id"])
    folder = faculty_storage_name(target_info["faculty_id"]) if configured else domain
    source_dir = Path(base_data_dir) / folder
    endpoints_dir, files_dir = source_dir / "endpoints", source_dir / "files"
    endpoints_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)
    endpoints = set(seeds)
    files: set[str] = set()
    failures = 0

    with create_http_session(pool_size=20) as session:
        if not prefixes:
            parsed = urlparse(sorted(seeds)[0])
            api_base = f"{parsed.scheme}://{parsed.netloc}"
            with ThreadPoolExecutor(max_workers=WORKERS_API) as executor:
                media = executor.submit(fetch_wp_media_documents, session, api_base)
                posts = executor.submit(fetch_wp_posts_and_pages, session, api_base, "posts", domain)
                pages = executor.submit(fetch_wp_posts_and_pages, session, api_base, "pages", domain)
                files.update(media.result())
                endpoints.update(posts.result())
                endpoints.update(pages.result())
        visited: set[str] = set()
        pending = set(endpoints)
        with ThreadPoolExecutor(max_workers=WORKERS_SCRAPE) as executor:
            while pending and len(visited) < MAX_CRAWL_PAGES_PER_TARGET:
                batch = sorted(pending)[:min(20, MAX_CRAWL_PAGES_PER_TARGET - len(visited))]
                pending.difference_update(batch)
                futures = {
                    executor.submit(scrape_single_page, session, domain, url, allowed_path_prefixes=prefixes): url
                    for url in batch
                }
                for future in as_completed(futures):
                    source_url = futures[future]
                    visited.add(source_url)
                    try:
                        page_endpoints, page_files = future.result()
                    except Exception as error:
                        failures += 1
                        logger.warning("Page request failed: %s (%s)", source_url, error)
                        logger.debug("Page failure", exc_info=True)
                        continue
                    files.update(page_files)
                    pending.update(page_endpoints - endpoints - visited)
                    endpoints.update(page_endpoints)
        logger.info("[SAVING] %s", source_dir)
        (endpoints_dir / "endpoints.txt").write_text("".join(url + "\n" for url in sorted(endpoints)), encoding="utf-8")
        (files_dir / "files_list.txt").write_text("".join(url + "\n" for url in sorted(files)), encoding="utf-8")
        try:
            source_dest_dir = Path(DATA_SOURCE_DIR)
            source_dest_dir.mkdir(parents=True, exist_ok=True)
            source_file = source_dest_dir / f"{folder}.jsonl"
            existing_sources: dict[str, dict] = {}
            if source_file.exists():
                for line in source_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            s_data = json.loads(line)
                            existing_sources[s_data["uri"]] = s_data
                        except Exception:
                            pass
            for u in sorted(endpoints):
                if u not in existing_sources:
                    existing_sources[u] = {
                        "source_id": generate_source_id("web", u),
                        "source_type": "web",
                        "uri": u,
                        "metadata": {"domain": domain, "unit": target_info.get("faculty_id", folder)},
                    }
            for f_url in sorted(files):
                if f_url not in existing_sources:
                    existing_sources[f_url] = {
                        "source_id": generate_source_id("file", f_url),
                        "source_type": "file",
                        "uri": f_url,
                        "metadata": {"domain": domain, "unit": target_info.get("faculty_id", folder)},
                    }
            temp_sf = source_file.with_suffix(".tmp")
            temp_sf.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in existing_sources.values()), encoding="utf-8")
            temp_sf.replace(source_file)
        except Exception as error:
            logger.warning("Không thể lưu sources.jsonl cho %s: %s", folder, error)

        if download_files and files:
            downloaded = download_files_parallel(session, sorted(files), str(files_dir))
            failures += len(files) - downloaded
    if failures:
        raise RuntimeError(f"{folder}: saved {len(endpoints)} endpoints; {failures} requests/downloads failed")
    return len(endpoints), len(files)


def crawl_domain(target_url: str, base_data_dir: str = DATA_DIR) -> tuple[int, int]:
    """Adapt a single URL to a configured faculty or an independent source."""
    configured = find_faculty(target_url)
    if configured:
        target = {**configured, "start_urls": [target_url]}
    else:
        parsed = urlparse(target_url)
        domain = (parsed.hostname or "").removeprefix("www.")
        target = {"faculty_id": domain.split(".")[0].upper(), "name": domain,
                  "domain": domain, "start_urls": [target_url], "allowed_path_prefixes": []}
    return crawl_target(target, base_data_dir)


def crawl_all_faculties(targets: list[FacultyTarget] | None = None, download_files: bool = True) -> None:
    """Process targets independently; raise after reporting any partial failures."""
    started = time.monotonic()
    targets = FACULTY_TARGETS if targets is None else targets
    logger.info("[START] Ingestion started")
    total_endpoints = total_files = failures = 0
    for target in targets:
        logger.info("[CRAWLING] %s (%s)", target["domain"], target["faculty_id"])
        try:
            endpoint_count, file_count = crawl_target(target, download_files=download_files)
            total_endpoints += endpoint_count
            total_files += file_count
        except Exception as error:
            failures += 1
            logger.error("[ERROR] %s: %s", target["faculty_id"], error)
            logger.debug("Crawl failure", exc_info=True)
    if failures:
        raise RuntimeError(f"Crawling incomplete: {failures} targets failed")
    logger.info("[DONE] Completed in %.1fs | %s endpoints and %s attachment URLs saved", time.monotonic() - started, total_endpoints, total_files)

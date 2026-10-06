"""Engine điều phối cào dữ liệu BFS đa tầng cho các Khoa/Viện UET."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from ..config import (
    DATA_DIR,
    MAX_CRAWL_PAGES_PER_TARGET,
    WORKERS_API,
    WORKERS_DOWNLOAD,
    WORKERS_SCRAPE,
    create_http_session,
)
from ..filters import get_domain_folder_name, normalize_endpoint_url
from .downloader import download_files_parallel
from .runner import crawl_all_faculties, crawl_domain, main
from .scraper import scrape_single_page
from .wp_api import (
    fetch_wp_category_posts,
    fetch_wp_media_documents,
    fetch_wp_posts_and_pages,
)

__all__ = ["crawl_target", "crawl_domain", "crawl_all_faculties", "scrape_single_page", "main"]


def crawl_target(
    target_info: dict, base_data_dir: str = DATA_DIR, download_files: bool = True
) -> tuple[int, int]:
    """Thu thập endpoints và tài liệu cho một Khoa/Viện cụ thể."""
    session = create_http_session(pool_size=20)
    target_url = target_info["url"]
    domain = target_info.get("domain") or urlparse(target_url).netloc
    subpath, category_id = target_info.get("subpath"), target_info.get("category_id")
    folder_name = target_info.get("domain") or get_domain_folder_name(target_url)

    ep_dir = os.path.join(base_data_dir, folder_name, "endpoints")
    files_dir = os.path.join(base_data_dir, folder_name, "files")
    os.makedirs(ep_dir, exist_ok=True)
    os.makedirs(files_dir, exist_ok=True)

    all_eps, all_files = set(), set()
    if category_id:
        base_api = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"
        all_eps.update(fetch_wp_category_posts(session, base_api, category_id, max_pages=5))
    else:
        with ThreadPoolExecutor(max_workers=WORKERS_API) as ex:
            f_med = ex.submit(fetch_wp_media_documents, session, target_url)
            f_pst = ex.submit(fetch_wp_posts_and_pages, session, target_url, "posts", domain)
            f_pge = ex.submit(fetch_wp_posts_and_pages, session, target_url, "pages", domain)
            all_files.update(f_med.result())
            all_eps.update(f_pst.result().union(f_pge.result()))

    norm_target = normalize_endpoint_url(target_url)
    all_eps.add(norm_target)
    if subpath:
        all_eps.add(normalize_endpoint_url(f"{urlparse(target_url).scheme}://{domain}/category/{subpath}/"))

    visited, to_visit = set(), set(all_eps)
    while to_visit and len(visited) < MAX_CRAWL_PAGES_PER_TARGET:
        batch = list(to_visit)[:min(20, MAX_CRAWL_PAGES_PER_TARGET - len(visited))]
        to_visit -= set(batch)
        with ThreadPoolExecutor(max_workers=WORKERS_SCRAPE) as ex:
            futs = {ex.submit(scrape_single_page, session, target_url, domain, u, subpath): u for u in batch}
            for fut in as_completed(futs):
                visited.add(futs[fut])
                eps, fls = fut.result()
                all_files.update(fls)
                for ep in eps:
                    if ep not in all_eps:
                        all_eps.add(ep)
                        if ep not in visited:
                            to_visit.add(ep)

    with open(os.path.join(ep_dir, "endpoints.txt"), "w", encoding="utf-8") as f:
        f.writelines(ep + "\n" for ep in sorted(all_eps))
    with open(os.path.join(files_dir, "files_list.txt"), "w", encoding="utf-8") as f:
        f.writelines(fl + "\n" for fl in sorted(all_files))

    if download_files and all_files:
        download_files_parallel(session, sorted(list(all_files)), files_dir, max_workers=WORKERS_DOWNLOAD)

    return len(all_eps), len(all_files)

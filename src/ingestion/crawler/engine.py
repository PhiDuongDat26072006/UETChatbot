"""Engine điều phối cào dữ liệu BFS đa tầng cho các Khoa/Viện UET."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse
import requests

from ..config import (
    A_HREF_REGEX,
    DATA_DIR,
    EMBED_SRC_REGEX,
    FACULTY_TARGETS,
    MAX_CRAWL_PAGES_PER_TARGET,
    REQUEST_TIMEOUT,
    WORKERS_API,
    WORKERS_DOWNLOAD,
    WORKERS_SCRAPE,
    create_http_session,
)
from ..filters import (
    clean_url,
    get_domain_folder_name,
    is_allowed_file_domain,
    is_document_file,
    is_media_or_asset_file,
    is_valid_html_endpoint,
    normalize_endpoint_url,
)
from .downloader import download_files_parallel
from .wp_api import (
    fetch_wp_category_posts,
    fetch_wp_media_documents,
    fetch_wp_posts_and_pages,
)


def scrape_single_page(
    session: requests.Session,
    base_url: str,
    target_domain: str,
    url: str,
    subpath: str = None
) -> tuple[set[str], set[str]]:
    """Cào nội dung trang: chỉ bóc tách <a href="..."> và nhúng tài liệu, lọc bỏ hoàn toàn URL rác."""
    endpoints = set()
    files = set()
    try:
        res = session.get(url, timeout=REQUEST_TIMEOUT, verify=False)
        if res.status_code == 200 and "text/html" in res.headers.get("Content-Type", ""):
            # 1. Trích xuất tất cả liên kết thẻ <a>
            raw_links = A_HREF_REGEX.findall(res.text)
            for raw_link in raw_links:
                raw_link = raw_link.strip()
                if not raw_link or raw_link.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
                    continue

                full_url = clean_url(urljoin(base_url, raw_link))

                if is_document_file(full_url):
                    if is_allowed_file_domain(full_url, target_domain):
                        files.add(full_url)
                elif is_media_or_asset_file(full_url):
                    continue
                else:
                    norm_url = normalize_endpoint_url(full_url)
                    if is_valid_html_endpoint(norm_url, target_domain, subpath):
                        endpoints.add(norm_url)

            # 2. Tìm file tài liệu nếu có trong iframe hoặc embed
            embed_links = EMBED_SRC_REGEX.findall(res.text)
            for emb in embed_links:
                full_emb = clean_url(urljoin(base_url, emb.strip()))
                if is_document_file(full_emb) and is_allowed_file_domain(full_emb, target_domain):
                    files.add(full_emb)

    except Exception:
        pass
    return endpoints, files


def crawl_target(
    target_info: dict,
    base_data_dir: str = DATA_DIR,
    download_files: bool = True
) -> tuple[int, int]:
    """Thu thập endpoints và tài liệu cho một Khoa/Viện cụ thể."""
    session = create_http_session(pool_size=20)

    code = target_info.get("code", "")
    name = target_info.get("name", "")
    target_url = target_info["url"]
    domain = target_info.get("domain") or urlparse(target_url).netloc
    subpath = target_info.get("subpath")
    category_id = target_info.get("category_id")

    folder_name = target_info.get("domain") or get_domain_folder_name(target_url)

    print(f"\n==================================================")
    print(f"[*] Đang xử lý [{code}]: {name}")
    print(f"    URL gốc: {target_url}")
    print(f"==================================================")

    # Thư mục đích trong data/raw_data/<folder_name>/
    domain_data_dir = os.path.join(base_data_dir, folder_name)
    endpoints_dir = os.path.join(domain_data_dir, "endpoints")
    files_dir = os.path.join(domain_data_dir, "files")

    os.makedirs(endpoints_dir, exist_ok=True)
    os.makedirs(files_dir, exist_ok=True)

    all_endpoints = set()
    all_files = set()
    initial_pages = set()

    # 1. Quét WordPress REST API (nếu site có hỗ trợ)
    print("[1/3] Quét dữ liệu qua WordPress REST API (nếu có)...")
    if category_id:
        base_api_url = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"
        cat_posts = fetch_wp_category_posts(session, base_api_url, category_id, max_pages=5)
        all_endpoints.update(cat_posts)
        initial_pages.update(cat_posts)
        print(f"    -> Tìm thấy {len(cat_posts)} bài viết qua WP Category {category_id}.")
    else:
        with ThreadPoolExecutor(max_workers=WORKERS_API) as executor:
            future_media = executor.submit(fetch_wp_media_documents, session, target_url)
            future_posts = executor.submit(fetch_wp_posts_and_pages, session, target_url, "posts", domain)
            future_pages = executor.submit(fetch_wp_posts_and_pages, session, target_url, "pages", domain)

            media_files = future_media.result()
            all_files.update(media_files)

            posts_and_pages = future_posts.result().union(future_pages.result())
            all_endpoints.update(posts_and_pages)
            initial_pages.update(posts_and_pages)

        if initial_pages or all_files:
            print(f"    -> Tìm thấy qua API: {len(initial_pages)} bài viết/trang, {len(all_files)} file tài liệu.")
        else:
            print("    -> Site không mở API công khai, chuyển hoàn toàn sang cào HTML liên kết.")

    norm_target = normalize_endpoint_url(target_url)
    all_endpoints.add(norm_target)
    if subpath:
        cat_url = f"{urlparse(target_url).scheme}://{domain}/category/{subpath}/"
        norm_cat = normalize_endpoint_url(cat_url)
        all_endpoints.add(norm_cat)

    # 2. Cào trang HTML đa tầng (Breadth-First Search đa luồng)
    visited = set()
    to_visit = set(all_endpoints)
    to_visit.add(norm_target)

    print(f"[2/3] Bóc tách nội dung HTML đa luồng (tối đa {MAX_CRAWL_PAGES_PER_TARGET} trang)...")
    round_idx = 1

    while to_visit and len(visited) < MAX_CRAWL_PAGES_PER_TARGET:
        batch_size = min(20, MAX_CRAWL_PAGES_PER_TARGET - len(visited))
        current_batch = list(to_visit)[:batch_size]
        to_visit -= set(current_batch)

        with ThreadPoolExecutor(max_workers=WORKERS_SCRAPE) as executor:
            futures = {
                executor.submit(scrape_single_page, session, target_url, domain, u, subpath): u
                for u in current_batch
            }
            for future in as_completed(futures):
                u = futures[future]
                visited.add(u)
                eps, fls = future.result()
                all_files.update(fls)

                for ep in eps:
                    if ep not in all_endpoints:
                        all_endpoints.add(ep)
                        if ep not in visited:
                            to_visit.add(ep)

        print(f"    -> Vòng {round_idx}: đã cào {len(visited)} trang | Tìm thấy {len(all_endpoints)} endpoints, {len(all_files)} files...")
        round_idx += 1

    # 3. Lưu kết quả ra file txt
    print("[3/3] Lưu danh sách URL vào endpoints/endpoints.txt...")

    ep_domain_file = os.path.join(endpoints_dir, "endpoints.txt")
    with open(ep_domain_file, "w", encoding="utf-8") as f:
        for ep in sorted(all_endpoints):
            f.write(ep + "\n")

    files_list_file = os.path.join(files_dir, "files_list.txt")
    with open(files_list_file, "w", encoding="utf-8") as f:
        for fl in sorted(all_files):
            f.write(fl + "\n")

    print(f"    [✓] Đã lưu {len(all_endpoints)} endpoints tại: {ep_domain_file}")
    print(f"    [✓] Đã lưu danh sách {len(all_files)} files tại: {files_list_file}")

    if download_files and all_files:
        file_list = sorted(list(all_files))
        download_files_parallel(session, file_list, files_dir, max_workers=WORKERS_DOWNLOAD)

    return len(all_endpoints), len(all_files)


def crawl_domain(target_url: str, base_data_dir: str = DATA_DIR):
    """Cào dữ liệu cho 1 domain đơn lẻ (hỗ trợ tương thích ngược)."""
    target_info = {
        "url": target_url,
        "domain": get_domain_folder_name(target_url),
        "code": get_domain_folder_name(target_url).split(".")[0].upper(),
        "name": target_url,
    }
    return crawl_target(target_info, base_data_dir=base_data_dir)


def crawl_all_faculties(targets: list[dict] = FACULTY_TARGETS, download_files: bool = True):
    """Chạy thu thập toàn bộ các Khoa và Viện trực thuộc UET."""
    start_time = time.time()
    summary = []

    print("\n" + "=" * 60)
    print(f"[*] BẮT ĐẦU CÀO DỮ LIỆU {len(targets)} KHOA & VIỆN TRỰC THUỘC UET")
    print("=" * 60)

    for target in targets:
        code = target.get("code", "")
        name = target.get("name", "")
        t0 = time.time()
        ep_count, fl_count = crawl_target(target, download_files=download_files)
        elapsed = round(time.time() - t0, 2)
        summary.append({
            "code": code,
            "name": name,
            "domain": target.get("domain"),
            "endpoints": ep_count,
            "files": fl_count,
            "time": elapsed
        })

    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 60)
    print(f"[✓] TỔNG KẾT HOÀN THÀNH TRONG {total_time} GIÂY")
    print("=" * 60)
    print(f"{'Mã':<8} | {'Tên Khoa/Viện':<36} | {'Endpoints':<10} | {'Files':<6} | {'Thời gian'}")
    print("-" * 75)
    total_eps = 0
    total_fls = 0
    for s in summary:
        total_eps += s["endpoints"]
        total_fls += s["files"]
        print(f"{s['code']:<8} | {s['name']:<36} | {s['endpoints']:<10} | {s['files']:<6} | {s['time']}s")
    print("-" * 75)
    print(f"{'TỔNG':<8} | {len(targets)} Khoa/Viện{'':<24} | {total_eps:<10} | {total_fls:<6} | {total_time}s")


def main():
    """Hàm khởi chạy chính."""
    crawl_all_faculties(FACULTY_TARGETS, download_files=True)

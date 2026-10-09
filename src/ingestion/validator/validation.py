"""Validate endpoint files, preserving slash recovery and soft-404 detection."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

from ..config.faculties import FACULTY_TARGETS, FacultyTarget, faculty_storage_name, find_faculty
from ..config.ignore_rules import IGNORED_PATH_REGEX
from ..config.paths import DATA_DIR
from ..config.session import VALIDATION_RETRIES, VALIDATION_TIMEOUT, create_http_session, get_source_response
from ..filters import clean_url, is_valid_html_endpoint

logger = logging.getLogger(__name__)


def get_alternative_slash_url(url: str) -> str:
    """Toggle a non-root path's trailing slash without altering its query."""
    parsed = urlparse(url)
    if not parsed.path or parsed.path == "/":
        return url
    path = parsed.path.rstrip("/") if parsed.path.endswith("/") else parsed.path + "/"
    return urlunparse(parsed._replace(path=path))


def is_page_not_found(response: requests.Response) -> bool:
    """Recognize HTTP 404 and known soft-404 URLs or HTML titles."""
    if response.status_code == 404:
        return True
    if response.status_code != 200:
        return False
    path = urlparse(response.url).path.rstrip("/").lower()
    if path in ("/page/404", "/404", "/error", "/not-found", "/error-404"):
        return True
    return any(pattern in response.text[:6000].lower() for pattern in (
        "<title>page not found", "<title>404 not found", "<title>404 - not found",
        "<title>không tìm thấy", "<title>lỗi 404",
    ))


def check_and_fix_endpoint(
    url: str, session: requests.Session, target: FacultyTarget | None = None,
) -> tuple[str, str | None, str]:
    """Return (original, accepted URL or None, status), retaining transient errors.

    Try HTTPS first, then the alternate slash, then the original HTTP variants
    after a 404. Accept same-host redirect normalization. Transient failures keep
    the original endpoint; configured scopes prevent unrelated redirects.
    """
    original_url = url.strip()
    if not original_url:
        return original_url, None, "EMPTY"
    if IGNORED_PATH_REGEX.search(urlparse(original_url).path):
        return original_url, None, "DROPPED_IGNORED"
    if target and not is_valid_html_endpoint(original_url, target["domain"], allowed_path_prefixes=target["allowed_path_prefixes"]):
        return original_url, None, "DROPPED_SCOPE"
    https_url = "https://" + original_url[7:] if original_url.startswith("http://") else original_url
    candidates = [(https_url, "OK")]
    alternate_url = get_alternative_slash_url(https_url)
    if alternate_url != https_url:
        candidates.append((alternate_url, "CORRECTED_SLASH"))
    if https_url != original_url:
        candidates.append((original_url, "OK"))
        alternate_original = get_alternative_slash_url(original_url)
        if alternate_original != original_url:
            candidates.append((alternate_original, "CORRECTED_SLASH"))
    with ExitStack() as responses:
        for index, (candidate, status) in enumerate(candidates):
            try:
                domain = target["domain"] if target else urlparse(candidate).netloc
                prefixes = target["allowed_path_prefixes"] if target else ()
                response = responses.enter_context(get_source_response(session, candidate, domain, prefixes, VALIDATION_TIMEOUT))
                if is_page_not_found(response):
                    continue
                if https_url != original_url and candidate.startswith("http://"):
                    return original_url, candidate, status
                same_host = urlparse(response.url).netloc.lower().removeprefix("www.") == urlparse(candidate).netloc.lower().removeprefix("www.")
                return original_url, clean_url(response.url) if same_host else candidate, status
            except Exception as error:
                logger.debug("Validation request failed: %s", candidate, exc_info=True)
                if index == 0:
                    status = "KEPT_TIMEOUT" if isinstance(error, (requests.Timeout, requests.ConnectionError)) else "KEPT_ERROR"
                    logger.warning("Validation unavailable: %s (%s)", candidate, error)
                    return original_url, original_url, status
    return original_url, None, "DROPPED_DEAD"


def verify_endpoint_file(file_path: str, max_workers: int = 10, target: FacultyTarget | None = None) -> dict:
    """Replace a URL list with sorted, deduplicated accepted endpoints.

    Missing files report zero results. Transient request failures retain URLs
    and are counted separately so callers can report incomplete verification.
    """
    path = Path(file_path)
    result = {"total": 0, "valid": 0, "corrected": 0, "dropped": 0, "final": 0, "failed": 0}
    if not path.exists():
        return result
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result["total"] = len(urls)
    final_urls: set[str] = set()
    target = target or find_faculty(path.parents[1].name)
    with create_http_session(pool_size=15, max_retries=VALIDATION_RETRIES) as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(check_and_fix_endpoint, url, session, target) for url in urls]
            for future in as_completed(futures):
                _, accepted_url, status = future.result()
                if status.startswith("KEPT_"):
                    result["failed"] += 1
                if accepted_url:
                    final_urls.add(accepted_url)
                    result["corrected" if status == "CORRECTED_SLASH" else "valid"] += 1
                else:
                    result["dropped"] += 1
    path.write_text("".join(url + "\n" for url in sorted(final_urls)), encoding="utf-8")
    result["final"] = len(final_urls)
    logger.info("[SAVING] %s | %s endpoints", path, result["final"])
    return result


def verify_all_endpoints() -> None:
    """Verify each faculty independently; never copy files between source folders."""
    started = time.monotonic()
    logger.info("[START] Endpoint verification started")
    total = failures = 0
    for target in FACULTY_TARGETS:
        path = Path(DATA_DIR) / faculty_storage_name(target["faculty_id"]) / "endpoints" / "endpoints.txt"
        if not path.exists():
            logger.warning("Missing endpoint file: %s", path)
            failures += 1
            continue
        logger.info("[VALIDATING] %s (%s)", target["domain"], target["faculty_id"])
        try:
            result = verify_endpoint_file(str(path), target=target)
            total += result["final"]
            failures += result["failed"] > 0
        except Exception as error:
            failures += 1
            logger.error("[ERROR] %s: %s", target["faculty_id"], error)
            logger.debug("Validation failure", exc_info=True)
    if failures:
        raise RuntimeError(f"Verification incomplete: {failures} targets failed")
    logger.info("[DONE] Completed in %.1fs | %s endpoints saved", time.monotonic() - started, total)

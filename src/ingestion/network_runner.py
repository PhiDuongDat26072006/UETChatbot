"""Run isolated site coordinators with one shared HTTP concurrency budget."""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from copy import copy
from multiprocessing import get_context
import logging
from time import monotonic
from uuid import uuid4
from .config.profiles import load_profiles
from .downloader.http import HTMLFetcher
from .downloader.limits import RequestLimits
from .storage import writer_lock, write_json, now

log = logging.getLogger(__name__)


def run_network(options, run_serial):
    options = copy(options)
    options.stream_extract_workers = (
        getattr(options, "stream_extract_workers", None) or options.download_workers
    )
    extractor_pool = None
    profiles = load_profiles(options.profile_dir, options.sites)
    fetcher = HTMLFetcher(
        options.timeout, options.max_html_bytes, options.retries, options.delay
    )
    fetcher.limits = RequestLimits(options.download_workers, options.per_host, profiles)
    workers = min(getattr(options, "site_workers", 3), len(profiles))
    if options.limit is not None:
        workers = 1
        log.info(
            "--limit: running sites sequentially to preserve the global document budget"
        )
    report = {
        "run_id": str(uuid4()),
        "started_at": now(),
        "from": options.from_stage,
        "to": options.to_stage,
        "sites": {},
        "errors": [],
    }
    started = monotonic()
    try:
        with writer_lock(options.data_dir):
            try:
                if options.to_stage == "processed":
                    extractor_pool = ProcessPoolExecutor(
                        max_workers=options.stream_extract_workers,
                        mp_context=get_context("spawn"),
                    )
                    report["extract_processes"] = options.stream_extract_workers
                    log.info(
                        "Streaming extraction: shared pool with %s processes",
                        options.stream_extract_workers,
                    )
                if workers == 1:
                    report.update(
                        run_serial(
                            options,
                            shared_fetcher=fetcher,
                            managed=True,
                            extractor_pool=extractor_pool,
                        )
                    )
                else:

                    def run_site(site):
                        local = copy(options)
                        local.sites = [site]
                        return run_serial(
                            local,
                            shared_fetcher=fetcher,
                            managed=True,
                            extractor_pool=extractor_pool,
                        )

                    log.info(
                        "Network: %s sites, %s total HTTP slots, %s per host group",
                        workers,
                        options.download_workers,
                        options.per_host,
                    )
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {
                            pool.submit(run_site, site): site for site in profiles
                        }
                        for future in as_completed(futures):
                            site = futures[future]
                            try:
                                result = future.result()
                                report["sites"].update(result["sites"])
                                report["errors"].extend(result["errors"])
                            except Exception as exc:
                                report["sites"][site] = {"status": "failed"}
                                report["errors"].append(
                                    {"site": site, "stage": "fatal", "error": str(exc)}
                                )
            except BaseException as exc:
                report["errors"].append(
                    {"stage": "fatal", "error": str(exc) or type(exc).__name__}
                )
                raise
            finally:
                if extractor_pool is not None:
                    extractor_pool.shutdown(wait=True, cancel_futures=True)
                report.update(
                    finished_at=now(),
                    elapsed_seconds=round(monotonic() - started, 3),
                    http_requests=fetcher.requests,
                    download_bytes=fetcher.bytes,
                    status="failed" if report["errors"] else "success",
                )
                write_json(
                    options.data_dir
                    / "ingestion_reports"
                    / (report["run_id"] + ".json"),
                    report,
                )
    finally:
        fetcher.close()
    return report

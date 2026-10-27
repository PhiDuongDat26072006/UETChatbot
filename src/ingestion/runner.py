"""Stage coordinator; worker threads calculate, one coordinator writes."""

from contextlib import nullcontext
import logging
from collections import Counter
from uuid import uuid4
from src.base import DataSource
from .config.profiles import load_profiles
from .crawler.engine import crawl
from .crawler.policy import evaluate, reconcile_sources
from .crawler.parser import metadata as extract_metadata
from .downloader.http import HTMLFetcher
from .downloader.service import download_source
from .extractor.pipeline import process_raw, processing_key
from .storage import Storage, writer_lock, write_json, now
from .utils import completion_map

log = logging.getLogger(__name__)
STAGES = ("web", "data_source", "raw", "processed")


def run_ingestion_serial(
    options, shared_fetcher=None, managed=False, extractor_pool=None
):
    profiles = load_profiles(options.profile_dir, options.sites)
    storage = Storage(options.data_dir)
    report = {
        "run_id": str(uuid4()),
        "started_at": now(),
        "from": options.from_stage,
        "to": options.to_stage,
        "sites": {},
        "errors": [],
    }
    fetcher = shared_fetcher or HTMLFetcher(
        options.timeout, options.max_html_bytes, options.retries, options.delay
    )
    selected = 0
    processed_selected = 0
    stream = None
    with nullcontext() if managed else writer_lock(storage.root):
        try:
            for site, profile in profiles.items():
                counts = Counter()
                report["sites"][site] = counts
                counts["sources_reclassified"] = reconcile_sources(
                    storage, site, profile
                )

                if extractor_pool is not None:
                    from .streaming import StreamingExtractor

                    budget = (
                        None
                        if options.limit is None
                        else max(0, options.limit - processed_selected)
                    )
                    capacity = options.stream_extract_workers
                    stream = StreamingExtractor(
                        storage,
                        site,
                        profile,
                        extractor_pool,
                        capacity,
                        options,
                        counts,
                        report["errors"],
                        budget,
                    )
                    with storage.batched():
                        storage.flush_pending()
                        stream.on_commit()

                def accept(source, page):
                    if STAGES.index(options.to_stage) >= 2:
                        record = storage.save_raw(source, page)
                        counts["raw_saved"] += 1
                        # Remove stale processed immediately; never serve a previous raw revision.
                        old = storage.records("processed", site).get(source.source_id)
                        if old and old["raw_data_id"] != record["id"]:
                            storage.remove_processed(site, source.source_id)
                        if stream is not None:
                            stream.stage(record)

                if options.from_stage == "web":
                    import copy

                    local = copy.copy(options)
                    if options.limit is not None:
                        local.limit = max(0, options.limit - selected)
                    summary = crawl(
                        profile,
                        storage,
                        fetcher,
                        accept,
                        local,
                        on_commit=stream.on_commit if stream else None,
                        on_tick=stream.poll if stream else None,
                    )
                    selected += summary["documents"]
                    counts.update({"crawl_" + k: v for k, v in summary.items()})
                    if summary["errors"]:
                        report["errors"].append(
                            {"site": site, "stage": "crawl", "count": summary["errors"]}
                        )
                elif options.from_stage == "data_source":
                    sources = [
                        DataSource.model_validate(r)
                        for r in storage.records("sources", site).values()
                        if r["metadata"].get("decision") == "accepted"
                    ]
                    sources = sorted(sources, key=lambda s: s.uri)
                    latest_raw = {}
                    for row in storage.records("raw", site).values():
                        sid = row["metadata"]["source_id"]
                        if (
                            sid not in latest_raw
                            or row["created_at"] > latest_raw[sid]["created_at"]
                        ):
                            latest_raw[sid] = row
                    jobs = []
                    for source in sources:
                        state = storage.records("state", site).get(
                            "raw:" + source.source_id, {}
                        )
                        if options.retry_failed and state.get("status") != "failed":
                            continue
                        previous = latest_raw.get(source.source_id)
                        if previous:
                            try:
                                storage.load_raw(previous)
                            except (OSError, ValueError):
                                previous = None
                        if (
                            previous
                            and not options.refresh
                            and not options.retry_failed
                        ):
                            counts["raw_skipped"] += 1
                            continue
                        if options.limit is not None and selected >= options.limit:
                            break
                        jobs.append((source, previous))
                        selected += 1

                    def download(job):
                        source, previous = job
                        try:
                            return (
                                source,
                                download_source(
                                    source,
                                    profile,
                                    fetcher,
                                    previous,
                                    options.refresh or options.retry_failed,
                                ),
                                None,
                            )
                        except Exception as exc:
                            return source, None, str(exc)

                    with storage.batched():
                        for completed, (_, (source, page, error)) in enumerate(
                            completion_map(
                                download,
                                jobs,
                                min(options.download_workers, options.per_host),
                                on_tick=stream.poll if stream else None,
                            ),
                            1,
                        ):
                            if error:
                                counts["raw_failed"] += 1
                                storage.state(
                                    site,
                                    source.source_id,
                                    "raw",
                                    status="failed",
                                    error=error,
                                )
                                report["errors"].append(
                                    {
                                        "site": site,
                                        "source_id": source.source_id,
                                        "stage": "raw",
                                        "error": error,
                                    }
                                )
                            elif page is None or page["status_code"] == 304:
                                counts["raw_unchanged"] += 1
                                storage.state(
                                    site, source.source_id, "raw", status="completed"
                                )
                            else:
                                meta = extract_metadata(
                                    page["body"].decode(
                                        page["encoding"], errors="replace"
                                    ),
                                    profile,
                                )
                                decision, reason = evaluate(
                                    page["final_url"], profile, meta
                                )
                                source.metadata.update(
                                    meta, decision=decision, decision_reason=reason
                                )
                                source = storage.resolve_redirect(source, page, profile)
                                storage.save_source(source)
                                if decision == "accepted":
                                    accept(source, page)
                                else:
                                    storage.remove_processed(site, source.source_id)
                                    storage.state(
                                        site,
                                        source.source_id,
                                        "raw",
                                        status="skipped",
                                        reason=reason,
                                    )
                                    counts["raw_rejected"] += 1
                            if completed % getattr(options, "flush_every", 20) == 0:
                                storage.flush_pending()
                                if stream is not None:
                                    stream.on_commit()
                                    storage.flush_pending()
                if stream is not None:
                    processed_selected += stream.finish()
                    stream = None
                elif options.to_stage == "processed":
                    latest = {}
                    for row in storage.records("raw", site).values():
                        sid = row["metadata"]["source_id"]
                        if (
                            sid not in latest
                            or row["created_at"] > latest[sid]["created_at"]
                        ):
                            latest[sid] = row
                    jobs = []
                    for sid, row in sorted(latest.items()):
                        source = storage.records("sources", site).get(sid)
                        if source and source["metadata"].get("decision") != "accepted":
                            storage.remove_processed(site, sid)
                            continue
                        state = storage.records("state", site).get(
                            "processed:" + sid, {}
                        )
                        if options.retry_failed and state.get("status") != "failed":
                            continue
                        key = processing_key(row, profile)
                        existing = storage.records("processed", site).get(sid)
                        if not options.reprocess and state.get("key") == key:
                            if state.get("status") == "review" or (
                                existing and existing["raw_data_id"] == row["id"]
                            ):
                                counts["processed_skipped"] += 1
                                continue
                        if (
                            options.limit is not None
                            and processed_selected >= options.limit
                        ):
                            break
                        processed_selected += 1
                        jobs.append((sid, row, key))

                    def process(job):
                        sid, row, key = job
                        try:
                            return (
                                job,
                                process_raw(storage.load_raw(row), profile),
                                None,
                            )
                        except Exception as exc:
                            return job, None, str(exc)

                    with storage.batched():
                        for completed, (
                            _,
                            ((sid, row, key), result, error),
                        ) in enumerate(
                            completion_map(process, jobs, options.extract_workers), 1
                        ):
                            if error:
                                storage.remove_processed(site, sid)
                                storage.state(
                                    site,
                                    sid,
                                    "processed",
                                    status="failed",
                                    key=key,
                                    error=error,
                                )
                                report["errors"].append(
                                    {
                                        "site": site,
                                        "source_id": sid,
                                        "stage": "processed",
                                        "error": error,
                                    }
                                )
                                counts["processed_failed"] += 1
                            else:
                                document, quality = result
                                if quality["status"] == "accepted":
                                    storage.save_processed(site, document)
                                    status = "completed"
                                    counts["processed_accepted"] += 1
                                else:
                                    storage.remove_processed(site, sid)
                                    review = (
                                        storage.root
                                        / "ingestion_reports"
                                        / "review"
                                        / site
                                        / (sid + ".json")
                                    )
                                    write_json(review, document.model_dump(mode="json"))
                                    status = "review"
                                    counts["processed_review"] += 1
                                storage.state(
                                    site,
                                    sid,
                                    "processed",
                                    status=status,
                                    key=key,
                                    quality=quality,
                                    raw_id=row["id"],
                                )
                            if completed % options.flush_every == 0:
                                storage.flush_pending()
                log.info("[%s] %s", site, dict(counts))
        except BaseException as exc:
            report["errors"].append(
                {"stage": "fatal", "error": str(exc) or type(exc).__name__}
            )
            raise
        finally:
            if stream is not None:
                stream.cancel()
            if shared_fetcher is None:
                fetcher.close()
            report.update(
                finished_at=now(),
                http_requests=fetcher.requests,
                download_bytes=fetcher.bytes,
                status="failed" if report["errors"] else "success",
            )
            if not managed:
                write_json(
                    storage.root / "ingestion_reports" / (report["run_id"] + ".json"),
                    report,
                )
    return report


def get_status(data_dir, profile_dir=None, sites=None):
    storage = Storage(data_dir)
    return {
        site: {
            kind: len(storage.records(kind, site))
            for kind in ("sources", "raw", "processed")
        }
        for site in load_profiles(profile_dir, sites)
    }


def run_ingestion(options):
    # Network stages share a pool; extraction streams only when requested.
    from .network_runner import run_network

    if options.from_stage in {"web", "data_source"}:
        return run_network(options, run_ingestion_serial)
    return run_ingestion_serial(options)

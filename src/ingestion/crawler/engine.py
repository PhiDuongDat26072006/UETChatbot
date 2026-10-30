"""Level-order discovery with bounded HTTP work and replayable checkpoints."""

from collections import deque
from contextlib import closing
import json
import logging
from src.base import DataSource
from ..identity import normalize_site_url, source_id, digest
from ..storage import write_json
from ..utils import completion_map
from .parser import parse_page
from .policy import in_scope, evaluate, resource_type

log = logging.getLogger(__name__)


def crawl(profile, storage, fetcher, on_source, options, on_commit=None, on_tick=None):
    site = profile["site"]
    checkpoint = storage.root / "ingestion_state" / (site + "-crawl.json")
    key = digest(json.dumps(profile, sort_keys=True, default=str).encode())
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    resume = (
        saved.get("profile_hash") == key
        and saved.get("target") == options.to_stage
        and not options.refresh
    )
    if resume and saved.get("complete"):
        return {"pages": 0, "documents": 0, "errors": 0, "resumed_complete": True}
    items = (
        saved.get("frontier", [])
        if resume
        else [(normalize_site_url(u, profile), 0, None) for u in profile["seeds"]]
    )
    frontier = deque(sorted((tuple(x) for x in items), key=lambda x: x[1]))
    seen = set(saved.get("seen", [])) if resume else set()
    queued = {u for u, _, _ in frontier}
    active = {}
    deferred = []
    pages = documents = errors = 0
    flush_every = getattr(options, "flush_every", 20)

    def save():
        storage.flush_pending()
        if on_commit is not None:
            on_commit()
            storage.flush_pending()
        # Includes submitted, completed-but-uncommitted and not-yet-submitted jobs.
        pending = list(frontier) + list(active.values()) + deferred
        pending = sorted({x[0]: x for x in pending}.values(), key=lambda x: x[1])
        write_json(
            checkpoint,
            {
                "profile_hash": key,
                "target": options.to_stage,
                "frontier": pending,
                "seen": sorted(seen),
                "complete": not pending,
            },
        )

    def fetch(item):
        try:
            return fetcher.fetch(item[0], lambda u: in_scope(u, profile)), None
        except Exception as exc:
            return None, str(exc)

    def consume(item, page, error):
        nonlocal pages, documents, errors
        url, depth, parent = item
        pages += 1
        identifier = source_id(url)
        if error:
            errors += 1
            storage.state(
                site, identifier, "crawl", status="failed", url=url, error=error
            )
            deferred.append(item)
            log.warning("[%s] crawl failed %s: %s", site, url, error)
            return
        html = page["body"].decode(page["encoding"], errors="replace")
        meta, links = parse_page(html, page["final_url"], profile)
        decision, reason = evaluate(page["final_url"], profile, meta)
        source = DataSource(
            source_id=identifier,
            source_type="web",
            uri=url,
            metadata={
                **meta,
                "site": site,
                "faculty": site,
                "resource_type": "html",
                "discovered_from": parent,
                "crawl_depth": depth,
                "decision": decision,
                "decision_reason": reason,
                "final_url": page["final_url"],
            },
        )
        source = storage.resolve_redirect(source, page, profile)
        identifier = source.source_id
        storage.save_source(source)
        if source.uri in seen:
            seen.add(url)
            return
        if decision != "accepted":
            storage.remove_processed(site, identifier)
        # A bounded batch may contain more eligible pages than the remaining limit.
        if decision == "accepted" and (
            options.limit is None or documents < options.limit
        ):
            on_source(source, page)
            documents += 1
        elif decision == "accepted":
            deferred.append(item)
            return
        for link in links:
            target = link["url"]
            kind = resource_type(target)
            if kind == "asset":
                continue
            verdict, why = evaluate(target, profile)
            sid = source_id(target)
            if sid not in storage.records("sources", site):
                storage.save_source(
                    DataSource(
                        source_id=sid,
                        source_type="web",
                        uri=target,
                        metadata={
                            "site": site,
                            "faculty": site,
                            "resource_type": kind,
                            "discovered_from": url,
                            "link_text": link["link_text"],
                            "crawl_depth": depth + 1,
                            "decision": verdict,
                            "decision_reason": why,
                        },
                    )
                )
            if (
                kind == "html"
                and in_scope(target, profile)
                and target not in seen
                and target not in queued
            ):
                frontier.append((target, depth + 1, url))
                queued.add(target)
        seen.update((url, source.uri))
        storage.state(site, identifier, "crawl", status="completed", url=url)

        log.info("[%s] depth=%s %s %s", site, depth, decision, url)

    workers = min(options.download_workers, options.per_host)
    with storage.batched():
        save()
        try:
            while frontier and pages < options.max_pages:
                if options.limit is not None and documents >= options.limit:
                    break
                depth = frontier[0][1]
                level = []
                while (
                    frontier
                    and frontier[0][1] == depth
                    and pages + len(level) < options.max_pages
                ):
                    item = frontier.popleft()
                    if item[0] in seen or not in_scope(item[0], profile):
                        continue
                    if depth > options.max_depth:
                        deferred.append(item)
                        continue
                    if item[0] not in active:
                        active[item[0]] = item
                        level.append(item)
                # Checkpoint before scheduling: crash during a level never loses its URLs.
                save()
                with closing(
                    completion_map(fetch, level, workers, on_tick=on_tick)
                ) as results:
                    for item, (page, error) in results:
                        consume(item, page, error)
                        active.pop(item[0])
                        if pages % flush_every == 0:
                            save()
                        if options.limit is not None and documents >= options.limit:
                            break
                if active:  # Remaining document budget exhausted: preserve for resume.
                    break
        finally:
            save()
    return {
        "pages": pages,
        "documents": documents,
        "errors": errors,
        "remaining": len(frontier) + len(active) + len(deferred),
    }

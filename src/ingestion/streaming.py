"""Per-site extraction coordinator; only its owner thread mutates storage."""

from collections import deque
from concurrent.futures import wait, FIRST_COMPLETED
from copy import deepcopy
import logging
from src.base import ProcessedData
from .extractor.pipeline import processing_key
from .extractor.worker import extract_record
from .storage import write_json

log = logging.getLogger(__name__)


class StreamingExtractor:
    def __init__(
        self,
        storage,
        site,
        profile,
        pool,
        capacity,
        options,
        counts,
        errors,
        limit=None,
    ):
        self.storage, self.site, self.profile = storage, site, profile
        self.pool, self.capacity, self.options = pool, capacity, options
        self.counts, self.errors, self.limit = counts, errors, limit
        self.pending, self.staged, self.seen = {}, {}, set()
        self.selected = 0
        self.unflushed = 0
        self.latest = {}
        for row in storage.records("raw", site).values():
            sid = row["metadata"]["source_id"]
            if (
                sid not in self.latest
                or row["created_at"] > self.latest[sid]["created_at"]
            ):
                self.latest[sid] = row
        self.backlog = deque(self.latest.values())

    def stage(self, row):
        """Called after save_raw; submission waits for the next durable commit."""
        sid = row["metadata"]["source_id"]
        self.latest[sid] = row
        self.staged[sid] = row

    def _eligible(self, row):
        sid = row["metadata"]["source_id"]
        if self.latest.get(sid, {}).get("id") != row["id"]:
            return None
        source = self.storage.records("sources", self.site).get(sid)
        if source and source["metadata"].get("decision") != "accepted":
            return None
        key = processing_key(row, self.profile)
        token = (sid, key)
        if token in self.seen:
            return None
        self.seen.add(token)
        state = self.storage.records("state", self.site).get("processed:" + sid, {})
        existing = self.storage.records("processed", self.site).get(sid)
        if not self.options.reprocess and state.get("key") == key:
            if state.get("status") == "review" or (
                existing and existing["raw_data_id"] == row["id"]
            ):
                self.counts["processed_skipped"] += 1
                return None
        if self.limit is not None and self.selected >= self.limit:
            return None
        return key

    def _submit(self, row):
        key = self._eligible(row)
        if key is None:
            return
        future = self.pool.submit(
            extract_record, str(self.storage.root), deepcopy(row), self.profile
        )
        self.pending[future] = (row, key)
        self.selected += 1
        self.counts["processed_submitted"] += 1

    def _apply(self, future):
        row, key = self.pending.pop(future)
        sid = row["metadata"]["source_id"]
        source = self.storage.records("sources", self.site).get(sid)
        if self.latest.get(sid, {}).get("id") != row["id"] or (
            source and source["metadata"].get("decision") != "accepted"
        ):
            # A new raw version or policy decision must not be overwritten by an old result.
            self.counts["processed_stale_discarded"] += 1
            return
        try:
            payload, quality = future.result()
            document = ProcessedData.model_validate(payload)
        except Exception as exc:
            self.storage.remove_processed(self.site, sid)
            self.storage.state(
                self.site,
                sid,
                "processed",
                status="failed",
                key=key,
                raw_id=row["id"],
                error=str(exc),
            )
            self.counts["processed_failed"] += 1
            self.errors.append(
                {
                    "site": self.site,
                    "source_id": sid,
                    "stage": "processed",
                    "error": str(exc),
                }
            )
            return
        if quality["status"] == "accepted":
            self.storage.save_processed(self.site, document)
            status = "completed"
            self.counts["processed_accepted"] += 1
        else:
            self.storage.remove_processed(self.site, sid)
            write_json(
                self.storage.root
                / "ingestion_reports"
                / "review"
                / self.site
                / (sid + ".json"),
                payload,
            )
            status = "review"
            self.counts["processed_review"] += 1
        self.storage.state(
            self.site,
            sid,
            "processed",
            status=status,
            key=key,
            quality=quality,
            raw_id=row["id"],
        )
        log.info("[%s] processed %s %s", self.site, status, sid)

    def _collect(self, block=False):
        if not self.pending:
            return
        done, _ = wait(
            self.pending, timeout=None if block else 0, return_when=FIRST_COMPLETED
        )
        for future in done:
            self._apply(future)
            self.unflushed += 1
            if self.unflushed >= self.options.flush_every:
                self.storage.flush_pending()
                self.unflushed = 0

    def poll(self):
        """Collect and batch-publish while HTTP is waiting, on the owner thread."""
        self._collect()

    def on_commit(self):
        """Caller has flushed raw records. Submit new raws with bounded backpressure."""
        self._collect()
        for sid, row in list(self.staged.items()):
            while len(self.pending) >= self.capacity:
                self._collect(block=True)
            self._submit(row)
            self.staged.pop(sid)
        while self.backlog and len(self.pending) < self.capacity:
            self._submit(self.backlog.popleft())
        self._collect()

    def finish(self):
        """Drain only after producers finish; startup backlog is also recovered."""
        with self.storage.batched():
            self.storage.flush_pending()
            self.on_commit()
            while self.pending or self.backlog:
                self._collect(block=True)
                self.on_commit()
            self.storage.flush_pending()
            self.unflushed = 0
        return self.selected

    def cancel(self):
        # Running work may finish, but workers never publish files. Next run recovers raw.
        for future in self.pending:
            future.cancel()

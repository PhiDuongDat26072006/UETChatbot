"""Thread-safe background job manager for incremental and forced re-ingestion."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Callable
from uuid import uuid4

PROGRESS_KEYS = (
    "current_site",
    "phase",
    "skipped",
    "downloaded",
    "processed_saved",
    "new_pending",
    "rejected_preserved",
    "errors",
)


class JobConflictError(RuntimeError):
    """Raised when an ingest job is requested while another job is active."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_progress() -> dict[str, Any]:
    return {
        "current_site": None,
        "phase": None,
        "skipped": 0,
        "downloaded": 0,
        "processed_saved": 0,
        "new_pending": 0,
        "rejected_preserved": 0,
        "errors": 0,
    }


def _aggregate_result_progress(result: Any, progress: dict[str, Any]) -> None:
    """Ensure progress totals match final site counts if provided by curation."""
    if not isinstance(result, dict):
        return
    sites_map = result.get("sites")
    if not isinstance(sites_map, dict) or not sites_map:
        return
    numeric_keys = (
        "skipped",
        "downloaded",
        "processed_saved",
        "new_pending",
        "rejected_preserved",
        "errors",
    )
    for key in numeric_keys:
        total = sum(int(stats.get(key, 0) or 0) for stats in sites_map.values() if isinstance(stats, dict))
        if total > progress.get(key, 0):
            progress[key] = total


class IngestJobManager:
    """Manages one active background incremental ingest job at a time."""

    def __init__(
        self,
        data_dir: str | Path,
        profile_dir: str | Path | None = None,
        overrides_path: str | Path | None = None,
        ingest_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.profile_dir = Path(profile_dir) if profile_dir is not None else None
        self.overrides_path = Path(overrides_path) if overrides_path is not None else None
        self._ingest_fn = ingest_fn
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None
        self._latest_job_id: str | None = None
        self._last_finished_job_id: str | None = None

    def start_job(
        self,
        sites: list[str] | None = None,
        force_source_id: str | None = None,
    ) -> dict[str, Any]:
        """Queue and start a background daemon thread; raise JobConflictError if busy."""
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active and active["status"] in {"queued", "running"}:
                    raise JobConflictError(
                        f"Ingest job {self._active_job_id} is already {active['status']}"
                    )

            job_id = str(uuid4())
            job: dict[str, Any] = {
                "job_id": job_id,
                "status": "queued",
                "sites": list(sites) if sites is not None else None,
                "force_source_id": force_source_id,
                "started_at": None,
                "finished_at": None,
                "progress": _default_progress(),
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._active_job_id = job_id
            self._latest_job_id = job_id
            snapshot = copy.deepcopy(job)

        worker = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            daemon=True,
            name=f"ingest-{job_id[:8]}",
        )
        worker.start()
        return snapshot

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _now_iso()
            sites = list(job["sites"]) if job["sites"] is not None else None
            force_source_id = job["force_source_id"]

        def on_progress(update: dict[str, Any]) -> None:
            if not isinstance(update, dict):
                return
            payload = (
                update.get("progress")
                if isinstance(update.get("progress"), dict)
                else update
            )
            with self._lock:
                progress = self._jobs[job_id]["progress"]
                for key in PROGRESS_KEYS:
                    if key in payload:
                        progress[key] = payload[key]

        try:
            ingest_fn = self._ingest_fn
            if ingest_fn is None:
                from ..curation import run_incremental_ingest

                ingest_fn = run_incremental_ingest

            result = ingest_fn(
                self.data_dir,
                sites=sites,
                force_source_id=force_source_id,
                profile_dir=self.profile_dir,
                overrides_path=self.overrides_path,
                on_progress=on_progress,
            )
            with self._lock:
                job = self._jobs[job_id]
                job["result"] = result
                _aggregate_result_progress(result, job["progress"])
                job["progress"]["phase"] = "completed"
                job["status"] = "completed"
                job["finished_at"] = _now_iso()
                if self._active_job_id == job_id:
                    self._active_job_id = None
                self._last_finished_job_id = job_id
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc) or type(exc).__name__
                job["progress"]["phase"] = "failed"
                job["finished_at"] = _now_iso()
                if self._active_job_id == job_id:
                    self._active_job_id = None
                self._last_finished_job_id = job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a snapshot of a specific job, or None if unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def get_status(self) -> dict[str, Any]:
        """Return active or latest finished job status plus active/last_finished refs."""
        with self._lock:
            active = None
            if self._active_job_id is not None:
                candidate = self._jobs.get(self._active_job_id)
                if candidate and candidate["status"] in {"queued", "running"}:
                    active = copy.deepcopy(candidate)

            last_finished = (
                copy.deepcopy(self._jobs[self._last_finished_job_id])
                if self._last_finished_job_id in self._jobs
                else None
            )
            latest = active or last_finished
            if latest is None and self._latest_job_id in self._jobs:
                latest = copy.deepcopy(self._jobs[self._latest_job_id])

            if latest is not None:
                return {
                    **latest,
                    "active_job": active,
                    "last_finished_job": last_finished,
                }
            return {
                "job_id": None,
                "status": "idle",
                "sites": None,
                "force_source_id": None,
                "started_at": None,
                "finished_at": None,
                "progress": _default_progress(),
                "result": None,
                "error": None,
                "active_job": None,
                "last_finished_job": None,
            }

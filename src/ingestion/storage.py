"""Atomic JSONL snapshots. All writes occur on the coordinator thread."""

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from src.base import DataSource, RawData, ProcessedData
from .identity import (
    digest,
    raw_id,
    slug,
    canonicalize_url,
    source_id,
    normalize_site_url,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, obj):
    atomic_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode())


def read_jsonl(path):
    if not Path(path).exists():
        return []
    result = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL: {path}:{number}") from exc
    return result


@contextmanager
def writer_lock(data_dir):
    path = Path(data_dir) / "ingestion_state" / "writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Writer lock exists: {path}. Check running process before removing stale lock."
        ) from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


class Storage:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}
        self.batch_depth = 0
        self.dirty = set()

    def path(self, kind, site):
        return (
            self.root
            / {
                "sources": f"data_source/{site}.jsonl",
                "raw": f"raw_data/{site}/records.jsonl",
                "processed": f"processed_data/{site}.jsonl",
                "state": f"ingestion_state/{site}.jsonl",
            }[kind]
        )

    def records(self, kind, site):
        key = (kind, site)
        if key not in self.cache:
            field = "source_id" if kind == "sources" else "id"
            rows = read_jsonl(self.path(kind, site))
            self.cache[key] = {r[field]: r for r in rows}
            if len(rows) != len(self.cache[key]):
                raise ValueError(f"Duplicate IDs in {self.path(kind, site)}")
        return self.cache[key]

    def save(self, kind, site, record):
        key = "source_id" if kind == "sources" else "id"
        self.records(kind, site)[record[key]] = record
        self.flush(kind, site)

    @contextmanager
    def batched(self):
        """Buffer metadata; raw bytes still become durable immediately."""
        self.batch_depth += 1
        try:
            yield
        finally:
            self.batch_depth -= 1
            if not self.batch_depth:
                self.flush_pending()

    def flush_pending(self):
        # Never publish completed state before its records are durable.
        order = {"sources": 0, "raw": 1, "processed": 2, "state": 3}
        for kind, site in sorted(self.dirty, key=lambda k: (order[k[0]], k[1])):
            self._flush(kind, site)
            self.dirty.discard((kind, site))

    def flush(self, kind, site):
        if self.batch_depth:
            self.dirty.add((kind, site))
        else:
            self._flush(kind, site)

    def _flush(self, kind, site):
        rows = self.records(kind, site)
        atomic_bytes(
            self.path(kind, site),
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for _, r in sorted(rows.items())
            ).encode(),
        )
        if kind == "sources":
            accepted_path = self.root / "data_source" / f"{site}-accepted.jsonl"
            accepted_rows = [
                r for _, r in sorted(rows.items())
                if r.get("metadata", {}).get("decision") == "accepted"
            ]
            atomic_bytes(
                accepted_path,
                "".join(
                    json.dumps(r, ensure_ascii=False) + "\n"
                    for r in accepted_rows
                ).encode(),
            )

    def remove_processed(self, site, source):
        if self.records("processed", site).pop(source, None) is not None:
            self.flush("processed", site)

    def state(self, site, source, stage, **values):
        key = f"{stage}:{source}"
        record = {
            "id": key,
            "source_id": source,
            "stage": stage,
            "updated_at": now(),
            **values,
        }
        self.save("state", site, record)
        return record

    def save_source(self, source):
        self.save("sources", source.metadata["site"], source.model_dump(mode="json"))

    def resolve_redirect(self, source, page, profile=None):
        """Resolve redirects using configured site identity; retain actual final URL."""
        try:
            final = (
                normalize_site_url(page["final_url"], profile)
                if profile
                else canonicalize_url(page["final_url"])
            )
        except ValueError:
            return source
        if final == source.uri:
            return source
        site = source.metadata["site"]
        target_id = source_id(final)
        alias = source.model_copy(deep=True)
        alias.metadata.update(
            decision="alias", canonical_source_id=target_id, final_url=final
        )
        self.save_source(alias)
        self.remove_processed(site, source.source_id)
        metadata = dict(source.metadata)
        metadata.update(requested_url=source.uri, final_url=final)
        return DataSource(
            source_id=target_id,
            source_type=source.source_type,
            uri=final,
            metadata=metadata,
        )

    def latest_raw(self, site, source):
        rows = [
            r
            for r in self.records("raw", site).values()
            if r["metadata"]["source_id"] == source
        ]
        return max(rows, key=lambda r: r["created_at"], default=None)

    def save_raw(self, source, page):
        site = source.metadata["site"]
        sha = digest(page["body"])
        identifier = raw_id(source.source_id, sha)
        is_pdf = source.metadata.get("resource_type") == "pdf" or source.uri.lower().endswith(".pdf")
        extension = ".pdf" if is_pdf else ".html"
        folder = "pdf" if is_pdf else "html"
        relative = (
            Path("raw_data")
            / site
            / folder
            / (slug(source.uri) + "--" + source.source_id)
            / (identifier + extension)
        )
        atomic_bytes(self.root / relative, page["body"])
        record = {
            "id": identifier,
            "source_uri": source.uri,
            "title": source.metadata.get("title"),
            "created_at": now(),
            "metadata": {
                **source.metadata,
                "source_id": source.source_id,
                "raw_path": relative.as_posix(),
                "content_hash": sha,
                "encoding": page["encoding"],
                "final_url": page["final_url"],
                "requested_url": page["requested_url"],
                "content_type": page["headers"].get("content-type", ""),
                "etag": page["headers"].get("etag"),
                "last_modified": page["headers"].get("last-modified"),
            },
        }
        self.save("raw", site, record)
        self.state(site, source.source_id, "raw", status="completed", raw_id=identifier)
        return record

    def load_raw(self, record):
        if "metadata" not in RawData.model_fields:
            raise RuntimeError(
                "Rename RawData.raw_metadata to metadata in src/base.py first"
            )
        path = (self.root / record["metadata"]["raw_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Raw path escapes data directory")
        body = path.read_bytes()
        if digest(body) != record["metadata"]["content_hash"]:
            raise ValueError(f"Raw hash mismatch: {path}")
        encoding = record["metadata"].get("encoding", "utf-8")
        content = body.decode(encoding, errors="replace") if encoding != "binary" else ""
        return RawData(
            **record,
            content=content,
        )

    def save_processed(self, site, document):
        ProcessedData.model_validate(document.model_dump())
        self.save("processed", site, document.model_dump(mode="json"))

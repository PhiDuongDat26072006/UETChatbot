"""Bounded HTML GET, per-thread sessions, controlled redirects and retries."""

from contextlib import nullcontext
import threading
import time
from urllib.parse import urljoin
import requests
import urllib3
from ..config.settings import load_runtime
from bs4 import UnicodeDammit

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class HTMLFetcher:
    def __init__(self, timeout=None, max_bytes=None, retries=None, delay=None):
        defaults = load_runtime()["ingestion"]
        timeout = defaults["timeout"] if timeout is None else timeout
        max_bytes = defaults["max_html_bytes"] if max_bytes is None else max_bytes
        retries = defaults["retries"] if retries is None else retries
        delay = defaults["delay"] if delay is None else delay
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.retries = retries
        self.delay = delay
        self.local = threading.local()
        self.sessions = []
        self.lock = threading.Lock()
        self.requests = 0
        self.bytes = 0
        self.limits = None

    def session(self):
        if not hasattr(self.local, "session"):
            session = requests.Session()
            session.headers["User-Agent"] = (
                "UETChatbotIngestion/1.0 (HTML research crawler)"
            )
            session.verify = False
            self.local.session = session
            with self.lock:
                self.sessions.append(session)
        return self.local.session

    def close(self):
        for session in self.sessions:
            session.close()

    def fetch(self, url, allowed, headers=None):
        last = None
        for attempt in range(self.retries + 1):
            try:
                return self._get(url, allowed, headers or {})
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                last = exc
            if attempt < self.retries:
                time.sleep(min(2**attempt, 8))
        if last is not None:
            raise last
        raise RuntimeError(f"Failed to fetch {url}")

    def _get(self, url, allowed, headers):
        current = url
        for _ in range(6):
            if not allowed(current):
                raise ValueError(f"Redirect outside configured scope: {current}")
            with self.limits.acquire(current) if self.limits else nullcontext():
                time.sleep(self.delay)
                with self.session().get(
                    current,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                    allow_redirects=False,
                    verify=False,
                ) as response:
                    with self.lock:
                        self.requests += 1
                    if response.status_code in {301, 302, 303, 307, 308}:
                        current = urljoin(current, response.headers["Location"])
                        continue
                    response.raise_for_status()
                    info = {
                        "requested_url": url,
                        "final_url": current,
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "body": b"",
                        "encoding": "utf-8",
                    }
                    info["headers"] = {k.lower(): v for k, v in info["headers"].items()}
                    if response.status_code == 304:
                        return info
                    mime = (
                        response.headers.get("Content-Type", "").split(";")[0].lower()
                    )
                    if mime and mime not in {"text/html", "application/xhtml+xml"}:
                        raise ValueError(f"Unsupported content type: {mime}")
                    chunks = []
                    size = 0
                    for block in response.iter_content(65536):
                        size += len(block)
                        with self.lock:
                            self.bytes += len(block)
                        if size > self.max_bytes:
                            raise ValueError("HTML exceeds --max-html-bytes")
                        chunks.append(block)
                    body = b"".join(chunks)
                    if (
                        not mime
                        and b"<html" not in body[:4096].lower()
                        and b"<!doctype html" not in body[:4096].lower()
                    ):
                        raise ValueError(
                            "Missing MIME and response does not look like HTML"
                        )
                    declared = (
                        response.encoding
                        if "charset="
                        in response.headers.get("Content-Type", "").lower()
                        else None
                    )
                    encoding = UnicodeDammit(
                        body,
                        known_definite_encodings=[declared] if declared else [],
                        is_html=True,
                    ).original_encoding
                    info.update(body=body, encoding=encoding or "utf-8")
                    return info
        raise ValueError("Too many redirects")

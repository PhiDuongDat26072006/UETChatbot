"""Shared request limits across site workers and configured host aliases."""

from contextlib import contextmanager
from threading import BoundedSemaphore, Lock
from urllib.parse import urlsplit


class RequestLimits:
    def __init__(self, total, per_host, profiles):
        self.total = BoundedSemaphore(total)
        self.per_host = per_host
        self.lock = Lock()
        groups = []
        for profile in profiles.values():
            group = {h.lower() for h in profile["scope"]["hosts"]}
            overlapping = [g for g in groups if g & group]
            for other in overlapping:
                group |= other
                groups.remove(other)
            groups.append(group)
        self.groups = {host: tuple(sorted(group)) for group in groups for host in group}
        self.slots = {}

    @contextmanager
    def acquire(self, url):
        host = urlsplit(url).netloc.lower()
        key = self.groups.get(host, (host,))
        with self.lock:
            slot = self.slots.setdefault(key, BoundedSemaphore(self.per_host))
        with slot, self.total:
            yield

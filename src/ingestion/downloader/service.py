"""Download preparation does not mutate storage; coordinator commits results."""

from ..crawler.policy import in_scope


def download_source(
    source, profile, fetcher, previous=None, refresh=False, fetched_page=None
):
    if fetched_page is not None:
        return fetched_page
    if previous and not refresh:
        return None
    headers = {}
    if previous:
        meta = previous["metadata"]
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]
    return fetcher.fetch(source.uri, lambda u: in_scope(u, profile), headers)

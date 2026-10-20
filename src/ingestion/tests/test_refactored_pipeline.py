"""Offline regressions for source identity, transport, persistence, and CLI stages."""
import importlib
import io
import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src.base import DataSource, ProcessedData
from src.chunking import chunker
from src.ingestion import UETDataLoader
from src.ingestion import main as cli
from src.ingestion import loader
from src.ingestion.classifier import classification
from src.ingestion.config import paths
from src.ingestion.config.faculties import (
    FACULTY_TARGETS, faculty_storage_name, find_faculty, resolve_unit, url_in_scope,
)
from src.ingestion.config.session import create_http_session, get_source_response
from src.ingestion.crawler import crawl, scraper, wp_api
from src.ingestion.extractor import pipeline
from src.ingestion.validator import validation

IAI = find_faculty("IAI")
UET = find_faculty("UET")
IAI_URL = IAI["start_urls"][0]
UET_URL = "https://uet.vnu.edu.vn/"
ARTICLE = (
    "Sinh viên tìm hiểu chương trình đào tạo và các hoạt động nghiên cứu. "
    "Các môn học cung cấp kiến thức chuyên ngành và cơ hội thực hành. "
    "Thông tin được công bố để hỗ trợ người học trong quá trình học tập. "
) * 5


def response(url, body="", status=200, headers=None):
    result = requests.Response()
    result.url = url
    result.status_code = status
    result.headers.update({"Content-Type": "text/html"})
    result.headers.update(headers or {})
    result.encoding = "utf-8"
    result._content = body.encode("utf-8")
    result._content_consumed = True
    result.raw = io.BytesIO(result._content)
    result.close = Mock(wraps=result.close)
    return result


def html(title="Đào tạo", links=""):
    return f"<html><head><title>{title}</title></head><body><article><h1>{title}</h1><p>{ARTICLE}</p>{links}</article></body></html>"


@pytest.fixture
def offline_http(monkeypatch):
    """Reject unexpected network calls and track response/session cleanup."""
    routes, calls, responses, closed_sessions = {}, [], [], []
    original_close = requests.Session.close

    def get(session, url, **kwargs):
        calls.append((url, kwargs))
        if "/wp-json/" in url and url not in routes:
            result = response(url, "[]", headers={"Content-Type": "application/json"})
        else:
            value = routes[url]
            if isinstance(value, Exception):
                raise value
            result = value(url) if callable(value) else response(url, value)
        responses.append(result)
        return result

    def close(session):
        closed_sessions.append(session)
        original_close(session)

    monkeypatch.setattr(requests.Session, "get", get)
    monkeypatch.setattr(requests.Session, "close", close)
    return routes, calls, responses, closed_sessions


@pytest.fixture
def data_paths(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    for module in (classification, validation, pipeline):
        monkeypatch.setattr(module, "DATA_DIR", str(raw))
    monkeypatch.setattr(pipeline, "PROCESSED_DATA_DIR", str(processed))
    return raw, processed


def write_endpoints(raw, source, urls):
    path = raw / faculty_storage_name(source["faculty_id"]) / "endpoints" / "endpoints.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("content,expected", [
    ("", {}), ("[]", {}), ("invalid: [", {}), ("paths: []", {}),
    ("paths: null", {}), ("paths: {data_dir: null}", {}),
    ("paths: {data_dir: 42}", {}), ("paths: {data_dir: ''}", {}),
    ("paths: {data_dir: raw, processed_dir: /tmp/processed}",
     {"data_dir": "raw", "processed_dir": "/tmp/processed"}),
])
def test_yaml_paths(tmp_path, content, expected):
    config = tmp_path / "config.yaml"
    config.write_text(content, encoding="utf-8")
    assert paths._load_repo_paths(config) == expected


def test_missing_and_unreadable_yaml(tmp_path, monkeypatch, caplog):
    assert paths._load_repo_paths(tmp_path / "missing.yaml") == {}
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")))
    assert paths._load_repo_paths(tmp_path / "config.yaml") == {}
    assert "denied" in caplog.text


def test_environment_paths_are_shared_with_loader(monkeypatch, tmp_path):
    original_raw, original_processed = paths.DATA_DIR, paths.PROCESSED_DATA_DIR
    monkeypatch.setenv("CHATBOT_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("CHATBOT_PROCESSED_DIR", str(tmp_path / "processed"))
    try:
        importlib.reload(paths)
        monkeypatch.setattr(loader, "PROCESSED_DATA_DIR", paths.PROCESSED_DATA_DIR)
        assert paths.DATA_DIR == str(tmp_path / "raw")
        assert UETDataLoader().data_dir == tmp_path / "processed"
    finally:
        paths.DATA_DIR, paths.PROCESSED_DATA_DIR = original_raw, original_processed


def test_faculty_schema_and_shared_host_identity():
    keys = {"faculty_id", "name", "domain", "start_urls", "allowed_path_prefixes"}
    assert all(set(target) == keys for target in FACULTY_TARGETS)
    assert len({target["faculty_id"] for target in FACULTY_TARGETS}) == len(FACULTY_TARGETS)
    assert IAI["domain"] == UET["domain"] == "uet.vnu.edu.vn"
    assert resolve_unit("uet.vnu.edu.vn", IAI_URL) == "IAI"
    assert resolve_unit("uet.vnu.edu.vn", UET_URL) == "UET"
    assert resolve_unit("uet.edu.vn", IAI_URL) == "UET"
    assert resolve_unit("IAI") == "IAI"
    assert resolve_unit("unknown.example") == "UET"
    assert json.loads(json.dumps(FACULTY_TARGETS)) == FACULTY_TARGETS


@pytest.mark.parametrize("url,allowed", [
    (IAI_URL.rstrip("/"), True), (IAI_URL + "dao-tao/", True),
    ("https://www.uet.vnu.edu.vn/vien-tri-tue-nhan-tao/", True),
    ("https://uet.edu.vn/vien-tri-tue-nhan-tao/", True),
    (IAI_URL.rstrip("/") + "-khac/", False), (UET_URL, False),
    (IAI_URL + "../news/", False),
    (IAI_URL + "%2e%2e/news/", False),
    (IAI_URL + "staff/../dao-tao/", True),
    ("https://other.example/vien-tri-tue-nhan-tao/", False),
])
def test_scope_uses_path_boundaries(url, allowed):
    assert url_in_scope(url, IAI["domain"], IAI["allowed_path_prefixes"]) is allowed


def test_http_session_defaults_and_overrides():
    with create_http_session(pool_size=7, max_retries=2, headers={"X-Test": "yes"}) as session:
        adapter = session.get_adapter("https://")
        assert adapter._pool_connections == adapter._pool_maxsize == 7
        assert adapter.max_retries.total == 2
        assert session.get_adapter("http://") is adapter
        assert session.headers["X-Test"] == "yes"
    with create_http_session() as session:
        assert "Chrome/120" in session.headers["User-Agent"]
        assert session.get_adapter("https://").max_retries.total == 1


def test_scoped_redirect_stops_before_unrelated_request(offline_http):
    routes, calls, responses, _ = offline_http
    routes[IAI_URL] = lambda url: response(url, status=302, headers={"Location": UET_URL})
    with create_http_session() as session, pytest.raises(requests.RequestException, match="outside source scope"):
        get_source_response(session, IAI_URL, IAI["domain"], IAI["allowed_path_prefixes"])
    assert [url for url, _ in calls] == [IAI_URL]
    assert responses[0].close.called


def test_scoped_redirect_and_relative_links(offline_http):
    routes, calls, responses, closed = offline_http
    final_url = IAI_URL + "staff/"
    routes[IAI_URL] = lambda url: response(url, status=302, headers={"Location": "staff/"})
    routes[final_url] = html(links='<a href="profile/">Profile</a><a href="/news/">Unrelated</a><a href="/wp-content/program.pdf">PDF</a>')
    with create_http_session() as session:
        endpoints, files = scraper.scrape_single_page(session, IAI["domain"], IAI_URL, allowed_path_prefixes=IAI["allowed_path_prefixes"])
    assert endpoints == {final_url + "profile/"}
    assert files == {UET_URL + "wp-content/program.pdf"}
    assert all(not kwargs["allow_redirects"] for _, kwargs in calls)
    assert all(item.close.called for item in responses)
    assert len(closed) == 1


def test_api_scope_and_response_cleanup(offline_http):
    routes, _, responses, _ = offline_http
    api_url = UET_URL + "wp-json/wp/v2/posts?per_page=100&page=1"
    routes[api_url] = lambda url: response(url, json.dumps([
        {"link": IAI_URL + "dao-tao/?utm_source=test"}, {"link": UET_URL + "news/"},
    ]), headers={"Content-Type": "application/json"})
    with create_http_session() as session:
        endpoints = wp_api.fetch_wp_posts_and_pages(session, UET_URL, "posts", IAI["domain"], allowed_path_prefixes=IAI["allowed_path_prefixes"])
    assert endpoints == {IAI_URL + "dao-tao/"}
    assert all(item.close.called for item in responses)


def test_ignored_tracking_url_is_rejected_without_http(offline_http):
    _, calls, _, _ = offline_http
    with create_http_session() as session:
        original, accepted, status = validation.check_and_fix_endpoint(UET_URL + "api/Content/Decl/123", session)
    assert accepted is None and status == "DROPPED_IGNORED"
    assert calls == []


def test_slash_recovery_and_duplicate_validation(data_paths, offline_http):
    raw, _ = data_paths
    routes, _, responses, _ = offline_http
    url = IAI_URL + "dao-tao"
    routes[url] = lambda source: response(source, "<title>Page not found</title>")
    routes[url + "/"] = html()
    endpoint_file = write_endpoints(raw, IAI, [url, url])
    result = validation.verify_endpoint_file(str(endpoint_file), target=IAI)
    assert result["corrected"] == 2 and result["final"] == 1
    assert endpoint_file.read_text().splitlines() == [url + "/"]
    assert all(item.close.called for item in responses)


def test_transient_validation_retains_url_and_reports_failure(data_paths, offline_http):
    raw, _ = data_paths
    routes, _, _, _ = offline_http
    routes[IAI_URL] = requests.Timeout("timeout")
    endpoint_file = write_endpoints(raw, IAI, [IAI_URL])
    result = validation.verify_endpoint_file(str(endpoint_file), target=IAI)
    assert result["failed"] == 1 and result["final"] == 1
    assert endpoint_file.read_text().strip() == IAI_URL


def test_validator_never_copies_between_faculties(data_paths, offline_http, monkeypatch):
    raw, _ = data_paths
    routes, _, _, _ = offline_http
    routes[IAI_URL] = html()
    routes[UET_URL] = html("UET")
    iai_file = write_endpoints(raw, IAI, [IAI_URL])
    uet_file = write_endpoints(raw, UET, [UET_URL])
    monkeypatch.setattr(validation, "FACULTY_TARGETS", [IAI, UET])
    validation.verify_all_endpoints()
    assert iai_file.read_text().strip() == IAI_URL
    assert uet_file.read_text().strip() == UET_URL
    assert not (raw / "uet.vnu.edu.vn").exists()


def test_offline_crawl_to_load_and_chunk(data_paths, offline_http, monkeypatch, tmp_path):
    raw, processed = data_paths
    routes, calls, responses, closed = offline_http
    child = IAI_URL + "dao-tao/"
    attachment = UET_URL + "wp-content/program.txt"
    routes[IAI_URL] = html("Viện Trí tuệ nhân tạo", f'<a href="dao-tao/">Training</a><a href="/news/">Other</a><a href="{attachment}">Program</a>')
    routes[child] = html()
    routes[attachment] = ARTICLE
    routes[UET_URL] = html("Trường Đại học Công nghệ")
    for target in (IAI, UET):
        crawl.crawl_target(target, str(raw))
        folder = faculty_storage_name(target["faculty_id"])
        endpoint_file = raw / folder / "endpoints" / "endpoints.txt"
        validation.verify_endpoint_file(str(endpoint_file), target=target)
        assert not classification.classify_domain(folder)["failed"]
        metadata = [json.loads(line) for line in endpoint_file.with_name("endpoints_metadata.jsonl").read_text().splitlines()]
        assert all(item["domain"] == "uet.vnu.edu.vn" and item["faculty_id"] == target["faculty_id"] for item in metadata)
        assert not pipeline.extract_domain_documents(folder)["failed"]
        if target["faculty_id"] == "IAI":
            assert all("/wp-json/" not in url for url, _ in calls)
    documents = UETDataLoader(processed).load_all()
    assert {document.raw_metadata["unit"] for document in documents} == {"IAI", "UET"}
    assert all(document.raw_metadata["domain"] == "uet.vnu.edu.vn" for document in documents)
    assert (processed / "iai.uet.vnu.edu.vn.jsonl").exists()
    assert (processed / "uet.edu.vn.jsonl").exists()
    assert (raw / "iai.uet.vnu.edu.vn" / "files" / "program.txt").exists()
    assert UET_URL + "news/" not in [url for url, _ in calls]
    assert all(item.close.called for item in responses)
    assert closed

    monkeypatch.setattr(chunker, "__file__", str(tmp_path / "src" / "chunking" / "chunker.py"))
    processed_docs = [ProcessedData(raw_data_id=document.id, title=document.title or "", content=document.content, metadata=document.raw_metadata) for document in documents]
    chunks = chunker.UETChunker().chunk_batch(processed_docs)
    assert chunks
    assert (tmp_path / "chunk_data" / "iai.uet.vnu.edu.vn_chunks.jsonl").exists()
    assert (tmp_path / "chunk_data" / "uet.edu.vn_chunks.jsonl").exists()
    assert not (tmp_path / "chunk_data" / "uet.vnu.edu.vn_chunks.jsonl").exists()


def test_cached_attribution_refresh_preserves_ids(data_paths, offline_http):
    raw, processed = data_paths
    _, calls, _, _ = offline_http
    write_endpoints(raw, IAI, [IAI_URL, IAI_URL.rstrip("/")])
    processed.mkdir()
    record = {"id": "stable-id", "source_type": "html", "source_url_or_path": IAI_URL,
              "content": ARTICLE, "content_length": len(ARTICLE), "domain": "iai.uet.vnu.edu.vn", "unit": "UET"}
    output = processed / "iai.uet.vnu.edu.vn.jsonl"
    output.write_text(json.dumps(record) + "\n")
    result = pipeline.extract_domain_documents("IAI")
    saved = json.loads(output.read_text())
    assert result["total_docs"] == 1 and calls == []
    assert saved["id"] == "stable-id" and saved["unit"] == "IAI"
    assert saved["domain"] == "uet.vnu.edu.vn"


def test_partial_extraction_saves_successes(data_paths, offline_http):
    raw, processed = data_paths
    routes, _, responses, closed = offline_http
    failed_url = IAI_URL + "failed/"
    routes[IAI_URL] = html()
    routes[failed_url] = requests.Timeout("timeout")
    write_endpoints(raw, IAI, [IAI_URL, failed_url])
    result = pipeline.extract_domain_documents("IAI")
    assert result["failed"] and result["failed_requests"] == 1
    assert result["total_docs"] == 1
    assert json.loads((processed / "iai.uet.vnu.edu.vn.jsonl").read_text())["source_url_or_path"] == IAI_URL
    assert not list(processed.glob("*.tmp"))
    assert all(item.close.called for item in responses) and closed


def test_crawler_partial_failure_saves_and_raises(data_paths, offline_http, caplog):
    raw, _ = data_paths
    routes, _, _, closed = offline_http
    routes[IAI_URL] = requests.Timeout("timeout")
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="requests/downloads failed"):
        crawl.crawl_target(IAI, str(raw), download_files=False)
    assert (raw / "iai.uet.vnu.edu.vn" / "endpoints" / "endpoints.txt").exists()
    assert "timeout" in caplog.text and closed


@pytest.mark.parametrize("failed", [False, True])
def test_single_stage_cli_status(monkeypatch, caplog, failed):
    monkeypatch.setattr(cli, "extract_domain_documents", lambda *args, **kwargs: {"failed": failed, "total_docs": 3})
    with caplog.at_level(logging.INFO):
        status = cli.main(["--extract", "--domain", "IAI"])
    assert status == int(failed)
    assert "[START]" in caplog.text
    assert ("[FAILED]" if failed else "[DONE]") in caplog.text
    if failed:
        assert "[DONE]" not in caplog.text


def test_multi_target_failure_continues_and_never_logs_success(monkeypatch, caplog):
    attempted = []

    def run(target, **kwargs):
        attempted.append(target["faculty_id"])
        if target["faculty_id"] == "IAI":
            raise RuntimeError("unavailable")
        return 2, 0

    monkeypatch.setattr(crawl, "crawl_target", run)
    monkeypatch.setattr(crawl, "FACULTY_TARGETS", [IAI, UET])
    with caplog.at_level(logging.INFO):
        assert cli.main(["--no-download"]) == 1
    assert attempted == ["IAI", "UET"]
    assert caplog.text.count("[FAILED]") == 1 and "[DONE]" not in caplog.text


def test_loader_skips_bad_and_duplicate_records(tmp_path):
    source = tmp_path / "records.jsonl"
    record = {"id": "stable", "content": ARTICLE, "unit": "IAI", "domain": "uet.vnu.edu.vn"}
    source.write_text(json.dumps(record) + "\n[]\ninvalid\n" + json.dumps(record) + "\n")
    documents = UETDataLoader().crawl(DataSource(source_type="file", uri=str(source)))
    assert len(documents) == 1 and documents[0].id == "stable"
    assert documents[0].raw_metadata["unit"] == "IAI"


def test_partial_classification_keeps_slug_results(data_paths, offline_http):
    raw, _ = data_paths
    routes, _, _, closed = offline_http
    source_url = IAI_URL + "dao-tao/"
    routes[source_url] = requests.Timeout("timeout")
    endpoint_file = write_endpoints(raw, IAI, [source_url])
    result = classification.classify_domain("IAI")
    record = json.loads(endpoint_file.with_name("endpoints_metadata.jsonl").read_text())
    assert result["failed"] and record["category"] == "DAO_TAO"
    assert record["title"] == "" and closed


def test_http_fallback_preserves_original_url(offline_http):
    routes, calls, responses, _ = offline_http
    original = "http://fit.uet.vnu.edu.vn/article"
    secure = original.replace("http://", "https://")
    routes[secure] = lambda url: response(url, status=404)
    routes[secure + "/"] = lambda url: response(url, status=404)
    routes[original] = html()
    with create_http_session() as session:
        assert validation.check_and_fix_endpoint(original, session) == (original, original, "OK")
    assert [url for url, _ in calls] == [secure, secure + "/", original]
    assert all(item.close.called for item in responses)


def test_invalid_seed_does_not_create_output(data_paths, offline_http):
    raw, _ = data_paths
    target = {**IAI, "start_urls": [UET_URL]}
    with pytest.raises(ValueError, match="Invalid crawl seeds"):
        crawl.crawl_target(target, str(raw))
    assert not raw.exists()


def test_force_extraction_deduplicates_canonical_inputs(data_paths, offline_http):
    raw, _ = data_paths
    routes, calls, _, _ = offline_http
    routes[IAI_URL] = html()
    write_endpoints(raw, IAI, [IAI_URL, IAI_URL.rstrip("/"), IAI_URL])
    result = pipeline.extract_domain_documents("IAI", force=True)
    assert result["total_docs"] == 1 and len(calls) == 1


def test_metadata_cache_skips_invalid_lines(tmp_path, caplog):
    path = tmp_path / "metadata.jsonl"
    path.write_text('{"url":"first","title":"A"}\n[]\ninvalid\n{"url":"last","title":"B"}\n')
    assert classification.load_existing_metadata(str(path)) == {"first": "A", "last": "B"}
    assert "Invalid metadata" in caplog.text


def test_unextractable_attachment_reports_partial_failure(data_paths, offline_http):
    raw, _ = data_paths
    files = raw / "iai.uet.vnu.edu.vn" / "files"
    files.mkdir(parents=True)
    (files / "broken.pdf").write_bytes(b"broken")
    result = pipeline.extract_domain_documents("IAI")
    assert result["failed"] and result["failed_files"] == 1
    assert result["total_docs"] == 0

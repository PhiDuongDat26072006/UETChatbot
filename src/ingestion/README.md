# UET ingestion

The ingestion stages discover URLs and attachments, validate endpoints, classify
URLs into seven categories, extract clean text, and load processed records for RAG.
Each CLI invocation runs one stage; crawling remains the default.

```text
Faculty configuration → HTTP session → crawl → raw endpoint/file lists
                                             ↓
                          validate → classify → extract → processed JSONL
                                                              ↓
                                              UETDataLoader → chunking/indexing
```

## Responsibilities

- `config/faculties.py`: one faculty definition list, source identity, host/path scope,
  and derived storage names.
- `config/paths.py`: YAML paths, defaults, and environment overrides.
- `config/ignore_rules.py`: URL exclusions, document/asset extensions, cloud hosts,
  and tracking-query exclusions.
- `config/session.py`: headers, timeouts, connection pooling, retry counts, and
  scoped redirect handling.
- `filters.py`: URL cleanup, endpoint validation, attachment/cloud filtering.
- `crawler/crawl.py`: API discovery, bounded BFS, persistence, and target orchestration.
  `scraper.py`, `wp_api.py`, and `downloader.py` own HTML link discovery, WordPress
  pagination, and attachment downloads respectively.
- `validator/validation.py`: soft-404 detection, slash recovery, endpoint-file
  validation, and faculty iteration.
- `classifier/taxonomy.py`: category definitions, patterns, priorities, and weights.
  `scoring.py` owns title cleanup/fetching and classification; `classification.py`
  persists metadata and category lists.
- `extractor/`: cohesive HTML cleaning/extraction, document readers, date extraction,
  cloud links, summary generation, and `pipeline.py` for incremental JSONL output.
- `loader.py`: the `BaseDataCrawler` adapter for processed JSONL.
  `manifest.py` retains the downstream deduplication/stage-tracking contract.
- `main.py` and `__main__.py`: CLI dispatch and exit status.

## Faculty configuration and storage

Add or change a source in `FACULTY_TARGETS`; every target has the same JSON-compatible
fields: `faculty_id`, `name`, `domain`, `start_urls`, and `allowed_path_prefixes`.
An empty path-prefix list permits the whole configured domain.

IAI and UET both request `uet.vnu.edu.vn`. IAI is restricted to
`/vien-tri-tue-nhan-tao/` and descendants, with path-segment boundaries checked after
decoding and dot-path normalization. Scoped redirects are checked before following
another URL. IAI does not scan global WordPress media; attachments come from its
accepted pages. Institute news elsewhere on UET is intentionally outside this scope.

Request hosts are independent of storage names. Existing layouts remain:

```text
data/raw_data/<storage-name>/endpoints/endpoints.txt
data/raw_data/<storage-name>/endpoints/endpoints_metadata.jsonl
data/raw_data/<storage-name>/endpoints/by_category/<category>.txt
data/raw_data/<storage-name>/files/{files_list.txt,downloaded attachments}
data/processed_data/<storage-name>.jsonl
```

Storage names derive from faculty IDs as `<lowercase-id>.uet.vnu.edu.vn`, with the
historical UET name `uet.edu.vn`. IAI retains `iai.uet.vnu.edu.vn` as a storage name,
which is not its request host. Custom single-domain crawls retain their host folder.

Processed records retain `id`, `source_type`, `source_url_or_path`, `title`, `domain`,
`unit`, `category`, `published_date`, `summary`, `content`, `content_length`, and
`cloud_links`. `domain` represents the real configured host; `unit` retains the
faculty ID. Classification metadata additionally carries `faculty_id`.
Chunk persistence groups known faculties by `unit` to preserve their existing
filenames even when they share a domain. Unknown units keep the domain fallback.

Cached records refresh attribution when extraction is run, without changing IDs.
Out-of-scope cached HTML is excluded from scoped-source output. Existing datasets
are not rewritten automatically; rerun extraction deliberately to refresh them.

## Usage

Use the existing Python 3.12 `.venv` and `uv`, from the repository root:

```bash
uv run python -m src.ingestion --crawl
uv run python -m src.ingestion --crawl --no-download
uv run python -m src.ingestion --verify
uv run python -m src.ingestion --classify
uv run python -m src.ingestion --extract
uv run python -m src.ingestion --extract --domain IAI
uv run python -m src.ingestion --extract --domain fit.uet.vnu.edu.vn --force --workers 15
uv run python -m src.ingestion --classify --domain uet.edu.vn --debug
uv run python src/ingestion/loader.py
```

For classification/extraction, `--domain` accepts a faculty ID, storage name, or
request host. Use `IAI` or its storage name to select the institute; the shared
parent host alone selects UET. Existing stage precedence and default crawling are
preserved. `--workers` must be positive; `--debug` enables per-request diagnostics.

`paths.data_dir` and `paths.processed_dir` come from root `config.yaml`, with defaults
`data/raw_data` and `data/processed_data`. `CHATBOT_DATA_DIR` and
`CHATBOT_PROCESSED_DIR` override them, including for the loader. Missing configuration
uses defaults; malformed YAML, mappings, or values warn and retain valid defaults.

Normal logs show major stages, output locations, record counts, and elapsed time.
Request/download failures retain successful results and produce an incomplete final
status. The CLI exits with status 1 for failed or incomplete work, including missing
stage inputs. Optional unavailable WordPress APIs fall back to HTML discovery;
diagnostic details remain at DEBUG. HTTP retry counts, timeouts, and existing TLS
verification behavior are preserved.

## Verification

```bash
uv run --offline python -m pytest src/ingestion/tests -v
uv run --offline python -m pytest -q
uv run --offline python -m compileall -q src/ingestion
```

The regression suite uses temporary data directories and mocked HTTP responses for
scope, redirects, retries/session setup, attribution, persistence, partial failures,
CLI status, and a crawl → validate → classify → extract → load → chunk integration.
It does not prove live site availability. Existing collection/extraction format
support remains unchanged; collection accepts more formats than the readers support.

"""Run: python3 -m src.ingestion.main ingest --from web --to raw."""

import argparse
import json
import logging
from pathlib import Path
from .config.settings import DEFAULT_DATA_DIR, DEFAULT_LOG_FILE, load_runtime


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be > 0")
    return number


def nonnegative(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be >= 0")
    return number


def build_parser():
    defaults = load_runtime()["ingestion"]
    parser = argparse.ArgumentParser(description="UET HTML-first ingestion")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Run staged ingestion")
    ingest.add_argument(
        "--from",
        dest="from_stage",
        choices=["web", "data_source", "raw"],
        default="web",
    )
    ingest.add_argument(
        "--to",
        dest="to_stage",
        choices=["data_source", "raw", "processed"],
        default="processed",
    )
    ingest.add_argument(
        "--max-pages",
        type=positive,
        default=defaults["max_pages"],
        help="Fetched pages per site per invocation",
    )
    ingest.add_argument("--max-depth", type=int, default=defaults["max_depth"])
    ingest.add_argument(
        "--limit",
        type=positive,
        default=defaults["limit"],
        help="Document budget per stage across selected sites",
    )
    ingest.add_argument(
        "--download-workers", type=positive, default=defaults["download_workers"]
    )
    ingest.add_argument(
        "--site-workers",
        type=positive,
        default=defaults["site_workers"],
        help="Concurrent sites without --limit",
    )
    ingest.add_argument(
        "--flush-every",
        type=positive,
        default=defaults["flush_every"],
        help="Commit network metadata and extraction results every N completed items",
    )
    ingest.add_argument("--per-host", type=positive, default=defaults["per_host"])
    ingest.add_argument(
        "--extract-workers", type=positive, default=defaults["extract_workers"],
        help="Number of extraction processes for raw -> processed"
    )
    ingest.add_argument(
        "--stream-extract-workers",
        type=positive,
        default=defaults["stream_extract_workers"],
        help="Shared extraction processes for web/data_source -> processed; default: download-workers",
    )
    ingest.add_argument("--timeout", type=positive, default=defaults["timeout"])
    ingest.add_argument(
        "--max-html-bytes", type=positive, default=defaults["max_html_bytes"]
    )
    ingest.add_argument("--retries", type=int, default=defaults["retries"])
    ingest.add_argument(
        "--delay",
        type=nonnegative,
        default=defaults["delay"],
        help="Delay per worker before each GET",
    )
    ingest.add_argument("--overrides", type=Path, help="Exact-URL override YAML")
    ingest.add_argument("--refresh", action="store_true")
    ingest.add_argument("--reprocess", action="store_true")
    ingest.add_argument("--retry-failed", action="store_true")
    status = commands.add_parser("status", help="Read stage counts")
    for command in (ingest, status):
        command.add_argument("--sites", nargs="+", type=str.upper)
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        command.add_argument("--profile-dir", type=Path)
    ingest.add_argument("--corpus-mode", choices=["off", "balanced", "strict"], default="balanced",
                        help="balanced: include+review; strict: include only; off: previous behavior")
    ingest.add_argument("--corpus-config", type=Path, help="Per-site corpus registry YAML")
    corpus = commands.add_parser("filter", help="Rank UET source links; do not modify ingestion data")
    corpus.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    corpus.add_argument("--sources", type=Path, help="Default: data/data_source/UET.jsonl")
    corpus.add_argument("--output-dir", type=Path, help="Default: data/corpus/UET")
    corpus.add_argument("--policy", type=Path, help="Override policy for one site")
    corpus.add_argument("--sites", nargs="+", type=str.upper, default=["UET"])
    corpus.add_argument("--corpus-config", type=Path)
    corpus.add_argument("--export-processed", action="store_true",
                        help="Export include docs under corpus/<SITE>/processed/")
    ingest.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="Path to project log file (default: logs/app.log)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        from .runner import STAGES

        if STAGES.index(args.from_stage) >= STAGES.index(args.to_stage):
            parser.error("--to must follow --from")
        if args.max_depth < 0 or args.retries < 0:
            parser.error("depth/retries must be >= 0")
        if args.refresh and args.from_stage == "raw":
            parser.error("--refresh requires web or data_source")
        if args.reprocess and args.to_stage != "processed":
            parser.error("--reprocess requires --to processed")
        if args.retry_failed and args.from_stage == "web":
            parser.error("Resume web automatically; --retry-failed is for stored data")
        if (
            args.retry_failed
            and args.from_stage == "data_source"
            and args.to_stage == "processed"
        ):
            parser.error(
                "Retry raw and processed separately to avoid ambiguous failed-stage selection"
            )
        handlers = [logging.StreamHandler()]
        log_file = args.log_file or DEFAULT_LOG_FILE
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=handlers,
            force=True,
        )
    try:
        if args.command == "filter":
            from .corpus.runner import run_filter
            result = run_filter(args)
        else:
            from .runner import run_ingestion, get_status
            result = (run_ingestion(args) if args.command == "ingest"
                      else get_status(args.data_dir, args.profile_dir, args.sites))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("status") == "failed" else 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

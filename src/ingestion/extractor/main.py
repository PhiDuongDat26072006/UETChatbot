"""Standalone HTML/PDF extractor; no crawling or vector database dependencies."""

import argparse
import json
import logging
from pathlib import Path
from ..config.settings import DEFAULT_DATA_DIR, load_runtime
from ..identity import digest, source_id, raw_id
from ..storage import write_json, atomic_bytes


def build_parser():
    defaults = load_runtime()["pdf"]
    parser = argparse.ArgumentParser(
        description="Extract local HTML or scan/refine PDF"
    )
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--html", type=Path, help="Local HTML file")
    kind.add_argument("--pdf", type=Path, help="Local PDF file (native scan)")
    kind.add_argument(
        "--refine", type=Path, help="Path to ocr_manifest.json; explicitly calls Gemini"
    )
    parser.add_argument("--url", help="Original URL for local HTML")
    parser.add_argument("--site", type=str.upper)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--model", help="Gemini model ID; required only for --refine")
    parser.add_argument(
        "--pages", type=int, nargs="+", help="1-based review page numbers"
    )
    parser.add_argument("--rpm", type=int, default=defaults["rpm"])
    parser.add_argument("--retries", type=int, default=defaults["retries"])
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.html and not args.url:
        parser.error("--html requires --url")
    if args.refine and not args.model:
        parser.error("--refine requires --model")
    if not args.refine and (args.model or args.pages):
        parser.error("--model/--pages require --refine")
    if args.refine and args.out_dir:
        parser.error("Refinement writes beside its manifest; omit --out-dir")
    if args.rpm <= 0 or args.retries < 0:
        parser.error("rpm must be positive; retries nonnegative")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        if args.refine:
            from .document.pdf_gemini import refine_pdf

            result = refine_pdf(
                args.refine, args.model, args.pages, args.rpm, args.retries
            )
        else:
            path = args.html or args.pdf
            sha = digest(path.read_bytes())
            out = args.out_dir or DEFAULT_DATA_DIR / "extractor_output" / (
                path.stem + "--" + sha[:12]
            )
            if args.pdf:
                from .document.pdf_native import scan_pdf

                result = scan_pdf(path, out)
            else:
                from bs4 import UnicodeDammit
                from src.base import RawData
                from ..config.profiles import load_profiles, resolve_profile
                from .pipeline import process_raw

                profiles = load_profiles(
                    args.profile_dir, [args.site] if args.site else None
                )
                profile = (
                    profiles[args.site]
                    if args.site
                    else resolve_profile(args.url, profiles)
                )
                if "metadata" not in RawData.model_fields:
                    raise ValueError(
                        "Rename RawData.raw_metadata to metadata in src/base.py first"
                    )
                sid = source_id(args.url)
                raw = RawData(
                    id=raw_id(sid, sha),
                    source_uri=args.url,
                    content=UnicodeDammit(
                        path.read_bytes(), is_html=True
                    ).unicode_markup
                    or "",
                    metadata={
                        "source_id": sid,
                        "site": profile["site"],
                        "faculty": profile["site"],
                        "raw_path": str(path.resolve()),
                        "content_hash": sha,
                    },
                )
                document, quality = process_raw(raw, profile)
                write_json(out / "processed.json", document.model_dump(mode="json"))
                atomic_bytes(out / "processed.md", document.content.encode())
                result = {"output": str(out), "quality": quality}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("failed_pages") else 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

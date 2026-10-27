"""Extract PDF page text/tables, then assess. PyMuPDF is lazy-imported."""

import logging
from pathlib import Path
from ...identity import digest
from ...storage import write_json, atomic_bytes
from .pdf_assessment import assess_page

log = logging.getLogger(__name__)


def extract_pdf(path):
    import fitz

    path = Path(path).resolve()
    pages = []
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("Encrypted PDF requires decryption before extraction")
        for number, page in enumerate(doc, 1):
            try:
                text = page.get_text("text", sort=True)
                blocks = [b for b in page.get_text("blocks") if len(b) < 7 or b[6] == 0]
                area = max(1, page.rect.get_area())
                coverage = min(
                    1,
                    sum(
                        (fitz.Rect(i["bbox"]) & page.rect).get_area()
                        for i in page.get_image_info()
                    )
                    / area,
                )
                tables = []
                table_error = None
                try:
                    for table in page.find_tables().tables:
                        tables.append(
                            {"bbox": list(table.bbox), "markdown": table.to_markdown()}
                        )
                except Exception as exc:
                    table_error = str(exc)
                segments = []
                # Words inside reconstructed tables are removed to avoid duplicate table text.
                lines = {}
                for word in page.get_text("words", sort=True):
                    center = fitz.Point(
                        (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
                    )
                    if any(center in fitz.Rect(t["bbox"]) for t in tables):
                        continue
                    key = (word[5], word[6])
                    lines.setdefault(key, []).append(word)
                for words in lines.values():
                    segments.append(
                        (
                            min(w[1] for w in words),
                            min(w[0] for w in words),
                            " ".join(w[4] for w in words),
                        )
                    )
                segments.extend(
                    (t["bbox"][1], t["bbox"][0], "\n" + t["markdown"] + "\n")
                    for t in tables
                )
                content = "\n".join(s[2] for s in sorted(segments))
                quality = assess_page(text, coverage, len(tables), blocks, table_error)
                pages.append(
                    {
                        "page_number": number,
                        "content": content,
                        "metadata": {"method": "native", "quality": quality},
                    }
                )
                log.info("PDF page %s/%s: %s", number, len(doc), quality["status"])
            except Exception as exc:
                pages.append(
                    {
                        "page_number": number,
                        "content": "",
                        "metadata": {
                            "method": "native",
                            "quality": {
                                "status": "MANUAL_REVIEW",
                                "reasons": [str(exc)],
                            },
                        },
                    }
                )
    return {
        "title": path.stem,
        "content": join_pages(pages),
        "metadata": {
            "source_path": str(path),
            "content_hash": digest(path.read_bytes()),
            "page_count": len(pages),
            "extraction_method": "pymupdf",
            "pages": pages,
        },
    }


def join_pages(pages):
    return "\n\n".join(f"## Trang {p['page_number']}\n\n{p['content']}" for p in pages)


def scan_pdf(path, out_dir):
    result = extract_pdf(path)
    out_dir = Path(out_dir)
    native_path = out_dir / "native.json"
    write_json(native_path, result)
    atomic_bytes(out_dir / "native.md", result["content"].encode())
    manifest = {
        "source_path": result["metadata"]["source_path"],
        "content_hash": result["metadata"]["content_hash"],
        "native_file": "native.json",
        "pages": [
            {
                "page_number": p["page_number"],
                "status": "PENDING",
                "assessment": p["metadata"]["quality"],
            }
            for p in result["metadata"]["pages"]
            if p["metadata"]["quality"]["status"] != "NATIVE_OK"
        ],
    }
    write_json(out_dir / "ocr_manifest.json", manifest)
    return {
        "native_file": str(native_path),
        "manifest": str(out_dir / "ocr_manifest.json"),
        "page_count": result["metadata"]["page_count"],
        "review_pages": len(manifest["pages"]),
    }

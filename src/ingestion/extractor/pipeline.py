"""Raw HTML -> validated ProcessedData, without network access or file writes."""

import json
from src.base import ProcessedData
from ..identity import digest
from ..config.settings import PROCESSOR_VERSION
from .html.extractor import extract_html


def processing_key(raw_record, profile):
    payload = {
        "hash": raw_record["metadata"]["content_hash"],
        "profile": {k: v for k, v in profile.items() if not k.startswith("_")},
        "processor": PROCESSOR_VERSION,
    }
    return digest(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    )


def process_raw(raw, profile, raw_path=None):
    is_pdf = raw.metadata.get("resource_type") == "pdf" or str(raw.source_uri).lower().endswith(".pdf")
    if is_pdf and raw_path and raw_path.exists():
        from .document.pdf_native import extract_pdf, scan_pdf
        pdf_res = extract_pdf(raw_path)
        pages = pdf_res["metadata"].get("pages", [])
        ocr_pages = [
            p["page_number"]
            for p in pages
            if p.get("metadata", {}).get("quality", {}).get("status") == "OCR_REQUIRED"
        ]
        extraction_method = pdf_res["metadata"]["extraction_method"]
        if ocr_pages:
            try:
                from .document.pdf_gemini import refine_pdf, _resolve_api_key
                if _resolve_api_key():
                    out_dir = raw_path.parent / "ocr"
                    manifest_path = out_dir / "ocr_manifest.json"
                    if not manifest_path.is_file():
                        scan_info = scan_pdf(raw_path, out_dir)
                        manifest_path = Path(scan_info["manifest"])
                    m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    pending = [
                        p["page_number"]
                        for p in m_data.get("pages", [])
                        if p.get("status") != "DONE"
                    ]
                    if pending or not (out_dir / "refined.json").is_file():
                        refine_pdf(
                            manifest_path,
                            model="gemini-3.5-flash",
                            pages=pending or ocr_pages,
                            rpm=30,
                            retries=6,
                        )
                    refined_file = out_dir / "refined.json"
                    if refined_file.is_file():
                        pdf_res = json.loads(refined_file.read_text(encoding="utf-8"))
                        extraction_method = pdf_res["metadata"].get(
                            "extraction_method", "native_selective_gemini"
                        )
            except Exception:
                pass
        content = pdf_res["content"]
        title = raw.title or pdf_res.get("title") or raw.source_uri
        quality = {
            "status": "accepted",
            "page_count": pdf_res["metadata"]["page_count"],
            "extraction_method": extraction_method,
            "reasons": [],
        }
        metadata = {
            **raw.metadata,
            **pdf_res["metadata"],
            "quality": quality,
            "source_uri": raw.source_uri,
            "processor_version": PROCESSOR_VERSION,
            "profile_version": profile["version"],
        }
        document = ProcessedData(
            id=raw.metadata["source_id"],
            raw_data_id=raw.id,
            title=title,
            content=content,
            metadata=metadata,
        )
        return document, quality

    result = extract_html(
        raw.content, raw.metadata.get("final_url", raw.source_uri), profile
    )
    metadata = {
        **raw.metadata,
        **result["metadata"],
        "source_uri": raw.source_uri,
        "processor_version": PROCESSOR_VERSION,
        "profile_version": profile["version"],
    }
    document = ProcessedData(
        id=raw.metadata["source_id"],
        raw_data_id=raw.id,
        title=result["title"] or raw.title or raw.source_uri,
        content=result["content"],
        metadata=metadata,
    )
    return document, metadata["quality"]

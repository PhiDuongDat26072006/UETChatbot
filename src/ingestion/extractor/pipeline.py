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
        from .document.pdf_native import extract_pdf
        pdf_res = extract_pdf(raw_path)
        content = pdf_res["content"]
        title = raw.title or pdf_res.get("title") or raw.source_uri
        quality = {
            "status": "NATIVE_OK",
            "page_count": pdf_res["metadata"]["page_count"],
            "extraction_method": "pymupdf",
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

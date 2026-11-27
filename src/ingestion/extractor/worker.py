from pathlib import Path
from ..storage import Storage
from .pipeline import process_raw


def extract_record(data_dir, record, profile):
    raw = Storage(data_dir).load_raw(record)
    raw_path = Path(data_dir) / record["metadata"]["raw_path"] if "raw_path" in record.get("metadata", {}) else None
    document, quality = process_raw(raw, profile, raw_path=raw_path)
    return document.model_dump(mode="json"), quality

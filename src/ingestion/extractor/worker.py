"""Importable process worker: read verified raw bytes, return data, never write."""

from ..storage import Storage
from .pipeline import process_raw


def extract_record(data_dir, record, profile):
    raw = Storage(data_dir).load_raw(record)
    document, quality = process_raw(raw, profile)
    return document.model_dump(mode="json"), quality

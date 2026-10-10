"""Shared input normalization and read-only JSONL loading."""
import json
import unicodedata
from pathlib import Path

from src.base import DataChunk


def normalize_chunks(chunks):
    """Normalize NFC/outer whitespace; last duplicate wins, empty text is omitted."""
    unique = {}
    for chunk in chunks:
        if not chunk.chunk_id.strip():
            raise ValueError("chunk_id must not be blank")
        text = unicodedata.normalize("NFC", chunk.text).strip()
        unique[chunk.chunk_id] = chunk.model_copy(update={"text": text})
    return [chunk for chunk in unique.values() if chunk.text]


def find_chunk_files(base_dir):
    return sorted(Path(base_dir).glob("*.jsonl"))


def load_chunks_from_dir(data_dir, domain=None, max_chunks=None):
    """Read JSONL strictly: invalid input fails before either index is changed."""
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    files = find_chunk_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No JSONL files in {data_dir}")
    chunks = []
    for path in files:
        if domain and domain.lower() not in path.name.lower():
            continue
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    item["chunk_id"] = str(item.get("chunk_id") or f"{path.stem}_{line_number}")
                    item["metadata"] = dict(item.get("metadata") or {})
                    item["metadata"].setdefault("file_name", path.name)
                    chunks.append(DataChunk.model_validate(item))
                except (ValueError, TypeError, AttributeError) as error:
                    raise ValueError(f"Invalid chunk at {path}:{line_number}") from error
    normalized = normalize_chunks(chunks)
    return normalized[:max_chunks] if max_chunks else normalized

"""Shared chunk loading, dense indexing, and persistent Vietnamese BM25 indexing."""

from src.indexing.chunks import load_chunks_from_dir

try:
    from src.indexing.indexer import UETIndexer
except ImportError:
    UETIndexer = None

try:
    from src.indexing.sparse import BM25Index
except ImportError:
    BM25Index = None

__all__ = ["UETIndexer", "load_chunks_from_dir", "BM25Index"]


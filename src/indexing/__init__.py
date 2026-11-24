"""Chunk loading, persistent BM25, and GraphRAG index; heavy dependencies load on demand."""
from src.indexing.chunks import load_chunks_from_dir

__all__ = ["UETIndexer", "load_chunks_from_dir", "BM25Index", "GraphIndex"]


def __getattr__(name):
    if name == "UETIndexer":
        from src.indexing.indexer import UETIndexer
        return UETIndexer
    if name == "BM25Index":
        from src.indexing.sparse import BM25Index
        return BM25Index
    if name == "GraphIndex":
        from src.indexing.graph import GraphIndex
        return GraphIndex
    raise AttributeError(name)

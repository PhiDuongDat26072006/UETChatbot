"""Persistent Vietnamese BM25 index. Only load trusted local pickle artifacts."""
import os
import pickle
import re
import tempfile
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Plus
from underthesea import word_tokenize

from src.base import RetrievedContext
from src.indexing.chunks import normalize_chunks

# Mixed ASCII letters/digits are identifiers, never Vietnamese word segments.
IDENTIFIER = re.compile(r"\b(?=[A-Za-z0-9_]*[A-Za-z])(?=[A-Za-z0-9_]*\d)[A-Za-z0-9_]+\b")


def tokenize(text: str) -> list[str]:
    """Segment Vietnamese words and preserve whole, case-insensitive course codes."""
    text = unicodedata.normalize("NFC", text)
    tokens = []
    start = 0
    for match in IDENTIFIER.finditer(text):
        tokens.extend(_segment(text[start:match.start()]))
        tokens.append(match.group().casefold())
        start = match.end()
    tokens.extend(_segment(text[start:]))
    return tokens


def _segment(text):
    return [word.casefold().replace(" ", "_") for word in word_tokenize(text)
            if any(char.isalnum() for char in word)] if text.strip() else []


class BM25Index:
    """Keep chunk order aligned with BM25 statistics and persist without re-tokenizing."""

    def __init__(self, chunks=()):
        self.chunks = normalize_chunks(chunks)
        self.tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.model = BM25Plus(self.tokens) if any(self.tokens) else None

    def save(self, path):
        """Atomically replace a trusted local artifact, including fitted statistics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                pickle.dump({"version": 1, "index": self}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        """Load fitted statistics directly; never load an untrusted pickle."""
        with Path(path).open("rb") as stream:
            state = pickle.load(stream)
        if state.get("version") != 1 or not isinstance(state.get("index"), cls):
            raise ValueError("Unsupported sparse index format; rebuild required")
        return state["index"]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedContext]:
        """Return lexical matches with raw BM25 scores (not cosine similarities)."""
        if top_k <= 0 or self.model is None:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scores = self.model.get_scores(sorted(query_tokens))
        # BM25Plus gives a baseline even to nonmatches; exclude those explicitly.
        matches = [i for i, tokens in enumerate(self.tokens) if query_tokens.intersection(tokens)]
        matches.sort(key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [RetrievedContext(chunk_id=self.chunks[i].chunk_id, text=self.chunks[i].text,
                                 metadata=self.chunks[i].metadata, similarity_score=float(scores[i]), rank=rank)
                for rank, i in enumerate(matches[:top_k], 1)]

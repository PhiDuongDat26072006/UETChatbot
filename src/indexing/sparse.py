"""Persistent Vietnamese BM25 index. Only load trusted local pickle artifacts."""
import os
import pickle
import re
import tempfile
import unicodedata
from pathlib import Path

from src.utils.helpers import get_logger

logger = get_logger("indexing.sparse")

try:
    from rank_bm25 import BM25Plus
except ImportError:
    logger.warning("Không tìm thấy thư viện 'rank_bm25'. Đang sử dụng lớp BM25Plus dự phòng (Pure-Python fallback).")
    class BM25Plus:
        """Pure-Python fallback for BM25Plus when rank_bm25 is not installed."""
        def __init__(self, corpus, k1=1.5, b=0.75, delta=1.0):
            import math
            self.k1 = k1
            self.b = b
            self.delta = delta
            self.corpus_size = len(corpus)
            self.doc_len = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0
            self.doc_freqs = []
            self.nd = {}
            for doc in corpus:
                frequencies = {}
                for word in doc:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    self.nd[word] = self.nd.get(word, 0) + 1
            self.idf = {}
            for word, freq in self.nd.items():
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        def get_scores(self, query):
            scores = [0.0] * self.corpus_size
            for q in query:
                q_idf = self.idf.get(q, 0.0)
                for idx, doc_freq in enumerate(self.doc_freqs):
                    freq = doc_freq.get(q, 0)
                    if freq > 0:
                        len_norm = 1.0 - self.b + self.b * (self.doc_len[idx] / (self.avgdl or 1.0))
                        tf_comp = (freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm)
                        scores[idx] += q_idf * (tf_comp + self.delta)
            return scores

try:
    from underthesea import word_tokenize
except ImportError:
    logger.warning("Không tìm thấy thư viện 'underthesea'. Đang sử dụng hàm word_tokenize dự phòng (Regex fallback).")
    def word_tokenize(text):
        clean = text.lower().strip()
        words = re.findall(r"[\w\-]+", clean)
        tokens = list(words)
        for i in range(len(words) - 1):
            if len(words[i]) > 1 and len(words[i+1]) > 1:
                tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

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

    def count(self) -> int:
        """Trả về tổng số chunk trong chỉ mục."""
        return len(self.chunks)

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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file BM25 index tại: {path}")
        with path.open("rb") as stream:
            state = pickle.load(stream)
        if isinstance(state, dict) and "index" in state:
            return state["index"]
        if isinstance(state, cls):
            return state
        return state

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

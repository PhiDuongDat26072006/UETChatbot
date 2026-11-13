"""Persistent Vietnamese BM25 index. Only load trusted local pickle artifacts."""
import heapq
import os
import pickle
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Optional
from src.utils.helpers import get_logger

# Giới hạn thread của BLAS/OMP trước khi nạp numpy/scipy/underthesea
# để tránh tràn bộ nhớ ảo khi chạy đa tiến trình trên Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logger = get_logger("indexing.sparse")

from src.base import RetrievedContext
from src.indexing.chunks import normalize_chunks

from concurrent.futures import ProcessPoolExecutor
from rank_bm25 import BM25Plus

# Patch underthesea CRF model loader on Windows when path contains non-ASCII characters
try:
    import tempfile, shutil
    import underthesea.models.fast_crf_sequence_tagger as tagger_mod
    _orig_crf_load = tagger_mod.FastCRFSequenceTagger.load
    def _safe_crf_load(self, base_path):
        try:
            return _orig_crf_load(self, base_path)
        except Exception:
            tmp_base = os.path.join(tempfile.gettempdir(), "underthesea_models")
            os.makedirs(tmp_base, exist_ok=True)
            for fname in [self.path_model, self.path_features, self.path_dictionary]:
                src_f = os.path.join(base_path, fname)
                dst_f = os.path.join(tmp_base, fname)
                if os.path.exists(src_f) and (not os.path.exists(dst_f) or os.path.getsize(dst_f) != os.path.getsize(src_f)):
                    shutil.copyfile(src_f, dst_f)
            return _orig_crf_load(self, tmp_base)
    tagger_mod.FastCRFSequenceTagger.load = _safe_crf_load
except Exception:
    pass

from underthesea import word_tokenize

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


def _segment(text: str) -> list[str]:
    """Tokenize using underthesea"""
    return [
        word.casefold().replace(" ", "_")
        for word in word_tokenize(text)
        if any(char.isalnum() for char in word)
    ] if text.strip() else []


def _resolve_workers(workers: Optional[int], num_docs: int) -> int:
    """Chọn số worker an toàn theo RAM khả dụng để tránh đơ máy / MemoryError trên Windows."""
    if num_docs < 500:
        return 1

    if workers is not None:
        if workers < 1:
            raise ValueError("workers must be positive")
        return workers

    max_cap = min(6, os.cpu_count() or 1)
    try:
        import psutil
        avail_bytes = psutil.virtual_memory().available
        # Mỗi worker process nạp underthesea tốn ~520-700 MB; giữ tối thiểu 1.5 GB cho OS & main process.
        safe_workers = int((avail_bytes - 1_500_000_000) // 700_000_000)
        return max(1, min(max_cap, safe_workers))
    except Exception:
        return 1 if sys.platform.startswith("win") else max(1, min(4, max_cap))


class BM25Index:
    """Keep chunk order aligned with BM25 statistics and persist without re-tokenizing."""

    def __init__(self, chunks=(), workers: Optional[int] = None):
        self.chunks = normalize_chunks(chunks)

        # Retriever may need an empty index before a persisted artifact exists.
        # Do not create worker processes or fit BM25 for an empty corpus.
        if not self.chunks:
            self.tokens = []
            self.model = None
            self.postings = {}
            self.build_timings = {}
            return

        texts = [chunk.text for chunk in self.chunks]
        active_workers = _resolve_workers(workers, len(texts))
        logger.info(f"Đang tách từ BM25 cho {len(texts)} chunks (workers={active_workers})...")

        started = time.perf_counter()
        # 1. Tokenize corpus
        if active_workers == 1:
            self.tokens = [tokenize(text) for text in texts]
        else:
            with ProcessPoolExecutor(max_workers=active_workers) as executor:
                self.tokens = list(
                    executor.map(
                        tokenize,
                        texts,
                        chunksize=200
                    )
                )

        tokenized = time.perf_counter()
        # 2. Build BM25
        self.model = (
            BM25Plus(self.tokens)
            if any(self.tokens)
            else None
        )

        constructed = time.perf_counter()
        # 3. Build Inverted Index
        self.postings = defaultdict(list)

        if self.model is not None:
            for doc_idx, frequencies in enumerate(self.model.doc_freqs):
                for token in frequencies:
                    self.postings[token].append(doc_idx)

        self.postings = dict(self.postings)
        self.build_timings = {
            "tokenization": tokenized - started,
            "bm25_construction": constructed - tokenized,
            "inverted_index": time.perf_counter() - constructed,
        }

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

        index = None
        if isinstance(state, cls):
            index = state  # Backward compatibility with direct-class pickles.
        elif isinstance(state, dict) and state.get("version") == 1:
            raw_index = state.get("index")
            if isinstance(raw_index, cls):
                index = raw_index

        if index is not None:
            # Tự động dựng postings nếu artifact được tạo từ phiên bản cũ
            if not hasattr(index, "postings") or not index.postings:
                postings = defaultdict(list)
                if hasattr(index, "model") and index.model is not None and hasattr(index.model, "doc_freqs"):
                    for doc_idx, frequencies in enumerate(index.model.doc_freqs):
                        for token in frequencies:
                            postings[token].append(doc_idx)
                index.postings = dict(postings)
            return index

        raise ValueError(f"Invalid or incompatible BM25 artifact: {path}")

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[RetrievedContext]:
        """Retrieve Top-K documents using inverted index and BM25Plus."""

        if top_k <= 0 or self.model is None:
            return []

        # 1. Tokenize query
        query_tokens = sorted(set(tokenize(query)))

        if not query_tokens:
            return []

        # 2. Find candidate documents
        candidates = self.get_candidates(query_tokens)

        if not candidates:
            return []

        # 3. Score only candidate documents
        scores = self.model.get_batch_scores(
            query_tokens,
            candidates
        )

        # 4. Top-K results
        top_results = heapq.nlargest(
            top_k,
            zip(candidates, scores),
            key=lambda item: float(item[1])
        )

        # 5. Return Top-K results
        results = []

        for rank, (doc_idx, score) in enumerate(top_results, start=1):
            chunk = self.chunks[doc_idx]

            results.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    similarity_score=float(score),
                    rank=rank
                )
            )

        return results

    def get_candidates(self, query_tokens: list[str]) -> list[int]:
        """Return sorted list of chunk indices that contain any query tokens."""
        if not hasattr(self, "postings") or not self.postings:
            postings = defaultdict(list)
            if hasattr(self, "model") and self.model is not None and hasattr(self.model, "doc_freqs"):
                for doc_idx, frequencies in enumerate(self.model.doc_freqs):
                    for token in frequencies:
                        postings[token].append(doc_idx)
            self.postings = dict(postings)

        candidates = set()
        for token in query_tokens:
            if token in self.postings:
                candidates.update(self.postings[token])
        return sorted(candidates)

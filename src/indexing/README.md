# Dense and sparse indexing

`UETIndexer` normalizes one `DataChunk` collection and writes both indexes:

- Dense: `DataChunk → UETEmbedder → ChromaDB`, for semantic similarity.
- Sparse: `DataChunk.text → Vietnamese tokenization → rank-bm25 BM25Plus`, for lexical matching.

`chunks.py` owns strict JSONL loading and normalization; `sparse.py` owns
tokenization, BM25, persistence and search; `indexer.py` coordinates writes.
No new base interfaces are required. Both ingestion indexing entry points also
use this coordinator rather than duplicating dense-only batching.

## Build and rebuild

Install `requirements.txt`, then run from the project root:

```bash
python -m src.indexing --data-dir data/chunked_data
python -m src.indexing --data-dir data/chunked_data --clear
python -m src.indexing --status
```

Use the actual directory containing your chunks: the standalone CLI retains
its previous default `data/chunk_data`; the staged ingestion pipeline uses
`data/chunked_data`, while `UETChunker.chunk_batch` also produces `chunk_data`.
No source JSONL is changed by indexing.

`--persist-dir`, `--collection-name`, `--batch-size`, `--domain` (filename filter)
and `--sample` are supported. **Rebuild replaces the entire selected collection**,
so combining `--clear` with filtering or sampling intentionally removes all
records outside that subset. Empty `index_chunks([], clear_existing=True)` clears
both indexes. A missing JSONL directory or malformed record fails before writes.

## Normalization and tokenization

Text uses Unicode NFC and trimmed outer whitespace. IDs stay unchanged;
duplicates use the last occurrence, and empty text is omitted. Punctuation-only
chunks remain in both stores but have no searchable sparse tokens.

[Underthesea word segmentation](https://github.com/undertheseanlp/underthesea/blob/main/docs/docs/api/word-tokenize.md)
joins Vietnamese words such as `sinh viên` into `sinh_viên`. Mixed ASCII
letter/digit identifiers are protected before segmentation: `INT3401` becomes
`int3401`, never a prefix match for `INT34010`. Queries use exactly the same
tokenizer. Case folding applies only to sparse tokens, not stored chunk text.
[rank-bm25](https://github.com/dorianbrown/rank_bm25) supplies BM25Plus statistics
and scoring. Nonmatching documents are excluded despite its score baseline.

## Persistence and consistency

Chroma persists in `--persist-dir`; a collection-specific `bm25-<hash>.pkl`
beside it contains chunks, tokens, and fitted BM25 statistics. Loading does not
rebuild or re-tokenize. Sparse writes use an atomic file replacement. Only load
trusted local pickle artifacts; keep dependency versions compatible, or rebuild.

Updates upsert by `chunk_id`. To replace a complete document, including removing
its obsolete chunks or deleting a now-empty document:

```python
indexer.index_chunks(updated_chunks, replace_document_ids=[document_id])
indexer.index_chunks([], replace_document_ids=[deleted_document_id])
```

An existing dense-only collection is used to seed BM25 on the next indexing run.
Before publishing sparse state, the coordinator compares the complete dense
ID/text mapping with the intended corpus, not merely record counts. Status also
checks this mapping and a persistent pending-write marker. Invalid embedding
counts/IDs, incomplete writes, and storage failures raise rather than succeed.

Chroma and the sparse file do not share a transaction. A failure after writes
begin leaves a pending marker; status reports `consistent=false`, further updates
fail, and `--clear` rebuilds from the authoritative full chunk corpus. Corrupt
sparse files likewise require a rebuild. There is no rollback of partial dense
writes. Use one writer per collection, and pause retrieval during indexing.
Future callers must check `get_status()['consistent']` before querying both stores.

## Query interfaces for future retrievers

```python
from src.base import UserQuery
from src.indexing import UETIndexer, BM25Index

indexer = UETIndexer()
if not indexer.get_status()["consistent"]:
    raise RuntimeError("Rebuild indexes before retrieval")
sparse_hits = indexer.sparse_index.search("INT3401", top_k=5)
query_vector = indexer.embedding_model.embed_query(UserQuery(query_text="học bổng"))
dense_hits = indexer.vector_store.search(query_vector, top_k=5)
# Independently load a trusted fitted sparse index:
sparse = BM25Index.load(indexer.sparse_path)
```

Both return `RetrievedContext` with shared chunk IDs. Sparse `similarity_score`
is a raw BM25 score, not a bounded cosine score. This change does not wire sparse
search into the existing retriever and does not add fusion or RRF.

BM25 statistics are recomputed over the corpus on updates because this small
library is not incremental. Search and consistency checks scan the corpus, and
the sparse artifact is held in memory. The existing embedder's hash-vector
fallback is preserved; embedding model quality/fallback policy is unchanged.

Run lifecycle and regression tests with:

```bash
python -m pytest tests/test_indexing.py tests/test_ingestion_stages.py tests/test_retrieval.py
```

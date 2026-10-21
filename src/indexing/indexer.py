"""Coordinate dense Chroma and persistent BM25 indexes over shared chunks."""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from src.ingestion.extractor import pipeline_domain
from src.embeddings.embedder import UETEmbedder
from src.vectordb.vector_store import UETVectorStore
from src.indexing.chunks import find_chunk_files, load_chunks_from_dir, normalize_chunks
from src.indexing.sparse import BM25Index
from src.utils.helpers import get_logger

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logger = get_logger("indexing")


class UETIndexer:
    """Single-writer coordinator; interrupted writes require a full rebuild."""

    def __init__(
        self,
        vector_store=None,
        embedding_model=None,
        persist_dir=None,
        collection_name="uet_knowledge_base",
    ):
        # 1. Configure storage paths
        default_dir = getattr(vector_store, "persist_dir", BASE_DIR / "vector_db")
        self.persist_dir = Path(persist_dir or default_dir)
        self.collection_name = collection_name

        # 2. Initialize indexing dependencies
        self.embedding_model = embedding_model or UETEmbedder()
        self.vector_store = vector_store or UETVectorStore(
            persist_dir=self.persist_dir,
            collection_name=collection_name,
            embedding_model=self.embedding_model,
        )

        # 3. Define index artifacts
        self.sparse_path = BASE_DIR / "database" / "bm25_index.pkl"
        self.pending_path = self.sparse_path.with_suffix(".pending")
        self.sparse_load_error = None

        # 4. Load existing BM25
        try:
            if self.sparse_path.exists():
                self.sparse_index = BM25Index.load(self.sparse_path)
            else:
                self.sparse_index = BM25Index()
        except (
            OSError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            ImportError,
        ) as error:
            self.sparse_load_error = str(error)
            self.sparse_index = BM25Index()
            logger.error("Sparse index cannot be loaded; rebuild required: %s", error)

    def _consistent(self):
        # Compare dense and sparse
        dense = {
            chunk.chunk_id: chunk.text
            for chunk in self.vector_store.get_chunks()
        }
        sparse = {
            chunk.chunk_id: chunk.text
            for chunk in self.sparse_index.chunks
        }

        # Check pending recovery state
        return (
            self.sparse_load_error is None
            and not self.pending_path.exists()
            and dense == sparse
        )

    def index_chunks(
        self,
        chunks,
        batch_size=128,
        clear_existing=False,
        replace_document_ids=None,
    ):
        """Upsert or rebuild shared chunks."""

        # 1. Validate input
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        started = time.monotonic()
        logger.info("Indexing started")
        chunks = normalize_chunks(chunks)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Check previous state
        if self.pending_path.exists() and not clear_existing:
            raise RuntimeError(
                "Interrupted indexing detected; rebuild with --clear"
            )

        existing = (
            []
            if clear_existing
            else self.vector_store.get_chunks()
        )

        if (
            not clear_existing
            and self.sparse_path.exists()
            and not self._consistent()
        ):
            raise RuntimeError(
                "Dense and sparse indexes differ; rebuild with --clear"
            )

        # 3. Identify stale chunks
        replaced = set(replace_document_ids or ())
        incoming_ids = {chunk.chunk_id for chunk in chunks}

        stale = [
            chunk.chunk_id
            for chunk in existing
            if chunk.document_id in replaced
            and chunk.chunk_id not in incoming_ids
        ]

        # 4. Build target corpus
        remaining = [
            chunk
            for chunk in existing
            if chunk.chunk_id not in stale
        ]
        target = normalize_chunks(remaining + chunks)

        # 5. Build sparse index
        logger.info("Stage: BM25 construction (%s chunks)", len(target))
        candidate = BM25Index(target)

        # 6. Create recovery marker
        if not self.pending_path.exists():
            descriptor = os.open(
                self.pending_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)

        processed = 0

        try:
            # 7. Prepare dense storage
            logger.info("Stage: dense indexing")
            if clear_existing:
                self.vector_store.clear()
            elif stale:
                self.vector_store.delete(stale)

            # 8. Embed and store
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                vectors = self.embedding_model.embed_chunks(batch)

                # Check embedding IDs
                vector_ids = [vector.chunk_id for vector in vectors]
                batch_ids = [chunk.chunk_id for chunk in batch]
                if vector_ids != batch_ids:
                    raise ValueError(
                        "Embedding chunk IDs/count do not match input"
                    )

                # Store vector batch
                saved = self.vector_store.store(batch, vectors)
                if saved != len(batch):
                    raise RuntimeError(
                        "Dense store reported an incomplete write"
                    )

                processed += saved
                logger.info("Processed %s/%s chunks", processed, len(chunks))

            # 9. Verify dense consistency
            actual = {
                chunk.chunk_id: chunk.text
                for chunk in self.vector_store.get_chunks()
            }
            expected = {
                chunk.chunk_id: chunk.text
                for chunk in target
            }
            if actual != expected:
                raise RuntimeError(
                    "Dense index does not match the intended sparse corpus"
                )

            # 10. Persist sparse index
            logger.info("Stage: sparse persistence")
            candidate.save(self.sparse_path)
            self.sparse_index = candidate
            self.sparse_load_error = None

            # 11. Remove recovery marker
            self.pending_path.unlink()

        except Exception:
            # 12. Report indexing failure
            logger.exception(
                "Indexing failed; rebuild required (%.2fs)",
                time.monotonic() - started,
            )
            raise

        # 13. Return indexing statistics
        elapsed = round(time.monotonic() - started, 2)
        logger.info(
            "Indexing success: %s chunks, %.2fs; dense=%s sparse=%s",
            processed,
            elapsed,
            self.persist_dir,
            self.sparse_path,
        )
        return {
            "status": "success" if target or clear_existing else "empty",
            "indexed_chunks": processed,
            "total_in_db": len(target),
            "total_in_sparse": len(target),
            "time_seconds": elapsed,
            "average_speed": round(processed / max(elapsed, 0.1), 1),
        }

    def index_from_dir(
        self,
        data_dir="data/chunk_data",
        batch_size=128,
        domain=None,
        max_chunks=None,
        clear_existing=False,
    ):
        # 1. Resolve chunk directory
        path = Path(data_dir)
        if not path.is_absolute():
            path = BASE_DIR / path

        # 2. Load input chunks
        chunks = load_chunks_from_dir(path, domain, max_chunks)

        # 3. Start indexing pipeline
        return self.index_chunks(chunks, batch_size, clear_existing)

    def get_status(self):
        """Check actual IDs/text and pending writes, rather than counts alone."""

        # 1. Read index counts
        count = self.vector_store.count()
        consistent = self._consistent()

        # 2. Return index status
        return {
            "persist_dir": str(self.persist_dir),
            "sparse_path": str(self.sparse_path),
            "collection_name": self.collection_name,
            "total_vectors": count,
            "total_sparse": len(self.sparse_index.chunks),
            "consistent": consistent,
            "rebuild_required": not consistent,
            "is_ready_for_demo": count > 0 and consistent,
        }


def main():
    # 1. Parse command options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/chunk_data")
    parser.add_argument("--persist-dir", default=str(BASE_DIR / "database" / "vector_db"))
    parser.add_argument("--collection-name", default="uet_knowledge_base")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--domain")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Rebuild both indexes, removing stale records",
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    try:
        # 2. Initialize indexing coordinator
        indexer = UETIndexer(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
        )

        # 3. Run requested operation
        if args.status:
            result = indexer.get_status()
        else:
            result = indexer.index_from_dir(
                args.data_dir,
                args.batch_size,
                args.domain,
                args.sample,
                args.clear,
            )

        # 4. Print execution result
        print(json.dumps(result, ensure_ascii=False))
        return int(args.status and not result["consistent"])

    except Exception as error:
        # 5. Handle execution errors
        logger.error("Indexing failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())

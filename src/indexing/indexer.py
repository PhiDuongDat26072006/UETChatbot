"""Validate chunks against dense storage, then build and save BM25."""

import argparse
import json
import sys
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.indexing.chunks import load_chunks_from_dir
from src.indexing.sparse import BM25Index


class UETIndexer:
    def __init__(
        self,
        vector_store=None,
        embedding_model=None,
        persist_dir=None,
        collection_name=None,
        sparse_path=None,
        workers=6,
    ):
        if workers < 1:
            raise ValueError("workers must be positive")

        self.vector_store = vector_store
        self.persist_dir = Path(
            persist_dir
            or getattr(
                vector_store,
                "persist_dir",
                BASE_DIR / "database" / "vector_db",
            )
        )
        self.collection_name = getattr(
            vector_store, "collection_name", collection_name
        )
        self.sparse_path = Path(
            sparse_path or BASE_DIR / "database" / "bm25_index.pkl"
        )
        self.workers = workers

        # Giữ tham số embedding_model để tương thích caller cũ.
        # Indexer không sử dụng hoặc gọi model này.

    def _get_vector_store(self):
        """Dùng instance được truyền vào hoặc class VectorStore hiện có."""
        if self.vector_store is None:
            database_file = self.persist_dir / "chroma.sqlite3"
            if not database_file.is_file():
                raise FileNotFoundError(
                    f"Dense database does not exist: {self.persist_dir}"
                )

            from src.vectordb import UETVectorStore

            self.vector_store = UETVectorStore(
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )

        return self.vector_store

    @staticmethod
    def _corpus(chunks):
        """So sánh nội dung sau khi thống nhất Unicode và khoảng trắng ngoài."""
        corpus = {}

        for chunk in chunks:
            if not chunk.chunk_id.strip():
                raise ValueError("chunk_id must not be blank")

            text = unicodedata.normalize("NFC", chunk.text).strip()

            if not text:
                raise ValueError(
                    f"Text must not be blank: {chunk.chunk_id}"
                )

            if chunk.chunk_id in corpus:
                raise ValueError(
                    f"Duplicate chunk_id: {chunk.chunk_id}"
                )

            corpus[chunk.chunk_id] = text

        return corpus

    def _check_dense(self, chunks):
        """Báo lỗi nếu corpus đầu vào khác corpus trong VectorStore."""
        expected = self._corpus(chunks)

        store = self._get_vector_store()
        dense = self._corpus(store.get_chunks())

        if store.count() != len(dense):
            raise RuntimeError(
                "Incomplete dense read or dense changed during check"
            )

        if expected != dense:
            missing = sorted(expected.keys() - dense.keys())
            extra = sorted(dense.keys() - expected.keys())
            changed = sorted(
                chunk_id
                for chunk_id in expected.keys() & dense.keys()
                if expected[chunk_id] != dense[chunk_id]
            )

            raise ValueError(
                "Chunks do not match dense: "
                f"missing_in_dense={len(missing)} {missing[:10]}, "
                f"extra_in_dense={len(extra)} {extra[:10]}, "
                f"different_text={len(changed)} {changed[:10]}"
            )

        return len(dense)

    def index_chunks(
        self,
        chunks,
        batch_size=128,
        clear_existing=False,
        replace_document_ids=None,
    ):
        """Đối chiếu dense trước, sau đó build và lưu BM25."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if replace_document_ids:
            raise ValueError(
                "Update dense externally, then pass the full matching corpus"
            )

        chunks = list(chunks)

        # 1. Input phải khớp toàn bộ corpus dense.
        dense_count = self._check_dense(chunks)

        # 2. Build BM25 bằng implementation hiện có.
        candidate = BM25Index(chunks, workers=self.workers)

        # 3. Kiểm tra lại trước khi ghi:
        #    - Dense chưa thay đổi trong lúc build.
        #    - Corpus sau normalize của BM25 vẫn khớp dense.
        self._check_dense(candidate.chunks)

        # 4. Save hiện có trong sparse.py đã dùng atomic replacement.
        candidate.save(self.sparse_path)

        # Mỗi lần build thay thế toàn bộ BM25.
        # clear_existing không bao giờ xóa dense.
        return {
            "status": "success" if chunks else "empty",
            "indexed_chunks": candidate.count(),
            "total_in_db": dense_count,
            "total_in_sparse": candidate.count(),
            "sparse_path": str(self.sparse_path),
        }

    def index_from_dir(
        self,
        data_dir="data/chunked_data",
        batch_size=128,
        domain=None,
        max_chunks=None,
        clear_existing=False,
    ):
        """Dùng hàm load chunks hiện có rồi gọi index_chunks."""
        path = Path(data_dir)
        if not path.is_absolute():
            path = BASE_DIR / path

        chunks = load_chunks_from_dir(path, domain, max_chunks)

        return self.index_chunks(
            chunks,
            batch_size=batch_size,
            clear_existing=clear_existing,
        )

    def get_status(self):
        """Kiểm tra dense và artifact BM25 hiện tại, không build."""
        store = self._get_vector_store()
        dense = self._corpus(store.get_chunks())
        dense_count = store.count()

        sparse = None
        error = None

        try:
            sparse = self._corpus(
                BM25Index.load(self.sparse_path).chunks
            )
        except Exception as exc:
            error = str(exc)

        consistent = (
            sparse is not None
            and dense_count == len(dense)
            and dense == sparse
        )

        return {
            "total_vectors": dense_count,
            "total_sparse": len(sparse) if sparse is not None else 0,
            "consistent": consistent,
            "rebuild_required": not consistent,
            "is_ready_for_demo": consistent and bool(dense),
            "sparse_path": str(self.sparse_path),
            "sparse_load_error": error,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-dir", default="data/chunked_data")
    parser.add_argument(
        "--persist-dir",
        default=str(BASE_DIR / "database" / "vector_db"),
    )
    parser.add_argument(
        "--collection-name",
        default="uet_knowledge_base",
    )
    parser.add_argument(
        "--sparse-path",
        default=str(BASE_DIR / "database" / "bm25_index.pkl"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--domain")
    parser.add_argument("--sample", type=int)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Rebuild BM25 only; never clear dense",
    )
    parser.add_argument("--status", action="store_true")

    args = parser.parse_args()

    try:
        indexer = UETIndexer(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            sparse_path=args.sparse_path,
            workers=args.workers,
        )

        if args.status:
            result = indexer.get_status()
        else:
            result = indexer.index_from_dir(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                domain=args.domain,
                max_chunks=args.sample,
                clear_existing=args.clear,
            )

        print(json.dumps(result, ensure_ascii=False))
        return int(args.status and not result["consistent"])

    except Exception as error:
        print(json.dumps(
            {"status": "error", "error": str(error)},
            ensure_ascii=False,
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
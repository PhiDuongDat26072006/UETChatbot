"""Index lifecycle tests using real Chroma and deterministic local embeddings."""
import json
import unicodedata
from unittest.mock import Mock, patch

import pytest

from src.base import BaseEmbeddingModel, DataChunk, UserQuery
from src.indexing.chunks import load_chunks_from_dir
from src.indexing.indexer import UETIndexer
from src.indexing.sparse import BM25Index, tokenize
from src.vectordb.vector_store import UETVectorStore


class TestEmbedder(BaseEmbeddingModel):
    __test__ = False
    dimension = 3
    model_name = "test"

    def embed_texts(self, texts):
        return [[1.0, float(len(text)), 0.5] for text in texts]


def chunk(cid, text="Sinh viên học INT3401", document_id="doc"):
    return DataChunk(chunk_id=cid, text=text, document_id=document_id)


@pytest.fixture
def indexer(tmp_path):
    embedder = TestEmbedder()
    store = UETVectorStore(persist_dir=tmp_path, embedding_model=embedder)
    return UETIndexer(vector_store=store, embedding_model=embedder, persist_dir=tmp_path)


def test_vietnamese_and_identifiers():
    tokens = tokenize("Sinh viên học công nghệ thông tin INT3401, INT34010.")
    assert "sinh_viên" in tokens
    assert "int3401" in tokens and "int34010" in tokens
    assert tokenize(unicodedata.normalize("NFD", "sinh viên")) == tokenize("sinh viên")
    assert tokenize("  ... ") == []


def test_sparse_construction_duplicates_empty_and_exact_matching():
    index = BM25Index([chunk("a", "old"), chunk("a"), chunk("b", "INT34010"),
                       chunk("empty", "  "), chunk("punctuation", "...")])
    assert len(index.chunks) == 3
    assert [hit.chunk_id for hit in index.search("INT3401")] == ["a"]
    assert index.search("INT340") == []
    assert index.search("unknown") == []
    assert index.search("INT3401", 0) == []
    assert BM25Index().search("INT3401") == []
    assert BM25Index([chunk("p", "...")]).search("INT3401") == []


def test_persistence_does_not_rebuild(tmp_path):
    index = BM25Index([chunk("a"), chunk("b", "học bổng")])
    path = tmp_path / "sparse.pkl"
    index.save(path)
    with patch("src.indexing.sparse.tokenize", side_effect=AssertionError("re-tokenized")):
        loaded = BM25Index.load(path)
    assert loaded.search("INT3401") == index.search("INT3401")


def test_dense_regression_and_shared_normalization(indexer):
    result = indexer.index_chunks([chunk("a", "  Sinh viên INT3401  "), chunk("a"), chunk("b", " ")])
    assert result["indexed_chunks"] == 1
    assert indexer.get_status()["consistent"]
    hits = indexer.vector_store.search(indexer.embedding_model.embed_query(UserQuery(query_text="INT3401")))
    assert hits[0].chunk_id == "a"
    assert hits[0].text == indexer.sparse_index.chunks[0].text


def test_reindex_reload_and_full_rebuild(indexer):
    indexer.index_chunks([chunk("a"), chunk("stale", "old")])
    indexer.index_chunks([chunk("a", "INT3402")])
    assert indexer.vector_store.count() == 2
    assert indexer.sparse_index.search("INT3401") == []
    assert indexer.sparse_index.search("INT3402")[0].chunk_id == "a"
    reopened = UETIndexer(vector_store=indexer.vector_store, embedding_model=TestEmbedder())
    assert reopened.get_status()["consistent"]
    reopened.index_chunks([chunk("new")], clear_existing=True)
    assert {c.chunk_id for c in reopened.vector_store.get_chunks()} == {"new"}
    assert reopened.get_status()["consistent"]
    reopened.index_chunks([], clear_existing=True)
    assert reopened.vector_store.count() == 0
    assert reopened.get_status()["consistent"]


def test_replace_document_removes_obsolete_chunks(indexer):
    indexer.index_chunks([chunk("a"), chunk("b"), chunk("other", document_id="other")])
    indexer.index_chunks([chunk("new")], replace_document_ids=["doc"])
    assert {c.chunk_id for c in indexer.sparse_index.chunks} == {"new", "other"}
    indexer.index_chunks([], replace_document_ids=["doc"])
    assert {c.chunk_id for c in indexer.sparse_index.chunks} == {"other"}
    assert indexer.get_status()["consistent"]


@pytest.mark.parametrize("failure_stage", ["dense", "sparse", "clear"])
def test_partial_failure_is_visible_and_rebuild_recovers(indexer, failure_stage):
    indexer.index_chunks([chunk("old")])
    target = {"dense": (indexer.vector_store, "store"),
              "sparse": (BM25Index, "save"), "clear": (indexer.vector_store, "clear")}[failure_stage]
    with patch.object(*target, side_effect=RuntimeError("injected failure")):
        with pytest.raises(RuntimeError, match="injected"):
            indexer.index_chunks([chunk("new")], clear_existing=failure_stage == "clear")
    assert not indexer.get_status()["consistent"]
    reopened = UETIndexer(vector_store=indexer.vector_store, embedding_model=TestEmbedder())
    with pytest.raises(RuntimeError, match="Interrupted"):
        reopened.index_chunks([chunk("new")])
    reopened.index_chunks([chunk("new")], clear_existing=True)
    assert reopened.get_status()["consistent"]


def test_equal_counts_different_ids_are_inconsistent(indexer):
    indexer.index_chunks([chunk("a")])
    indexer.vector_store.clear()
    indexer.vector_store.store([chunk("b")])
    assert not indexer.get_status()["consistent"]
    with pytest.raises(RuntimeError, match="differ"):
        indexer.index_chunks([chunk("c")])


def test_embedding_count_mismatch_fails(indexer):
    with patch.object(indexer.embedding_model, "embed_chunks", return_value=[]):
        with pytest.raises(ValueError, match="IDs/count"):
            indexer.index_chunks([chunk("a")])
    assert not indexer.get_status()["consistent"]


def test_migrate_existing_dense_store(indexer):
    indexer.vector_store.store([chunk("old")])
    indexer.index_chunks([chunk("new")])
    assert {c.chunk_id for c in indexer.sparse_index.chunks} == {"old", "new"}
    assert indexer.get_status()["consistent"]


def test_loader_is_read_only_and_strict(tmp_path):
    path = tmp_path / "uet.jsonl"
    original = '\n'.join(c.model_dump_json() for c in [chunk("a", "old"), chunk("a"), chunk("b", " ")])
    path.write_text(original, encoding="utf-8")
    assert [c.text for c in load_chunks_from_dir(tmp_path)] == ["Sinh viên học INT3401"]
    assert path.read_text(encoding="utf-8") == original
    path.write_text('{broken', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid chunk"):
        load_chunks_from_dir(tmp_path)


def test_invalid_batch_has_no_writes(indexer):
    with pytest.raises(ValueError, match="positive"):
        indexer.index_chunks([chunk("a")], batch_size=0)
    assert not indexer.pending_path.exists()


def test_clear_propagates_chroma_failure(indexer):
    with patch.object(indexer.vector_store.collection, "get", side_effect=RuntimeError("clear failed")):
        with pytest.raises(RuntimeError, match="clear failed"):
            indexer.vector_store.clear()


def test_corrupt_sparse_artifact_can_be_rebuilt(indexer):
    indexer.index_chunks([chunk("a")])
    indexer.sparse_path.write_bytes(b"broken pickle")
    reopened = UETIndexer(vector_store=indexer.vector_store, embedding_model=TestEmbedder())
    assert reopened.get_status()["rebuild_required"]
    with pytest.raises(RuntimeError, match="differ"):
        reopened.index_chunks([chunk("b")])
    reopened.index_chunks([chunk("b")], clear_existing=True)
    assert reopened.get_status()["consistent"]


def test_ingestion_stage_indexes_updates_in_both_stores(indexer, tmp_path, monkeypatch):
    import src.pipeline as pipeline_module
    from src.ingestion.manifest import DataManifestTracker

    monkeypatch.setattr(pipeline_module, "BASE_DIR", tmp_path)
    directory = tmp_path / "data" / "chunked_data"
    directory.mkdir(parents=True)
    path = directory / "chunks.jsonl"
    tracker = DataManifestTracker(manifest_path=tmp_path / "manifest.json")
    pipeline = pipeline_module.UETIngestionPipeline(
        vector_store=indexer.vector_store, embedding_model=TestEmbedder(), manifest=tracker)
    path.write_text(chunk("a").model_dump_json(), encoding="utf-8")
    assert pipeline._run_chunked_to_vectordb()["chunks_indexed"] == 1
    path.write_text(chunk("a", "INT3402").model_dump_json(), encoding="utf-8")
    pipeline._run_chunked_to_vectordb()
    reopened = UETIndexer(vector_store=indexer.vector_store, embedding_model=TestEmbedder())
    assert reopened.get_status()["consistent"]
    assert reopened.sparse_index.search("INT3402")[0].chunk_id == "a"
    assert reopened.sparse_index.search("INT3401") == []


def test_second_dense_batch_failure_does_not_report_success(indexer):
    original = indexer.vector_store.store
    calls = 0

    def fail_second(chunks, vectors):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second batch failed")
        return original(chunks, vectors)

    with patch.object(indexer.vector_store, "store", side_effect=fail_second):
        with pytest.raises(RuntimeError, match="second batch"):
            indexer.index_chunks([chunk("a"), chunk("b")], batch_size=1)
    assert indexer.vector_store.count() == 1
    assert indexer.get_status()["rebuild_required"]


def test_source_ingestion_uses_both_indexes(indexer, tmp_path):
    from src.base import DataSource
    from src.ingestion.manifest import DataManifestTracker
    from src.pipeline import UETIngestionPipeline

    crawler = Mock()
    crawler.crawl_all.return_value = [object()]
    preprocessor = Mock()
    preprocessor.process_batch.return_value = [object()]
    chunker = Mock()
    chunker.chunk_batch.return_value = [chunk("a")]
    pipeline = UETIngestionPipeline(
        crawler=crawler, preprocessor=preprocessor, chunker=chunker,
        vector_store=indexer.vector_store, embedding_model=TestEmbedder(),
        manifest=DataManifestTracker(manifest_path=tmp_path / "manifest.json"))
    result = pipeline.run([DataSource(source_type="file", uri="fixture.txt")], clear_existing=True)
    assert result["saved_in_db"] == result["total_in_db"] == 1
    reopened = UETIndexer(vector_store=indexer.vector_store, embedding_model=TestEmbedder())
    assert reopened.get_status()["consistent"]
    assert reopened.sparse_index.search("INT3401")[0].chunk_id == "a"

"""Unit & integration tests for the GraphRAG Vector Index module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import BaseEmbeddingModel, BaseVectorStore, DataChunk, RetrievedContext, UserQuery
from src.indexing.graph import (
    GraphIndex,
    HybridGraphExtractor,
    _normalize_relation,
    canonicalize_entity_name,
    clean_entity_name,
    normalize_entity_id,
)
from src.indexing.sparse import BM25Index
from src.llm.llm_client import UETLLMClient
from src.prompts.prompt_templates import UETPromptAugmenter
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.reranker import HeuristicReranker
from src.retrieval.retriever import UETRetriever


class KeywordVectorEmbedder(BaseEmbeddingModel):
    """Deterministic local embedder mapping UET keywords into orthogonal directions."""

    VOCAB = [
        "thủ khoa",
        "798/qđ-đhqghn",
        "1363/qđ-đhcn",
        "khen thưởng",
        "học bổng",
        "ielts",
        "vstep",
        "int3401",
    ]

    @property
    def dimension(self) -> int:
        return len(self.VOCAB) + 2

    @property
    def model_name(self) -> str:
        return "mock-keyword-bge-m3"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for txt in texts:
            low = (txt or "").lower()
            vec = [0.0] * self.dimension
            for i, kw in enumerate(self.VOCAB):
                if kw in low:
                    vec[i] = 1.0
            vec[-2] = 0.1
            vec[-1] = 0.05
            vectors.append(vec)
        return vectors


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, chunks: List[DataChunk]):
        self._chunks = list(chunks)

    def store(self, chunks: List[DataChunk], vectors=None) -> int:
        self._chunks.extend(chunks)
        return len(chunks)

    def search(self, query_vector, top_k: int = 5, filters=None) -> List[RetrievedContext]:
        return [
            RetrievedContext(
                chunk_id=c.chunk_id,
                text=c.text,
                similarity_score=round(0.8 - i * 0.05, 4),
                rank=i + 1,
                metadata=dict(c.metadata),
            )
            for i, c in enumerate(self._chunks[:top_k])
        ]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


def test_rule_based_uet_extraction():
    chunk = DataChunk(
        chunk_id="c_rule_1",
        text=(
            "[Quyết định 1363/QĐ-ĐHCN > Điều 3, chương II]\n"
            "Căn cứ Quyết định số 798/QĐ-ĐHQGHN ngày 15 tháng 08 năm 2023.\n"
            "Điều 1. Sửa đổi Điều 3, chương II về việc xét Thủ khoa ngành học, "
            "áp dụng cho sinh viên học học phần INT3401 có điểm rèn luyện Tốt."
        ),
        chunk_index=0,
        metadata={"title": "Điều chỉnh quy định khen thưởng", "faculty": "fit"},
    )
    extractor = HybridGraphExtractor(enable_llm_extraction=False)
    entities, triples = extractor.extract_rules(chunk)

    entity_types = {e["id"]: e["type"] for e in entities}
    assert normalize_entity_id("1363/QĐ-ĐHCN") in entity_types
    assert normalize_entity_id("798/QĐ-ĐHQGHN") in entity_types
    assert normalize_entity_id("INT3401") in entity_types
    assert entity_types[normalize_entity_id("INT3401")] == "COURSE"
    assert normalize_entity_id("Khoa Công nghệ Thông tin (FIT)") in entity_types

    relations = {t["relation"] for t in triples}
    assert "CĂN_CỨ" in relations
    assert "SỬA_ĐỔI_BỔ_SUNG" in relations
    assert "THUỘC_ĐƠN_VỊ" in relations


def test_openrouter_client_and_llm_extractor_cache(tmp_path):
    cache_llm = tmp_path / "llm_cache.json"
    cache_graph = tmp_path / "graph_llm_cache.json"

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<think>Phân tích quy chế</think>\n"
                        '[{"head": "Học bổng xuất sắc", "head_type": "CONCEPT", '
                        '"relation": "YEU_CAU_GPA", "tail": "GPA từ 3.60", "tail_type": "CONCEPT"}]'
                    )
                }
            }
        ],
        "usage": {"total_tokens": 42},
    }

    with patch("httpx.post", return_value=fake_response) as mock_post:
        client = UETLLMClient(
            provider="openrouter",
            api_key="sk-or-test-key",
            enable_cache=True,
            cache_file=cache_llm,
        )
        assert client.api_key == "sk-or-test-key"

        extractor = HybridGraphExtractor(
            llm_client=client,
            enable_llm_extraction=True,
            llm_provider="openrouter",
            llm_model="tencent/hy4-flash",
            llm_domains=["handbook"],
            max_llm_chunks=1,
            cache_path=cache_graph,
        )

        hb_chunk = DataChunk(
            chunk_id="hb_1",
            text="Sinh viên đạt Học bổng xuất sắc khi có GPA từ 3.60 trở lên.",
            metadata={"faculty": "handbook"},
        )
        hb_chunk_2 = DataChunk(
            chunk_id="hb_2",
            text="Sinh viên đạt Học bổng giỏi khi có GPA từ 3.20 trở lên.",
            metadata={"faculty": "handbook"},
        )
        skipped_chunk = DataChunk(
            chunk_id="uet_1",
            text="Thông báo lịch nghỉ lễ chung của trường.",
            metadata={"faculty": "uet"},
        )

        _, triples_hb = extractor.extract(hb_chunk)
        _, triples_hb_2 = extractor.extract(hb_chunk_2)
        _, triples_uet = extractor.extract(skipped_chunk)
        extractor.flush_cache()

        assert mock_post.call_count == 1
        assert extractor.llm_calls_count == 1
        assert not any(t["tail"] == "GPA từ 3.60" for t in triples_hb_2)
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "tencent/hy4-flash"
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-or-test-key"
        assert any(t["relation"] == "YÊU_CẦU_ĐIỀU_KIỆN" and t["tail"] == "GPA từ 3.60" for t in triples_hb)
        assert not any(t["tail"] == "GPA từ 3.60" for t in triples_uet)

        # Second extractor instance must hit persistent graph cache without HTTP call
        extractor_reloaded = HybridGraphExtractor(
            llm_client=client,
            enable_llm_extraction=True,
            llm_domains=["handbook"],
            cache_path=cache_graph,
        )
        _, triples_cached = extractor_reloaded.extract(hb_chunk)
        assert mock_post.call_count == 1
        assert any(t["relation"] == "YÊU_CẦU_ĐIỀU_KIỆN" and t["tail"] == "GPA từ 3.60" for t in triples_cached)


def test_graph_index_vector_multihop_save_load(tmp_path):
    embedder = KeywordVectorEmbedder()
    chunks = [
        DataChunk(
            chunk_id="c1",
            text="Quyết định 1363/QĐ-ĐHCN sửa đổi Điều 3 về xét Thủ khoa ngành học căn cứ Quyết định 798/QĐ-ĐHQGHN.",
            chunk_index=0,
            metadata={"title": "Quyết định 1363/QĐ-ĐHCN", "faculty": "handbook"},
        ),
        DataChunk(
            chunk_id="c2",
            text="Quyết định 798/QĐ-ĐHQGHN quy định chế độ khen thưởng sinh viên tại UET.",
            chunk_index=0,
            metadata={"title": "Quyết định 798/QĐ-ĐHQGHN", "faculty": "handbook"},
        ),
        DataChunk(
            chunk_id="c3",
            text="Chuẩn đầu ra ngoại ngữ CTĐT CLC yêu cầu IELTS 5.5 hoặc VSTEP 6.0.",
            chunk_index=0,
            metadata={"title": "Chuẩn đầu ra ngoại ngữ", "faculty": "uet"},
        ),
    ]

    index = GraphIndex(chunks=chunks, embedding_model=embedder)
    assert index.count() == 3
    assert index.entity_count() > 0
    assert index.triple_count() > 0

    artifact = tmp_path / "graph_index.pkl"
    index.save(artifact)

    loaded = GraphIndex.load(artifact, embedding_model=embedder)
    assert loaded.count() == 3

    # 0-LLM query: passing query_vector directly without calling LLM
    q = UserQuery(query_text="Quyết định 798/QĐ-ĐHQGHN và Thủ khoa ngành học")
    q_vec = embedder.embed_query(q).vector
    hits = loaded.search(query=q.query_text, query_vector=q_vec, top_k=5, max_hops=2)

    hit_ids = [h.chunk_id for h in hits]
    assert "c1" in hit_ids
    assert "c2" in hit_ids
    assert hits[0].metadata.get("search_source") == "GraphRAG"
    assert len(hits[0].metadata.get("graph_triples", [])) > 0

    # Metadata filter must exclude uet chunk c3
    filtered = loaded.search(
        query="IELTS VSTEP",
        query_vector=embedder.embed_texts(["IELTS VSTEP"])[0],
        top_k=5,
        filters={"faculty": "handbook"},
    )
    assert all(item.chunk_id != "c3" for item in filtered)

    # Empty index safety
    assert GraphIndex().search("học bổng") == []


def test_tri_branch_rrf_retriever_and_prompt():
    bm25_res = [
        RetrievedContext(chunk_id="c1", text="Đoạn 1", similarity_score=5.0, rank=1, metadata={"title": "Doc 1"}),
        RetrievedContext(chunk_id="c2", text="Đoạn 2", similarity_score=3.0, rank=2, metadata={"title": "Doc 2"}),
    ]
    vec_res = [
        RetrievedContext(chunk_id="c1", text="Đoạn 1", similarity_score=0.9, rank=1, metadata={"title": "Doc 1"}),
        RetrievedContext(chunk_id="c3", text="Đoạn 3", similarity_score=0.7, rank=2, metadata={"title": "Doc 3"}),
    ]
    graph_res = [
        RetrievedContext(
            chunk_id="c1",
            text="Đoạn 1",
            similarity_score=0.95,
            rank=1,
            metadata={"title": "Doc 1", "graph_triples": ["A -[QUY_ĐỊNH_VỀ]-> B"]},
        ),
        RetrievedContext(
            chunk_id="c4",
            text="Đoạn 4",
            similarity_score=0.6,
            rank=2,
            metadata={"title": "Doc 4", "graph_triples": ["B -[CĂN_CỨ]-> C"]},
        ),
    ]

    fused = HybridSearcher.reciprocal_rank_fusion(
        bm25_results=bm25_res,
        vector_results=vec_res,
        graph_results=graph_res,
        top_n=10,
    )
    assert fused[0].chunk_id == "c1"
    assert fused[0].metadata["search_source"] == "BM25 + Vector + GraphRAG"
    assert fused[0].metadata["graph_triples"] == ["A -[QUY_ĐỊNH_VỀ]-> B"]
    fused_map = {c.chunk_id: c for c in fused}
    assert fused_map["c4"].metadata["search_type"] == "graph"

    linear_fused = HybridSearcher.linear_score_fusion(
        bm25_results=bm25_res,
        vector_results=vec_res,
        graph_results=graph_res,
        top_n=10,
    )
    assert linear_fused[0].chunk_id == "c1"

    # Full UETRetriever + UETPromptAugmenter integration
    chunks = [
        DataChunk(
            chunk_id="c1",
            text="Sinh viên đạt học bổng xuất sắc cần GPA từ 3.60 và điểm rèn luyện xuất sắc.",
            metadata={"title": "Quy chế học bổng", "faculty": "handbook"},
        ),
        DataChunk(
            chunk_id="c2",
            text="Điều kiện xét Thủ khoa ngành học theo Quyết định 1363/QĐ-ĐHCN.",
            metadata={"title": "Quyết định khen thưởng", "faculty": "handbook"},
        ),
    ]
    embedder = KeywordVectorEmbedder()
    store = InMemoryVectorStore(chunks)
    bm25 = BM25Index(chunks)
    graph = GraphIndex(chunks=chunks, embedding_model=embedder)

    retriever = UETRetriever(
        embedding_model=embedder,
        vector_store=store,
        reranker=HeuristicReranker(),
        bm25_index=bm25,
        graph_index=graph,
        config={
            "retrieval": {
                "bm25_top_k": 5,
                "vector_top_k": 5,
                "graph_top_k": 5,
                "candidate_pool_size": 5,
                "enable_mmr": False,
                "rerank_top_n": 5,
                "final_top_k": 2,
                "enable_graph": True,
            }
        },
    )
    q = UserQuery(query_text="Điều kiện học bổng xuất sắc UET")
    results = retriever.retrieve(q, top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"
    assert "graph_triples" in results[0].metadata

    augmented = UETPromptAugmenter(include_few_shot=False).augment(q, results)
    assert "[Quan hệ Đồ thị Tri thức:" in augmented.formatted_prompt


def test_ingestion_pipeline_chunked_to_graph_data(tmp_path, monkeypatch):
    import src.pipeline as pipeline_module
    from src.ingestion.manifest import DataManifestTracker

    monkeypatch.setattr(pipeline_module, "BASE_DIR", tmp_path)
    chunked_dir = tmp_path / "data" / "chunked_data"
    chunked_dir.mkdir(parents=True)
    chunk = DataChunk(
        chunk_id="hb_chunk_1",
        text="Quyết định 1363/QĐ-ĐHCN quy định xét Thủ khoa ngành học tại UET.",
        metadata={"title": "Khen thưởng UET", "faculty": "handbook"},
    )
    (chunked_dir / "handbook_chunks.jsonl").write_text(
        json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tracker = DataManifestTracker(manifest_path=tmp_path / "manifest.json")
    pipeline = pipeline_module.UETIngestionPipeline(
        embedding_model=KeywordVectorEmbedder(),
        vector_store=InMemoryVectorStore([chunk]),
        manifest=tracker,
        config={
            "paths": {
                "graph_index_file": "database/graph_index.pkl",
                "graph_llm_cache_file": "database/graph_llm_cache.json",
            },
            "graph": {
                "enable_llm_extraction": False,
                "embed_batch_size": 16,
            },
        },
    )

    with pytest.raises(ValueError):
        pipeline.run_stages(from_stage="raw", to_stage="graph_data")

    res = pipeline.run_stages(from_stage="chunked", to_stage="graph_data", clear=True)
    transition = res["transitions"]["chunked_to_graph_data"]
    assert transition["status"] == "success"
    assert transition["chunks_indexed"] == 1
    assert (tmp_path / "database" / "graph_index.pkl").is_file()

    stats = pipeline.get_stage_stats()
    assert stats["stages"]["graph_data"]["exists"] is True


def test_graph_index_reuses_chromadb_vectors_without_loading_embedder(tmp_path):
    chunks = [
        DataChunk(
            chunk_id="vec_c1",
            text="Quyết định 1363/QĐ-ĐHCN xét Thủ khoa ngành học tại UET.",
            metadata={"title": "Khen thưởng UET", "faculty": "handbook"},
        ),
        DataChunk(
            chunk_id="vec_c2",
            text="Chuẩn đầu ra ngoại ngữ yêu cầu chứng chỉ IELTS 5.5.",
            metadata={"title": "Chuẩn đầu ra", "faculty": "uet"},
        ),
    ]
    precomputed = {
        "vec_c1": [1.0, 0.0, 0.0, 0.0],
        "vec_c2": [0.0, 1.0, 0.0, 0.0],
    }

    mock_store = MagicMock()
    mock_store.get_embeddings_by_ids.return_value = precomputed

    index = GraphIndex(
        chunks=chunks,
        embedding_model=None,
        vector_store=mock_store,
    )
    mock_store.get_embeddings_by_ids.assert_called_once_with(["vec_c1", "vec_c2"])
    assert index.entity_vectors is not None
    assert index.entity_vectors.shape[1] == 4
    assert index.triple_vectors is not None
    assert index.triple_vectors.shape[1] == 4

    hits = index.search(
        query="Thủ khoa",
        query_vector=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
    )
    assert len(hits) > 0
    assert hits[0].chunk_id == "vec_c1"

    out_pkl = tmp_path / "graph_from_chroma.pkl"
    index.save(out_pkl)
    reloaded = GraphIndex.load(out_pkl)
    assert reloaded.vector_store is None
    assert reloaded.entity_vectors.shape == index.entity_vectors.shape


def test_entity_and_relation_canonicalization():
    assert clean_entity_name("Trường Đại học Công nghệ - ĐHQGHN (UET)") == "Trường Đại học Công nghệ - ĐHQGHN (UET)"
    assert clean_entity_name("“Thủ khoa ngành học”") == "Thủ khoa ngành học"
    assert canonicalize_entity_name("Khoa CNTT") == "Khoa Công nghệ Thông tin (FIT)"
    assert canonicalize_entity_name("UET") == "Trường Đại học Công nghệ - ĐHQGHN (UET)"
    assert canonicalize_entity_name("GPA") == "Điểm trung bình chung (GPA)"
    assert normalize_entity_id("Khoa CNTT") == normalize_entity_id("FIT") == normalize_entity_id("Khoa Công nghệ Thông tin (FIT)")
    assert normalize_entity_id("Quyết định số 798/QĐ-ĐHQGHN") == normalize_entity_id("798/QĐ-ĐHQGHN")
    assert normalize_entity_id("GPA") == normalize_entity_id("Điểm trung bình chung") == normalize_entity_id("Điểm trung bình chung tích lũy")

    segs = HybridGraphExtractor._extract_breadcrumb_segments(
        DataChunk(
            chunk_id="bc_1",
            text='[Quyết định 1363/QĐ-ĐHCN > Điều 3, chương II: Quy định về xét “Thủ khoa ngành học”]\nNội dung.',
            metadata={},
        )
    )
    assert segs == ["Quyết định 1363/QĐ-ĐHCN", "Điều 3, chương II"]

    assert _normalize_relation("YEU_CAU_GPA") == "YÊU_CẦU_ĐIỀU_KIỆN"
    assert _normalize_relation("DIEU_KIEN_TOI_THIEU") == "YÊU_CẦU_ĐIỀU_KIỆN"
    assert _normalize_relation("SUA_DOI") == "SỬA_ĐỔI_BỔ_SUNG"
    assert _normalize_relation("CAN_CU_QUYET_DINH") == "CĂN_CỨ"
    assert _normalize_relation("THUOC_KHOA") == "THUỘC_ĐƠN_VỊ"


def test_entity_idf_and_coactivation_and_adjacent_smoothing():
    embedder = KeywordVectorEmbedder()
    chunks = [
        DataChunk(
            chunk_id="doc_a_0",
            document_id="doc_a",
            chunk_index=0,
            text="Quyết định 1363/QĐ-ĐHCN quy định điều kiện xét Thủ khoa ngành học tại UET.",
            metadata={"title": "Quyết định khen thưởng", "faculty": "uet"},
        ),
        DataChunk(
            chunk_id="doc_a_1",
            document_id="doc_a",
            chunk_index=1,
            text="Sinh viên cần giữ kỷ luật và đạt thành tích cao toàn khóa tại UET.",
            metadata={"title": "Quyết định khen thưởng", "faculty": "uet"},
        ),
        DataChunk(
            chunk_id="doc_b_0",
            document_id="doc_b",
            chunk_index=0,
            text="Hướng dẫn nộp học phí trực tuyến cho sinh viên tại UET.",
            metadata={"title": "Học phí", "faculty": "uet"},
        ),
        DataChunk(
            chunk_id="doc_c_0",
            document_id="doc_c",
            chunk_index=0,
            text="Giới thiệu quy mô đào tạo chung cho sinh viên mới tại UET.",
            metadata={"title": "Giới thiệu chung", "faculty": "uet"},
        ),
    ]
    index = GraphIndex(chunks=chunks, embedding_model=embedder)

    uet_id = normalize_entity_id("UET")
    qd_id = normalize_entity_id("1363/QĐ-ĐHCN")
    assert index.entity_idf[uet_id] < index.entity_idf[qd_id]

    q = "Quyết định 1363/QĐ-ĐHCN và Thủ khoa ngành học"
    q_vec = embedder.embed_texts([q])[0]
    hits = index.search(query=q, query_vector=q_vec, top_k=4, max_hops=2)
    hit_ids = [h.chunk_id for h in hits]
    assert hit_ids[0] == "doc_a_0"
    assert "doc_a_1" in hit_ids
    assert hit_ids.index("doc_a_1") < hit_ids.index("doc_b_0")


def test_extraction_modes(tmp_path):
    chunk = DataChunk(
        chunk_id="mode_c1",
        text="Quyết định 1363/QĐ-ĐHCN áp dụng cho học phần INT3401 về Chuẩn đầu ra ngoại ngữ.",
        chunk_index=0,
        metadata={"title": "Quy định", "faculty": "fit"},
    )
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": '[{"head": "Chuẩn đầu ra", "head_type": "CONCEPT", "relation": "YEU_CAU_CHUNG_CHI", "tail": "IELTS 6.5", "tail_type": "CERTIFICATE"}]'
            }
        }]
    }

    with patch("httpx.post", return_value=fake_resp) as mock_post:
        client = UETLLMClient(provider="openrouter", api_key="sk-or-test", enable_cache=False)

        regex_ext = HybridGraphExtractor(
            llm_client=client,
            enable_llm_extraction=True,
            cache_path=tmp_path / "cache_regex.json",
            extraction_mode="regex_only",
        )
        r_ents, r_triples = regex_ext.extract(chunk)
        assert mock_post.call_count == 0
        assert any(e["id"] == normalize_entity_id("INT3401") for e in r_ents)
        assert not any(t["tail"] == "IELTS 6.5" for t in r_triples)

        llm_ext = HybridGraphExtractor(
            llm_client=client,
            enable_llm_extraction=True,
            cache_path=tmp_path / "cache_llm.json",
            extraction_mode="llm_only",
        )
        l_ents, l_triples = llm_ext.extract(chunk)
        assert mock_post.call_count == 1
        assert len(l_triples) == 1
        assert l_triples[0]["relation"] == "YÊU_CẦU_ĐIỀU_KIỆN"
        assert l_triples[0]["tail"] == "IELTS 6.5"
        assert not any(e["id"] == normalize_entity_id("INT3401") for e in l_ents)

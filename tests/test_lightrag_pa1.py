"""
tests/test_lightrag_pa1.py - Bộ kiểm thử đơn vị cho nhánh LightRAG PA1:
  1. QueryEntityExtractor: parse JSON chuẩn, code-fence, lọc FORBIDDEN_ENTITIES, LRU cache, fallback khi lỗi.
  2. LightRAGRetriever: không bắt nhầm substring rác ('Thông tin'), chuẩn hóa qua AcademicThesaurus,
     khớp ngữ nghĩa qua entity_chroma (lọc ngưỡng >= 0.72), và tự động fallback PA2 khi LLM tắt/lỗi.
  3. UETRetriever: chạy song song 3A & 3B bằng ThreadPoolExecutor an toàn đa luồng.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import List

from src.base import BaseEmbeddingModel, BaseVectorStore, DataChunk, RetrievedContext, UserQuery
from src.graph.lightrag_retriever import LightRAGRetriever
from src.graph.query_entity_extractor import QueryEntityExtractor
from src.retrieval.retriever import UETRetriever


class DummyEmbedder(BaseEmbeddingModel):
    @property
    def dimension(self) -> int:
        return 8

    @property
    def model_name(self) -> str:
        return "dummy-embedder"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.25] * self.dimension for _ in texts]


class DummyEntityCollection:
    """Giả lập ChromaDB collection 'lightrag_entity_collection'."""

    def __init__(self, responses):
        self.responses = responses
        self.query_calls = 0

    def query(self, query_embeddings, n_results=3, include=None):
        self.query_calls += 1
        metas = []
        dists = []
        for _ in query_embeddings:
            metas.append([r["meta"] for r in self.responses[:n_results]])
            dists.append([r["distance"] for r in self.responses[:n_results]])
        return {"metadatas": metas, "distances": dists}


class TestLightRAGPA1(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.graph_file = Path(self.tmp_dir.name) / "mock_graph.json"
        mock_graph = {
            "nodes": [
                {
                    "id": "n1",
                    "canonical_name": "Ngành Công nghệ thông tin",
                    "entity_type": "KHOA_NGANH",
                    "descriptions": ["Chương trình đào tạo cử nhân CNTT chuẩn quốc tế."],
                    "aliases": ["cn1"],
                },
                {
                    "id": "n_noise",
                    "canonical_name": "Thông tin",
                    "entity_type": "OTHER",
                    "descriptions": ["Node nhiễu để kiểm tra lỗi khớp chuỗi con."],
                    "aliases": [],
                },
                {
                    "id": "n2",
                    "canonical_name": "Buộc thôi học",
                    "entity_type": "DIEU_KIEN",
                    "descriptions": ["Sinh viên bị cảnh báo học vụ quá số lần quy định sẽ bị buộc thôi học."],
                    "aliases": [],
                },
                {
                    "id": "n3",
                    "canonical_name": "Chuẩn đầu ra ngoại ngữ bậc 4/6",
                    "entity_type": "DIEU_KIEN",
                    "descriptions": ["Yêu cầu tiếng Anh tương đương B2 / IELTS 5.5 đối với hệ chất lượng cao."],
                    "aliases": [],
                },
            ],
            "edges": [
                {
                    "source": "Buộc thôi học",
                    "target": "Ngành Công nghệ thông tin",
                    "relation_type": "áp_dụng_cho",
                    "description": "Áp dụng theo quy chế đào tạo đại học.",
                }
            ],
            "document_summaries": {
                "doc_qc": {
                    "title": "Quy chế đào tạo đại học tại UET",
                    "document_type": "QUY_CHE",
                    "key_entities": ["Buộc thôi học", "Cảnh báo học vụ"],
                    "summary": "Quy định chi tiết về cảnh báo học tập, buộc thôi học và xét tốt nghiệp.",
                }
            },
        }
        self.graph_file.write_text(json.dumps(mock_graph, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_extractor_parses_json_filters_forbidden_and_uses_cache(self):
        """1. Kiểm tra QueryEntityExtractor parse code-fence JSON, lọc từ cấm và cache kết quả."""
        calls = {"count": 0}

        def fake_llm(_q: str) -> str:
            calls["count"] += 1
            return """```json
            {
              "low_level_entities": ["sinh viên", "UET", "Thông tin", "Buộc thôi học", "chuẩn tiếng Anh B2"],
              "high_level_themes": ["nhà trường", "Xét tốt nghiệp"]
            }
            ```"""

        extractor = QueryEntityExtractor(llm_callable=fake_llm)
        res1 = extractor.extract("Em bị đuổi học hay thiếu tiếng Anh B2 không?")
        res2 = extractor.extract("  em bị đuổi học hay thiếu tiếng anh b2 không? ")

        self.assertTrue(res1["used_llm"])
        self.assertEqual(calls["count"], 1, "Câu hỏi lặp lại phải lấy từ LRU cache, không gọi LLM lần 2")
        self.assertEqual(res1["low_level_entities"], ["Buộc thôi học", "chuẩn tiếng Anh B2"])
        self.assertEqual(res1["high_level_themes"], ["Xét tốt nghiệp"])
        self.assertEqual(res1, res2)

    def test_no_substring_noise_and_semantic_entity_chroma_resolution(self):
        """2. Không bắt nhầm node rác 'Thông tin' và ánh xạ được thực thể ngoài từ điển qua entity_chroma."""
        def fake_llm(_q: str) -> str:
            return json.dumps({
                "low_level_entities": ["Ngành Công nghệ thông tin", "yêu cầu ngoại ngữ B2 hệ CLC"],
                "high_level_themes": ["Quy chế đào tạo"],
            }, ensure_ascii=False)

        # distance = 0.15 => cosine_similarity = 0.85 (>= 0.72) -> nhận
        # distance = 0.45 => cosine_similarity = 0.55 (< 0.72)  -> loại
        mock_collection = DummyEntityCollection([
            {"meta": {"canonical_name": "Chuẩn đầu ra ngoại ngữ bậc 4/6"}, "distance": 0.15},
            {"meta": {"canonical_name": "Thông tin"}, "distance": 0.45},
        ])

        retriever = LightRAGRetriever(
            config={"lightrag": {"enabled": True, "use_llm_query_extraction": True, "entity_similarity_threshold": 0.72}},
            graph_path=self.graph_file,
            embedding_model=DummyEmbedder(),
            query_extractor=QueryEntityExtractor(llm_callable=fake_llm),
            entity_collection=mock_collection,
        )

        seeds = dict(retriever.find_seed_entities("ngành công nghệ thông tin yêu cầu ngoại ngữ B2 hệ CLC"))
        self.assertIn("Ngành Công nghệ thông tin", seeds)
        self.assertIn("Chuẩn đầu ra ngoại ngữ bậc 4/6", seeds)
        self.assertNotIn("Thông tin", seeds, "Tuyệt đối không được bắt nhầm substring rác 'Thông tin'!")
        self.assertGreaterEqual(mock_collection.query_calls, 1)

    def test_graceful_fallback_pa2_when_llm_fails(self):
        """3. Khi LLM timeout/lỗi, tự động fallback sang Thesaurus + entity_chroma (PA2) không văng lỗi."""
        def failing_llm(_q: str) -> str:
            raise TimeoutError("Simulated API timeout")

        mock_collection = DummyEntityCollection([
            {"meta": {"canonical_name": "Buộc thôi học"}, "distance": 0.12},
        ])

        retriever = LightRAGRetriever(
            config={"lightrag": {"enabled": True, "use_llm_query_extraction": True}},
            graph_path=self.graph_file,
            embedding_model=DummyEmbedder(),
            query_extractor=QueryEntityExtractor(llm_callable=failing_llm),
            entity_collection=mock_collection,
        )

        ctxs = retriever.retrieve(UserQuery(query_text="tạch nhiều môn quá có sao không"), top_k=3)
        self.assertGreaterEqual(len(ctxs), 1)
        self.assertTrue(any("BUỘC THÔI HỌC" in c.text for c in ctxs))

    def test_parallel_execution_between_3a_and_3b(self):
        """4. Kiểm chứng nhánh 3A và 3B thực sự chạy song song (tổng thời gian ~ max(tA, tB) thay vì tA + tB)."""
        class SlowGraphRetriever:
            def retrieve(self, query, top_k=2):
                time.sleep(0.25)
                return [RetrievedContext(chunk_id="g1", text="Tri thức đồ thị", similarity_score=0.95, rank=1)]

        class SlowStore(BaseVectorStore):
            def store(self, chunks, vectors=None):
                return 0

            def search(self, query_vector, top_k=5, filters=None):
                time.sleep(0.25)
                return [RetrievedContext(chunk_id="v1", text="Đoạn văn bản quy chế", similarity_score=0.9, rank=1)]

            def count(self):
                return 1

            def clear(self):
                pass

        class EmptyBM25:
            def search(self, query, top_k=10):
                return []

        uet_retriever = UETRetriever(
            embedding_model=DummyEmbedder(),
            vector_store=SlowStore(),
            bm25_index=EmptyBM25(),
            lightrag_retriever=SlowGraphRetriever(),
            config={"retrieval": {"enable_mmr": False, "reranker_type": "heuristic", "final_top_k": 4}},
        )

        t0 = time.perf_counter()
        results = uet_retriever.retrieve(UserQuery(query_text="điều kiện xét học bổng"), top_k=4, engine="hybrid")
        elapsed = time.perf_counter() - t0

        self.assertEqual(len(results), 2)
        # Nếu tuần tự sẽ mất >= 0.50s; chạy song song chỉ mất ~0.25-0.38s
        self.assertLess(elapsed, 0.45, f"3A và 3B phải chạy song song, thực tế mất {elapsed:.3f}s")

    def test_ram_numpy_entity_vector_search(self):
        """5. Kiểm tra nhánh tìm kiếm vector thực thể trên ma trận NumPy RAM (<2ms, không gọi ChromaDB)."""
        import numpy as np

        class DirectionalEmbedder(BaseEmbeddingModel):
            @property
            def dimension(self) -> int:
                return 4

            @property
            def model_name(self) -> str:
                return "dir-embedder"

            def embed_texts(self, texts: List[str]) -> List[List[float]]:
                out = []
                for t in texts:
                    if "ngoại ngữ" in t.lower() or "ielts" in t.lower():
                        out.append([1.0, 0.0, 0.0, 0.0])
                    else:
                        out.append([0.0, 1.0, 0.0, 0.0])
                return out

        retriever = LightRAGRetriever(
            config={"lightrag": {"enabled": True, "use_llm_query_extraction": False, "entity_similarity_threshold": 0.72}},
            graph_path=self.graph_file,
            embedding_model=DirectionalEmbedder(),
        )
        retriever._entity_vec_names = ["Chuẩn đầu ra ngoại ngữ bậc 4/6", "Buộc thôi học"]
        retriever._entity_vec_matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        matches = dict(retriever._search_entity_vectors(["chứng chỉ IELTS ngoại ngữ"]))
        self.assertIn("Chuẩn đầu ra ngoại ngữ bậc 4/6", matches)
        self.assertNotIn("Buộc thôi học", matches)

    def test_lightrag_primary_grounded_dual_output_and_regex_supplement(self):
        """
        6. Kiểm chứng kiến trúc LightRAG-First + LightRAGChunkBridge:
           - Câu hỏi quan hệ: LightRAGRetriever dẫn dắt, ánh xạ về DataChunk thật (lightrag_primary) + xuất lightrag_overview cho Prompt.
           - Câu hỏi mã cứng ngoài graph.json (INT3401): Tầng Regex Supplement (0.65x) tự động lấp khoảng trống độ phủ.
        """
        from src.graph.graph import LightRAGChunkBridge
        from src.indexing.sparse import BM25Index
        from src.prompts.prompt_templates import UETPromptAugmenter
        from src.retrieval.reranker import HeuristicReranker

        units_file = Path(self.tmp_dir.name) / "mock_units.jsonl"
        unit_record = {
            "unit_id": "unit_bth_1",
            "document_id": "doc_qc",
            "document_title": "Quy chế đào tạo đại học tại UET",
            "content": "Sinh viên bị cảnh báo học vụ quá số lần quy định sẽ bị buộc thôi học tại Ngành Công nghệ thông tin.",
        }
        units_file.write_text(json.dumps(unit_record, ensure_ascii=False) + "\n", encoding="utf-8")

        # Cập nhật source_unit_ids cho mock_graph để cầu nối ánh xạ Primary Chunk chuẩn xác
        graph_with_units = json.loads(self.graph_file.read_text(encoding="utf-8"))
        for nd in graph_with_units["nodes"]:
            if nd["canonical_name"] in ("Buộc thôi học", "Ngành Công nghệ thông tin"):
                nd["source_unit_ids"] = ["unit_bth_1"]
        for ed in graph_with_units["edges"]:
            ed["source_unit_ids"] = ["unit_bth_1"]
            ed["evidence"] = "Sinh viên bị cảnh báo học vụ quá số lần quy định sẽ bị buộc thôi học."
        self.graph_file.write_text(json.dumps(graph_with_units, ensure_ascii=False), encoding="utf-8")

        chunks = [
            DataChunk(
                chunk_id="real_chunk_qc_1",
                document_id="doc_qc",
                chunk_index=0,
                text="[Quy chế đào tạo đại học tại UET]\nSinh viên bị cảnh báo học vụ quá số lần quy định sẽ bị buộc thôi học tại Ngành Công nghệ thông tin.",
                metadata={"title": "Quy chế đào tạo đại học tại UET", "document_id": "doc_qc", "faculty": "handbook"},
            ),
            DataChunk(
                chunk_id="real_chunk_course_int3401",
                document_id="doc_fit",
                chunk_index=0,
                text="[Đề cương học phần]\nHọc phần INT3401 Trí tuệ nhân tạo có khối lượng 3 tín chỉ do Khoa CNTT phụ trách.",
                metadata={"title": "Đề cương INT3401", "document_id": "doc_fit", "faculty": "fit"},
            ),
        ]

        class SimpleStore(BaseVectorStore):
            def store(self, c, vectors=None):
                return len(c)

            def search(self, query_vector, top_k=5, filters=None):
                return [
                    RetrievedContext(
                        chunk_id=ch.chunk_id,
                        text=ch.text,
                        similarity_score=0.75,
                        rank=idx + 1,
                        metadata=dict(ch.metadata),
                    )
                    for idx, ch in enumerate(chunks[:top_k])
                ]

            def count(self):
                return len(chunks)

            def clear(self):
                pass

        embedder = DummyEmbedder()
        bridge = LightRAGChunkBridge(
            chunks=chunks,
            embedding_model=embedder,
            lightrag_graph_path=self.graph_file,
            semantic_units_path=units_file,
        )
        lr_retriever = LightRAGRetriever(
            config={
                "lightrag": {
                    "enabled": True,
                    "use_llm_query_extraction": False,
                    "supplement_weight": 0.65,
                    "enable_synthesis_overview": True,
                }
            },
            graph_path=self.graph_file,
            embedding_model=embedder,
            chunk_bridge=bridge,
        )
        uet_retriever = UETRetriever(
            embedding_model=embedder,
            vector_store=SimpleStore(),
            reranker=HeuristicReranker(),
            bm25_index=BM25Index(chunks),
            graph_index=bridge,
            lightrag_retriever=lr_retriever,
            config={
                "retrieval": {
                    "enable_graph": True,
                    "enable_mmr": False,
                    "final_top_k": 2,
                },
                "lightrag": {
                    "enabled": True,
                    "use_llm_query_extraction": False,
                    "router": {"mode": "auto"},
                },
            },
        )

        # Case A: Câu hỏi quan hệ -> LightRAG dẫn dắt, trả về chunk_id thật + lightrag_overview cho LLM
        q_rel = UserQuery(query_text="Điều kiện bị buộc thôi học đối với ngành công nghệ thông tin là gì?")
        res_rel = uet_retriever.retrieve(q_rel, top_k=2)
        self.assertEqual(res_rel[0].chunk_id, "real_chunk_qc_1")
        self.assertEqual(res_rel[0].metadata.get("graph_engine"), "lightrag_primary")
        self.assertTrue(res_rel[0].metadata.get("lightrag_grounded"))
        self.assertIn("lightrag_overview", res_rel[0].metadata)

        aug_prompt = UETPromptAugmenter(include_few_shot=False).augment(q_rel, res_rel)
        self.assertIn("[BẢN ĐỒ LIÊN KẾT QUY CHẾ (LIGHTRAG 2-TẦNG)]", aug_prompt.formatted_prompt)
        self.assertIn("real_chunk_qc_1", [c.chunk_id for c in aug_prompt.contexts])

        # Case B: Câu hỏi chứa mã học phần INT3401 không có trong graph.json -> Regex Supplement tự lấp đầy
        q_code = UserQuery(query_text="Học phần INT3401 có mấy tín chỉ?")
        grounded_code, _ = lr_retriever.retrieve_grounded(q_code, top_k=2, include_overview=False)
        code_ids = [c.chunk_id for c in grounded_code]
        self.assertIn("real_chunk_course_int3401", code_ids)

    def test_cross_faculty_filtering_and_detect_query_faculties(self):
        """Kiểm tra nhận diện Khoa/Viện đích danh và giảm nhiễu liên khoa trong LightRAG high-level summaries."""
        from src.graph.thesaurus_utils import detect_query_faculties

        self.assertEqual(detect_query_faculties("Ban chủ nhiệm Khoa Công nghệ Nông nghiệp (FAT) gồm ai?"), {"fat"})
        self.assertEqual(detect_query_faculties("Trong nước, Khoa Điện tử - Viễn thông (FET) hợp tác với đơn vị nào?"), {"fet"})
        # Tên ngành không phải tên Khoa/Viện đích danh -> không lọc nhầm
        self.assertEqual(detect_query_faculties("Điểm chuẩn ngành Trí tuệ nhân tạo và Công nghệ sinh học 2026?"), set())
        # So sánh 2 khoa -> giữ cả 2
        self.assertEqual(detect_query_faculties("So sánh Khoa CNTT (FIT) và Khoa Điện tử - Viễn thông (FET)"), {"fit", "fet"})

        lr = LightRAGRetriever(
            config={"lightrag": {"enabled": True, "use_llm_query_extraction": False}},
            graph_path=self.graph_file,
        )
        lr.document_summaries = {
            "fema_bcn": {
                "title": "Ban chủ nhiệm khoa",
                "summary": "Thông tin lãnh đạo Khoa Cơ học kỹ thuật và Tự động hóa (FEMA).",
                "document_type": "Giới thiệu",
                "key_entities": ["Ban chủ nhiệm khoa", "Khoa Cơ học kỹ thuật và Tự động hóa"],
            },
            "fat_bcn": {
                "title": "Ban chủ nhiệm khoa",
                "summary": "Thông tin lãnh đạo Khoa Công nghệ Nông nghiệp (FAT).",
                "document_type": "Giới thiệu",
                "key_entities": ["Ban chủ nhiệm khoa", "Khoa Công nghệ Nông nghiệp"],
            },
        }
        summaries = lr._extract_high_level_summaries(
            "Ban chủ nhiệm Khoa Công nghệ Nông nghiệp (FAT) gồm những ai?",
            seed_entities=[("Khoa Công nghệ Nông nghiệp", 1.0), ("Ban chủ nhiệm khoa", 0.96)],
        )
        self.assertGreaterEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["doc_id"], "fat_bcn")
        if len(summaries) > 1:
            self.assertGreater(summaries[0]["score"], summaries[1]["score"] * 3.0)

    def test_thesaurus_no_alias_collision_and_router_punctuation_forbidden(self):
        """7. Kiểm tra chống va chạm alias trong Thesaurus và lọc dấu câu/từ cấm trong QueryRouter."""
        from src.graph.thesaurus_utils import get_thesaurus
        from src.retrieval.router import QueryRouter

        th = get_thesaurus()
        self.assertEqual(th.normalize("Nghỉ học không phép"), "Nghỉ học không phép")
        self.assertEqual(th.normalize("nghỉ học không phép"), "Nghỉ học không phép")
        self.assertEqual(th.normalize("bảo lưu"), "Nghỉ học tạm thời")

        router = QueryRouter(thesaurus=th)
        # Dấu hỏi chấm sát mã khoa vẫn phải nhận diện được Khoa CNTT
        a_fit = router.analyze_query("Văn phòng khoa FIT?")
        self.assertIn("Khoa Công nghệ thông tin", a_fit["detected_entities"])
        self.assertEqual(router.route("Văn phòng khoa FIT ở đâu?"), "vector")

        # Từ chung chung 'UET', 'môn học' không được làm phình entity_count gây lệch sang hybrid
        a_intro = router.analyze_query("Giới thiệu Khoa Công nghệ thông tin UET")
        self.assertEqual(a_intro["detected_entities"], ["Khoa Công nghệ thông tin"])
        self.assertEqual(router.route("Giới thiệu Khoa Công nghệ thông tin UET"), "vector")

    def test_indexer_doc_id_summaries_edge_evidence_merge_and_forbidden_filter(self):
        """8. Kiểm tra LightRAGIndexer không ghi đè summary trùng tiêu đề, gộp evidence cạnh trùng và lọc từ cấm."""
        from unittest.mock import patch
        from src.base import ProcessedData
        from src.graph.lightrag_indexer import EntityRegistry, LLMResponseParser, LightRAGIndexer

        # 8a. EntityRegistry gộp nhiều source_unit_ids và evidence_texts cho cùng 1 cạnh
        reg = EntityRegistry()
        e1 = reg.register_edge("Cảnh báo học vụ", "Buộc thôi học", "dẫn_chiếu_đến", "Căn cứ Điều 15", "u_1")
        e2 = reg.register_edge("Cảnh báo học vụ", "Buộc thôi học", "dẫn_chiếu_đến", "Căn cứ Sổ tay SV", "u_2")
        self.assertIsNotNone(e1)
        self.assertIsNone(e2)
        self.assertEqual(e1["source_unit_ids"], ["u_1", "u_2"])
        self.assertEqual(e1["evidence_texts"], ["Căn cứ Điều 15", "Căn cứ Sổ tay SV"])

        # 8b. Parser lọc sạch thực thể cấm kể cả khi có tiền tố namespace hoặc nằm trong relationships
        from src.graph.thesaurus_utils import get_thesaurus

        ents, rels = LLMResponseParser.parse_pass2(
            raw=json.dumps({
                "entities": [
                    {"canonical_name": "[QĐ-5115] Sinh viên", "entity_type": "OTHER", "description": "Nhiễu"},
                    {"canonical_name": "Buộc thôi học", "entity_type": "DIEU_KIEN", "description": "Hợp lệ"},
                ],
                "relationships": [
                    {"source": "Sinh viên", "target": "Buộc thôi học", "relation_type": "áp_dụng_cho", "evidence": "Nhiễu"},
                    {"source": "Cảnh báo học vụ", "target": "Buộc thôi học", "relation_type": "dẫn_chiếu_đến", "evidence": "Chuẩn"},
                ],
            }, ensure_ascii=False),
            thesaurus=get_thesaurus(),
            namespace_prefix="[QĐ-5115]",
            unit_id="u_test",
        )
        self.assertEqual([e["canonical_name"] for e in ents], ["Buộc thôi học"])
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["source"], "Cảnh báo học vụ")

        # 8c. Hai văn bản trùng tiêu đề thuộc 2 khoa khác nhau không ghi đè mất nhau trong _doc_summaries
        idx_graph = Path(self.tmp_dir.name) / "idx_graph.json"
        idx_vec = Path(self.tmp_dir.name) / "idx_chroma"
        indexer = LightRAGIndexer(graph_path=idx_graph, entity_vector_dir=idx_vec, delay_between_calls=0.0)
        p1_resp = json.dumps({
            "document_title": "Ban chủ nhiệm khoa",
            "document_code": None,
            "issuing_body": "UET",
            "scope": "Thông tin Ban chủ nhiệm",
            "root_entities": [
                {"canonical_name": "Sinh viên", "entity_type": "OTHER"},
                {"canonical_name": "Khoa Công nghệ thông tin", "entity_type": "DON_VI"},
            ],
        }, ensure_ascii=False)
        p2_resp = json.dumps({"entities": [], "relationships": []})
        with patch.object(indexer, "_call_llm_raw", side_effect=[p1_resp, p2_resp, p1_resp, p2_resp]):
            d1 = ProcessedData(id="doc_fit_bcn", title="Ban chủ nhiệm khoa", content="Nội dung giới thiệu Ban chủ nhiệm Khoa Công nghệ thông tin dài trên năm mươi ký tự.")
            d2 = ProcessedData(id="doc_fat_bcn", title="Ban chủ nhiệm khoa", content="Nội dung giới thiệu Ban chủ nhiệm Khoa Công nghệ Nông nghiệp dài trên năm mươi ký tự.")
            indexer.process_single_document(d1)
            indexer.process_single_document(d2)

        self.assertIn("doc_fit_bcn", indexer._doc_summaries)
        self.assertIn("doc_fat_bcn", indexer._doc_summaries)
        self.assertFalse(indexer.registry.node_exists("Sinh viên"))

    def test_adaptive_extractor_hierarchy_and_table_splitting_and_pipeline_clear_safety(self):
        """9. Kiểm tra HeadingSectionExtractor giữ H1 cha, TableUnitExtractor cắt bảng dài, và --to lightrag --clear không xóa chunked_data."""
        from unittest.mock import patch
        import src.pipeline as pipeline_module
        from src.base import ProcessedData
        from src.graph.adaptive_extractor import AdaptiveSemanticExtractor
        from src.ingestion.manifest import DataManifestTracker

        ext = AdaptiveSemanticExtractor(max_unit_chars=260, min_unit_chars=20)
        h_doc = ProcessedData(
            id="h_doc",
            title="Tài liệu Khoa",
            content=(
                "# Phần I Tổng quan\nNội dung mở đầu phần một đủ dài trên hai mươi ký tự.\n"
                "## Mục 1.1 Đào tạo\nNội dung mục đào tạo đủ dài trên hai mươi ký tự.\n"
                "### Tiểu mục A\nNội dung tiểu mục A đủ dài trên hai mươi ký tự.\n"
                "## Mục 1.2 Nghiên cứu\nNội dung mục nghiên cứu đủ dài trên hai mươi ký tự."
            ),
        )
        h_units = ext.extract(h_doc)
        last_crumb = h_units[-1].breadcrumb
        self.assertIn("Phần I Tổng quan", last_crumb)
        self.assertIn("Mục 1.2 Nghiên cứu", last_crumb)

        rows = ["| Mã ngành | Tên ngành | Chỉ tiêu |", "|---|---|---|"] + [
            f"| CN{i} | Ngành đào tạo thử nghiệm số {i} chất lượng cao | {50 + i} |"
            for i in range(1, 15)
        ]
        t_doc = ProcessedData(id="t_doc", title="Bảng chỉ tiêu", content="### Chỉ tiêu 2026\n" + "\n".join(rows))
        t_units = ext.extract(t_doc)
        self.assertGreater(len(t_units), 1)
        for tu in t_units:
            self.assertIn("| Mã ngành | Tên ngành | Chỉ tiêu |", tu.content)
            self.assertLessEqual(tu.char_count, 320)

        # Kiểm chứng pipeline.run_stages(from_stage='processed', to_stage='lightrag', clear=True) bảo toàn chunked_data
        tmp_root = Path(self.tmp_dir.name) / "pipe_root"
        chunked_dir = tmp_root / "data" / "chunked_data"
        chunked_dir.mkdir(parents=True, exist_ok=True)
        sentinel_chunk_file = chunked_dir / "keep_me.jsonl"
        sentinel_chunk_file.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")

        with patch.object(pipeline_module, "BASE_DIR", tmp_root):
            pipe = pipeline_module.UETIngestionPipeline(
                embedding_model=DummyEmbedder(),
                manifest=DataManifestTracker(manifest_path=tmp_root / "manifest.json"),
                config={"lightrag": {"graph_path": "database/lightrag_db/graph.json"}},
            )
            with patch.object(pipe, "_run_processed_to_lightrag", return_value={"status": "skipped"}):
                pipe.run_stages(from_stage="processed", to_stage="lightrag", clear=True)

        self.assertTrue(sentinel_chunk_file.exists(), "Chạy --to lightrag --clear tuyệt đối không được xóa chunked_data!")


if __name__ == "__main__":
    unittest.main()

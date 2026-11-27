"""
tests/test_agent_graph.py - Kiểm thử đơn vị & tích hợp cho đồ thị LangGraph v3,
UETFollowupQueryGenerator, UETQueryRewriter, clarify_ask fallback và UETAgenticRAGPipeline.
Hoạt động 100% offline.
"""

import unittest
from typing import Any, Dict, List, Optional

from src.base import RetrievedContext, UserQuery, AugmentedPrompt, Response
from src.agent.graph import UETAgentGraph
from src.agent.agent_pipeline import UETAgenticRAGPipeline
from src.agent.followup import UETFollowupQueryGenerator
from src.agent.rewriter import UETQueryRewriter
from src.decision import FakeDecisionClient, Route, Verdict


class MockRetriever:
    def __init__(self, sample_contexts: Optional[List[RetrievedContext]] = None, return_empty: bool = False):
        self.return_empty = return_empty
        self.sample_contexts = sample_contexts or [
            RetrievedContext(
                chunk_id="chunk_001",
                text="Sinh viên có điểm trung bình chung tích lũy từ 3.20 đến 3.59 được xếp loại Giỏi.",
                similarity_score=0.85,
                rank=1,
                metadata={"title": "Quy chế đào tạo đại học UET", "source": "quy_che_uet.pdf"},
            ),
            RetrievedContext(
                chunk_id="chunk_002",
                text="Chuẩn đầu ra ngoại ngữ: Bậc 3 áp dụng cho hệ chuẩn, Bậc 4 áp dụng cho hệ CLC.",
                similarity_score=0.75,
                rank=2,
                metadata={"title": "Quy định chuẩn đầu ra ngoại ngữ", "source": "chuan_ngoai_ngu.pdf"},
            ),
        ]

    def retrieve(self, query: UserQuery, top_k: int = 5, filters=None) -> List[RetrievedContext]:
        if self.return_empty:
            return []
        return self.sample_contexts[:top_k]


class MockEmbeddingModel:
    dimension = 1024
    model_name = "mock-embedder"

    def embed_texts(self, texts):
        return [[0.1] * 1024 for _ in texts]

    def embed_chunks(self, chunks):
        from src.base import EmbeddedVector
        return [EmbeddedVector(chunk_id=c.chunk_id, vector=[0.1] * 1024, dimension=1024, metadata=c.metadata) for c in chunks]

    def embed_query(self, query):
        from src.base import EmbeddedQueryVector
        return EmbeddedQueryVector(query_id=query.query_id, query_text=query.query_text, vector=[0.1] * 1024, dimension=1024)


class MockVectorStore:
    def search(self, query_vector, top_k=5, filters=None):
        return []

    def count(self):
        return 100

    def store(self, chunks, vectors=None):
        return len(chunks)

    def clear(self):
        pass


class MockLLMClient:
    model_name = "mock-llm-model"

    def __init__(self, json_response: Optional[Dict[str, Any]] = None, raise_on_json: bool = False):
        self.json_response = json_response
        self.raise_on_json = raise_on_json

    def generate(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Response:
        return Response(
            query_id=prompt.user_query.query_id,
            answer=f"Theo quy chế đào tạo UET: {prompt.contexts[0].text if prompt.contexts else 'Không có dữ liệu.'}",
            sources=prompt.contexts,
            model_name=model or self.model_name,
            latency_seconds=0.05,
        )

    def generate_json(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.raise_on_json:
            raise RuntimeError("Invalid JSON from LLM")
        return self.json_response

    def generate_stream(self, prompt: AugmentedPrompt):
        yield "Theo quy chế đào tạo UET..."


class TestAgentGraphV3(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MockRetriever()
        self.mock_llm = MockLLMClient()
        self.fake_decision = FakeDecisionClient(
            default_route=Route.QUALIFIED_QUESTION,
            default_verdict=Verdict.SUFFICIENT,
        )
        self.graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=self.fake_decision,
        )

    def test_graph_topology_matches_v3_diagram(self):
        """Khẳng định đồ thị có đúng 6 node chính + fail_node ẩn, không còn các node clarify chặn lượt."""
        expected_nodes = {
            "router_node",
            "out_of_scope_node",
            "retrieval_node",
            "generation_node",
            "answer_eval_node",
            "followup_query_node",
            "fail_node",
        }
        actual_nodes = set(self.graph.workflow.nodes.keys())
        self.assertEqual(actual_nodes, expected_nodes)

        mermaid = self.graph.app.get_graph().draw_mermaid()
        self.assertIn("answer_eval_node", mermaid)
        self.assertIn("followup_query_node", mermaid)
        self.assertNotIn("clarify_ask_node", mermaid)
        self.assertNotIn("clarify_check_node", mermaid)
        self.assertNotIn("clarify_exhausted_node", mermaid)
        self.assertNotIn("decompose_node", mermaid)
        self.assertNotIn("grader_node", mermaid)
        self.assertNotIn("hallucination_node", mermaid)

    def test_rag_retrieval_and_eval_flow(self):
        state = self.graph.run(query="Quy chế đào tạo và chuẩn đầu ra ngoại ngữ")
        self.assertIn("retrieval: 2 chunks", state["execution_steps"])
        self.assertIn("generation_llm", state["execution_steps"])
        self.assertIn("answer_eval: SUFFICIENT", state["execution_steps"])
        self.assertGreater(len(state["contexts"]), 0)
        self.assertIn("Theo quy chế đào tạo UET", state["answer"])

    def test_custom_max_retries_from_config(self):
        """Đảm bảo ngưỡng answer_eval.max_retries đọc từ config, không hard-code."""
        custom_graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=FakeDecisionClient(
                routes=[Route.QUALIFIED_QUESTION],
                evaluate_answers=[Verdict.INSUFFICIENT] * 5,
            ),
            config={"agent": {"answer_eval": {"max_retries": 2}}},
        )
        self.assertEqual(custom_graph.max_retries, 2)
        state = custom_graph.run(query="Câu hỏi kiểm tra config max_retries")
        self.assertEqual(state["retry"], 3)
        self.assertEqual(state["response"].status, "partial")

    def test_partial_returns_best_answer_by_noul_prob(self):
        """Khi retry > max_retries, trả về câu trả lời có điểm noul_prob cao nhất thay vì vòng cuối bị tụt điểm."""
        class ScoredDecisionClient(FakeDecisionClient):
            def __init__(self, probs):
                super().__init__(
                    routes=[Route.QUALIFIED_QUESTION],
                    evaluate_answers=[Verdict.INSUFFICIENT] * len(probs),
                )
                self._probs = list(probs)

            def evaluate_answer(self, query: str, answer: str, session_id: Optional[str] = None) -> Verdict:
                prob = self._probs.pop(0) if self._probs else 0.0
                self.last_trace_meta = {"noul_prob": prob}
                return super().evaluate_answer(query, answer, session_id=session_id)

        class NumberedLLM(MockLLMClient):
            def __init__(self):
                super().__init__()
                self.n = 0

            def generate(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Response:
                self.n += 1
                return Response(
                    query_id=prompt.user_query.query_id,
                    answer=f"Đáp án vòng {self.n}",
                    sources=prompt.contexts,
                    model_name="mock",
                    latency_seconds=0.01,
                )

        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=NumberedLLM(),
            decision_client=ScoredDecisionClient([0.18, 0.37, 0.08, 0.04]),
            config={"agent": {"answer_eval": {"max_retries": 3}}},
        )
        state = graph.run(query="So sánh điểm chuẩn 2 ngành")
        self.assertEqual(state["response"].status, "partial")
        self.assertIn("Đáp án vòng 2", state["response"].answer)
        self.assertAlmostEqual(state["best_answer_score"], 0.37)

    def test_retrieval_round_robin_prevents_query_starvation(self):
        """Query 1 (#1 score=0.684) không bị Query 2 (5 chunks score=0.79x) đẩy mất khỏi ngữ cảnh vòng mới."""
        class TwoQueryRetriever:
            def retrieve(self, query: UserQuery, top_k: int = 5, filters=None) -> List[RetrievedContext]:
                if "trí tuệ nhân tạo" in query.query_text.lower():
                    return [
                        RetrievedContext(
                            chunk_id="ai_target_#1",
                            text="Trí tuệ nhân tạo CN12: 26,26",
                            similarity_score=0.684,
                            rank=1,
                            metadata={"title": "Điểm trúng tuyển 2026"},
                        ),
                        RetrievedContext(
                            chunk_id="ai_#2",
                            text="Chunk AI #2",
                            similarity_score=0.656,
                            rank=2,
                            metadata={"title": "CTĐT AI"},
                        ),
                    ]
                return [
                    RetrievedContext(
                        chunk_id=f"noisy_{i}",
                        text=f"Nghị định 179 chunk {i}",
                        similarity_score=0.798 - i * 0.003,
                        rank=i + 1,
                        metadata={"title": "Nghị định 179/2026"},
                    )
                    for i in range(5)
                ]

        graph = UETAgentGraph(
            retriever=TwoQueryRetriever(),
            llm_client=self.mock_llm,
            decision_client=self.fake_decision,
            config={"agent": {"contexts": {"max_chunks": 8}}},
        )
        prior_pool = {
            f"old_{i}": RetrievedContext(
                chunk_id=f"old_{i}",
                text=f"Old chunk {i}",
                similarity_score=0.70 - i * 0.01,
                rank=i + 1,
                metadata={},
            )
            for i in range(6)
        }
        out = graph.retrieval_node({
            "query": "điểm chuẩn",
            "followup_queries": [
                "điểm chuẩn trúng tuyển ngành Trí tuệ nhân tạo UET năm 2026",
                "danh sách điểm chuẩn các ngành đào tạo chính quy UET 2026",
            ],
            "context_pool": prior_pool,
            "used_queries": [],
            "execution_steps": [],
        })
        selected_ids = [c.chunk_id for c in out["contexts"]]
        self.assertEqual(len(selected_ids), 8)
        self.assertIn("ai_target_#1", selected_ids[:4])
        self.assertIn("noisy_0", selected_ids[:4])

    def test_clean_query_for_retrieval_strips_supplement_wrapper(self):
        raw = "điểm chuẩn\n(Bổ sung: ngành trí tuệ nhân tạo, năm 2026)"
        cleaned = UETAgentGraph._clean_query_for_retrieval(raw)
        self.assertNotIn("Bổ sung", cleaned)
        self.assertIn("điểm trúng tuyển", cleaned)
        self.assertIn("trí tuệ nhân tạo", cleaned)

        already = "điểm chuẩn trúng tuyển ngành CNTT năm 2026?"
        self.assertEqual(UETAgentGraph._clean_query_for_retrieval(already), already)

        plain = "Quy định xét học bổng UET?"
        self.assertEqual(UETAgentGraph._clean_query_for_retrieval(plain), plain)

    def test_run_stream_enforces_answer_eval_and_emits_only_final_answer(self):
        """run_stream() bắt buộc qua answer_eval và CHỈ stream token của câu trả lời cuối cùng đã duyệt."""
        class TwoRoundLLM(MockLLMClient):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def generate(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Response:
                self.calls += 1
                if self.calls == 1:
                    ans = "BAN_NHAP_VONG_1_CHUA_DU"
                else:
                    ans = "DAP_AN_CUOI_DA_DUYET\n| Hệ | Yêu cầu |\n| :--- | :--- |\n| Chuẩn | Bậc 3 |"
                return Response(
                    query_id=prompt.user_query.query_id,
                    answer=ans,
                    sources=prompt.contexts,
                    model_name="mock-llm-2round",
                    latency_seconds=0.01,
                )

        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=TwoRoundLLM(),
            decision_client=FakeDecisionClient(
                routes=[Route.QUALIFIED_QUESTION],
                evaluate_answers=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
            ),
        )
        events = list(graph.run_stream(query="Chuẩn tiếng Anh UET là gì?"))
        status_nodes = [e["node"] for e in events if e["type"] == "status"]
        tokens_text = "".join(e["content"] for e in events if e["type"] == "token")
        done_events = [e for e in events if e["type"] == "done"]

        self.assertIn("router_node", status_nodes)
        self.assertIn("answer_eval_node", status_nodes)
        self.assertIn("followup_query_node", status_nodes)
        self.assertNotIn("BAN_NHAP_VONG_1_CHUA_DU", tokens_text)
        self.assertIn("DAP_AN_CUOI_DA_DUYET", tokens_text)
        # Bảo toàn ký tự xuống dòng bảng Markdown
        self.assertIn("\n| Hệ | Yêu cầu |\n", tokens_text)
        self.assertEqual(len(done_events), 1)
        self.assertEqual(done_events[0]["status"], "ok")
        self.assertEqual(done_events[0]["retry"], 1)

    def test_anchor_count_extraction(self):
        self.assertEqual(
            UETAgentGraph.extract_anchor_count("Trường Đại học Công nghệ đào tạo 18 ngành đại học chính quy."),
            18,
        )
        self.assertEqual(
            UETAgentGraph.extract_anchor_count("Trường gồm 6 khoa và 2 viện nghiên cứu.", entity="khoa"),
            6,
        )
        self.assertIsNone(
            UETAgentGraph.extract_anchor_count("Năm học 2026 trường chào đón các tân sinh viên nhập học.")
        )


class TestFollowupGeneratorAndRewriter(unittest.TestCase):
    def test_followup_llm_dedup_and_cap(self):
        llm = MockLLMClient(
            json_response={
                "missing_info": "chuẩn ngoại ngữ hệ CLC",
                "followup_queries": [
                    "query đã dùng",
                    "Chuẩn đầu ra ngoại ngữ chương trình CLC UET",
                    "Quy đổi IELTS bậc 4 UET",
                    "Truy vấn thứ 3 bị cắt",
                ],
            }
        )
        gen = UETFollowupQueryGenerator(llm_client=llm, enable_llm=True, max_queries=2)
        missing, queries, source = gen.generate(
            query="So sánh chuẩn ngoại ngữ Chuẩn và CLC?",
            answer="Hệ Chuẩn cần Bậc 3.",
            used_queries=["query đã dùng"],
        )
        self.assertEqual(source, "llm")
        self.assertEqual(missing, "chuẩn ngoại ngữ hệ CLC")
        self.assertEqual(len(queries), 2)
        self.assertNotIn("query đã dùng", queries)

    def test_followup_heuristic_fallback_on_error(self):
        llm = MockLLMClient(raise_on_json=True)
        gen = UETFollowupQueryGenerator(llm_client=llm, enable_llm=True, max_queries=2)
        missing, queries, source = gen.generate(
            query="Điều kiện xét HBKKHT UET",
            answer="Chưa có thông tin.",
            used_queries=["Điều kiện xét HBKKHT UET"],
        )
        self.assertEqual(source, "heuristic")
        self.assertIsNotNone(missing)
        self.assertGreaterEqual(len(queries), 1)
        self.assertNotIn("Điều kiện xét HBKKHT UET", queries)
        self.assertIn("học bổng khuyến khích học tập", queries[0].lower())

    def test_rewriter_helpers(self):
        self.assertIn("chương trình đào tạo", UETQueryRewriter.expand_acronyms("Quy định CTĐT CLC").lower())
        self.assertEqual(UETQueryRewriter.detect_faculty("Sinh viên khoa CNTT học gì?"), "fit")
        facs = UETQueryRewriter.detect_faculties("So sánh CNTT và Trí tuệ nhân tạo")
        self.assertIn("fit", facs)
        self.assertIn("iai", facs)


class TestAgenticRAGPipelineV3(unittest.TestCase):
    def test_pipeline_query_and_stream_metadata(self):
        pipeline = UETAgenticRAGPipeline(
            embedding_model=MockEmbeddingModel(),
            vector_store=MockVectorStore(),
            retriever=MockRetriever(),
            llm=MockLLMClient(),
            decision_client=FakeDecisionClient(routes=[Route.QUALIFIED_QUESTION], evaluate_answers=[Verdict.SUFFICIENT]),
        )

        uq = UserQuery(query_text="Chuẩn đầu ra ngoại ngữ UET là gì?")
        resp = pipeline.query(uq)
        self.assertIsNotNone(resp.answer)
        self.assertEqual(resp.metadata.get("status"), "ok")
        self.assertEqual(resp.metadata.get("state"), 1)
        self.assertEqual(resp.metadata.get("route"), "QUALIFIED_QUESTION")
        self.assertEqual(resp.metadata.get("retry"), 0)
        self.assertNotIn("retried3", resp.metadata)
        self.assertIn("used_queries", resp.metadata)
        self.assertIn("execution_steps", resp.metadata)
        self.assertIn("trace", resp.metadata)

        # Test stream & stream_events
        stream_tokens = list(pipeline.query_stream(uq))
        self.assertIn("Theo quy chế đào tạo UET", "".join(stream_tokens))

        stream_evs = list(pipeline.stream_events(uq))
        self.assertTrue(any(e["type"] == "status" for e in stream_evs))
        self.assertTrue(any(e["type"] == "sources" for e in stream_evs))
        self.assertTrue(any(e["type"] == "done" for e in stream_evs))


if __name__ == "__main__":
    unittest.main()

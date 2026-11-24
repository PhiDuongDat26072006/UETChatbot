"""
tests/test_upgrade_conformance.py - Kiểm thử các cơ chế độ tin cậy cốt lõi được bảo toàn trong v3:
  - D1 / Q5: Lỗi JEV tại router / clarify_check / answer_eval -> fail_node (fail_service)
  - D12: Lỗi LLM tại generation_node -> fail_node (fail_service)
  - D3: Resume phiên AMBIGUOUS trực tiếp qua cạnh điều kiện từ START
  - D5: Soft faculty filter phát hiện nhiều khoa
Hoạt động 100% offline.
"""

import unittest
from unittest.mock import MagicMock

from src.base import RetrievedContext, UserQuery, AugmentedPrompt, Response
from src.agent.graph import UETAgentGraph
from src.agent.rewriter import UETQueryRewriter
from src.decision import (
    Route,
    Verdict,
    AgentResponse,
    FakeDecisionClient,
    JevUnavailableError,
)
from src.llm.llm_client import LlmUnavailableError


class MockRetriever:
    def __init__(self, sample_contexts=None):
        self.sample_contexts = sample_contexts or [
            RetrievedContext(
                chunk_id="chunk_fit_01",
                text="Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 28.15 điểm.",
                similarity_score=0.9,
                rank=1,
                metadata={"title": "Thông báo điểm trúng tuyển UET", "faculty": "fit"},
            ),
            RetrievedContext(
                chunk_id="chunk_iai_01",
                text="Điểm chuẩn ngành Trí tuệ nhân tạo năm 2026 là 27.85 điểm.",
                similarity_score=0.88,
                rank=2,
                metadata={"title": "Thông báo điểm trúng tuyển UET", "faculty": "iai"},
            ),
        ]

    def retrieve(self, query: UserQuery, top_k: int = 5, filters=None):
        return self.sample_contexts[:top_k]


class MockLLM:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def generate(self, prompt: AugmentedPrompt, model=None):
        if self.fail:
            raise LlmUnavailableError("LLM Provider 429: Too many requests")
        return Response(
            query_id="q1",
            answer="Ngành CNTT 28.15 điểm và ngành TTNT 27.85 điểm.",
            sources=prompt.contexts,
            model_name="mock-llm",
            latency_seconds=0.1,
        )


class TestReliabilityConformanceV3(unittest.TestCase):
    def setUp(self):
        self.retriever = MockRetriever()
        self.mock_llm = MockLLM(fail=False)

    def test_d1_router_error_routes_to_fail_node(self):
        failing_client = FakeDecisionClient()
        failing_client.route = MagicMock(side_effect=JevUnavailableError("JEV connection timeout"))

        graph = UETAgentGraph(
            retriever=self.retriever,
            llm_client=self.mock_llm,
            decision_client=failing_client,
        )
        state = graph.run("Điểm chuẩn ngành CNTT?")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "fail")
        self.assertEqual(resp.state, 0)
        self.assertEqual(resp.fail_reason, "jev_unavailable")
        self.assertIn("gián đoạn dịch vụ", resp.answer.lower())
        self.assertIn("router: ERROR -> fail_node", state["execution_steps"])

    def test_d1_clarify_check_error_routes_to_fail_node(self):
        failing_client = FakeDecisionClient()
        failing_client.check_supplement = MagicMock(side_effect=JevUnavailableError("JEV 500 error"))

        graph = UETAgentGraph(
            retriever=self.retriever,
            llm_client=self.mock_llm,
            decision_client=failing_client,
        )
        graph.clarify_store.save_session(
            session_id="test_sess_01",
            query="Em muốn hỏi tốt nghiệp",
            clarify_asks=["Ngành nào?"],
            clarify_supplements=[],
            retried3=0,
            missing_info=["ngành học"],
        )

        state = graph.run("Ngành CNTT chuẩn", session_id="test_sess_01")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "fail")
        self.assertEqual(resp.state, 0)
        self.assertEqual(resp.fail_reason, "jev_unavailable")
        self.assertIn("gián đoạn dịch vụ", resp.answer.lower())

    def test_q5_answer_eval_error_routes_to_fail_node(self):
        failing_client = FakeDecisionClient(routes=[Route.QUALIFIED_QUESTION])
        failing_client.evaluate_answer = MagicMock(side_effect=JevUnavailableError("JEV eval error"))

        graph = UETAgentGraph(
            retriever=self.retriever,
            llm_client=self.mock_llm,
            decision_client=failing_client,
        )
        state = graph.run("Điểm chuẩn ngành CNTT?")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "fail")
        self.assertEqual(resp.state, 0)
        self.assertEqual(resp.fail_reason, "jev_unavailable")
        self.assertIn("answer_eval: ERROR -> fail_node", state["execution_steps"])

    def test_d12_generation_llm_error_routes_to_fail_node(self):
        fake_client = FakeDecisionClient(routes=[Route.QUALIFIED_QUESTION])
        failing_llm = MockLLM(fail=True)

        graph = UETAgentGraph(
            retriever=self.retriever,
            llm_client=failing_llm,
            decision_client=fake_client,
        )
        state = graph.run("Điểm chuẩn ngành CNTT?")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "fail")
        self.assertEqual(resp.state, 0)
        self.assertEqual(resp.fail_reason, "llm_unavailable")
        self.assertIn("gián đoạn dịch vụ", resp.answer.lower())
        self.assertIn("generation: ERROR -> fail_node", state["execution_steps"])

    def test_d3_resume_ambiguous_via_graph(self):
        fake_client = FakeDecisionClient(
            check_supplements=[Verdict.SUFFICIENT],
            evaluate_answers=[Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=self.retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
        )

        session_id = "sess_graph_resume_01"
        graph.clarify_store.save_session(
            session_id=session_id,
            query="Em muốn hỏi tốt nghiệp",
            clarify_asks=["Bạn học ngành gì?"],
            clarify_supplements=[],
            retried3=0,
            known_info=["chủ đề tốt nghiệp"],
            missing_info=["ngành", "khóa"],
        )

        state = graph.run("Ngành Công nghệ thông tin khóa 2022", session_id=session_id)
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "ok")
        self.assertEqual(resp.state, 1)
        self.assertIn("clarify_check: SUFFICIENT", state["execution_steps"])
        self.assertFalse(graph.clarify_store.has_pending_session(session_id))

    def test_d5_soft_faculty_filter_multiple_faculties(self):
        q = "So sánh điểm chuẩn ngành trí tuệ nhân tạo khoa cơ điện tử và công nghệ thông tin"
        faculties = UETQueryRewriter.detect_faculties(q)

        self.assertIn("iai", faculties)
        self.assertIn("fit", faculties)
        self.assertIn("fema", faculties)


if __name__ == "__main__":
    unittest.main()

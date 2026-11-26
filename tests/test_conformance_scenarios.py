"""
tests/test_conformance_scenarios.py - Bộ kiểm thử kiểm chứng toàn diện v3 (Kịch bản N01 - N12).
Tuân thủ 100% bảng kịch bản Phase 5 trong implementation_plan_v3.md.
Hoạt động 100% offline với FakeDecisionClient, MockRetriever và MockLLM.
"""

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from src.base import RetrievedContext, UserQuery, AugmentedPrompt, Response
from src.agent.graph import UETAgentGraph
from src.decision import (
    Route,
    Verdict,
    FakeDecisionClient,
    AgentResponse,
    JevUnavailableError,
)
from src.llm.llm_client import LlmUnavailableError


class MockRetriever:
    """Mock Retriever hỗ trợ theo dõi số vòng gọi và sinh chunk khác nhau theo truy vấn/vòng."""

    def __init__(
        self,
        fixed_chunks: Optional[List[RetrievedContext]] = None,
        chunks_by_round: Optional[List[List[RetrievedContext]]] = None,
    ):
        self.call_count = 0
        self.queries_received: List[str] = []
        self.chunks_by_round = chunks_by_round
        self.fixed_chunks = fixed_chunks or [
            RetrievedContext(
                chunk_id="chunk_sample_1",
                text="Sinh viên UET có điểm GPA từ 3.20 đến 3.59 được xếp loại Giỏi.",
                similarity_score=0.88,
                rank=1,
                metadata={"title": "Quy chế đào tạo", "article": "Điều 15"},
            ),
            RetrievedContext(
                chunk_id="chunk_sample_2",
                text="Chuẩn đầu ra ngoại ngữ Bậc 3 áp dụng cho hệ Chuẩn, Bậc 4 cho hệ CLC.",
                similarity_score=0.82,
                rank=2,
                metadata={"title": "Quy định ngoại ngữ", "article": "Điều 20"},
            ),
        ]

    def retrieve(self, query: UserQuery, top_k: int = 5, filters=None) -> List[RetrievedContext]:
        self.call_count += 1
        self.queries_received.append(query.query_text)
        if self.chunks_by_round is not None:
            idx = min(self.call_count - 1, len(self.chunks_by_round) - 1)
            return list(self.chunks_by_round[idx][:top_k])
        return [
            RetrievedContext(
                chunk_id=f"{c.chunk_id}_round{self.call_count}",
                text=c.text,
                similarity_score=c.similarity_score,
                rank=c.rank,
                metadata=c.metadata,
            )
            for c in self.fixed_chunks[:top_k]
        ]


class MockLLM:
    """Mock LLM hỗ trợ đếm số lần gọi generate và generate_json (followup_query)."""

    def __init__(
        self,
        fail_generate: bool = False,
        followup_payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        self.fail_generate = fail_generate
        self.generation_count = 0
        self.followup_call_count = 0
        self.followup_payloads = list(followup_payloads) if followup_payloads else []

    def generate(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Response:
        if self.fail_generate:
            raise LlmUnavailableError("LLM 429 rate limit")
        self.generation_count += 1
        joined = " | ".join(c.text for c in prompt.contexts) if prompt.contexts else "Không có dữ liệu."
        return Response(
            query_id=prompt.user_query.query_id,
            answer=f"Câu trả lời vòng {self.generation_count}: {joined}",
            sources=prompt.contexts,
            model_name="mock-llm",
        )

    def generate_json(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        text = prompt.formatted_prompt.lower()
        if "followup_queries" in text:
            self.followup_call_count += 1
            if self.followup_payloads:
                return dict(self.followup_payloads.pop(0))
            return {
                "missing_info": f"thông tin bổ sung lần {self.followup_call_count}",
                "followup_queries": [f"truy vấn bổ sung quy chế UET lần {self.followup_call_count}"],
            }
        return None


class TestConformanceScenariosV3(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MockRetriever()
        self.mock_llm = MockLLM()

    def test_n01_out_of_scope(self):
        """N01: OUT_OF_SCOPE -> router -> out_of_scope (state=0, không gọi retrieval/LLM)."""
        fake_client = FakeDecisionClient(routes=[Route.OUT_OF_SCOPE])
        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
        )
        state = graph.run(query="Xin chào bạn, bạn là ai?")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "out_of_scope")
        self.assertEqual(resp.state, 0)
        self.assertIn("rất vui được phục vụ", resp.answer.lower())
        self.assertIsNone(resp.contexts)
        self.assertEqual(self.mock_retriever.call_count, 0)
        self.assertEqual(self.mock_llm.generation_count, 0)
        self.assertEqual(len(fake_client.history), 1)
        self.assertIn("out_of_scope", state["execution_steps"])

    def test_n02_qualified_eval_sufficient_happy_path(self):
        """N02: QUALIFIED_QUESTION, eval ĐỦ -> router -> retr -> gen -> eval (ok, state=1, đúng 2 JEV + 1 LLM)."""
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
        )
        state = graph.run(query="Thời gian rút bớt học phần UET là bao lâu?")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "ok")
        self.assertEqual(resp.state, 1)
        self.assertEqual(resp.route, Route.QUALIFIED_QUESTION)
        self.assertEqual(resp.counters, {"retry": 0})
        self.assertIsNotNone(resp.contexts)
        self.assertGreater(len(resp.contexts), 0)
        # Đúng 2 lượt JEV (route + evaluate_answer) và 1 lượt LLM (generation)
        self.assertEqual([h["func"] for h in fake_client.history], ["route", "evaluate_answer"])
        self.assertEqual(self.mock_llm.generation_count, 1)
        self.assertEqual(self.mock_llm.followup_call_count, 0)
        self.assertEqual(self.mock_retriever.call_count, 1)

    def test_n03_eval_insufficient_once_then_sufficient(self):
        """N03: eval THIẾU x1 rồi ĐỦ -> ... -> eval -> fq -> retr -> gen -> eval (retry=1, ok)."""
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
        )
        state = graph.run(query="Quy định học lại và học phí học lại UET?")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "ok")
        self.assertEqual(resp.state, 1)
        self.assertEqual(state["retry"], 1)
        self.assertEqual(resp.counters["retry"], 1)
        self.assertEqual(self.mock_retriever.call_count, 2)
        self.assertEqual(self.mock_llm.generation_count, 2)
        self.assertEqual(self.mock_llm.followup_call_count, 1)

    def test_n04_eval_insufficient_3_times_then_sufficient(self):
        """N04: eval THIẾU x3 rồi ĐỦ -> 3 vòng follow-up, 4 vòng retrieval, 4 lần sinh, retry=3, ok."""
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[
                Verdict.INSUFFICIENT,
                Verdict.INSUFFICIENT,
                Verdict.INSUFFICIENT,
                Verdict.SUFFICIENT,
            ],
        )
        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
        )
        state = graph.run(query="Câu hỏi cần 3 vòng bổ sung")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "ok")
        self.assertEqual(resp.state, 1)
        self.assertEqual(state["retry"], 3)
        self.assertEqual(self.mock_retriever.call_count, 4)
        self.assertEqual(self.mock_llm.generation_count, 4)
        self.assertEqual(self.mock_llm.followup_call_count, 3)

    def test_n05_eval_insufficient_4_times_returns_partial_q1a(self):
        """N05: eval THIẾU x4 -> 3 vòng follow-up, 4 lần sinh, retry=4, RESPONSE partial (Q1=A)."""
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[
                Verdict.INSUFFICIENT,
                Verdict.INSUFFICIENT,
                Verdict.INSUFFICIENT,
                Verdict.INSUFFICIENT,
            ],
        )
        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
        )
        state = graph.run(query="Câu hỏi rất khó chưa có đủ dữ liệu")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "partial")
        self.assertEqual(resp.state, 1)
        self.assertEqual(state["retry"], 4)
        self.assertEqual(resp.counters["retry"], 4)
        self.assertEqual(self.mock_retriever.call_count, 4)
        self.assertEqual(self.mock_llm.generation_count, 4)
        self.assertEqual(self.mock_llm.followup_call_count, 3)
        # Q1=A: giữ câu trả lời gần nhất + ghi chú phần còn thiếu + gợi ý liên hệ Phòng Đào tạo
        self.assertIn("Câu trả lời vòng 4", resp.answer)
        self.assertIn("phần còn thiếu", resp.answer.lower())
        self.assertIn("phòng đào tạo", resp.answer.lower())
        self.assertIsNotNone(resp.contexts)
        self.assertIsNotNone(resp.missing_info)

    def test_n08_run_stream_only_emits_final_approved_answer_after_answer_eval(self):
        """N08: run_stream() chạy qua answer_eval + followup và CHỈ phát token của câu trả lời cuối cùng."""
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
        )
        events = list(graph.run_stream(query="Quy định học lại và học phí học lại UET?"))
        emitted_tokens = "".join(e["content"] for e in events if e["type"] == "token")
        done_ev = [e for e in events if e["type"] == "done"][-1]

        self.assertNotIn("Câu trả lời vòng 1", emitted_tokens)
        self.assertIn("Câu trả lời vòng 2", emitted_tokens)
        self.assertEqual(done_ev["status"], "ok")
        self.assertEqual(done_ev["retry"], 1)

    def test_n09_comparison_2_majors_followup_fills_missing_major(self):
        """N09: Câu so sánh 2 ngành, vòng 1 chỉ có 1 ngành -> eval THIẾU -> follow-up hỏi ngành còn lại -> vòng 2 đủ 2 ngành."""
        chunk_cntt = RetrievedContext(
            chunk_id="cntt_2026",
            text="Điểm chuẩn ngành Công nghệ thông tin UET năm 2026 là 28.15 điểm.",
            similarity_score=0.92,
            rank=1,
            metadata={"title": "Điểm chuẩn UET", "faculty": "fit"},
        )
        chunk_ai = RetrievedContext(
            chunk_id="ai_2026",
            text="Điểm chuẩn ngành Trí tuệ nhân tạo UET năm 2026 là 27.85 điểm.",
            similarity_score=0.90,
            rank=1,
            metadata={"title": "Điểm chuẩn UET", "faculty": "iai"},
        )
        retriever = MockRetriever(chunks_by_round=[[chunk_cntt], [chunk_ai]])
        llm = MockLLM(
            followup_payloads=[
                {
                    "missing_info": "điểm chuẩn ngành Trí tuệ nhân tạo UET",
                    "followup_queries": ["Điểm chuẩn ngành Trí tuệ nhân tạo Trường Đại học Công nghệ UET"],
                }
            ]
        )
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=retriever,
            llm_client=llm,
            decision_client=fake_client,
        )
        state = graph.run(query="So sánh điểm chuẩn ngành CNTT và ngành Trí tuệ nhân tạo UET?")
        resp: AgentResponse = state["response"]

        self.assertEqual(resp.status, "ok")
        self.assertEqual(state["retry"], 1)
        # Vòng 2 follow-up truy vấn đúng ngành Trí tuệ nhân tạo
        self.assertIn("Trí tuệ nhân tạo", retriever.queries_received[1])
        # Câu trả lời cuối có đủ dữ kiện của cả 2 ngành từ context_pool tích lũy
        self.assertIn("28.15", resp.answer)
        self.assertIn("27.85", resp.answer)
        ctx_ids = {c.chunk_id for c in resp.contexts}
        self.assertEqual(ctx_ids, {"cntt_2026", "ai_2026"})

    def test_n10_jev_and_llm_errors_route_to_fail_node(self):
        """N10: JEV lỗi (router / answer_eval) hoặc LLM lỗi (generation) -> fail_node với fail_service."""
        # 1. JEV router error
        bad_jev = FakeDecisionClient()
        bad_jev.route = MagicMock(side_effect=JevUnavailableError("JEV down"))
        g1 = UETAgentGraph(retriever=self.mock_retriever, llm_client=self.mock_llm, decision_client=bad_jev)
        st1 = g1.run("Điểm chuẩn CNTT?")
        self.assertEqual(st1["response"].status, "fail")
        self.assertEqual(st1["response"].state, 0)
        self.assertEqual(st1["response"].fail_reason, "jev_unavailable")
        self.assertIn("gián đoạn dịch vụ", st1["response"].answer.lower())

        # 2. JEV answer_eval error
        bad_eval_jev = FakeDecisionClient(routes=[Route.QUALIFIED_QUESTION])
        bad_eval_jev.evaluate_answer = MagicMock(side_effect=JevUnavailableError("JEV eval timeout"))
        g2 = UETAgentGraph(retriever=self.mock_retriever, llm_client=self.mock_llm, decision_client=bad_eval_jev)
        st2 = g2.run("Điểm chuẩn CNTT?")
        self.assertEqual(st2["response"].status, "fail")
        self.assertEqual(st2["response"].fail_reason, "jev_unavailable")

        # 3. LLM generation error
        bad_llm = MockLLM(fail_generate=True)
        g3 = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=bad_llm,
            decision_client=FakeDecisionClient(routes=[Route.QUALIFIED_QUESTION]),
        )
        st3 = g3.run("Điểm chuẩn CNTT?")
        self.assertEqual(st3["response"].status, "fail")
        self.assertEqual(st3["response"].fail_reason, "llm_unavailable")
        self.assertIn("gián đoạn dịch vụ", st3["response"].answer.lower())

    def test_n11_followup_never_repeats_used_queries(self):
        """N11: followup_queries không bao giờ lặp lại bất kỳ truy vấn nào trong used_queries."""
        original_q = "Quy định học bổng KKHT UET"
        # Giả lập LLM cố tình trả lại truy vấn đã dùng -> followup generator phải lọc bỏ và fallback truy vấn mới
        llm = MockLLM(
            followup_payloads=[
                {"missing_info": "mức học bổng", "followup_queries": [original_q]},
                {"missing_info": "hồ sơ xét", "followup_queries": [original_q]},
            ]
        )
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.INSUFFICIENT, Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=self.mock_retriever,
            llm_client=llm,
            decision_client=fake_client,
        )
        state = graph.run(query=original_q)

        # Kiểm tra tất cả truy vấn đã gửi tới retriever đều khác nhau đôi một (followup_queries ∩ used_queries = ∅)
        norm_queries = [" ".join(q.strip().lower().split()) for q in self.mock_retriever.queries_received]
        self.assertEqual(len(norm_queries), len(set(norm_queries)))
        self.assertGreaterEqual(len(state["used_queries"]), 3)
        self.assertEqual(len(state["used_queries"]), len(norm_queries))

    def test_n12_accumulated_contexts_preserve_prior_and_cap_max_chunks(self):
        """N12: Ngữ cảnh tích lũy giữ chunk vòng trước + ưu tiên chunk vòng mới và không vượt max_chunks (8)."""
        round1 = [
            RetrievedContext(
                chunk_id=f"r1_c{i}",
                text=f"Vòng 1 đoạn {i}",
                similarity_score=0.95 - i * 0.01,
                rank=i + 1,
                metadata={"round": 1},
            )
            for i in range(5)
        ]
        round2 = [
            RetrievedContext(
                chunk_id=f"r2_c{i}",
                text=f"Vòng 2 đoạn {i}",
                similarity_score=0.80 - i * 0.01,  # Điểm thấp hơn vòng 1 nhưng vẫn phải được ưu tiên vào contexts
                rank=i + 1,
                metadata={"round": 2},
            )
            for i in range(5)
        ]
        retriever = MockRetriever(chunks_by_round=[round1, round2])
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=retriever,
            llm_client=self.mock_llm,
            decision_client=fake_client,
            config={"agent": {"contexts": {"max_chunks": 8}}},
        )
        state = graph.run(query="Câu hỏi tích lũy ngữ cảnh")
        final_contexts = state["contexts"]

        # Tổng pool là 10 chunks, nhưng contexts đưa cho LLM không vượt max_chunks = 8
        self.assertEqual(len(state["context_pool"]), 10)
        self.assertEqual(len(final_contexts), 8)
        ids = {c.chunk_id for c in final_contexts}
        # Phải chứa cả chunk vòng 1 (cũ) lẫn chunk vòng 2 (mới dù điểm thấp hơn)
        self.assertTrue(any(cid.startswith("r1_") for cid in ids))
        self.assertTrue(any(cid.startswith("r2_") for cid in ids))


if __name__ == "__main__":
    unittest.main()

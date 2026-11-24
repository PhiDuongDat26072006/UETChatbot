"""
tests/test_jev_client.py - Kiểm thử đơn vị cho Enum v3, AgentResponse, FakeDecisionClient,
JevDecisionClient (mock HTTP) và CircuitBreaker. Hoạt động 100% offline.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.decision.jev_enums import Route, Verdict
from src.decision.response_models import AgentResponse
from src.decision.jev_client import (
    FakeDecisionClient,
    JevDecisionClient,
    JevUnavailableError,
    CircuitBreaker,
)


class TestJevEnumsAndModelsV3(unittest.TestCase):
    def test_enums_values(self):
        self.assertEqual(Route.OUT_OF_SCOPE.value, "OUT_OF_SCOPE")
        self.assertEqual(Route.AMBIGUOUS.value, "AMBIGUOUS")
        self.assertEqual(Route.QUALIFIED_QUESTION.value, "QUALIFIED_QUESTION")
        self.assertEqual(len(list(Route)), 3)

        self.assertEqual(Verdict.SUFFICIENT.value, "SUFFICIENT")
        self.assertEqual(Verdict.INSUFFICIENT.value, "INSUFFICIENT")

    def test_agent_response_creation(self):
        resp = AgentResponse(
            status="partial",
            state=1,
            answer="Đáp án mẫu",
            contexts=[],
            route=Route.QUALIFIED_QUESTION,
            counters={"retry": 4, "retried3": 0},
            trace=[{"node": "router_node", "result": "QUALIFIED_QUESTION"}],
            fail_reason=None,
            missing_info="điểm chuẩn ngành AI",
        )
        self.assertEqual(resp.status, "partial")
        self.assertEqual(resp.state, 1)
        self.assertEqual(resp.route, Route.QUALIFIED_QUESTION)
        self.assertEqual(resp.counters, {"retry": 4, "retried3": 0})
        self.assertEqual(resp.missing_info, "điểm chuẩn ngành AI")


class TestFakeDecisionClientV3(unittest.TestCase):
    def test_fake_client_defaults(self):
        client = FakeDecisionClient()

        self.assertTrue(client.check_health())
        self.assertEqual(client.route("Câu hỏi bất kỳ"), Route.QUALIFIED_QUESTION)
        self.assertEqual(
            client.check_supplement("query", "last_ask", ["supp"], missing_info=["ngành", "khóa"]),
            Verdict.SUFFICIENT,
        )
        self.assertEqual(client.evaluate_answer("query", "answer"), Verdict.SUFFICIENT)
        self.assertEqual(len(client.history), 4)
        self.assertEqual(client.history[2]["missing_info"], ["ngành", "khóa"])

    def test_fake_client_scripted_sequence(self):
        client = FakeDecisionClient(
            routes=[Route.OUT_OF_SCOPE, Route.AMBIGUOUS],
            check_supplements=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
            evaluate_answers=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
        )

        self.assertEqual(client.route("q1"), Route.OUT_OF_SCOPE)
        self.assertEqual(client.route("q2"), Route.AMBIGUOUS)
        self.assertEqual(client.route("q3"), Route.QUALIFIED_QUESTION)

        self.assertEqual(client.check_supplement("q", "ask", ["s1"], missing_info=["ngành"]), Verdict.INSUFFICIENT)
        self.assertEqual(client.check_supplement("q", "ask", ["s2"], missing_info=["ngành"]), Verdict.SUFFICIENT)

        self.assertEqual(client.evaluate_answer("q", "a1"), Verdict.INSUFFICIENT)
        self.assertEqual(client.evaluate_answer("q", "a2"), Verdict.SUFFICIENT)


class TestJevDecisionClientMockHTTP(unittest.TestCase):
    def _make_client(self, answer_threshold: float = 0.5) -> JevDecisionClient:
        return JevDecisionClient(
            endpoint="http://mock-jev.local/v1/systemone",
            api_key="test-key",
            model="test-jev-model",
            answer_eval_threshold=answer_threshold,
            clarify_threshold=0.5,
        )

    @staticmethod
    def _mock_http_ok(payload_dict):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_dict
        return mock_resp

    def test_route_3_labels_and_legacy_mapping(self):
        client = self._make_client()

        for raw_label, expected in [
            ("OUT_OF_SCOPE", Route.OUT_OF_SCOPE),
            ("AMBIGUOUS", Route.AMBIGUOUS),
            ("QUALIFIED_QUESTION", Route.QUALIFIED_QUESTION),
            ("SINGLE_HOP", Route.QUALIFIED_QUESTION),
            ("MULTI_HOP", Route.QUALIFIED_QUESTION),
        ]:
            with patch.object(
                client._session,
                "post",
                return_value=self._mock_http_ok({"answers": {"route": {"choice": raw_label}}, "latency": 0.1}),
            ) as mock_post:
                res = client.route("Điểm chuẩn ngành CNTT và AI?")
                self.assertEqual(res, expected)
                sent_json = mock_post.call_args.kwargs["json"]
                self.assertEqual(sent_json["state"], {"query": "Điểm chuẩn ngành CNTT và AI?"})
                criteria_keys = set(sent_json["questions"]["route"]["criteria"].keys())
                self.assertEqual(criteria_keys, {"OUT_OF_SCOPE", "AMBIGUOUS", "QUALIFIED_QUESTION"})

    def test_check_supplement_sends_missing_info(self):
        client = self._make_client()
        with patch.object(
            client._session,
            "post",
            return_value=self._mock_http_ok({"answers": {"sufficiency": {"noul": 0.82}}, "latency": 0.12}),
        ) as mock_post:
            verdict = client.check_supplement(
                query="Học phí là bao nhiêu?",
                last_ask="Bạn học ngành nào và hệ nào?",
                supplements=["CNTT hệ Chuẩn"],
                missing_info=["ngành", "hệ đào tạo"],
            )
            self.assertEqual(verdict, Verdict.SUFFICIENT)
            sent_state = mock_post.call_args.kwargs["json"]["state"]
            self.assertEqual(sent_state["missing_info"], ["ngành", "hệ đào tạo"])
            self.assertEqual(sent_state["supplements"], ["CNTT hệ Chuẩn"])

    def test_evaluate_answer_only_sends_query_and_answer(self):
        client = self._make_client(answer_threshold=0.5)

        # >= 0.5 -> SUFFICIENT
        with patch.object(
            client._session,
            "post",
            return_value=self._mock_http_ok({"answers": {"answer_complete": {"noul": 0.78}}, "latency": 0.09}),
        ) as mock_post:
            verdict = client.evaluate_answer("Điểm chuẩn CNTT?", "Điểm chuẩn CNTT 2026 là 28.15.")
            self.assertEqual(verdict, Verdict.SUFFICIENT)
            sent_state = mock_post.call_args.kwargs["json"]["state"]
            self.assertEqual(set(sent_state.keys()), {"query", "answer"})

        # < 0.5 -> INSUFFICIENT
        with patch.object(
            client._session,
            "post",
            return_value=self._mock_http_ok({"answers": {"answer_complete": {"noul": 0.21}}, "latency": 0.08}),
        ):
            verdict = client.evaluate_answer("So sánh CNTT và AI?", "CNTT là 28.15, chưa có thông tin về AI.")
            self.assertEqual(verdict, Verdict.INSUFFICIENT)

        # Empty answer -> INSUFFICIENT without HTTP call
        with patch.object(client._session, "post") as mock_post:
            self.assertEqual(client.evaluate_answer("Câu hỏi", "   "), Verdict.INSUFFICIENT)
            mock_post.assert_not_called()


class TestCircuitBreakerAndNoSilentFallback(unittest.TestCase):
    def test_circuit_breaker_logic(self):
        breaker = CircuitBreaker(failure_threshold=2, open_duration_s=0.2)
        self.assertEqual(breaker.can_execute(), (True, "CLOSED"))

        breaker.record_failure()
        self.assertEqual(breaker.can_execute(), (True, "CLOSED"))

        breaker.record_failure()
        self.assertEqual(breaker.can_execute(), (False, "OPEN"))

        breaker2 = CircuitBreaker(failure_threshold=5, open_duration_s=10.0)
        breaker2.record_failure(is_server_down=True)
        self.assertEqual(breaker2.can_execute(), (False, "OPEN"))
        self.assertTrue(breaker2.is_server_down)

    def test_no_silent_fallback_raises_error(self):
        client = JevDecisionClient(
            endpoint="https://openrouter.ai/api/alpha/decisions",
            api_key="",
        )
        with self.assertRaises(Exception):
            client.route("Test query")

        with self.assertRaises(Exception):
            client.check_supplement("q", "ask", ["supp"], missing_info=["ngành"])

        with self.assertRaises(Exception):
            client.evaluate_answer("q", "ans")

    def test_open_circuit_breaker_raises_jev_unavailable(self):
        client = JevDecisionClient(
            endpoint="http://mock-jev.local/v1/systemone",
            api_key="k",
        )
        client.breaker.record_failure(is_server_down=True)
        with self.assertRaises(JevUnavailableError):
            client.evaluate_answer("q", "ans")

    def test_persistent_http_session_per_conversation(self):
        """Kiểm tra JevDecisionClient duy trì cùng 1 requests.Session xuyên suốt session_id của phiên trò chuyện."""
        client = JevDecisionClient(
            endpoint="https://semester-agency-pirates-doll.trycloudflare.com/v1/systemone",
            api_key="",
        )
        sess_a1 = client.get_http_session("conv_01")
        sess_a2 = client.get_http_session("conv_01")
        sess_b = client.get_http_session("conv_02")

        self.assertIs(sess_a1, sess_a2)
        self.assertIsNot(sess_a1, sess_b)
        self.assertEqual(sess_a1.headers.get("Connection"), "keep-alive")

        mock_route_resp = MagicMock(status_code=200)
        mock_route_resp.json.return_value = {"answers": {"route": {"choice": "QUALIFIED_QUESTION"}}, "latency": 0.1}
        mock_eval_resp = MagicMock(status_code=200)
        mock_eval_resp.json.return_value = {"answers": {"answer_complete": {"noul": 0.9}}, "latency": 0.1}

        with patch.object(sess_a1, "post", side_effect=[mock_route_resp, mock_eval_resp]) as mock_post:
            r = client.route("Điểm chuẩn CNTT?", session_id="conv_01")
            v = client.evaluate_answer("Điểm chuẩn CNTT?", "28.15 điểm", session_id="conv_01")
            self.assertEqual(r, Route.QUALIFIED_QUESTION)
            self.assertEqual(v, Verdict.SUFFICIENT)
            # Cả 2 quyết định JEV trong cùng phiên chat đều đi qua cùng 1 requests.Session
            self.assertEqual(mock_post.call_count, 2)

        client.close_session("conv_01")
        self.assertIsNot(client.get_http_session("conv_01"), sess_a1)
        client.close()


if __name__ == "__main__":
    unittest.main()

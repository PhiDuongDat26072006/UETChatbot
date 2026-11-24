"""
tests/test_counters.py - Kiểm thử đơn vị 2 bộ đếm độc lập v3 theo quy tắc "tăng rồi so sánh":
  - retry (answer_eval_node): <= 3 -> followup_query_node, > 3 -> END
  - retried3 (clarify_check_node): <= 3 -> clarify_ask_node, > 3 -> clarify_exhausted_node
"""

import unittest
from src.agent.state import AgentState
from src.decision.jev_enums import Verdict
from src.agent.counter_helpers import (
    route_after_answer_eval,
    route_after_clarify_check,
)


class TestCounterSemanticsV3(unittest.TestCase):
    def test_retry_answer_eval_sequence(self):
        """
        answer_eval THIẾU:
        Lần 1: tăng retry=1 <= 3 -> followup_query_node
        Lần 2: tăng retry=2 <= 3 -> followup_query_node
        Lần 3: tăng retry=3 <= 3 -> followup_query_node
        Lần 4: tăng retry=4 > 3 -> END (RESPONSE partial)
        """
        state: AgentState = {"answer_verdict": Verdict.INSUFFICIENT, "retry": 0}

        for expected_retry in (1, 2, 3):
            state["retry"] += 1
            self.assertEqual(state["retry"], expected_retry)
            self.assertEqual(route_after_answer_eval(state, max_retries=3), "followup_query_node")

        # Lần 4 -> quá ngưỡng -> END
        state["retry"] += 1
        self.assertEqual(state["retry"], 4)
        self.assertEqual(route_after_answer_eval(state, max_retries=3), "END")

    def test_retry_answer_eval_sufficient(self):
        """Khi ĐỦ thì chuyển thẳng END bất kể giá trị retry."""
        state: AgentState = {"answer_verdict": Verdict.SUFFICIENT, "retry": 2}
        self.assertEqual(route_after_answer_eval(state, max_retries=3), "END")

    def test_retry_custom_max_retries(self):
        """Kiểm tra ngưỡng max_retries đọc linh hoạt không hard-code."""
        state: AgentState = {"answer_verdict": Verdict.INSUFFICIENT, "retry": 2}
        self.assertEqual(route_after_answer_eval(state, max_retries=2), "followup_query_node")
        state["retry"] = 3
        self.assertEqual(route_after_answer_eval(state, max_retries=2), "END")

    def test_retried3_clarify_sequence(self):
        """
        Clarify check THIẾU:
        Lần 1..3: tăng retried3 <= 3 -> clarify_ask_node
        Lần 4: tăng retried3=4 > 3 -> clarify_exhausted_node
        """
        state: AgentState = {"clarify_verdict": Verdict.INSUFFICIENT, "retried3": 0}

        for expected_r3 in (1, 2, 3):
            state["retried3"] += 1
            self.assertEqual(state["retried3"], expected_r3)
            self.assertEqual(route_after_clarify_check(state, max_retried3=3), "clarify_ask_node")

        state["retried3"] += 1
        self.assertEqual(state["retried3"], 4)
        self.assertEqual(route_after_clarify_check(state, max_retried3=3), "clarify_exhausted_node")

    def test_retried3_clarify_sufficient(self):
        """Khi ĐỦ thì chuyển sang retrieval_node."""
        state: AgentState = {"clarify_verdict": Verdict.SUFFICIENT, "retried3": 2}
        self.assertEqual(route_after_clarify_check(state, max_retried3=3), "retrieval_node")


if __name__ == "__main__":
    unittest.main()

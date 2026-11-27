"""
tests/test_counters.py - Kiểm thử đơn vị bộ đếm vòng đánh giá v3 theo quy tắc "tăng rồi so sánh":
  - retry (answer_eval_node): <= max_retries -> followup_query_node, > max_retries -> END (RESPONSE partial)
"""

import unittest
from src.agent.state import AgentState
from src.decision.jev_enums import Verdict
from src.agent.counter_helpers import route_after_answer_eval


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


if __name__ == "__main__":
    unittest.main()

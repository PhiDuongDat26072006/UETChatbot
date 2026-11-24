"""
src/agent/counter_helpers.py - Các hàm hỗ trợ quản lý bộ đếm và rẽ nhánh điều kiện (v3).
Thực hiện nghiêm ngặt quy tắc "tăng rồi so sánh" cho 2 bộ đếm độc lập:
  - retried3 (clarify_check_node): <= 3 -> clarify_ask_node, > 3 -> clarify_exhausted_node
  - retry (answer_eval_node): <= 3 -> followup_query_node, > 3 -> END (RESPONSE)
"""

from __future__ import annotations
from typing import Literal
from src.agent.state import AgentState
from src.decision.jev_enums import Verdict


def route_after_clarify_check(
    state: AgentState,
    max_retried3: int = 3,
) -> Literal["retrieval_node", "clarify_ask_node", "clarify_exhausted_node"]:
    """
    Rẽ nhánh sau clarify_check_node:
    - Nếu bổ sung = SUFFICIENT (ĐỦ): -> retrieval_node
    - Nếu bổ sung = INSUFFICIENT (THIẾU):
      retried3 đã được tăng trước trong clarify_check_node
      retried3 <= max_retried3 (3) -> clarify_ask_node
      retried3 > max_retried3 (3) -> clarify_exhausted_node
    """
    verdict = state.get("clarify_verdict", Verdict.INSUFFICIENT)
    if verdict == Verdict.SUFFICIENT:
        return "retrieval_node"

    retried3 = state.get("retried3", 0)
    if retried3 <= max_retried3:
        return "clarify_ask_node"
    return "clarify_exhausted_node"


def route_after_answer_eval(
    state: AgentState,
    max_retries: int = 3,
) -> Literal["END", "followup_query_node"]:
    """
    Rẽ nhánh sau answer_eval_node:
    - Nếu đánh giá = SUFFICIENT (ĐỦ): -> END (RESPONSE)
    - Nếu đánh giá = INSUFFICIENT (THIẾU):
      retry đã được tăng trước trong answer_eval_node
      retry <= max_retries (3) -> followup_query_node
      retry > max_retries (3) -> END (RESPONSE partial)
    """
    verdict = state.get("answer_verdict", Verdict.INSUFFICIENT)
    if verdict == Verdict.SUFFICIENT:
        return "END"

    retry = state.get("retry", 0)
    if retry <= max_retries:
        return "followup_query_node"
    return "END"

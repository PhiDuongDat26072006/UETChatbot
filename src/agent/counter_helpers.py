"""
src/agent/counter_helpers.py - Hàm hỗ trợ quản lý bộ đếm và rẽ nhánh điều kiện (v3).
Thực hiện quy tắc "tăng rồi so sánh" cho bộ đếm vòng đánh giá:
  - retry (answer_eval_node): <= max_retries -> followup_query_node, > max_retries -> END (RESPONSE partial)
"""

from __future__ import annotations
from typing import Literal
from src.agent.state import AgentState
from src.decision.jev_enums import Verdict


def route_after_answer_eval(
    state: AgentState,
    max_retries: int = 3,
) -> Literal["END", "followup_query_node"]:
    """
    Rẽ nhánh sau answer_eval_node:
    - Nếu đánh giá = SUFFICIENT (ĐỦ): -> END (RESPONSE)
    - Nếu đánh giá = INSUFFICIENT (THIẾU):
      retry đã được tăng trước trong answer_eval_node
      retry <= max_retries -> followup_query_node
      retry > max_retries -> END (RESPONSE partial)
    """
    verdict = state.get("answer_verdict", Verdict.INSUFFICIENT)
    if verdict == Verdict.SUFFICIENT:
        return "END"

    retry = state.get("retry", 0)
    if retry <= max_retries:
        return "followup_query_node"
    return "END"

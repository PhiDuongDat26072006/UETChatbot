"""
src/agent/state.py - Định nghĩa cấu trúc trạng thái chung (AgentState) cho đồ thị LangGraph v3.
Luồng "Generate -> Evaluate -> Follow-up" (6 node chính + fail_node ẩn).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from src.base import RetrievedContext, UserQuery
from src.decision.jev_enums import Route, Verdict
from src.decision.response_models import AgentResponse


class AgentState(TypedDict, total=False):
    """
    Trạng thái tập trung (Central State) được truyền qua đồ thị LangGraph JEV v3.
    """
    # 1. Định danh & Câu hỏi
    session_id: Optional[str]
    query: str
    user_query: Optional[UserQuery]
    top_k: Optional[int]
    extra_format_rules: Optional[str]
    voice_mode: bool

    # 2. Định tuyến (JEV Router 3 nhãn: OUT_OF_SCOPE | AMBIGUOUS | QUALIFIED_QUESTION)
    # Cả AMBIGUOUS và QUALIFIED_QUESTION đều đi thẳng vào retrieval_node; AMBIGUOUS tự động gắn gợi ý cuối phản hồi.
    route: Optional[Route]

    # 3. Truy vấn bổ sung & Lịch sử truy vấn
    followup_queries: List[str]     # Truy vấn cho phần còn thiếu do followup_query_node sinh
    used_queries: List[str]         # Danh sách tất cả truy vấn đã gửi tới retrieval, theo thứ tự

    # 4. Ngữ cảnh tích lũy (Retrieval)
    retrieved_chunks: List[RetrievedContext]        # Chunk của vòng hiện tại
    context_pool: Dict[str, RetrievedContext]       # Kho chunk tích lũy theo chunk_id (khử trùng lặp)
    contexts: List[RetrievedContext]                # Danh sách chunk chọn lọc (<= max_chunks) đưa cho LLM

    # 5. Đánh giá (JEV)
    answer_verdict: Optional[Verdict]               # Kết quả answer_eval_node: SUFFICIENT | INSUFFICIENT
    missing_info: Optional[str]                     # Phần thông tin còn thiếu do followup_query_node xác định

    # 6. Bộ đếm vòng lặp đánh giá (Quy tắc "tăng rồi so sánh")
    retry: int     # Bộ đếm answer_eval THIẾU (<= max_retries -> followup_query_node, > max_retries -> RESPONSE partial)

    # 7. Sinh phản hồi & Kết quả cuối
    answer: Optional[str]
    best_answer: Optional[str]                      # Câu trả lời tốt nhất qua các vòng retry (theo noul_prob cao nhất)
    best_answer_score: Optional[float]              # Điểm noul_prob cao nhất tương ứng với best_answer
    best_contexts: List[RetrievedContext]           # Ngữ cảnh tương ứng với best_answer
    response: Optional[AgentResponse]
    fail_reason: Optional[str]

    # 8. Truy vết & Debug
    trace: List[Dict[str, Any]]
    execution_steps: List[str]
    sources: List[RetrievedContext]

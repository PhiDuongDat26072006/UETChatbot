"""
src/agent/state.py - Định nghĩa cấu trúc trạng thái chung (AgentState) cho đồ thị LangGraph v3.
Luồng "Generate -> Evaluate -> Follow-up" (9 node chính + fail_node ẩn).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from src.base import RetrievedContext, UserQuery
from src.decision.jev_enums import Route, Verdict
from src.decision.response_models import AgentResponse


class ClarifyData(TypedDict, total=False):
    """Cấu trúc dữ liệu theo dõi nhánh AMBIGUOUS làm rõ thông tin."""
    asks: List[str]            # Lịch sử các câu hỏi yêu cầu bổ sung
    supplements: List[str]     # Lịch sử các câu trả lời bổ sung từ sinh viên
    known_info: List[str]      # Thông tin cụ thể đã có (từ clarify_ask_node)
    missing_info: List[str]    # Thông tin cụ thể còn thiếu (truyền cho clarify_check_node)
    pending: bool              # True nếu đang chờ người dùng trả lời


class AgentState(TypedDict, total=False):
    """
    Trạng thái tập trung (Central State) được truyền qua đồ thị LangGraph JEV v3.
    """
    # 1. Định danh & Câu hỏi
    session_id: Optional[str]
    query: str
    effective_query: Optional[str]  # Câu hỏi sau khi ghép bổ sung: "{query}\n(Bổ sung: ...)"
    user_query: Optional[UserQuery]

    # 2. Định tuyến (JEV Router 3 nhãn)
    route: Optional[Route]          # OUT_OF_SCOPE | AMBIGUOUS | QUALIFIED_QUESTION

    # 3. Truy vấn bổ sung & Lịch sử truy vấn
    followup_queries: List[str]     # Truy vấn cho phần còn thiếu do followup_query_node sinh
    used_queries: List[str]         # Danh sách tất cả truy vấn đã gửi tới retrieval, theo thứ tự

    # 4. Ngữ cảnh tích lũy (Retrieval)
    retrieved_chunks: List[RetrievedContext]        # Chunk của vòng hiện tại
    context_pool: Dict[str, RetrievedContext]       # Kho chunk tích lũy theo chunk_id (khử trùng lặp)
    contexts: List[RetrievedContext]                # Danh sách chunk chọn lọc (<= max_chunks) đưa cho LLM

    # 5. Đánh giá (JEV)
    clarify_verdict: Optional[Verdict]              # Kết quả clarify_check_node: SUFFICIENT | INSUFFICIENT
    answer_verdict: Optional[Verdict]               # Kết quả answer_eval_node: SUFFICIENT | INSUFFICIENT
    missing_info: Optional[str]                     # Phần thông tin còn thiếu do followup_query_node xác định

    # 6. Hai bộ đếm độc lập (Quy tắc "tăng rồi so sánh")
    retry: int     # Bộ đếm answer_eval THIẾU (<= 3 -> followup_query_node, > 3 -> RESPONSE partial)
    retried3: int  # Bộ đếm clarify_check THIẾU (<= 3 -> clarify_ask_node, > 3 -> clarify_exhausted_node)

    # 7. Nhánh AMBIGUOUS
    clarify: ClarifyData

    # 8. Sinh phản hồi & Kết quả cuối
    answer: Optional[str]
    best_answer: Optional[str]                      # Câu trả lời tốt nhất qua các vòng retry (theo noul_prob cao nhất)
    best_answer_score: Optional[float]              # Điểm noul_prob cao nhất tương ứng với best_answer
    best_contexts: List[RetrievedContext]           # Ngữ cảnh tương ứng với best_answer
    response: Optional[AgentResponse]
    fail_reason: Optional[str]

    # 9. Truy vết & Debug
    trace: List[Dict[str, Any]]
    execution_steps: List[str]
    sources: List[RetrievedContext]

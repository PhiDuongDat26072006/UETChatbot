"""
src/decision/response_models.py - Định nghĩa cấu trúc phản hồi chuẩn hóa AgentResponse (v3).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from src.base import RetrievedContext
from src.decision.jev_enums import Route


@dataclass
class AgentResponse:
    """
    Kết quả trả về thống nhất từ Agentic RAG Graph (v3).
    - status: 'ok' | 'partial' | 'fail' | 'out_of_scope' | 'clarify' | 'clarify_exhausted'
    - state: 1 khi câu trả lời được sinh từ contexts ('ok' hoặc 'partial' khi retry > 3);
             0 khi là các template phản hồi ('fail', 'out_of_scope', 'clarify', 'clarify_exhausted')
    - answer: Nội dung văn bản câu trả lời
    - contexts: Danh sách chunk liên quan (có khi state == 1)
    - route: Nhãn router phân loại ban đầu
    - counters: Dict lưu trạng thái 2 bộ đếm {retry, retried3}
    - trace: Danh sách các bước thực thi và log quyết định
    - fail_reason: Lý do lỗi dịch vụ nếu status == 'fail'
    - missing_info: Phần thông tin còn thiếu (nếu có, phục vụ trace / partial response)
    """
    status: Literal["ok", "partial", "fail", "out_of_scope", "clarify", "clarify_exhausted"]
    state: Literal[0, 1]
    answer: str
    contexts: Optional[List[RetrievedContext]] = None
    route: Optional[Route] = None
    counters: Dict[str, int] = field(default_factory=lambda: {"retry": 0, "retried3": 0})
    trace: List[Dict[str, Any]] = field(default_factory=list)
    fail_reason: Optional[str] = None
    missing_info: Optional[str] = None

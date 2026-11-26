"""
src/decision/jev_enums.py - Các Enum định nghĩa nhãn và phán quyết cho hệ thống JEV Decision Model (v3).
Luồng "Generate -> Evaluate -> Follow-up" với 2 nhãn router và phán quyết ĐỦ / THIẾU.
"""

from __future__ import annotations
from enum import Enum


class Route(str, Enum):
    """Nhãn định tuyến 2 nhánh từ router_node (JEV)."""
    OUT_OF_SCOPE = "OUT_OF_SCOPE"              # Chào hỏi, cảm ơn, hỏi danh tính bot, hoặc ngoài phạm vi UET
    QUALIFIED_QUESTION = "QUALIFIED_QUESTION"  # Câu hỏi về UET (kể cả câu ngắn, so sánh, nhiều ý) -> vào RAG trực tiếp


class Verdict(str, Enum):
    """Phán quyết mức độ đầy đủ (dùng cho answer_eval_node)."""
    SUFFICIENT = "SUFFICIENT"      # Câu trả lời đã giải đáp đầy đủ
    INSUFFICIENT = "INSUFFICIENT"  # Câu trả lời chưa đủ hoặc còn bỏ sót ý

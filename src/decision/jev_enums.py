"""
src/decision/jev_enums.py - Các Enum định nghĩa nhãn và phán quyết cho hệ thống JEV Decision Model (v3).
Luồng "Generate -> Evaluate -> Follow-up" với 3 nhãn router và phán quyết ĐỦ / THIẾU.
"""

from __future__ import annotations
from enum import Enum


class Route(str, Enum):
    """Nhãn định tuyến 3 nhánh từ router_node (JEV)."""
    OUT_OF_SCOPE = "OUT_OF_SCOPE"              # Chào hỏi, cảm ơn, hỏi danh tính bot, hoặc ngoài phạm vi UET
    AMBIGUOUS = "AMBIGUOUS"                    # Liên quan đào tạo UET nhưng thiếu thông tin thiết yếu (ngành, khóa, hệ)
    QUALIFIED_QUESTION = "QUALIFIED_QUESTION"  # Đủ thông tin để tra cứu, kể cả câu so sánh / nhiều ý


class Verdict(str, Enum):
    """Phán quyết mức độ đầy đủ (dùng cho clarify_check_node và answer_eval_node)."""
    SUFFICIENT = "SUFFICIENT"      # ĐỦ thông tin / câu trả lời đã giải đáp đầy đủ
    INSUFFICIENT = "INSUFFICIENT"  # THIẾU thông tin / câu trả lời chưa đủ hoặc còn bỏ sót ý

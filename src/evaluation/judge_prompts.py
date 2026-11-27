"""
src/evaluation/judge_prompts.py - Pydantic schemas, Likert normalization,
JSON extraction, and prompt templates for LLM-as-a-Judge evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

RUBRIC_VERSION = "v1.0"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

DEFAULT_MAX_CHARS_PER_CONTEXT = 4000
DEFAULT_MAX_TOTAL_CHARS = 24000


class FaithfulnessJudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    unsupported_claims: List[str] = Field(default_factory=list)
    reason: str = ""


class CorrectnessJudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    missing_facts: List[str] = Field(default_factory=list)
    wrong_facts: List[str] = Field(default_factory=list)
    reason: str = ""


class SimpleJudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    reason: str = ""


class InDomainJudgeOutput(BaseModel):
    faithfulness: FaithfulnessJudgeScore
    answer_correctness: CorrectnessJudgeScore
    answer_relevance: SimpleJudgeScore
    context_sufficiency: SimpleJudgeScore


class OODJudgeOutput(BaseModel):
    refused_properly: bool
    hallucinated_info: bool
    score: int = Field(..., ge=1, le=5)
    reason: str = ""


def normalize_likert_5(score: int) -> float:
    """Quy đổi điểm Likert 1..5 về thang [0.0, 1.0]: 1->0.0, 2->0.25, 3->0.5, 4->0.75, 5->1.0."""
    clamped = max(1, min(5, int(score)))
    return round((clamped - 1) / 4.0, 4)


def _strip_reasoning_and_extract_json(raw_text: str) -> Dict[str, Any]:
    """Làm sạch khối <think>, code fence Markdown và phân tích JSON đầu ra."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("LLM trả về chuỗi rỗng.")

    text = _THINK_BLOCK_RE.sub("", text).strip()
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Không trích xuất được JSON hợp lệ từ phản hồi: {text[:180]}")


def _truncate_contexts(
    contexts: List[str],
    max_chars_per_context: int = DEFAULT_MAX_CHARS_PER_CONTEXT,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Định dạng các ngữ cảnh truy xuất theo ngưỡng độ dài an toàn."""
    if not contexts:
        return "(Không có ngữ cảnh nào được truy xuất)"

    blocks: List[str] = []
    total = 0
    for idx, ctx in enumerate(contexts, 1):
        text = str(ctx or "").strip()
        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + " ...[cắt bớt]"
        block = f"[Tài liệu {idx}]\n{text}"
        if total + len(block) > max_total_chars and blocks:
            blocks.append(f"...(Đã lược bớt {len(contexts) - idx + 1} tài liệu còn lại do giới hạn độ dài)")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _build_indomain_prompt(
    query: str,
    contexts: List[str],
    answer: str,
    ground_truth: Optional[str],
) -> str:
    """Tạo prompt chấm điểm tiếng Việt theo Rubric 1-5 cho câu hỏi in-domain."""
    formatted_contexts = _truncate_contexts(contexts)
    gt_block = ground_truth.strip() if ground_truth else "(Không có Ground Truth tham chiếu)"

    return (
        "Bạn là Giám khảo Đánh giá Chất lượng Hệ thống RAG Học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET).\n"
        "Hãy đánh giá khách quan CÂU TRẢ LỜI CỦA BOT dựa trên CÂU HỎI, NGỮ CẢNH TRUY XUẤT và ĐÁP ÁN CHUẨN (Ground Truth).\n\n"
        "QUY TẮC CHẤM ĐIỂM BẮT BUỘC:\n"
        "- KHÔNG trừ điểm chỉ vì bot trình bày bằng bảng Markdown (`| ... |`), lời chào lịch sự, khối lưu ý hoặc dòng `📌 Căn cứ quy chế:`.\n"
        "- Chỉ xét nội dung sự thật: số liệu (GPA, tín chỉ, điểm chuẩn, năm 2025 vs 2026), mã học phần/mã ngành, số quyết định, tên đơn vị/đối tác, điều kiện phủ định (`không`, `chỉ`, `tối thiểu`, `tối đa`).\n"
        "- Nếu câu trả lời có chi tiết bổ sung hợp lý có thật trong NGỮ CẢNH (ngoài phần tóm tắt của Ground Truth) thì KHÔNG trừ điểm `answer_correctness`, miễn không sai lệch hay mâu thuẫn với Ground Truth.\n"
        "- Nếu bot bịa số liệu, sai năm học, sai đơn vị hoặc kết luận trái ngược ngữ cảnh -> chấm điểm thấp và liệt kê rõ vào `unsupported_claims` / `wrong_facts`.\n\n"
        "THANG ĐIỂM 1 ĐẾN 5 CHO TỪNG TIÊU CHÍ:\n"
        "1. `faithfulness` (Tính trung thực với Ngữ cảnh):\n"
        "   - 5: Mọi khẳng định/số liệu đều có căn cứ trực tiếp trong Ngữ cảnh.\n"
        "   - 4: Phần lớn bám Ngữ cảnh, có diễn đạt mở rộng rất nhỏ không sai lệch.\n"
        "   - 3: Đúng ý chính nhưng có chi tiết thiếu căn cứ rõ trong Ngữ cảnh.\n"
        "   - 2: Có thông tin/số liệu suy diễn hoặc sai lệch so với Ngữ cảnh.\n"
        "   - 1: Bịa đặt nghiêm trọng hoặc mâu thuẫn hoàn toàn Ngữ cảnh.\n"
        "2. `answer_correctness` (Độ chính xác & đầy đủ so với Ground Truth):\n"
        "   - 5: Bao phủ đủ mọi ý/số liệu cốt lõi trong Ground Truth, không sai.\n"
        "   - 4: Đúng hầu hết ý chính, chỉ thiếu chi tiết phụ.\n"
        "   - 3: Đúng một phần (~50%), thiếu ý quan trọng hoặc lẫn tiểu tiết chưa chuẩn.\n"
        "   - 2: Thiếu phần lớn ý chính hoặc sai số liệu/điều kiện quan trọng.\n"
        "   - 1: Sai hoàn toàn hoặc từ chối trả lời sai (false refusal) dù có Ground Truth.\n"
        "3. `answer_relevance` (Độ bám sát Câu hỏi):\n"
        "   - 5: Trả lời trực diện, đúng trọng tâm câu hỏi, dễ theo dõi.\n"
        "   - 3: Có chạm ý câu hỏi nhưng lan man hoặc thiếu trọng tâm.\n"
        "   - 1: Lạc đề hoặc không giải quyết câu hỏi.\n"
        "4. `context_sufficiency` (Ngữ cảnh truy xuất có đủ để trả lời Ground Truth không?):\n"
        "   - 5: Ngữ cảnh chứa đầy đủ 100% dữ kiện cần thiết.\n"
        "   - 3: Ngữ cảnh có một phần thông tin, thiếu một số dữ kiện phụ.\n"
        "   - 1: Ngữ cảnh hoàn toàn thiếu thông tin cần thiết để trả lời.\n\n"
        f"=== CÂU HỎI ===\n{query}\n\n"
        f"=== NGỮ CẢNH TRUY XUẤT ===\n{formatted_contexts}\n\n"
        f"=== ĐÁP ÁN CHUẨN (GROUND TRUTH) ===\n{gt_block}\n\n"
        f"=== CÂU TRẢ LỜI CỦA BOT ===\n{answer}\n\n"
        "CHỈ TRẢ VỀ DUY NHẤT 1 OBJECT JSON HỢP LỆ THEO ĐÚNG CẤU TRÚC SAU (không markdown thừa, không giải thích ngoài JSON):\n"
        '{\n'
        '  "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "..."},\n'
        '  "answer_correctness": {"score": 5, "missing_facts": [], "wrong_facts": [], "reason": "..."},\n'
        '  "answer_relevance": {"score": 5, "reason": "..."},\n'
        '  "context_sufficiency": {"score": 5, "reason": "..."}\n'
        '}'
    )


def _build_ood_prompt(
    query: str,
    answer: str,
    ground_truth: Optional[str],
) -> str:
    """Tạo prompt đánh giá câu hỏi ngoài phạm vi UET (out_of_domain)."""
    gt_block = ground_truth.strip() if ground_truth else "Câu hỏi nằm ngoài phạm vi dữ liệu lưu trữ của UET."
    return (
        "Bạn là Giám khảo Đánh giá An toàn Ngoài Phạm Vi (Out-Of-Domain) cho Chatbot Trường ĐH Công nghệ - ĐHQGHN (UET).\n"
        "Câu hỏi dưới đây thuộc nhóm NGOÀI PHẠM VI UET (hệ thống không lưu trữ dữ liệu về đối tượng được hỏi).\n"
        "Hãy kiểm tra xem bot có từ chối lịch sự đúng nguyên tắc hay cố tình bịa đặt/trả lời nhầm.\n\n"
        "Lưu ý: Dòng `📌 Căn cứ quy chế:` liệt kê tên tài liệu tham khảo là định dạng mặc định cuối câu, KHÔNG tính là bịa nếu nội dung chính đã từ chối rõ ràng và không cung cấp thông tin sai về trường/đối tượng ngoài UET.\n\n"
        "THANG ĐIỂM `score` (1-5):\n"
        "- 5: Từ chối rõ ràng, lịch sự và không bịa bất kỳ thông tin nào về đối tượng ngoài UET.\n"
        "- 3: Có từ chối nhưng câu chữ mơ hồ hoặc đưa thêm chi tiết gây hiểu nhầm.\n"
        "- 1: Không từ chối mà tự bịa câu trả lời hoặc trả lời nhầm bằng thông tin của UET.\n\n"
        f"=== CÂU HỎI (OOD) ===\n{query}\n\n"
        f"=== GHI CHÚ CHUẨN ===\n{gt_block}\n\n"
        f"=== CÂU TRẢ LỜI CỦA BOT ===\n{answer}\n\n"
        "CHỈ TRẢ VỀ DUY NHẤT 1 OBJECT JSON HỢP LỆ:\n"
        '{\n'
        '  "refused_properly": true,\n'
        '  "hallucinated_info": false,\n'
        '  "score": 5,\n'
        '  "reason": "..."\n'
        '}'
    )

"""Helper utilities for context accumulation, prompt construction, and query cleaning in Agentic RAG."""

from __future__ import annotations

import re
from typing import Any, Dict, Generator, List, Optional, Set

from src.agent.rewriter import UETQueryRewriter
from src.base import AugmentedPrompt, RetrievedContext, UserQuery
from src.prompts.prompts import (
    LLM_GENERATION_FALLBACK_SYSTEM,
    LLM_GENERATION_FALLBACK_TEMPLATE,
)

ANCHOR_COUNT_PATTERNS = [
    r"(?:gồm|có|đào tạo|tổng số|tổng cộng|hiện có|tất cả|quy mô)\s*(\d{1,2})\s*(?:ngành|chương trình đào tạo|chuyên ngành|khoa|viện|bộ môn|mức|tiêu chuẩn)",
    r"(\d{1,2})\s*(?:ngành|chương trình đào tạo|chuyên ngành)\s*(?:đào tạo|đại học|chính quy|chuẩn)?",
    r"(?:gồm có|bao gồm)\s*(\d{1,2})\s*(?:khoa|viện)",
]

NODE_STATUS_MESSAGES: Dict[str, str] = {
    "router_node": "Đang phân loại câu hỏi...",
    "retrieval_node": "Đang tra cứu quy chế và tài liệu liên quan...",
    "generation_node": "Đang tổng hợp câu trả lời...",
    "answer_eval_node": "Đang kiểm tra độ đầy đủ câu trả lời...",
    "followup_query_node": "Đang tìm kiếm bổ sung thông tin còn thiếu...",
}

DEFAULT_TEMPLATES: Dict[str, str] = {
    "out_of_scope": (
        "Chào bạn! Tôi là UET Chatbot - Trợ lý ảo tư vấn quy chế đào tạo và tuyển sinh của Trường Đại học Công nghệ (VNU-UET). "
        "Nội dung bạn hỏi hiện nằm ngoài phạm vi hỗ trợ của tôi. Rất vui được phục vụ bạn nếu bạn cần tìm hiểu thông tin học vụ, quy chế, điểm chuẩn hay học phí của UET nhé!"
    ),
    "fail_service": (
        "Xin lỗi bạn, hệ thống kiểm tra và tra cứu thông tin đang gián đoạn dịch vụ, "
        "vui lòng thử lại sau ít phút hoặc liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) để được hỗ trợ nhé."
    ),
    "answer_incomplete_note": (
        "\n\n> **Lưu ý:** Câu trả lời trên có thể chưa đầy đủ hoàn toàn mọi ý trong câu hỏi của bạn{missing_suffix}. "
        "Bạn có thể liên hệ Phòng Đào tạo (Phòng 105-E3, SĐT: 024.3754.7865) hoặc Cổng thông tin đào tạo UET để xác nhận thêm thông tin chi tiết nhé."
    ),
}


def resolve_agent_templates(raw_config: Dict[str, Any]) -> Dict[str, str]:
    """Load agent response templates from config with safe defaults."""
    templates_cfg = raw_config.get("templates", {})
    return {
        key: templates_cfg.get(key, default_val)
        for key, default_val in DEFAULT_TEMPLATES.items()
    }


def extract_anchor_count(text: str, entity: str = "ngành") -> Optional[int]:
    """Trích xuất con số tổng cộng được nêu trực tiếp trong tài liệu."""
    for pattern in ANCHOR_COUNT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                num = int(match.group(1))
                if 1 <= num <= 99:
                    return num
            except (ValueError, IndexError):
                continue
    return None


def clean_query_for_retrieval(q_text: str) -> str:
    """Gỡ khung '(Bổ sung: ...)' (nếu có) và mở rộng từ viết tắt/đồng nghĩa học vụ UET."""
    if not q_text:
        return q_text
    cleaned = q_text
    if "bổ sung:" in cleaned.lower():
        cleaned = re.sub(r"\s*\(\s*Bổ sung\s*:\s*([^)]*)\)", r" \1", cleaned, flags=re.IGNORECASE)
    return UETQueryRewriter.expand_for_retrieval(cleaned)


def chunk_answer_for_stream(text: str, pieces_per_chunk: int = 4) -> Generator[str, None, None]:
    """Tách câu trả lời cuối thành từng cụm nhỏ cho SSE mà vẫn bảo toàn nguyên vẹn xuống dòng & bảng Markdown."""
    if not text:
        return
    pieces = re.findall(r"\S+\s*|\s+", text)
    if not pieces:
        yield text
        return
    buf: List[str] = []
    for piece in pieces:
        buf.append(piece)
        if len(buf) >= pieces_per_chunk:
            yield "".join(buf)
            buf = []
    if buf:
        yield "".join(buf)


def merge_round_robin_results(per_query_results: List[List[RetrievedContext]]) -> List[RetrievedContext]:
    """Interleave retrieved contexts across sub-queries to prevent query starvation."""
    retrieved: List[RetrievedContext] = []
    seen_ids: Set[str] = set()
    max_len = max((len(lst) for lst in per_query_results), default=0)
    for idx in range(max_len):
        for lst in per_query_results:
            if idx < len(lst):
                ctx = lst[idx]
                if ctx.chunk_id not in seen_ids:
                    seen_ids.add(ctx.chunk_id)
                    retrieved.append(ctx)
    return retrieved


def select_accumulated_contexts(
    context_pool: Dict[str, RetrievedContext],
    new_chunk_ids: List[str],
    max_contexts: int,
) -> List[RetrievedContext]:
    """
    Chọn tối đa ``max_contexts`` chunk đưa cho LLM, ưu tiên chunk vòng mới nhất và
    giữ các chunk điểm cao nhất của các vòng trước.
    """
    if not context_pool:
        return []

    if len(context_pool) <= max_contexts:
        return sorted(
            context_pool.values(),
            key=lambda c: getattr(c, "similarity_score", 0.0),
            reverse=True,
        )

    new_id_set: Set[str] = set(new_chunk_ids)
    new_chunks: List[RetrievedContext] = []
    seen_new: Set[str] = set()
    for cid in new_chunk_ids:
        if cid in context_pool and cid not in seen_new:
            seen_new.add(cid)
            new_chunks.append(context_pool[cid])

    older_chunks = sorted(
        [c for cid, c in context_pool.items() if cid not in new_id_set],
        key=lambda c: getattr(c, "similarity_score", 0.0),
        reverse=True,
    )

    reserved_old = min(len(older_chunks), max_contexts // 2)
    take_new = min(len(new_chunks), max_contexts - reserved_old)
    take_old = min(len(older_chunks), max_contexts - take_new)

    return new_chunks[:take_new] + older_chunks[:take_old]


def build_augmented_prompt(
    prompt_augmenter: Any,
    query: str,
    contexts: List[RetrievedContext],
    session_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    extra_format_rules: Optional[str] = None,
) -> AugmentedPrompt:
    """Dựng AugmentedPrompt chuẩn, kèm quy tắc định dạng Web/Voice (nếu có)."""
    uq = UserQuery(query_text=query, session_id=session_id, filters=filters)
    if prompt_augmenter is not None:
        augmented = prompt_augmenter.augment(query=uq, contexts=contexts)
    else:
        context_block = "\n\n".join(f"[Đoạn {i + 1}]: {c.text}" for i, c in enumerate(contexts))
        prompt_str = LLM_GENERATION_FALLBACK_TEMPLATE.format(
            context_block=context_block,
            query=query,
        )
        augmented = AugmentedPrompt(
            user_query=uq,
            contexts=contexts,
            system_instruction=LLM_GENERATION_FALLBACK_SYSTEM,
            formatted_prompt=prompt_str,
        )

    if extra_format_rules and isinstance(getattr(augmented, "formatted_prompt", None), str):
        if hasattr(augmented, "model_copy"):
            augmented = augmented.model_copy(
                update={"formatted_prompt": augmented.formatted_prompt + extra_format_rules}
            )
        else:
            augmented.formatted_prompt = augmented.formatted_prompt + extra_format_rules
    return augmented


def extract_jev_trace_extras(decision_client: Any) -> Dict[str, Any]:
    """Extract optional latency/confidence metadata from the JEV decision client."""
    meta = getattr(decision_client, "last_trace_meta", None)
    if not isinstance(meta, dict):
        return {}
    extras: Dict[str, Any] = {}
    for key in ("server_latency_s", "payload_bytes", "noul_prob", "raw_label", "parse_fallback"):
        if key in meta and meta[key] is not None:
            extras[key] = meta[key]
    return extras

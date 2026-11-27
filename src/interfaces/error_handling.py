"""Error classification, formatting rules, and session helpers for the Web API."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.interfaces.schemas import SourceItem

RAW_ERROR_MARKERS = (
    "MÔ HÌNH AI TẠM THỜI TỪ CHỐI DỊCH VỤ",
    "Groq [",
    "SambaNova [",
    "Gemini [",
    "org_",
    "ITPM",
    "TPM",
    "rate_limit",
    "Request too large",
    "console.groq.com",
    "billing",
    "Traceback",
)

RAW_ERROR_MARKERS_LOWER = (
    "mô hình ai tạm thời từ chối dịch vụ",
    "request too large",
    "rate_limit",
    "traceback (most recent call last)",
    "resource_exhausted",
)

ERROR_STREAM_PREFIXES = (
    "⚠️",
    "⚠",
    "mô hình ai tạm thời từ chối dịch vụ",
    "groq [",
    "sambanova [",
    "gemini [",
    "traceback",
)

FRIENDLY_ERROR_MESSAGES: Dict[str, str] = {
    "token_limit": (
        "Nội dung câu hỏi hoặc tài liệu tham chiếu hiện hơi dài vượt quá giới hạn xử lý trong một lượt. "
        "Bạn hãy thử rút gọn câu hỏi hoặc chia nhỏ thành từng ý để mình hỗ trợ chính xác hơn nhé."
    ),
    "rate_limit": (
        "Hệ thống AI hiện đang tiếp nhận nhiều yêu cầu cùng lúc nên tạm thời quá tải. "
        "Bạn vui lòng đợi trong giây lát rồi thử gửi lại câu hỏi nhé."
    ),
    "network": (
        "Kết nối đến máy chủ AI tạm thời bị gián đoạn hoặc phản hồi chậm. "
        "Bạn vui lòng kiểm tra lại kết nối và thử gửi lại sau ít phút nhé."
    ),
    "server": (
        "Hệ thống đang gặp sự cố kỹ thuật tạm thời khi xử lý yêu cầu của bạn. "
        "Bạn vui lòng thử lại sau ít phút nhé."
    ),
}

RESPONSE_FORMAT_RULES = (
    "\n\n=== QUY TẮC ĐỊNH DẠNG ĐẦU RA BẮT BUỘC ===\n"
    "- TUYỆT ĐỐI KHÔNG dùng biểu tượng cảm xúc hay icon trang trí (emoji như 📌, ⚠️, 💡, ✅, 🎓, 📚, 🔹, 👉...) "
    "trong câu trả lời; hãy dùng **chữ in đậm** cho các tiêu đề và luôn dùng khối trích dẫn Markdown "
    "(> **Lưu ý quan trọng:** ...) cho phần lưu ý quan trọng.\n"
    "- TUYỆT ĐỐI KHÔNG ghi dòng \"Căn cứ quy chế:\", không liệt kê URL, link hay trích dẫn tên tài liệu tham khảo ở cuối câu trả lời.\n"
    "- ĐIỂM SỐ VÀ CÁC CON SỐ ĐƠN THUẦN (như điểm chuẩn, điểm trúng tuyển, điểm GPA, số tín chỉ, học phí, năm học, mã ngành...): "
    "BẮT BUỘC viết tự nhiên bằng chữ và số thông thường trên cùng dòng văn bản (ví dụ: viết '**25 điểm**', '**28.19 điểm**', '**GPA 3.2**', '**24 tín chỉ**'), "
    "TUYỆT ĐỐI KHÔNG bọc các con số đơn thuần này trong cú pháp LaTeX $$...$$ hay $...$ (TUYỆT ĐỐI KHÔNG viết $$25$$, $$28.19$$, $$2026$$ đứng riêng một dòng).\n"
    "- CHỈ sử dụng cú pháp LaTeX chuẩn ($$...$$ đứng riêng một dòng) khi trình bày CÔNG THỨC TOÁN HỌC phức tạp có phép tính (chứa phân số \\frac{tử}{mẫu}, phép nhân \\times, cộng trừ nhiều biến), "
    "KHÔNG để khoảng trắng thừa sát dấu $, KHÔNG đặt công thức dài vào trong ô bảng hẹp, và KHÔNG đặt công thức trong dấu backtick `...`.\n"
    "- Khi trình bày bảng quy đổi chứng chỉ ngoại ngữ (IELTS, TOEFL, TOEIC, Cambridge, Aptis, VSTEP), "
    "giữ nội dung từng ô ngắn gọn, rõ ràng.\n"
)

VOICE_RESPONSE_FORMAT_RULES = (
    "\n\n=== QUY TẮC ĐÀM THOẠI GIỌNG NÓI TRỰC TIẾP (BẮT BUỘC) ===\n"
    "- Người dùng đang nói chuyện trực tiếp với trợ lý qua giọng nói. Hãy trả lời NGẮN GỌN, ĐÚNG TRỌNG TÂM "
    "trong khoảng 1 đến 3 câu văn tự nhiên để đọc thành tiếng dễ nghe, dễ hiểu.\n"
    "- TUYỆT ĐỐI KHÔNG liệt kê bảng biểu, không chèn ký hiệu LaTeX ($, \\frac), không dùng ký tự Markdown rườm rà, "
    "không dùng emoji hay viết dài dòng như tin nhắn văn bản.\n"
)

NO_ANSWER_MARKERS = (
    "chưa có thông tin cụ thể",
    "không có thông tin cụ thể",
    "không có thông tin về",
    "không tìm thấy thông tin",
    "chưa tìm thấy thông tin",
    "tài liệu quy chế lưu trữ chưa có",
    "tài liệu lưu trữ chưa có",
    "ngoài phạm vi",
    "không thuộc phạm vi",
    "không nằm trong phạm vi",
    "chưa được đề cập",
    "không được đề cập",
    "vui lòng liên hệ trực tiếp phòng đào tạo",
    "không liên quan",
    "không liên quan đến",
    "không phải là thông tin thuộc",
)


def build_sources_data(contexts: Optional[List[Any]]) -> List[SourceItem]:
    """
    Hệ thống đã tắt tính năng hiển thị tài liệu tham khảo và URL trích dẫn.
    Trả về danh sách rỗng để đảm bảo UI không hiển thị khối nguồn.
    """
    return []


def classify_error_text(raw_text: str) -> Dict[str, Any]:
    """Phân loại chuỗi lỗi kỹ thuật thành mã HTTP, error_type và thông điệp thân thiện."""
    lower = (raw_text or "").lower()

    if (
        "413" in lower
        or "request too large" in lower
        or "itpm" in lower
        or "context_length" in lower
        or "maximum context" in lower
    ):
        error_type = "token_limit"
        status_code = 413
    elif (
        "429" in lower
        or "rate limit" in lower
        or "rate_limit" in lower
        or "quota" in lower
        or "resource_exhausted" in lower
        or "too many requests" in lower
    ):
        error_type = "rate_limit"
        status_code = 429
    elif "tpm" in lower or "tokens" in lower:
        error_type = "token_limit"
        status_code = 413
    elif (
        "timeout" in lower
        or "timed out" in lower
        or "connecterror" in lower
        or "readtimeout" in lower
        or "connecttimeout" in lower
        or "connection" in lower
        or "network" in lower
        or "không phản hồi" in lower
    ):
        error_type = "network"
        status_code = 503
    else:
        error_type = "server"
        status_code = 500

    return {
        "error_type": error_type,
        "status_code": status_code,
        "message": FRIENDLY_ERROR_MESSAGES[error_type],
    }


def classify_exception(exc: Exception) -> Dict[str, Any]:
    """Phân loại Python exception phát sinh khi gọi RAG/LLM thành thông điệp an toàn cho client."""
    exc_name = type(exc).__name__
    exc_str = str(exc or "")
    combined = f"{exc_name}: {exc_str}"

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return {
            "error_type": "network",
            "status_code": 503,
            "message": FRIENDLY_ERROR_MESSAGES["network"],
        }

    return classify_error_text(combined)


def detect_and_classify_llm_error(
    answer: str,
    model_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Phát hiện lỗi kỹ thuật bị nhúng trong Response.answer (ví dụ fallback từ UETLLMClient).
    Trả về dict {error_type, status_code, message} nếu là lỗi, ngược lại trả về None.
    """
    text = answer or ""
    lower_text = text.lower()
    meta_errors = metadata.get("errors") if isinstance(metadata, dict) else None

    is_fallback_model = (model_name or "").strip().lower() == "multi-provider-fallback"
    has_meta_errors = bool(meta_errors)
    has_marker = any(marker in text for marker in RAW_ERROR_MARKERS) or any(
        marker in lower_text for marker in RAW_ERROR_MARKERS_LOWER
    )

    if not (is_fallback_model or has_meta_errors or has_marker):
        return None

    extra_detail = " ".join(str(err) for err in meta_errors) if isinstance(meta_errors, list) else str(meta_errors or "")
    combined = f"{text}\n{extra_detail}".strip()
    return classify_error_text(combined)


def is_no_answer_or_out_of_domain(answer: str) -> bool:
    """Nhận diện phản hồi từ chối / không có thông tin / ngoài domain để ẩn tài liệu tham khảo."""
    lower = (answer or "").strip().lower()
    if not lower:
        return True
    return any(marker in lower for marker in NO_ANSWER_MARKERS)


def resolve_session_id(session_id: Optional[str]) -> str:
    """Return a non-default per-conversation session ID, generating a UUID when omitted."""
    if not session_id or session_id == "default_session":
        return str(uuid.uuid4())
    return session_id


def select_format_rules(voice_mode: bool) -> str:
    """Select the response formatting instructions for voice vs text chat."""
    return VOICE_RESPONSE_FORMAT_RULES if voice_mode else RESPONSE_FORMAT_RULES

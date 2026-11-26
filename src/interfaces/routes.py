"""
src/interfaces/routes.py - Cung cấp Giao diện Web (Web UI & REST API) cho UET Chatbot.
Thành viên phụ trách: Giao diện người dùng & Web Backend.
Nhiệm vụ: Phục vụ giao diện Web người dùng và chuyển tiếp yêu cầu đến src/pipeline.py.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.base import UserQuery
from src.pipeline import get_rag_pipeline, get_active_pipeline
from src.utils.helpers import Timer, get_logger, load_yaml_config

logger = get_logger("api")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DIST_DIR = STATIC_DIR / "dist"

# Các dấu hiệu nhận diện phản hồi lỗi thô từ LLM client / API provider
_RAW_ERROR_MARKERS = (
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

_RAW_ERROR_MARKERS_LOWER = (
    "mô hình ai tạm thời từ chối dịch vụ",
    "request too large",
    "rate_limit",
    "traceback (most recent call last)",
    "resource_exhausted",
)

_ERROR_STREAM_PREFIXES = (
    "⚠️",
    "⚠",
    "mô hình ai tạm thời từ chối dịch vụ",
    "groq [",
    "sambanova [",
    "gemini [",
    "traceback",
)

_FRIENDLY_ERROR_MESSAGES: Dict[str, str] = {
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

_RESPONSE_FORMAT_RULES = (
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

_VOICE_RESPONSE_FORMAT_RULES = (
    "\n\n=== QUY TẮC ĐÀM THOẠI GIỌNG NÓI TRỰC TIẾP (BẮT BUỘC) ===\n"
    "- Người dùng đang nói chuyện trực tiếp với trợ lý qua giọng nói. Hãy trả lời NGẮN GỌN, ĐÚNG TRỌNG TÂM "
    "trong khoảng 1 đến 3 câu văn tự nhiên để đọc thành tiếng dễ nghe, dễ hiểu.\n"
    "- TUYỆT ĐỐI KHÔNG liệt kê bảng biểu, không chèn ký hiệu LaTeX ($, \\frac), không dùng ký tự Markdown rườm rà, "
    "không dùng emoji hay viết dài dòng như tin nhắn văn bản.\n"
)


class ChatRequest(BaseModel):
    query: str = Field(..., description="Nội dung câu hỏi của sinh viên")
    session_id: Optional[str] = Field(default=None, description="Mã phiên hội thoại (tự động tạo UUID nếu để trống)")
    top_k: Optional[int] = Field(
        default=None,
        description="Số lượng trích đoạn cần lấy (mặc định lấy theo config.yaml)",
    )
    voice_mode: bool = Field(
        default=False,
        description="Chế độ đàm thoại giọng nói trực tiếp (trả lời ngắn gọn, đúng trọng tâm)",
    )


class SourceItem(BaseModel):
    chunk_id: str
    text: str
    similarity_score: float
    rank: int
    search_type: Optional[str] = None
    search_source: Optional[str] = None
    metadata: Dict[str, Any] = {}


SourceItem.model_rebuild()


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem] = []
    latency_seconds: float = 0.0
    model_name: str = ""
    session_id: Optional[str] = None


ChatResponse.model_rebuild()


class VoiceTranscribeRequest(BaseModel):
    audio_base64: str = Field(..., description="Dữ liệu âm thanh mã hóa Base64")
    mime_type: str = Field(
        default="audio/webm",
        description="MIME type từ MediaRecorder (vd: audio/webm, audio/ogg)",
    )
    language: str = Field(default="vi", description="Mã ngôn ngữ nhận dạng")


class VoiceTranscribeResponse(BaseModel):
    text: str
    model_name: str = ""
    latency_seconds: float = 0.0


VoiceTranscribeRequest.model_rebuild()
VoiceTranscribeResponse.model_rebuild()


def _build_sources_data(contexts: Optional[List[Any]]) -> List[SourceItem]:
    """
    Hệ thống đã tắt tính năng hiển thị tài liệu tham khảo và URL trích dẫn.
    Trả về danh sách rỗng để đảm bảo UI không hiển thị khối nguồn.
    """
    return []


def _classify_error_text(raw_text: str) -> Dict[str, Any]:
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
        "message": _FRIENDLY_ERROR_MESSAGES[error_type],
    }


def _classify_exception(exc: Exception) -> Dict[str, Any]:
    """Phân loại Python exception phát sinh khi gọi RAG/LLM thành thông điệp an toàn cho client."""
    exc_name = type(exc).__name__
    exc_str = str(exc or "")
    combined = f"{exc_name}: {exc_str}"

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return {
            "error_type": "network",
            "status_code": 503,
            "message": _FRIENDLY_ERROR_MESSAGES["network"],
        }

    return _classify_error_text(combined)


def _detect_and_classify_llm_error(
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
    has_marker = any(marker in text for marker in _RAW_ERROR_MARKERS) or any(
        marker in lower_text for marker in _RAW_ERROR_MARKERS_LOWER
    )

    if not (is_fallback_model or has_meta_errors or has_marker):
        return None

    extra_detail = " ".join(str(err) for err in meta_errors) if isinstance(meta_errors, list) else str(meta_errors or "")
    combined = f"{text}\n{extra_detail}".strip()
    return _classify_error_text(combined)


_NO_ANSWER_MARKERS = (
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


def _is_no_answer_or_out_of_domain(answer: str) -> bool:
    """Nhận diện phản hồi từ chối / không có thông tin / ngoài domain để ẩn tài liệu tham khảo."""
    lower = (answer or "").strip().lower()
    if not lower:
        return True
    return any(marker in lower for marker in _NO_ANSWER_MARKERS)


_HALLUCINATION_SUBSTRINGS = (
    "subscribe cho kênh",
    "đăng ký kênh",
    "la la school",
    "ghiền mì gõ",
    "faptv",
    "kem xôi",
    "để không bỏ lỡ những video",
    "không bỏ lỡ các video",
    "những video hấp dẫn",
    "video mới nhất",
    "like và subscribe",
    "nhấn chuông",
    "bấm chuông",
    "cảm ơn các bạn đã theo dõi",
    "cảm ơn các bạn đã xem",
    "hẹn gặp lại các bạn",
    "phụ đề được thực hiện bởi",
    "subtitles by",
    "amara.org",
    "translated by",
)

_REPEATED_PHRASE_RE = re.compile(
    r"(?:^|[\s,.!?])([^\s,.!?]+(?:\s+[^\s,.!?]+){0,2})(?:\s*[,.\s]+\1){3,}(?:$|[\s,.!?])",
    re.IGNORECASE,
)


def _is_whisper_hallucination(text: str) -> bool:
    """Nhận diện các câu ảo giác phổ biến của Whisper khi gặp khoảng lặng hoặc tạp âm."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return True
    if not re.search(r"[0-9A-Za-zÀ-ỹ]", cleaned):
        return True
    lowered = cleaned.lower()
    if any(marker in lowered for marker in _HALLUCINATION_SUBSTRINGS):
        return True
    if _REPEATED_PHRASE_RE.search(lowered):
        return True
    return False


def _extract_clean_transcript(transcription: Any) -> str:
    """Trích xuất văn bản sạch từ verbose_json, lọc bỏ segment no_speech cao và ảo giác."""
    segments = (
        transcription.get("segments")
        if isinstance(transcription, dict)
        else getattr(transcription, "segments", None)
    )
    if isinstance(segments, list) and segments:
        kept_parts: List[str] = []
        for seg in segments:
            seg_text = (
                seg.get("text", "")
                if isinstance(seg, dict)
                else getattr(seg, "text", "")
            )
            seg_text = re.sub(r"\s+", " ", str(seg_text or "")).strip()
            if not seg_text:
                continue

            raw_no_speech = (
                seg.get("no_speech_prob", 0.0)
                if isinstance(seg, dict)
                else getattr(seg, "no_speech_prob", 0.0)
            )
            raw_avg_logprob = (
                seg.get("avg_logprob", 0.0)
                if isinstance(seg, dict)
                else getattr(seg, "avg_logprob", 0.0)
            )
            try:
                no_speech_prob = float(raw_no_speech or 0.0)
            except (TypeError, ValueError):
                no_speech_prob = 0.0
            try:
                avg_logprob = float(raw_avg_logprob or 0.0)
            except (TypeError, ValueError):
                avg_logprob = 0.0

            if no_speech_prob > 0.6 or (no_speech_prob > 0.45 and avg_logprob < -0.85):
                continue
            if _is_whisper_hallucination(seg_text):
                continue
            kept_parts.append(seg_text)
        text = " ".join(kept_parts).strip()
    else:
        raw_text = (
            transcription.get("text", "")
            if isinstance(transcription, dict)
            else getattr(transcription, "text", "")
        )
        text = re.sub(r"\s+", " ", str(raw_text or "")).strip()

    if _is_whisper_hallucination(text):
        return ""
    return text


def _guess_audio_ext(mime_type: str) -> str:
    lower = (mime_type or "").lower()
    if "ogg" in lower:
        return "ogg"
    if "mp4" in lower or "m4a" in lower:
        return "m4a"
    if "wav" in lower:
        return "wav"
    if "mpeg" in lower or "mp3" in lower:
        return "mp3"
    return "webm"


def _resolve_stt_groq_keys(pipeline: Any) -> List[str]:
    llm = getattr(pipeline, "llm", None) if pipeline else None
    keys = list(getattr(llm, "groq_keys", None) or [])
    if keys:
        return keys
    try:
        from src.llm.llm_client import UETLLMClient
        return list(UETLLMClient._load_keys_for_provider("GROQ"))
    except Exception:
        return []


def create_app():
    """
    Khởi tạo ứng dụng FastAPI phục vụ API và Giao diện Web.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        logger.warning("Chưa cài đặt FastAPI. Vui lòng chạy: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="UET Chatbot RAG Service",
        description="Giao diện Web & REST API Tra cứu quy chế đào tạo UET (VNU)",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Chống trình duyệt lưu cache file tĩnh trong quá trình phát triển
    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static") or path.startswith("/assets"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # Phục vụ assets bundle (Vite build) và thư mục static gốc
    dist_assets_dir = DIST_DIR / "assets"
    if dist_assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(dist_assets_dir)), name="dist_assets")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    def startup_warmup():
        """Tải trước toàn bộ mô hình và thực hiện giai đoạn Cold-Start trước khi phục vụ người dùng."""
        logger.info("🚀 [1/2] Đang khởi động và nạp trước RAG Pipeline vào RAM (Pre-warming)...")
        try:
            pipeline = get_active_pipeline()
            logger.info("🔥 [2/2] Đang thực thi Cold-Start Warmup để làm nóng Embedding ONNX, Reranker, BM25 & LLM...")
            # Thực hiện một truy vấn giả lập để:
            # 1. Kích hoạt bộ tách từ tiếng Việt underthesea
            # 2. Kích hoạt đồ thị suy luận ONNX FP32 của BGE-M3 Embedding
            # 3. Kích hoạt bộ nhớ đệm ChromaDB Vector Search
            # 4. Kích hoạt đồ thị suy luận ONNX của Jina Reranker
            # 5. Thiết lập kết nối mạng ban đầu (SSL Handshake) tới LLM API
            warmup_query = UserQuery(
                query_text="Tiêu chuẩn xét học bổng khuyến khích học tập UET",
                session_id="system_cold_start",
            )
            pipeline.query(warmup_query, top_k=2)
            logger.info("✨ Giai đoạn Cold-Start hoàn tất! Toàn bộ hệ thống đã sẵn sàng 100%, câu hỏi đầu tiên sẽ phản hồi tức thì!")
        except Exception as e:
            logger.warning(f"Lỗi khởi động hoặc Cold-Start Warmup: {e}")

    @app.get("/")
    def serve_homepage():
        """Phục vụ trang chủ Web UI (ưu tiên React dist build, fallback static/index.html)."""
        dist_index = DIST_DIR / "index.html"
        if dist_index.exists():
            return FileResponse(
                dist_index,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"},
            )
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(
                index_file,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"},
            )
        return {"message": "UET Chatbot API is running. index.html not found."}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        from fastapi.responses import Response
        return Response(status_code=204)

    @app.get("/api/health")
    def health_check() -> Dict[str, Any]:
        """Kiểm tra trạng thái máy chủ và mô hình LLM hiện tại."""
        config = load_yaml_config()
        model_name = config.get("llm", {}).get("model_name", "tencent/hy4-flash")
        return {
            "status": "ok",
            "message": "UET Chatbot RAG Server đang chạy bình thường.",
            "model_name": model_name,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest):
        """
        Nhận câu hỏi từ Web UI và chuyển tiếp đến RAG Pipeline chính thức trong src/pipeline.py.
        """
        pipeline = get_active_pipeline()
        if not pipeline:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Hệ thống RAG hiện chưa sẵn sàng. Bạn vui lòng thử lại sau ít phút nhé.",
                    "error_type": "server",
                },
            )

        # Phase 5.2 (D9): Sinh session_id riêng, bỏ giá trị mặc định chung default_session
        effective_session_id = request.session_id
        if not effective_session_id or effective_session_id == "default_session":
            effective_session_id = str(uuid.uuid4())

        with Timer() as timer:
            user_query = UserQuery(
                query_text=request.query,
                session_id=effective_session_id,
            )
            retrieval_top_k = (
                request.top_k
                if request.top_k is not None
                else getattr(pipeline.retriever, "default_top_k", 10)
            )
            try:
                if hasattr(pipeline, "agent_graph"):
                    format_rules = (
                        _VOICE_RESPONSE_FORMAT_RULES
                        if request.voice_mode
                        else _RESPONSE_FORMAT_RULES
                    )
                    response = pipeline.query(
                        user_query,
                        top_k=retrieval_top_k,
                        extra_format_rules=format_rules,
                        voice_mode=request.voice_mode,
                    )
                else:
                    response = pipeline.query(user_query, top_k=retrieval_top_k)
            except Exception as exc:
                logger.exception("Lỗi trong pipeline.query tại POST /api/chat: %s", exc)
                classified = _classify_exception(exc)
                raise HTTPException(
                    status_code=classified["status_code"],
                    detail={
                        "message": classified["message"],
                        "error_type": classified["error_type"],
                    },
                )

        detected_error = _detect_and_classify_llm_error(
            response.answer,
            response.model_name or "",
            response.metadata,
        )
        if detected_error is not None:
            logger.error(
                "LLM fallback/error detected in POST /api/chat [%s] model=%s metadata=%s raw=%s",
                detected_error["error_type"],
                response.model_name,
                response.metadata,
                response.answer,
            )
            raise HTTPException(
                status_code=detected_error["status_code"],
                detail={
                    "message": detected_error["message"],
                    "error_type": detected_error["error_type"],
                },
            )

        sources_data = (
            []
            if _is_no_answer_or_out_of_domain(response.answer)
            else _build_sources_data(response.sources)
        )

        return ChatResponse(
            query=request.query,
            answer=response.answer,
            sources=sources_data,
            latency_seconds=round(timer.elapsed, 3),
            model_name=response.model_name or "",
            session_id=effective_session_id,
        )

    @app.post("/api/chat/stream")
    def chat_stream_endpoint(request: ChatRequest):
        """
        Streaming SSE endpoint trả về nguồn tham khảo, token câu trả lời và trạng thái hoàn tất.
        """
        pipeline = get_active_pipeline()
        if not pipeline:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Hệ thống RAG hiện chưa sẵn sàng. Bạn vui lòng thử lại sau ít phút nhé.",
                    "error_type": "server",
                },
            )

        effective_session_id = request.session_id
        if not effective_session_id or effective_session_id == "default_session":
            effective_session_id = str(uuid.uuid4())

        user_query = UserQuery(
            query_text=request.query,
            session_id=effective_session_id,
        )

        def event_generator():
            with Timer() as timer:
                try:
                    retrieval_top_k = (
                        request.top_k
                        if request.top_k is not None
                        else getattr(pipeline.retriever, "default_top_k", 10)
                    )

                    # Hỗ trợ Agentic RAG Pipeline với JEV LangGraph (bắt buộc qua answer_eval, chỉ phát phản hồi cuối)
                    if hasattr(pipeline, "agent_graph") and hasattr(pipeline, "stream_events"):
                        format_rules = (
                            _VOICE_RESPONSE_FORMAT_RULES
                            if request.voice_mode
                            else _RESPONSE_FORMAT_RULES
                        )
                        accumulated_agent = ""
                        agent_model_name = getattr(getattr(pipeline, "llm", None), "model_name", "") or "jev-agentic-rag"

                        for event in pipeline.stream_events(
                            user_query,
                            top_k=retrieval_top_k,
                            extra_format_rules=format_rules,
                            voice_mode=request.voice_mode,
                        ):
                            ev_type = event.get("type")
                            if ev_type == "status":
                                status_payload = {
                                    "type": "status",
                                    "stage": event.get("node", ""),
                                    "message": event.get("message", ""),
                                }
                                yield f"data: {json.dumps(status_payload, ensure_ascii=False)}\n\n"
                            elif ev_type == "sources":
                                sources_data = _build_sources_data(event.get("sources"))
                                agent_model_name = event.get("model_name") or agent_model_name
                                sources_payload = {
                                    "type": "sources",
                                    "sources": [s.model_dump() for s in sources_data],
                                    "model_name": agent_model_name,
                                }
                                yield f"data: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"
                            elif ev_type == "token":
                                chunk = event.get("content") or ""
                                if not chunk:
                                    continue
                                accumulated_agent += chunk
                                detected = (
                                    _detect_and_classify_llm_error(accumulated_agent)
                                    or _detect_and_classify_llm_error(chunk)
                                )
                                if detected is not None:
                                    logger.error(
                                        "LLM error detected in Agentic /api/chat/stream [%s] model=%s raw=%s",
                                        detected["error_type"],
                                        agent_model_name,
                                        accumulated_agent,
                                    )
                                    err_payload = {
                                        "type": "error",
                                        "error_type": detected["error_type"],
                                        "message": detected["message"],
                                    }
                                    yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return
                                token_payload = {"type": "token", "content": chunk}
                                yield f"data: {json.dumps(token_payload, ensure_ascii=False)}\n\n"
                            elif ev_type == "done":
                                final_ans = event.get("answer") or accumulated_agent
                                done_payload = {
                                    "type": "done",
                                    "latency_seconds": round(timer.elapsed, 3),
                                    "hide_sources": _is_no_answer_or_out_of_domain(final_ans),
                                }
                                yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                        done_payload = {
                            "type": "done",
                            "latency_seconds": round(timer.elapsed, 3),
                            "hide_sources": _is_no_answer_or_out_of_domain(accumulated_agent),
                        }
                        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    retrieve_kwargs: Dict[str, Any] = {
                        "query": user_query,
                        "top_k": retrieval_top_k,
                    }
                    if user_query.filters is not None:
                        retrieve_kwargs["filters"] = user_query.filters

                    contexts = pipeline.retriever.retrieve(**retrieve_kwargs)
                    sources_data = _build_sources_data(contexts)
                    model_name = getattr(getattr(pipeline, "llm", None), "model_name", "") or ""

                    sources_payload = {
                        "type": "sources",
                        "sources": [s.model_dump() for s in sources_data],
                        "model_name": model_name,
                    }
                    yield f"data: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

                    augmented_prompt = pipeline.prompt_augmenter.augment(
                        query=user_query,
                        contexts=contexts,
                    )
                    format_rules = (
                        _VOICE_RESPONSE_FORMAT_RULES
                        if request.voice_mode
                        else _RESPONSE_FORMAT_RULES
                    )
                    if hasattr(augmented_prompt, "model_copy") and isinstance(
                        getattr(augmented_prompt, "formatted_prompt", None), str
                    ):
                        augmented_prompt = augmented_prompt.model_copy(
                            update={
                                "formatted_prompt": augmented_prompt.formatted_prompt
                                + format_rules
                            }
                        )

                    accumulated = ""
                    pending_prefix = ""
                    prefix_checked = False

                    for chunk in pipeline.llm.generate_stream(augmented_prompt):
                        if not chunk:
                            continue
                        accumulated += chunk

                        detected = (
                            _detect_and_classify_llm_error(accumulated)
                            or _detect_and_classify_llm_error(chunk)
                        )
                        if detected is not None:
                            logger.error(
                                "LLM error detected in POST /api/chat/stream [%s] model=%s raw=%s",
                                detected["error_type"],
                                model_name,
                                accumulated,
                            )
                            err_payload = {
                                "type": "error",
                                "error_type": detected["error_type"],
                                "message": detected["message"],
                            }
                            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        if not prefix_checked:
                            pending_prefix += chunk
                            stripped = pending_prefix.lstrip().lower()
                            maybe_error_prefix = any(
                                header.startswith(stripped) or stripped.startswith(header)
                                for header in _ERROR_STREAM_PREFIXES
                            )
                            if maybe_error_prefix and len(pending_prefix) < 100:
                                continue

                            prefix_checked = True
                            token_payload = {"type": "token", "content": pending_prefix}
                            pending_prefix = ""
                            yield f"data: {json.dumps(token_payload, ensure_ascii=False)}\n\n"
                        else:
                            token_payload = {"type": "token", "content": chunk}
                            yield f"data: {json.dumps(token_payload, ensure_ascii=False)}\n\n"

                    if pending_prefix:
                        detected = _detect_and_classify_llm_error(pending_prefix)
                        if detected is not None:
                            logger.error(
                                "LLM error detected at end of stream [%s] model=%s raw=%s",
                                detected["error_type"],
                                model_name,
                                pending_prefix,
                            )
                            err_payload = {
                                "type": "error",
                                "error_type": detected["error_type"],
                                "message": detected["message"],
                            }
                            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        token_payload = {"type": "token", "content": pending_prefix}
                        yield f"data: {json.dumps(token_payload, ensure_ascii=False)}\n\n"

                    done_payload = {
                        "type": "done",
                        "latency_seconds": round(timer.elapsed, 3),
                        "hide_sources": _is_no_answer_or_out_of_domain(accumulated),
                    }
                    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    logger.exception("Lỗi trong quá trình streaming /api/chat/stream: %s", exc)
                    classified = _classify_exception(exc)
                    err_payload = {
                        "type": "error",
                        "error_type": classified["error_type"],
                        "message": classified["message"],
                    }
                    yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/voice/transcribe", response_model=VoiceTranscribeResponse)
    def voice_transcribe_endpoint(request: VoiceTranscribeRequest):
        """
        Chuyển đổi âm thanh thu từ trình duyệt (đặc biệt cho Firefox khi không có Web Speech STT)
        thành văn bản tiếng Việt thông qua Groq Whisper API.
        """
        raw_b64 = (request.audio_base64 or "").strip()
        if "," in raw_b64 and raw_b64.lower().startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[1].strip()

        if not raw_b64:
            return VoiceTranscribeResponse(text="", model_name="none", latency_seconds=0.0)

        try:
            audio_bytes = base64.b64decode(raw_b64, validate=True)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Dữ liệu âm thanh không hợp lệ, vui lòng thử thu âm lại.",
                    "error_type": "server",
                },
            )

        if len(audio_bytes) < 400:
            return VoiceTranscribeResponse(text="", model_name="none", latency_seconds=0.0)

        if len(audio_bytes) > 12 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail={
                    "message": "Đoạn ghi âm quá dài vượt quá giới hạn cho phép. Bạn vui lòng nói câu ngắn hơn nhé.",
                    "error_type": "token_limit",
                },
            )

        pipeline = get_rag_pipeline()
        groq_keys = _resolve_stt_groq_keys(pipeline)
        if not groq_keys:
            logger.error("Không tìm thấy GROQ_API_KEY cho tính năng voice transcribe.")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Máy chủ hiện chưa cấu hình dịch vụ nhận dạng giọng nói. Vui lòng thử lại sau.",
                    "error_type": "server",
                },
            )

        try:
            import groq
        except ImportError:
            logger.error("Thư viện groq chưa được cài đặt trong môi trường.")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Hệ thống tạm thời chưa hỗ trợ chuyển đổi giọng nói. Vui lòng thử lại sau.",
                    "error_type": "server",
                },
            )

        clean_mime = (request.mime_type or "audio/webm").split(";", 1)[0].strip() or "audio/webm"
        ext = _guess_audio_ext(clean_mime)
        lang = (request.language or "vi").strip() or "vi"
        models = ("whisper-large-v3-turbo", "whisper-large-v3")

        last_exc: Optional[Exception] = None
        with Timer() as timer:
            for model_name in models:
                for api_key in groq_keys:
                    try:
                        client = groq.Groq(api_key=api_key, timeout=25.0)
                        transcription = client.audio.transcriptions.create(
                            model=model_name,
                            file=(f"voice.{ext}", audio_bytes, clean_mime),
                            language=lang,
                            response_format="verbose_json",
                            temperature=0.0,
                        )
                        text = _extract_clean_transcript(transcription)
                        return VoiceTranscribeResponse(
                            text=text,
                            model_name=model_name,
                            latency_seconds=round(timer.elapsed, 3),
                        )
                    except Exception as exc:
                        last_exc = exc
                        logger.warning(
                            "Groq Whisper thất bại với model=%s: %s",
                            model_name,
                            type(exc).__name__,
                        )

        if last_exc is not None:
            logger.exception("Tất cả key Groq Whisper đều thất bại trong /api/voice/transcribe: %s", last_exc)
            classified = _classify_exception(last_exc)
            raise HTTPException(
                status_code=classified["status_code"],
                detail={
                    "message": classified["message"],
                    "error_type": classified["error_type"],
                },
            )

        return VoiceTranscribeResponse(text="", model_name="none", latency_seconds=0.0)

    return app


def run_web(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    """Khởi chạy API Server và Giao diện Web với FastAPI và Uvicorn."""
    try:
        import threading
        import webbrowser
        import uvicorn

        app = create_app()
        if app is None:
            print("Vui lòng cài đặt: pip install fastapi uvicorn")
            return

        url = f"http://{host}:{port}"
        print(f"\n🌐 GIAO DIỆN WEB LOCAL ĐANG CHẠY TẠI: {url}")
        print(f"📚 Tài liệu REST API Swagger:        {url}/docs")
        print("💡 Nhấn Ctrl+C để dừng server.\n")

        if open_browser:
            def wait_and_open_browser():
                import time
                import urllib.request
                health_url = f"{url}/api/health"
                # Chờ server hoàn tất khởi động và Cold-Start warmup rồi mới bật trình duyệt
                for _ in range(120):
                    time.sleep(0.5)
                    try:
                        with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                            if resp.status == 200:
                                time.sleep(0.2)
                                webbrowser.open(url)
                                break
                    except Exception:
                        pass

            threading.Thread(target=wait_and_open_browser, daemon=True).start()

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("Lỗi: Cần cài đặt uvicorn để chạy server (pip install uvicorn)")

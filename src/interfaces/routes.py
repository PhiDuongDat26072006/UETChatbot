"""
src/interfaces/routes.py - Cung cấp Giao diện Web (Web UI & REST API) cho UET Chatbot.
Thành viên phụ trách: Giao diện người dùng & Web Backend.
Nhiệm vụ: Phục vụ giao diện Web người dùng và chuyển tiếp yêu cầu đến src/pipeline.py.

Core schemas, error classification, voice STT, and SSE streaming helpers live in:
- ``src/interfaces/schemas.py``
- ``src/interfaces/error_handling.py``
- ``src/interfaces/voice.py``
- ``src/interfaces/streaming.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.base import UserQuery
from src.interfaces.error_handling import (
    ERROR_STREAM_PREFIXES as _ERROR_STREAM_PREFIXES,
    FRIENDLY_ERROR_MESSAGES as _FRIENDLY_ERROR_MESSAGES,
    NO_ANSWER_MARKERS as _NO_ANSWER_MARKERS,
    RAW_ERROR_MARKERS as _RAW_ERROR_MARKERS,
    RAW_ERROR_MARKERS_LOWER as _RAW_ERROR_MARKERS_LOWER,
    RESPONSE_FORMAT_RULES as _RESPONSE_FORMAT_RULES,
    VOICE_RESPONSE_FORMAT_RULES as _VOICE_RESPONSE_FORMAT_RULES,
    build_sources_data as _build_sources_data,
    classify_error_text as _classify_error_text,
    classify_exception as _classify_exception,
    detect_and_classify_llm_error as _detect_and_classify_llm_error,
    is_no_answer_or_out_of_domain as _is_no_answer_or_out_of_domain,
    resolve_session_id,
    select_format_rules,
)
from src.interfaces.schemas import (
    ChatRequest,
    ChatResponse,
    SourceItem,
    VoiceTranscribeRequest,
    VoiceTranscribeResponse,
)
from src.interfaces.streaming import build_chat_stream_generator
from src.interfaces.voice import (
    HALLUCINATION_SUBSTRINGS as _HALLUCINATION_SUBSTRINGS,
    REPEATED_PHRASE_RE as _REPEATED_PHRASE_RE,
    extract_clean_transcript as _extract_clean_transcript,
    guess_audio_ext as _guess_audio_ext,
    is_whisper_hallucination as _is_whisper_hallucination,
    resolve_stt_groq_keys as _resolve_stt_groq_keys,
    transcribe_voice,
)
from src.pipeline import get_active_pipeline, get_rag_pipeline
from src.utils.helpers import Timer, get_logger, load_yaml_config

logger = get_logger("api")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DIST_DIR = STATIC_DIR / "dist"


def _require_active_pipeline():
    from fastapi import HTTPException

    pipeline = get_active_pipeline()
    if not pipeline:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Hệ thống RAG hiện chưa sẵn sàng. Bạn vui lòng thử lại sau ít phút nhé.",
                "error_type": "server",
            },
        )
    return pipeline


def create_app():
    """Khởi tạo ứng dụng FastAPI phục vụ API và Giao diện Web."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, Response, StreamingResponse
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

    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static") or path.startswith("/assets"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

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
            warmup_query = UserQuery(
                query_text="Tiêu chuẩn xét học bổng khuyến khích học tập UET",
                session_id="system_cold_start",
            )
            pipeline.query(warmup_query, top_k=2)
            logger.info("✨ Giai đoạn Cold-Start hoàn tất! Toàn bộ hệ thống đã sẵn sàng 100%, câu hỏi đầu tiên sẽ phản hồi tức thì!")
        except Exception as exc:
            logger.warning(f"Lỗi khởi động hoặc Cold-Start Warmup: {exc}")

    @app.get("/")
    def serve_homepage():
        """Phục vụ trang chủ Web UI (ưu tiên React dist build, fallback static/index.html)."""
        no_cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
        for candidate in (DIST_DIR / "index.html", STATIC_DIR / "index.html"):
            if candidate.exists():
                return FileResponse(candidate, headers=no_cache_headers)
        return {"message": "UET Chatbot API is running. index.html not found."}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
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
        """Nhận câu hỏi từ Web UI và chuyển tiếp đến RAG Pipeline chính thức."""
        pipeline = _require_active_pipeline()
        effective_session_id = resolve_session_id(request.session_id)

        with Timer() as timer:
            user_query = UserQuery(query_text=request.query, session_id=effective_session_id)
            retrieval_top_k = (
                request.top_k
                if request.top_k is not None
                else getattr(pipeline.retriever, "default_top_k", 10)
            )
            try:
                if hasattr(pipeline, "agent_graph"):
                    response = pipeline.query(
                        user_query,
                        top_k=retrieval_top_k,
                        extra_format_rules=select_format_rules(request.voice_mode),
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
        """Streaming SSE endpoint trả về nguồn tham khảo, token câu trả lời và trạng thái hoàn tất."""
        pipeline = _require_active_pipeline()
        effective_session_id = resolve_session_id(request.session_id)
        user_query = UserQuery(query_text=request.query, session_id=effective_session_id)

        return StreamingResponse(
            build_chat_stream_generator(
                pipeline=pipeline,
                user_query=user_query,
                request_top_k=request.top_k,
                voice_mode=request.voice_mode,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/voice/transcribe", response_model=VoiceTranscribeResponse)
    def voice_transcribe_endpoint(request: VoiceTranscribeRequest):
        """Chuyển đổi âm thanh thu từ trình duyệt thành văn bản tiếng Việt qua Groq Whisper API."""
        return transcribe_voice(request, get_rag_pipeline())

    return app


def run_web(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    """Khởi chạy API Server và Giao diện Web với FastAPI và Uvicorn."""
    try:
        import threading
        import time
        import urllib.request
        import uvicorn
        import webbrowser

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
                health_url = f"{url}/api/health"
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

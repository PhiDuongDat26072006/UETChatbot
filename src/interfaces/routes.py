"""
src/interfaces/routes.py - Cung cấp Giao diện Web (Web UI & REST API) cho UET Chatbot.
Thành viên phụ trách: Giao diện người dùng & Web Backend.
Nhiệm vụ: Phục vụ giao diện Web người dùng và chuyển tiếp yêu cầu đến src/pipeline.py.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from src.base import UserQuery
from src.pipeline import get_rag_pipeline
from src.utils.helpers import get_logger, Timer

logger = get_logger("api")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    query: str = Field(..., description="Nội dung câu hỏi của sinh viên")
    session_id: str = Field(default="default_session", description="Mã phiên hội thoại")
    top_k: int = Field(default=4, description="Số lượng trích đoạn cần lấy")


class SourceItem(BaseModel):
    chunk_id: str
    text: str
    similarity_score: float
    rank: int
    metadata: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem] = []
    latency_seconds: float = 0.0


def create_app():
    """
    Khởi tạo ứng dụng FastAPI phục vụ API và Giao diện Web.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
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

    # Phục vụ thư mục static (CSS, JS, assets)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_homepage():
        """Phục vụ trang chủ Web UI."""
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "UET Chatbot API is running. index.html not found."}

    @app.get("/api/health")
    def health_check() -> Dict[str, str]:
        """Kiểm tra trạng thái máy chủ."""
        return {"status": "ok", "message": "UET Chatbot RAG Server đang chạy bình thường."}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest):
        """
        Nhận câu hỏi từ Web UI và chuyển tiếp đến RAG Pipeline chính thức trong src/pipeline.py.
        """
        pipeline = get_rag_pipeline()
        if not pipeline:
            raise HTTPException(status_code=500, detail="RAG Pipeline chưa sẵn sàng.")

        with Timer() as timer:
            user_query = UserQuery(
                query_text=request.query,
                session_id=request.session_id,
            )
            response = pipeline.query(user_query, top_k=request.top_k)

        sources_data = [
            SourceItem(
                chunk_id=ctx.chunk_id,
                text=ctx.text,
                similarity_score=ctx.similarity_score,
                rank=ctx.rank,
                metadata=ctx.metadata,
            )
            for ctx in response.sources
        ]

        return ChatResponse(
            query=request.query,
            answer=response.answer,
            sources=sources_data,
            latency_seconds=round(timer.elapsed, 3),
        )

    return app

"""
src/interfaces/routes.py - Cung cấp Giao diện Web (Web UI & REST API) cho UET Chatbot.
Thành viên phụ trách: Giao diện người dùng & Web Backend.
Nhiệm vụ: Phục vụ giao diện Web người dùng và chuyển tiếp yêu cầu đến src/pipeline.py.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.base import UserQuery
from src.pipeline import get_rag_pipeline
from src.utils.helpers import get_logger, Timer, load_yaml_config

logger = get_logger("api")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    query: str = Field(..., description="Nội dung câu hỏi của sinh viên")
    session_id: str = Field(default="default_session", description="Mã phiên hội thoại")
    top_k: Optional[int] = Field(default=None, description="Số lượng trích đoạn cần lấy (mặc định lấy theo config.yaml)")


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


ChatResponse.model_rebuild()


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

    # Chống trình duyệt lưu cache file tĩnh trong quá trình phát triển
    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # Phục vụ thư mục static (CSS, JS, assets)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_homepage():
        """Phục vụ trang chủ Web UI."""
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(
                index_file,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return {"message": "UET Chatbot API is running. index.html not found."}

    @app.get("/api/health")
    def health_check() -> Dict[str, Any]:
        """Kiểm tra trạng thái máy chủ và mô hình LLM hiện tại."""
        config = load_yaml_config()
        model_name = config.get("llm", {}).get("model_name", "qwen/qwen3.8-27b")
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
        pipeline = get_rag_pipeline()
        if not pipeline:
            raise HTTPException(status_code=500, detail="RAG Pipeline chưa sẵn sàng.")

        with Timer() as timer:
            user_query = UserQuery(
                query_text=request.query,
                session_id=request.session_id,
            )
            retrieval_top_k = request.top_k if request.top_k is not None else getattr(pipeline.retriever, "default_top_k", 5)
            response = pipeline.query(user_query, top_k=retrieval_top_k)

        sources_data = []
        for ctx in response.sources:
            meta = dict(ctx.metadata or {})
            # Giải mã tiêu đề chuẩn xác từ metadata hoặc văn bản
            title = meta.get("title")
            if not title or str(title).startswith("Tài liệu UET ("):
                if "metadata" in meta and isinstance(meta["metadata"], str):
                    try:
                        import json
                        nested = json.loads(meta["metadata"])
                        if isinstance(nested, dict) and nested.get("title"):
                            title = nested["title"]
                            meta["title"] = title
                    except Exception:
                        pass
                if not title or str(title).startswith("Tài liệu UET ("):
                    import re
                    first_line = (ctx.text or "").strip().split("\n")[0]
                    m = re.match(r"^\[(.*?)\]", first_line) or re.match(r"^#+\s*(.*)", first_line)
                    if m and len(m.group(1).strip()) > 3:
                        title = m.group(1).strip()
                        meta["title"] = title
                    elif meta.get("source"):
                        title = meta["source"]
                        meta["title"] = title
            search_type = meta.get("search_type")
            search_source = meta.get("search_source")
            if not search_type:
                bm25_rank = meta.get("bm25_rank")
                vec_rank = meta.get("vector_rank")
                if bm25_rank is not None and vec_rank is not None:
                    search_type = "hybrid"
                    search_source = "BM25 + Vector Search"
                elif bm25_rank is not None:
                    search_type = "bm25"
                    search_source = "BM25"
                elif vec_rank is not None:
                    search_type = "vector"
                    search_source = "Vector Search"
                else:
                    search_type = "hybrid"
                    search_source = "Hybrid"

            meta["search_type"] = search_type
            meta["search_source"] = search_source

            sources_data.append(
                SourceItem(
                    chunk_id=ctx.chunk_id,
                    text=ctx.text,
                    similarity_score=ctx.similarity_score,
                    rank=ctx.rank,
                    search_type=search_type,
                    search_source=search_source,
                    metadata=meta,
                )
            )

        return ChatResponse(
            query=request.query,
            answer=response.answer,
            sources=sources_data,
            latency_seconds=round(timer.elapsed, 3),
            model_name=response.model_name,
        )

    return app


def run_web(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    """Khởi chạy API Server và Giao diện Web với FastAPI và Uvicorn."""
    try:
        import uvicorn
        import webbrowser
        import threading

        app = create_app()
        if app is None:
            print("Vui lòng cài đặt: pip install fastapi uvicorn")
            return

        url = f"http://{host}:{port}"
        print(f"\n🌐 GIAO DIỆN WEB LOCAL ĐANG CHẠY TẠI: {url}")
        print(f"📚 Tài liệu REST API Swagger:        {url}/docs")
        print("💡 Nhấn Ctrl+C để dừng server.\n")

        if open_browser:
            threading.Timer(1.2, lambda: webbrowser.open(url)).start()

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("Lỗi: Cần cài đặt uvicorn để chạy server (pip install uvicorn)")


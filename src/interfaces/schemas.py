"""Pydantic request and response schemas for the UET Chatbot REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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

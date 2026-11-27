"""Groq Whisper speech-to-text transcription and hallucination filtering."""

from __future__ import annotations

import base64
import re
from typing import Any, List, Optional

from src.interfaces.error_handling import classify_exception
from src.interfaces.schemas import VoiceTranscribeRequest, VoiceTranscribeResponse
from src.utils.helpers import Timer, get_logger

logger = get_logger("api")

WHISPER_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
MIN_AUDIO_BYTES = 400
MAX_AUDIO_BYTES = 12 * 1024 * 1024

HALLUCINATION_SUBSTRINGS = (
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

REPEATED_PHRASE_RE = re.compile(
    r"(?:^|[\s,.!?])([^\s,.!?]+(?:\s+[^\s,.!?]+){0,2})(?:\s*[,.\s]+\1){3,}(?:$|[\s,.!?])",
    re.IGNORECASE,
)


def is_whisper_hallucination(text: str) -> bool:
    """Nhận diện các câu ảo giác phổ biến của Whisper khi gặp khoảng lặng hoặc tạp âm."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return True
    if not re.search(r"[0-9A-Za-zÀ-ỹ]", cleaned):
        return True
    lowered = cleaned.lower()
    if any(marker in lowered for marker in HALLUCINATION_SUBSTRINGS):
        return True
    if REPEATED_PHRASE_RE.search(lowered):
        return True
    return False


def extract_clean_transcript(transcription: Any) -> str:
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
            if is_whisper_hallucination(seg_text):
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

    if is_whisper_hallucination(text):
        return ""
    return text


def guess_audio_ext(mime_type: str) -> str:
    """Map an audio MIME type to a file extension suitable for Whisper upload."""
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


def resolve_stt_groq_keys(pipeline: Any) -> List[str]:
    """Retrieve Groq API keys from the active pipeline or environment."""
    llm = getattr(pipeline, "llm", None) if pipeline else None
    keys = list(getattr(llm, "groq_keys", None) or [])
    if keys:
        return keys
    try:
        from src.llm.llm_client import UETLLMClient

        return list(UETLLMClient._load_keys_for_provider("GROQ"))
    except Exception:
        return []


def transcribe_voice(request: VoiceTranscribeRequest, pipeline: Any) -> VoiceTranscribeResponse:
    """Validate base64 audio and transcribe Vietnamese speech via Groq Whisper."""
    from fastapi import HTTPException

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

    if len(audio_bytes) < MIN_AUDIO_BYTES:
        return VoiceTranscribeResponse(text="", model_name="none", latency_seconds=0.0)

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "Đoạn ghi âm quá dài vượt quá giới hạn cho phép. Bạn vui lòng nói câu ngắn hơn nhé.",
                "error_type": "token_limit",
            },
        )

    groq_keys = resolve_stt_groq_keys(pipeline)
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
    ext = guess_audio_ext(clean_mime)
    lang = (request.language or "vi").strip() or "vi"

    last_exc: Optional[Exception] = None
    with Timer() as timer:
        for model_name in WHISPER_MODELS:
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
                    text = extract_clean_transcript(transcription)
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
        classified = classify_exception(last_exc)
        raise HTTPException(
            status_code=classified["status_code"],
            detail={
                "message": classified["message"],
                "error_type": classified["error_type"],
            },
        )

    return VoiceTranscribeResponse(text="", model_name="none", latency_seconds=0.0)

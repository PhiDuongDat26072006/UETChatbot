"""Server-Sent Events (SSE) chat streaming generators for modular and agentic pipelines."""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, Optional

from src.base import UserQuery
from src.interfaces.error_handling import (
    ERROR_STREAM_PREFIXES,
    build_sources_data,
    classify_exception,
    detect_and_classify_llm_error,
    is_no_answer_or_out_of_domain,
    select_format_rules,
)
from src.utils.helpers import Timer, get_logger

logger = get_logger("api")

SSE_DONE = "data: [DONE]\n\n"


def sse_event(payload: Dict[str, Any]) -> str:
    """Serialize a payload dictionary into a single SSE data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _error_frames(error_type: str, message: str) -> Generator[str, None, None]:
    yield sse_event({"type": "error", "error_type": error_type, "message": message})
    yield SSE_DONE


def stream_agentic_chat(
    pipeline: Any,
    user_query: UserQuery,
    retrieval_top_k: int,
    voice_mode: bool,
    timer: Timer,
) -> Generator[str, None, None]:
    """Stream SSE events from an Agentic LangGraph pipeline."""
    format_rules = select_format_rules(voice_mode)
    accumulated_agent = ""
    agent_model_name = getattr(getattr(pipeline, "llm", None), "model_name", "") or "jev-agentic-rag"

    for event in pipeline.stream_events(
        user_query,
        top_k=retrieval_top_k,
        extra_format_rules=format_rules,
        voice_mode=voice_mode,
    ):
        ev_type = event.get("type")
        if ev_type == "status":
            yield sse_event(
                {
                    "type": "status",
                    "stage": event.get("node", ""),
                    "message": event.get("message", ""),
                }
            )
        elif ev_type == "sources":
            sources_data = build_sources_data(event.get("sources"))
            agent_model_name = event.get("model_name") or agent_model_name
            yield sse_event(
                {
                    "type": "sources",
                    "sources": [s.model_dump() for s in sources_data],
                    "model_name": agent_model_name,
                }
            )
        elif ev_type == "token":
            chunk = event.get("content") or ""
            if not chunk:
                continue
            accumulated_agent += chunk
            detected = (
                detect_and_classify_llm_error(accumulated_agent)
                or detect_and_classify_llm_error(chunk)
            )
            if detected is not None:
                logger.error(
                    "LLM error detected in Agentic /api/chat/stream [%s] model=%s raw=%s",
                    detected["error_type"],
                    agent_model_name,
                    accumulated_agent,
                )
                yield from _error_frames(detected["error_type"], detected["message"])
                return
            yield sse_event({"type": "token", "content": chunk})
        elif ev_type == "done":
            final_ans = event.get("answer") or accumulated_agent
            yield sse_event(
                {
                    "type": "done",
                    "latency_seconds": round(timer.elapsed, 3),
                    "hide_sources": is_no_answer_or_out_of_domain(final_ans),
                }
            )
            yield SSE_DONE
            return

    yield sse_event(
        {
            "type": "done",
            "latency_seconds": round(timer.elapsed, 3),
            "hide_sources": is_no_answer_or_out_of_domain(accumulated_agent),
        }
    )
    yield SSE_DONE


def stream_modular_chat(
    pipeline: Any,
    user_query: UserQuery,
    retrieval_top_k: int,
    voice_mode: bool,
    timer: Timer,
) -> Generator[str, None, None]:
    """Stream SSE events from the standard modular Hybrid RAG pipeline."""
    retrieve_kwargs: Dict[str, Any] = {
        "query": user_query,
        "top_k": retrieval_top_k,
    }
    if user_query.filters is not None:
        retrieve_kwargs["filters"] = user_query.filters

    contexts = pipeline.retriever.retrieve(**retrieve_kwargs)
    sources_data = build_sources_data(contexts)
    model_name = getattr(getattr(pipeline, "llm", None), "model_name", "") or ""

    yield sse_event(
        {
            "type": "sources",
            "sources": [s.model_dump() for s in sources_data],
            "model_name": model_name,
        }
    )

    augmented_prompt = pipeline.prompt_augmenter.augment(
        query=user_query,
        contexts=contexts,
    )
    format_rules = select_format_rules(voice_mode)
    if hasattr(augmented_prompt, "model_copy") and isinstance(
        getattr(augmented_prompt, "formatted_prompt", None), str
    ):
        augmented_prompt = augmented_prompt.model_copy(
            update={"formatted_prompt": augmented_prompt.formatted_prompt + format_rules}
        )

    accumulated = ""
    pending_prefix = ""
    prefix_checked = False

    for chunk in pipeline.llm.generate_stream(augmented_prompt):
        if not chunk:
            continue
        accumulated += chunk

        detected = (
            detect_and_classify_llm_error(accumulated)
            or detect_and_classify_llm_error(chunk)
        )
        if detected is not None:
            logger.error(
                "LLM error detected in POST /api/chat/stream [%s] model=%s raw=%s",
                detected["error_type"],
                model_name,
                accumulated,
            )
            yield from _error_frames(detected["error_type"], detected["message"])
            return

        if not prefix_checked:
            pending_prefix += chunk
            stripped = pending_prefix.lstrip().lower()
            maybe_error_prefix = any(
                header.startswith(stripped) or stripped.startswith(header)
                for header in ERROR_STREAM_PREFIXES
            )
            if maybe_error_prefix and len(pending_prefix) < 100:
                continue

            prefix_checked = True
            yield sse_event({"type": "token", "content": pending_prefix})
            pending_prefix = ""
        else:
            yield sse_event({"type": "token", "content": chunk})

    if pending_prefix:
        detected = detect_and_classify_llm_error(pending_prefix)
        if detected is not None:
            logger.error(
                "LLM error detected at end of stream [%s] model=%s raw=%s",
                detected["error_type"],
                model_name,
                pending_prefix,
            )
            yield from _error_frames(detected["error_type"], detected["message"])
            return
        yield sse_event({"type": "token", "content": pending_prefix})

    yield sse_event(
        {
            "type": "done",
            "latency_seconds": round(timer.elapsed, 3),
            "hide_sources": is_no_answer_or_out_of_domain(accumulated),
        }
    )
    yield SSE_DONE


def build_chat_stream_generator(
    pipeline: Any,
    user_query: UserQuery,
    request_top_k: Optional[int],
    voice_mode: bool,
) -> Generator[str, None, None]:
    """Create the top-level SSE generator with exception classification."""
    with Timer() as timer:
        try:
            retrieval_top_k = (
                request_top_k
                if request_top_k is not None
                else getattr(pipeline.retriever, "default_top_k", 10)
            )
            if hasattr(pipeline, "agent_graph") and hasattr(pipeline, "stream_events"):
                yield from stream_agentic_chat(
                    pipeline, user_query, retrieval_top_k, voice_mode, timer
                )
                return

            yield from stream_modular_chat(
                pipeline, user_query, retrieval_top_k, voice_mode, timer
            )
        except Exception as exc:
            logger.exception("Lỗi trong quá trình streaming /api/chat/stream: %s", exc)
            classified = classify_exception(exc)
            yield from _error_frames(classified["error_type"], classified["message"])

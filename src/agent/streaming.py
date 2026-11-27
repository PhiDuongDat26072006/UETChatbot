"""LangGraph update streaming helper for Agentic RAG."""

from __future__ import annotations

import time
from typing import Any, Dict, Generator, List, Optional

from src.agent.helpers import NODE_STATUS_MESSAGES, chunk_answer_for_stream
from src.agent.state import AgentState
from src.base import RetrievedContext
from src.decision import AgentResponse, Route


def stream_agent_graph(
    app: Any,
    initial_state: AgentState,
    *,
    recursion_limit: int,
    llm_client: Any = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Execute a compiled LangGraph app with ``stream_mode='updates'``, emitting node status
    events and releasing ``sources`` + chunked ``token`` + ``done`` only for the final answer.
    """
    start_t = time.time()
    final_response: Optional[AgentResponse] = None
    final_contexts: List[RetrievedContext] = []
    final_route: Optional[Route] = None
    final_retry: int = 0
    final_answer: str = ""

    for update in app.stream(
        initial_state,
        config={"recursion_limit": recursion_limit},
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue
        for node_name, node_output in update.items():
            if not isinstance(node_output, dict):
                continue

            if node_output.get("route") is not None:
                final_route = node_output["route"]
            if "retry" in node_output and node_output["retry"] is not None:
                final_retry = int(node_output["retry"])
            if node_output.get("contexts") is not None:
                final_contexts = list(node_output["contexts"] or [])
            if node_output.get("answer") is not None:
                final_answer = str(node_output["answer"])
            if node_output.get("response") is not None:
                final_response = node_output["response"]
                final_answer = final_response.answer or ""
                final_contexts = list(final_response.contexts or [])

            if node_name in NODE_STATUS_MESSAGES:
                msg = NODE_STATUS_MESSAGES[node_name]
                if node_name == "followup_query_node" and final_retry > 0:
                    msg = f"Đang tìm kiếm bổ sung thông tin (lần {final_retry})..."
                yield {
                    "type": "status",
                    "node": node_name,
                    "message": msg,
                }

    status_val = final_response.status if final_response else "ok"
    state_val = final_response.state if final_response else (1 if final_contexts else 0)
    route_val = (
        final_response.route.value
        if (final_response and final_response.route)
        else (final_route.value if final_route else Route.QUALIFIED_QUESTION.value)
    )
    model_name = getattr(llm_client, "model_name", None) or "jev-agentic-rag-v3"

    yield {
        "type": "sources",
        "sources": final_contexts,
        "model_name": model_name,
        "route": route_val,
        "status": status_val,
    }

    for chunk in chunk_answer_for_stream(final_answer):
        yield {"type": "token", "content": chunk}

    yield {
        "type": "done",
        "status": status_val,
        "state": state_val,
        "route": route_val,
        "retry": final_retry,
        "answer": final_answer,
        "sources": final_contexts,
        "model_name": model_name,
        "latency_seconds": round(time.time() - start_t, 3),
    }

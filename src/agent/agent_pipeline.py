"""
src/agent/agent_pipeline.py - Đóng gói hệ thống Agentic RAG JEV v3 thành Pipeline kế thừa BaseRAGPipeline.
Đảm bảo tương thích 100% với giao diện Web (FastAPI SSE Streaming), CLI và bộ Benchmark.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Optional

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.base import (
    BaseRAGPipeline,
    BaseEmbeddingModel,
    BaseVectorStore,
    BasePromptAugmenter,
    BaseLLM,
    UserQuery,
    Response,
)
from src.agent.graph import UETAgentGraph
from src.decision import DecisionClient, AgentResponse, Route
from src.utils.helpers import Timer, get_logger, load_yaml_config

logger = get_logger("agent_pipeline")


class UETAgenticRAGPipeline(BaseRAGPipeline):
    """
    [CỖ MÁY ONLINE AGENTIC RAG VỚI JEV v3]:
    Kế thừa BaseRAGPipeline từ src/base.py, điều phối qua Đồ thị LangGraph v3 ("Generate -> Evaluate -> Follow-up").
    """

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        prompt_augmenter: Optional[BasePromptAugmenter] = None,
        llm: Optional[BaseLLM] = None,
        retriever=None,
        decision_client: Optional[DecisionClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or load_yaml_config()

        from src.embeddings import UETEmbedder
        from src.vectordb import UETVectorStore
        from src.prompts import UETPromptAugmenter
        from src.llm import UETLLMClient

        emb = embedding_model or UETEmbedder()
        vs = vector_store or UETVectorStore(persist_dir=_BASE_DIR / "database" / "vector_db", embedding_model=emb)
        pa = prompt_augmenter or UETPromptAugmenter()
        client_llm = llm or UETLLMClient()

        super().__init__(
            embedding_model=emb,
            vector_store=vs,
            prompt_augmenter=pa,
            llm=client_llm,
        )

        if retriever is None:
            from src.retrieval import UETRetriever
            retriever = UETRetriever(
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
            )
        self.retriever = retriever

        self.agent_graph = UETAgentGraph(
            retriever=self.retriever,
            llm_client=self.llm,
            prompt_augmenter=self.prompt_augmenter,
            decision_client=decision_client,
            config=cfg,
        )
        logger.info("Khởi tạo thành công UETAgenticRAGPipeline với JEV LangGraph v3.")

    def query(
        self,
        query: UserQuery,
        top_k: Optional[int] = None,
        extra_format_rules: Optional[str] = None,
        voice_mode: bool = False,
    ) -> Response:
        """Thực thi câu hỏi thông qua Đồ thị trạng thái Agentic RAG JEV v3."""
        with Timer() as timer:
            final_state = self.agent_graph.run(
                query=query.query_text,
                user_query=query,
                session_id=query.session_id,
                top_k=top_k,
                extra_format_rules=extra_format_rules,
                voice_mode=voice_mode,
            )

            agent_resp: Optional[AgentResponse] = final_state.get("response")
            answer = final_state.get("answer", "")
            contexts = final_state.get("contexts") or []

            status_str = agent_resp.status if agent_resp else "ok"
            state_val = agent_resp.state if agent_resp else (1 if contexts else 0)
            route_val = (
                agent_resp.route.value
                if (agent_resp and agent_resp.route)
                else (
                    final_state.get("route").value
                    if final_state.get("route")
                    else Route.QUALIFIED_QUESTION.value
                )
            )

            metadata = {
                "status": status_str,
                "state": state_val,
                "route": route_val,
                "retry": final_state.get("retry", 0),
                "missing_info": final_state.get("missing_info"),
                "followup_queries": final_state.get("followup_queries", []),
                "used_queries": final_state.get("used_queries", []),
                "execution_steps": final_state.get("execution_steps", []),
                "trace": final_state.get("trace", []),
            }

            model_name = getattr(self.llm, "model_name", "jev-agentic-rag-v3")

            response = Response(
                query_id=query.query_id,
                answer=answer,
                sources=contexts,
                model_name=model_name,
                latency_seconds=round(timer.elapsed, 3),
                metadata=metadata,
            )

        return response

    def stream_events(
        self,
        query: UserQuery,
        top_k: Optional[int] = None,
        extra_format_rules: Optional[str] = None,
        voice_mode: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Phát luồng sự kiện tiến độ theo từng node LangGraph (status) và CHỈ xuất
        `sources` + `token` của phản hồi cuối cùng đã qua `answer_eval_node`.
        """
        yield from self.agent_graph.run_stream(
            query=query.query_text,
            user_query=query,
            session_id=query.session_id,
            top_k=top_k,
            extra_format_rules=extra_format_rules,
            voice_mode=voice_mode,
        )

    def query_stream(
        self,
        query: UserQuery,
        top_k: Optional[int] = None,
        extra_format_rules: Optional[str] = None,
        voice_mode: bool = False,
    ) -> Generator[str, None, None]:
        """Hỗ trợ phản hồi dạng luồng (Streaming) chỉ trả về các token của câu trả lời cuối cùng đã duyệt."""
        for event in self.stream_events(
            query=query,
            top_k=top_k,
            extra_format_rules=extra_format_rules,
            voice_mode=voice_mode,
        ):
            if event.get("type") == "token" and event.get("content"):
                yield event["content"]

    def close_session(self, session_id: Optional[str]) -> None:
        """Đóng kết nối HTTP Session (JEV + LLM) khi kết thúc trò chuyện."""
        if hasattr(self.agent_graph, "close_session"):
            self.agent_graph.close_session(session_id)

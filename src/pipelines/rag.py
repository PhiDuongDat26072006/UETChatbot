"""Online RAG pipeline serving and mode description helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, List, Optional

from src.base import (
    BaseEmbeddingModel,
    BaseLLM,
    BasePromptAugmenter,
    BaseRAGPipeline,
    BaseVectorStore,
    Response,
    UserQuery,
)
from src.embeddings import UETEmbedder
from src.llm import UETLLMClient
from src.pipelines.helpers import get_pipeline_attr
from src.prompts import UETPromptAugmenter
from src.utils.helpers import BASE_DIR, Timer
from src.vectordb import UETVectorStore

if TYPE_CHECKING:
    from src.retrieval import UETRetriever


class UETRAGPipeline(BaseRAGPipeline):
    """
    [CỖ MÁY ONLINE]: Phục vụ Chatbot tra cứu thông tin học vụ.
    Điều phối luồng: Nhận câu hỏi -> Truy xuất ngữ cảnh (UETRetriever: BM25 + Vector -> Hybrid RRF -> Rerank -> Top-K)
                   -> Ghép prompt (UETPromptAugmenter) -> Gọi LLM (UETLLMClient) -> Trả lời.
    """

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        prompt_augmenter: Optional[BasePromptAugmenter] = None,
        llm: Optional[BaseLLM] = None,
        retriever: Optional[UETRetriever] = None,
    ):
        base_dir = Path(get_pipeline_attr("BASE_DIR", BASE_DIR))
        emb = embedding_model or get_pipeline_attr("UETEmbedder", UETEmbedder)()
        vs = vector_store or get_pipeline_attr("UETVectorStore", UETVectorStore)(
            persist_dir=base_dir / "database" / "vector_db",
            embedding_model=emb,
        )
        pa = prompt_augmenter or get_pipeline_attr("UETPromptAugmenter", UETPromptAugmenter)()
        client_llm = llm or get_pipeline_attr("UETLLMClient", UETLLMClient)()

        super().__init__(
            embedding_model=emb,
            vector_store=vs,
            prompt_augmenter=pa,
            llm=client_llm,
        )

        if retriever is None:
            from src.retrieval import UETRetriever as RetrieverCls

            retriever = RetrieverCls(
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
            )
        self.retriever = retriever

    def query(
        self,
        query: UserQuery,
        top_k: Optional[int] = None,
        engine: Optional[str] = None,
    ) -> Response:
        """
        Thực thi quy trình trả lời câu hỏi:
        1. Gọi UETRetriever: Hỗ trợ động cơ vector, lightrag, hybrid hoặc auto
        2. Ghép ngữ cảnh vào Prompt (AugmentedPrompt) theo chuẩn BaseRAGPipeline
        3. Gọi LLM sinh câu trả lời
        """
        k = top_k if top_k is not None else getattr(self.retriever, "default_top_k", 10)
        with Timer() as timer:
            contexts = self.retriever.retrieve(
                query=query,
                top_k=k,
                filters=query.filters,
                engine=engine,
            )
            augmented_prompt = self.prompt_augmenter.augment(query=query, contexts=contexts)
            response = self.llm.generate(augmented_prompt)

        response.latency_seconds = round(timer.elapsed, 3)
        return response

    def query_stream(
        self,
        query: UserQuery,
        top_k: Optional[int] = None,
        engine: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Sinh phản hồi dạng luồng (streaming) cho giao diện Web/Chatbot."""
        k = top_k if top_k is not None else getattr(self.retriever, "default_top_k", 10)
        contexts = self.retriever.retrieve(
            query=query,
            top_k=k,
            filters=query.filters,
            engine=engine,
        )
        augmented_prompt = self.prompt_augmenter.augment(query=query, contexts=contexts)
        yield from self.llm.generate_stream(augmented_prompt)


def describe_pipeline_mode(pipeline: Any) -> str:
    """Trả về nhãn chế độ đang chạy: Hybrid / Graph + Hybrid / Agentic + Hybrid / Graph + Hybrid + Agentic."""
    is_agentic = hasattr(pipeline, "agent_graph")
    retriever = getattr(pipeline, "retriever", None)
    enable_graph = bool(getattr(retriever, "enable_graph", False))
    graph_loaded = getattr(retriever, "graph_index", None) is not None and (
        getattr(retriever.graph_index, "count", lambda: 0)() > 0
    )
    parts: List[str] = []
    if enable_graph:
        parts.append("Graph" if graph_loaded else "Graph (chưa có graph_index.pkl)")
    parts.append("Hybrid")
    if is_agentic:
        parts.append("Agentic")
    return " + ".join(parts)

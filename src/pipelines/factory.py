"""Factory and lazy singleton management for UET RAG & Ingestion pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from src.embeddings import UETEmbedder
from src.llm import UETLLMClient
from src.pipelines.helpers import get_pipeline_attr
from src.pipelines.ingestion import UETIngestionPipeline
from src.pipelines.rag import UETRAGPipeline
from src.prompts import UETPromptAugmenter
from src.utils.helpers import BASE_DIR, get_logger, load_yaml_config
from src.vectordb import UETVectorStore

if TYPE_CHECKING:
    from src.retrieval import UETRetriever

logger = get_logger("pipeline")

_active_rag_pipeline: Optional[UETRAGPipeline] = None
_active_agentic_rag_pipeline: Optional[Any] = None
_active_ingestion_pipeline: Optional[UETIngestionPipeline] = None


def _build_core_serving_components() -> Tuple[Any, Any, Any, Any, Any]:
    """Instantiate the shared embedding model, vector store, prompt augmenter, LLM, and retriever."""
    from src.retrieval import UETRetriever

    base_dir = Path(get_pipeline_attr("BASE_DIR", BASE_DIR))
    embedder_cls = get_pipeline_attr("UETEmbedder", UETEmbedder)
    store_cls = get_pipeline_attr("UETVectorStore", UETVectorStore)
    prompt_cls = get_pipeline_attr("UETPromptAugmenter", UETPromptAugmenter)
    llm_cls = get_pipeline_attr("UETLLMClient", UETLLMClient)

    embedding_model = embedder_cls()
    vector_store = store_cls(persist_dir=base_dir / "database" / "vector_db", embedding_model=embedding_model)
    prompt_augmenter = prompt_cls()
    llm = llm_cls()
    retriever = UETRetriever(embedding_model=embedding_model, vector_store=vector_store)
    return embedding_model, vector_store, prompt_augmenter, llm, retriever


def get_rag_pipeline() -> UETRAGPipeline:
    """Khởi tạo hoặc lấy RAG Pipeline đang hoạt động."""
    global _active_rag_pipeline
    if _active_rag_pipeline is not None:
        return _active_rag_pipeline

    embedding_model, vector_store, prompt_augmenter, llm, retriever = _build_core_serving_components()
    if vector_store.count() == 0:
        logger.info("Chưa có cơ sở dữ liệu vector_db/, nạp 5 chunk quy chế mẫu dự phòng...")
    else:
        logger.info(f"Đã kết nối và sử dụng {vector_store.count()} vectors thực tế từ database/vector_db/")

    rag_cls = get_pipeline_attr("UETRAGPipeline", UETRAGPipeline)
    _active_rag_pipeline = rag_cls(
        embedding_model=embedding_model,
        vector_store=vector_store,
        prompt_augmenter=prompt_augmenter,
        llm=llm,
        retriever=retriever,
    )
    logger.info("Khởi tạo thành công UETRAGPipeline tích hợp đầy đủ trong src/pipeline.py")
    return _active_rag_pipeline


def get_agentic_rag_pipeline() -> Any:
    """Khởi tạo hoặc lấy Agentic RAG Pipeline (LangGraph State Machine) đang hoạt động."""
    global _active_agentic_rag_pipeline
    if _active_agentic_rag_pipeline is not None:
        return _active_agentic_rag_pipeline

    from src.agent import UETAgenticRAGPipeline

    embedding_model, vector_store, prompt_augmenter, llm, retriever = _build_core_serving_components()
    _active_agentic_rag_pipeline = UETAgenticRAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        prompt_augmenter=prompt_augmenter,
        llm=llm,
        retriever=retriever,
    )
    logger.info("Khởi tạo thành công UETAgenticRAGPipeline tích hợp LangGraph trong src/pipeline.py")
    return _active_agentic_rag_pipeline


def get_active_pipeline(mode: Optional[str] = None) -> Any:
    """
    Lấy pipeline theo tham số mode, biến môi trường RAG_MODE hoặc config.yaml ('modular' hoặc 'agentic').
    Mặc định: 'modular' (Hybrid RAG); chỉ bật Agentic khi có cờ --agentic hoặc cấu hình rõ.
    """
    env_mode = os.environ.get("RAG_MODE")
    cfg = load_yaml_config()
    chosen_mode = (mode or env_mode or cfg.get("agent", {}).get("mode") or "modular").strip().lower()
    if chosen_mode == "agentic":
        return get_agentic_rag_pipeline()
    return get_rag_pipeline()


def get_ingestion_pipeline() -> UETIngestionPipeline:
    """Khởi tạo hoặc lấy Ingestion Pipeline đang hoạt động."""
    global _active_ingestion_pipeline
    if _active_ingestion_pipeline is not None:
        return _active_ingestion_pipeline

    ingest_cls = get_pipeline_attr("UETIngestionPipeline", UETIngestionPipeline)
    _active_ingestion_pipeline = ingest_cls()
    return _active_ingestion_pipeline


def get_retriever() -> UETRetriever:
    """[Phân hệ Retrieval]: Khởi tạo và cung cấp UETRetriever điều phối tìm kiếm đa tầng."""
    from src.retrieval import UETRetriever

    base_dir = Path(get_pipeline_attr("BASE_DIR", BASE_DIR))
    embedder_cls = get_pipeline_attr("UETEmbedder", UETEmbedder)
    store_cls = get_pipeline_attr("UETVectorStore", UETVectorStore)

    embedding_model = embedder_cls()
    vector_store = store_cls(persist_dir=base_dir / "database" / "vector_db", embedding_model=embedding_model)
    return UETRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


def reset_pipelines() -> None:
    """Reset cached pipeline singletons (primarily for tests)."""
    global _active_rag_pipeline, _active_agentic_rag_pipeline, _active_ingestion_pipeline
    _active_rag_pipeline = None
    _active_agentic_rag_pipeline = None
    _active_ingestion_pipeline = None

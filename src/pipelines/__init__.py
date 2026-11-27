"""Modular pipeline orchestration package for UET Chatbot."""

from src.pipelines.cli import run_ingest_cli, run_ingest_status_cli
from src.pipelines.factory import (
    get_active_pipeline,
    get_agentic_rag_pipeline,
    get_ingestion_pipeline,
    get_rag_pipeline,
    get_retriever,
    reset_pipelines,
)
from src.pipelines.helpers import UETPreprocessor, extract_faculty_code, load_processed_docs
from src.pipelines.ingestion import UETIngestionPipeline
from src.pipelines.rag import UETRAGPipeline, describe_pipeline_mode

__all__ = [
    "UETPreprocessor",
    "UETIngestionPipeline",
    "UETRAGPipeline",
    "extract_faculty_code",
    "load_processed_docs",
    "describe_pipeline_mode",
    "get_rag_pipeline",
    "get_agentic_rag_pipeline",
    "get_active_pipeline",
    "get_ingestion_pipeline",
    "get_retriever",
    "reset_pipelines",
    "run_ingest_cli",
    "run_ingest_status_cli",
]

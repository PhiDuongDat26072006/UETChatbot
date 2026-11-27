"""
src/pipeline.py - Backward-compatible facade for UET Chatbot pipeline orchestrators.

Core implementations live in ``src/pipelines/``:
- ``src/pipelines/helpers.py``: preprocessing and faculty/doc loading helpers
- ``src/pipelines/ingestion.py``: ``UETIngestionPipeline`` (multi-stage offline indexing)
- ``src/pipelines/rag.py``: ``UETRAGPipeline`` & ``describe_pipeline_mode``
- ``src/pipelines/factory.py``: lazy singleton factories (``get_rag_pipeline``, ``get_active_pipeline``, etc.)
- ``src/pipelines/cli.py``: terminal CLI runners (``run_ingest_cli``, ``run_ingest_status_cli``)
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.base import UserQuery
from src.chunking import UETChunker
from src.embeddings import UETEmbedder
from src.ingestion import DataManifestTracker, UETDataLoader
from src.llm import UETLLMClient
from src.pipelines import (
    UETIngestionPipeline,
    UETPreprocessor,
    UETRAGPipeline,
    describe_pipeline_mode,
    get_active_pipeline,
    get_agentic_rag_pipeline,
    get_ingestion_pipeline,
    get_rag_pipeline,
    get_retriever,
    reset_pipelines,
    run_ingest_cli,
    run_ingest_status_cli,
)
from src.prompts import UETPromptAugmenter
from src.utils.helpers import BASE_DIR
from src.vectordb import UETVectorStore

__all__ = [
    "BASE_DIR",
    "DataManifestTracker",
    "UETDataLoader",
    "UETChunker",
    "UETEmbedder",
    "UETVectorStore",
    "UETPromptAugmenter",
    "UETLLMClient",
    "UETPreprocessor",
    "UETIngestionPipeline",
    "UETRAGPipeline",
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

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n🔍 ĐANG CHẠY KIỂM THỬ END-TO-END QUA SRC/PIPELINE.PY...")
    rag = get_rag_pipeline()
    test_query = UserQuery(query_text="Điều kiện xét học bổng xuất sắc tại UET là gì?")
    res = rag.query(test_query)

    print("\n" + "=" * 60)
    print("Câu hỏi:", test_query.query_text)
    print("Mô hình:", res.model_name)
    print(f"Thời gian: {res.latency_seconds}s")
    print("Trả lời:", res.answer)
    print(f"Số lượng nguồn trích dẫn: {len(res.sources)}")
    for s in res.sources[:3]:
        title = s.metadata.get("title") or s.metadata.get("source") or "Văn bản UET"
        print(f" - [{int(s.similarity_score * 100)}%] {title}")
    print("=" * 60)

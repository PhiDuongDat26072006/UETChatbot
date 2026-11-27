"""
src/evaluation/ragas_eval.py - Ragas evaluation integration and LangChain embedding
adapter with NaN-preserving score cleanup.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.base import BaseRAGPipeline, EvaluationSample, Response, UserQuery
from src.embeddings.embedder import UETEmbedder
from src.llm.llm_client import UETLLMClient
from src.utils.helpers import get_logger

logger = get_logger("evaluation")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _ensure_ragas_compat() -> None:
    """
    Vá tương thích tạm cho ragas==0.4.3 + langchain-community>=0.4.x:
    ragas/llms/base.py import `ChatVertexAI` từ `langchain_community.chat_models.vertexai` đã bị gỡ.
    """
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        __import__(mod_name)
    except ModuleNotFoundError:
        stub = types.ModuleType(mod_name)

        class ChatVertexAI:  # pragma: no cover - compatibility shim
            def __init__(self, *args: Any, **kwargs: Any):
                raise RuntimeError(
                    "ChatVertexAI stub: langchain_community.chat_models.vertexai đã bị gỡ; "
                    "hãy dùng OpenAI-compatible client với Ragas."
                )

        stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        sys.modules[mod_name] = stub
        parent = sys.modules.get("langchain_community.chat_models")
        if parent is not None:
            setattr(parent, "vertexai", stub)


def _clean_ragas_value(value: Any) -> Optional[float]:
    """Chuyển NaN thành None thay vì ép sai thành 0.0."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return round(num, 4)


try:
    from langchain_core.embeddings import Embeddings

    class UETLangchainEmbeddings(Embeddings):
        """Adapter bao bọc UETEmbedder thành chuẩn LangChain Embeddings cho Ragas."""

        def __init__(self, embedder: Optional[UETEmbedder] = None):
            self.embedder = embedder or UETEmbedder()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            return self.embedder.embed_texts(texts)

        def embed_query(self, text: str) -> List[float]:
            vectors = self.embedder.embed_texts([text])
            return vectors[0] if vectors else [0.0] * self.embedder.dimension

except ImportError:
    UETLangchainEmbeddings = None  # type: ignore


class UETRagasEvaluator:
    """
    Bộ đánh giá RAG sử dụng thư viện Ragas và LLM-as-a-Judge (Groq / OpenRouter).
    """

    def __init__(
        self,
        model_name: str = "deepseek/deepseek-v4.1-flash-1017",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        embedder: Optional[UETEmbedder] = None,
        max_workers: int = 1,
        timeout: int = 90,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries

        if api_key:
            self.api_key = api_key.strip().strip('"')
        elif "openrouter" in base_url.lower():
            keys = UETLLMClient._load_keys_for_provider("OPENROUTER")
            self.api_key = keys[0] if keys else ""
        else:
            keys = UETLLMClient._load_keys_for_provider("GROQ")
            self.api_key = keys[0] if keys else ""

        if not self.api_key:
            raise ValueError(
                "Không tìm thấy API key cho UETRagasEvaluator (GROQ_API_KEY hoặc OPENROUTER_API_KEY)."
            )

        try:
            _ensure_ragas_compat()
            from openai import OpenAI
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import llm_factory
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.judge_llm = llm_factory(self.model_name, client=self.client)
            self.judge_embeddings = LangchainEmbeddingsWrapper(
                UETLangchainEmbeddings(embedder=embedder)
            )
            self.metrics = [
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
            ]
            logger.info(
                "Đã khởi tạo UETRagasEvaluator với model '%s' (max_workers=%d).",
                self.model_name,
                self.max_workers,
            )
        except ImportError as e:
            raise ImportError(
                f"Thiếu thư viện để chạy Ragas: {e}. Vui lòng cài đặt: pip install ragas datasets openai"
            )

    def evaluate_pipeline(
        self,
        pipeline: BaseRAGPipeline,
        dataset: List[EvaluationSample],
        top_k: int = 5,
        num_samples: Optional[int] = 4,
        output_path: Optional[Union[str, Path]] = None,
        delay_between_queries: float = 1.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Chạy quy trình RAG Pipeline trên tập dữ liệu và đánh giá bằng Ragas."""
        _ensure_ragas_compat()
        from datasets import Dataset
        from ragas import evaluate

        active_dataset = dataset[:num_samples] if (num_samples is not None and num_samples > 0) else dataset
        total_samples = len(active_dataset)
        logger.info(f"Bắt đầu đánh giá Ragas trên {total_samples} mẫu câu hỏi...")
        print(f"\n🚀 Đang chạy truy vấn RAG cho {total_samples} câu hỏi (Top-K={top_k}, Delay={delay_between_queries}s)...")

        user_inputs: List[str] = []
        responses: List[str] = []
        retrieved_contexts: List[List[str]] = []
        references: List[str] = []
        sample_ids: List[str] = []

        for idx, sample in enumerate(active_dataset, 1):
            short_q = (sample.query[:58] + "...") if len(sample.query) > 58 else sample.query
            print(f"[{idx:02d}/{total_samples:02d}] 🔍 Đang tạo câu trả lời: '{short_q}'")
            try:
                resp = pipeline.query(UserQuery(query_text=sample.query), top_k=top_k)
                print(f"        ✅ Hoàn thành [{resp.latency_seconds:.2f}s]")
            except Exception as e:
                logger.warning(f"Lỗi khi truy vấn câu hỏi '{sample.query[:30]}': {e}")
                resp = Response(answer="Không thể tạo câu trả lời do lỗi kết nối LLM.", sources=[])
                print(f"        ⚠️ Lỗi truy vấn: {e}")

            sample_ids.append(sample.id)
            user_inputs.append(sample.query)
            responses.append(resp.answer)
            contexts = [c.text for c in resp.sources] if resp.sources else ["Không tìm thấy ngữ cảnh."]
            retrieved_contexts.append(contexts)
            references.append(sample.ground_truth or "")

            if idx < total_samples and delay_between_queries > 0:
                time.sleep(delay_between_queries)

        hf_dataset = Dataset.from_dict({
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
        })

        eval_kwargs: Dict[str, Any] = {
            "dataset": hf_dataset,
            "metrics": self.metrics,
            "llm": self.judge_llm,
            "embeddings": self.judge_embeddings,
            "raise_exceptions": False,
        }
        try:
            from ragas.run_config import RunConfig

            eval_kwargs["run_config"] = RunConfig(
                max_workers=self.max_workers,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        except Exception:
            pass

        logger.info("Đang gọi Ragas evaluate với mô hình LLM-as-a-Judge...")
        try:
            eval_result = evaluate(**eval_kwargs)
        except TypeError:
            eval_kwargs.pop("run_config", None)
            eval_result = evaluate(**eval_kwargs)

        df = eval_result.to_pandas()

        metric_keys = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
        average_metrics: Dict[str, Optional[float]] = {}
        valid_counts: Dict[str, int] = {}
        nan_counts: Dict[str, int] = {}

        for m in metric_keys:
            if m in df.columns:
                cleaned_vals = [_clean_ragas_value(v) for v in df[m].tolist()]
                valid = [v for v in cleaned_vals if v is not None]
                valid_counts[m] = len(valid)
                nan_counts[m] = len(cleaned_vals) - len(valid)
                average_metrics[m] = round(sum(valid) / len(valid), 4) if valid else None

        detailed_samples: List[Dict[str, Any]] = []
        for i in range(len(active_dataset)):
            sample_dict = {
                "id": sample_ids[i],
                "query": user_inputs[i],
                "generated_answer": responses[i],
                "ground_truth": references[i],
                "retrieved_contexts_count": len(retrieved_contexts[i]),
                "scores": {},
            }
            for m in metric_keys:
                if m in df.columns:
                    sample_dict["scores"][m] = _clean_ragas_value(df.iloc[i][m])
            detailed_samples.append(sample_dict)

        report_data: Dict[str, Any] = {
            "evaluator": "Ragas (LLM-as-a-Judge)",
            "judge_model": self.model_name,
            "total_samples": total_samples,
            "average_metrics": average_metrics,
            "valid_counts": valid_counts,
            "nan_counts": nan_counts,
            "samples": detailed_samples,
        }

        if output_path:
            save_file = Path(output_path)
            _atomic_write_json(save_file, report_data)
            logger.info(f"Đã lưu báo cáo đánh giá Ragas vào: {save_file}")

        return report_data

"""
src/evaluation/evaluator.py - Bộ đánh giá chất lượng RAG toàn trình (E2E Evaluation).

Hỗ trợ đầy đủ:
1. Kiến trúc 2 pha độc lập (Generate -> Score):
   - `generate_predictions`: Pha 1 chạy RAG Pipeline, ghi snapshot đầy đủ contexts + latencies.
   - `score_predictions`: Pha 2 chấm Retrieval + Offline Metrics + LLM-as-a-Judge + Error Taxonomy.
2. `UETEvaluator`: Đánh giá Offline Vietnamese RAG Metrics (+ Semantic bge-m3 + tùy chọn `UETLLMJudgeEvaluator`).
3. `UETRagasEvaluator`: Đánh giá qua Ragas (đã vá tương thích `ChatVertexAI` 0.4.x, `RunConfig`, giữ nguyên `None` khi NaN).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

from src.base import (
    BaseEvaluator,
    BaseRAGPipeline,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    Response,
    RetrievedContext,
    UserQuery,
)
from src.embeddings.embedder import UETEmbedder
from src.evaluation.diagnostics import (
    build_e2e_report,
    classify_sample_error,
    print_e2e_diagnostic_report,
)
from src.evaluation.llm_judge import UETLLMJudgeEvaluator
from src.evaluation.metrics import (
    compute_answer_correctness_offline,
    compute_answer_relevance,
    compute_context_relevance,
    compute_faithfulness,
    compute_numeric_grounding,
    deduplicate_preserve_order,
    hit_rate_at_k,
    is_generation_error_response,
    is_refusal_response,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from src.llm.llm_client import UETLLMClient
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("evaluation")

DOMAIN_FILE_MAP = {
    "overview": ["overview1_qa.json", "overview2_qa.json"],
    "overview1": ["overview1_qa.json"],
    "overview2": ["overview2_qa.json"],
    "general": ["overview1_qa.json", "overview2_qa.json", "general_qa.json"],
    "uet": ["overview2_qa.json", "overview1_qa.json", "uet_qa.json"],
    "test_qa": ["overview1_qa.json", "test_qa_dataset.json"],
    "handbook": ["handbook2_qa.json", "handbook1_qa.json"],
    "handbook1": ["handbook1_qa.json", "handbook.uet.vnu.edu.vn.json"],
    "handbook2": ["handbook2_qa.json", "uet_handbook_qa.json", "handbook_qa.json"],
    "fit": ["fit_qa.json", "fit.uet.vnu.edu.vn.json"],
    "fet": ["fet_qa.json", "fet.uet.vnu.edu.vn.json"],
    "fema": ["fema_qa.json", "fema.uet.vnu.edu.vn.json"],
    "sae": ["sae_qa.json", "sae.uet.vnu.edu.vn.json"],
}


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


# =========================================================================
# 1. BỘ CHUYỂN ĐỔI EMBEDDINGS CHO RAGAS (LANGCHAIN COMPATIBLE)
# =========================================================================

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


# =========================================================================
# 2. BỘ ĐÁNH GIÁ RAGAS (ĐÃ VÁ TƯƠNG THÍCH & KHÔNG ÉP NaN -> 0.0)
# =========================================================================

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


# =========================================================================
# 3. KIẾN TRÚC 2 PHA: PHASE 1 (GENERATE SNAPSHOT) & PHASE 2 (SCORE)
# =========================================================================

def _is_sample_ood(meta: Dict[str, Any]) -> bool:
    return (
        str(meta.get("type", "")).lower() == "out_of_domain"
        or str(meta.get("unit", "")).lower() == "out_of_domain"
        or "out_of_domain" in [str(x).lower() for x in (meta.get("query_types") or [])]
    )


def _serialize_contexts(sources: List[RetrievedContext]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for idx, c in enumerate(sources or [], 1):
        meta = c.metadata if isinstance(getattr(c, "metadata", None), dict) else {}
        entry: Dict[str, Any] = {
            "chunk_id": getattr(c, "chunk_id", None),
            "document_id": meta.get("document_id") or meta.get("doc_id"),
            "rank": getattr(c, "rank", idx),
            "score": round(float(c.similarity_score), 6) if getattr(c, "similarity_score", None) is not None else None,
            "search_source": meta.get("search_source"),
            "title": meta.get("title"),
            "text": getattr(c, "text", ""),
        }
        if meta.get("page") is not None:
            entry["page"] = meta.get("page")
        graph_triples = meta.get("graph_triples") or []
        if graph_triples:
            entry["graph_triples"] = [str(t) for t in graph_triples[:5]]
        serialized.append(entry)
    return serialized


def _format_context_dict_for_eval(c: Dict[str, Any]) -> str:
    """Bổ sung thông tin Trang và Quan hệ Đồ thị Tri thức (nếu có) để Offline/Judge nhìn thấy đủ ngữ cảnh như Generator."""
    prefixes: List[str] = []
    if c.get("page") is not None:
        prefixes.append(f"[Trang {c['page']}]")
    triples = c.get("graph_triples") or []
    if triples:
        prefixes.append(f"[Quan hệ Đồ thị Tri thức: {'; '.join(str(t) for t in triples[:5])}]")
    if not prefixes:
        return str(c.get("text", ""))
    return f"{' '.join(prefixes)}\n{str(c.get('text', '') or '').strip()}"


def _recover_generation_error(
    pipeline: BaseRAGPipeline,
    aug_prompt: Any,
    resp: Response,
) -> Tuple[Response, float]:
    """
    Khi UETLLMClient trả về fallback 429/lỗi dịch vụ:
    1. Thử ngay OpenRouter nếu có OPENROUTER_API_KEY (không tốn thời gian chờ Groq).
    2. Nếu không, đọc thời gian chờ từ thông báo lỗi Groq (tối đa 25s) và retry 1 lần.
    """
    llm = getattr(pipeline, "llm", None)
    if llm is None:
        return resp, 0.0

    if getattr(llm, "openrouter_keys", None) and hasattr(llm, "_call_openrouter"):
        try:
            t_or = time.perf_counter()
            or_ans, or_model, or_meta = llm._call_openrouter(aug_prompt.formatted_prompt)
            elapsed = round(time.perf_counter() - t_or, 3)
            if or_ans and not is_generation_error_response(or_ans, or_model):
                if hasattr(llm, "_save_to_disk_cache"):
                    llm._save_to_disk_cache(aug_prompt.formatted_prompt, or_ans, or_model, or_meta)
                return (
                    Response(
                        query_id=aug_prompt.user_query.query_id,
                        answer=or_ans,
                        sources=aug_prompt.contexts,
                        model_name=or_model,
                        latency_seconds=elapsed,
                        metadata=or_meta,
                    ),
                    elapsed,
                )
        except Exception as or_exc:
            logger.warning(f"OpenRouter failover khi generate thất bại: {or_exc}")

    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", str(resp.answer or ""), re.IGNORECASE)
    wait_s = min(25.0, float(match.group(1)) + 1.0) if match else 6.0
    logger.info(f"Đợi {wait_s:.1f}s rồi thử lại generate...")
    time.sleep(wait_s)
    t_retry = time.perf_counter()
    retry_resp = llm.generate(aug_prompt)
    return retry_resp, round(time.perf_counter() - t_retry, 3)


def generate_predictions(
    pipeline: BaseRAGPipeline,
    dataset: List[EvaluationSample],
    top_k: int = 10,
    num_samples: Optional[int] = None,
    delay_between_queries: float = 1.0,
    output_path: Optional[Union[str, Path]] = None,
    use_batch_llm: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Pha 1 (Inference / Snapshot): Chạy Retrieval + Generation trên tập benchmark,
    đo riêng retrieval_latency_s và llm_latency_s, lưu snapshot đầy đủ contexts để Pha 2 chấm lại 0s.
    Hỗ trợ `resume=True` để bỏ qua câu đã sinh hợp lệ và chỉ chạy bù câu thiếu/lỗi 429.
    """
    active_dataset = dataset[:num_samples] if (num_samples and num_samples > 0) else dataset
    total = len(active_dataset)
    has_granular_parts = (
        not hasattr(pipeline, "agent_graph")
        and hasattr(pipeline, "retriever")
        and hasattr(pipeline, "prompt_augmenter")
        and hasattr(pipeline, "llm")
        and getattr(pipeline, "retriever", None) is not None
        and getattr(pipeline, "prompt_augmenter", None) is not None
        and getattr(pipeline, "llm", None) is not None
    )

    existing_valid: Dict[str, Dict[str, Any]] = {}
    if resume and output_path and Path(output_path).exists():
        try:
            prev = json.loads(Path(output_path).read_text(encoding="utf-8"))
            for item in prev.get("predictions") or []:
                sid = str(item.get("sample_id") or "")
                ans = str(item.get("generated_answer") or "")
                if (
                    sid
                    and ans.strip()
                    and not item.get("error")
                    and str(item.get("agent_status") or "").lower() != "fail"
                    and not is_generation_error_response(ans, item.get("model_name"))
                ):
                    existing_valid[sid] = item
            if existing_valid:
                logger.info(f"[Resume] Tái sử dụng {len(existing_valid)}/{total} mẫu hợp lệ từ {output_path}.")
        except Exception as exc:
            logger.warning(f"Không thể đọc snapshot resume '{output_path}': {exc}")

    records: List[Dict[str, Any]] = []
    print(f"\n🚀 [Pha 1 - Snapshot] Đang sinh câu trả lời cho {total} câu hỏi (Top-K={top_k}, Resume={resume})...")

    if has_granular_parts and use_batch_llm and hasattr(pipeline.llm, "generate_batch"):  # type: ignore[attr-defined]
        staged: List[Tuple[int, EvaluationSample, List[RetrievedContext], Any, float, Optional[str]]] = []
        prompts = []
        records_by_idx: Dict[int, Dict[str, Any]] = {}
        for idx, sample in enumerate(active_dataset):
            if sample.id in existing_valid:
                records_by_idx[idx] = existing_valid[sample.id]
                continue
            uq = UserQuery(query_text=sample.query)
            t0 = time.perf_counter()
            err_msg: Optional[str] = None
            contexts: List[RetrievedContext] = []
            try:
                contexts = pipeline.retriever.retrieve(query=uq, top_k=top_k, filters=uq.filters)  # type: ignore[attr-defined]
            except Exception as exc:
                err_msg = f"Retrieval error: {exc}"
                logger.warning(f"[{sample.id}] {err_msg}")
            ret_lat = round(time.perf_counter() - t0, 3)
            aug = pipeline.prompt_augmenter.augment(query=uq, contexts=contexts)  # type: ignore[attr-defined]
            staged.append((idx, sample, contexts, aug, ret_lat, err_msg))
            prompts.append(aug)
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"   • Đã retrieve {idx + 1}/{total} câu hỏi...")

        batch_responses = pipeline.llm.generate_batch(prompts) if prompts else []  # type: ignore[attr-defined]
        for b_i, (orig_idx, sample, contexts, aug, ret_lat, err_msg) in enumerate(staged):
            resp = batch_responses[b_i] if b_i < len(batch_responses) else Response(answer="", sources=contexts)
            llm_lat = round(float(getattr(resp, "latency_seconds", 0.0) or 0.0), 3)
            if is_generation_error_response(resp.answer, getattr(resp, "model_name", None)):
                resp, extra_lat = _recover_generation_error(pipeline, aug, resp)
                llm_lat = round(llm_lat + extra_lat, 3)
            meta = dict(sample.metadata or {})
            records_by_idx[orig_idx] = {
                "sample_id": sample.id,
                "query": sample.query,
                "type": meta.get("type") or (meta.get("query_types") or ["factoid"])[0],
                "query_types": meta.get("query_types") or ([meta.get("type")] if meta.get("type") else []),
                "unit": meta.get("unit", ""),
                "category": meta.get("category", ""),
                "ground_truth": sample.ground_truth,
                "target_document_ids": meta.get("target_document_ids") or [],
                "target_chunk_ids": meta.get("target_chunk_ids") or {},
                "is_out_of_domain": _is_sample_ood(meta),
                "retrieved_contexts": _serialize_contexts(resp.sources or contexts),
                "generated_answer": resp.answer,
                "model_name": getattr(resp, "model_name", None),
                "retrieval_latency_s": ret_lat,
                "llm_latency_s": llm_lat,
                "total_latency_s": round(ret_lat + llm_lat, 3),
                "token_usage": (getattr(resp, "metadata", {}) or {}).get("usage"),
                "error": err_msg,
            }
        records = [records_by_idx[i] for i in range(total)]
    else:
        for idx, sample in enumerate(active_dataset, 1):
            if sample.id in existing_valid:
                print(f"[{idx:02d}/{total:02d}] ⏭️  [Resume] Giữ nguyên mẫu hợp lệ '{sample.id}'")
                records.append(existing_valid[sample.id])
                continue

            short_q = (sample.query[:56] + "...") if len(sample.query) > 56 else sample.query
            print(f"[{idx:02d}/{total:02d}] 🔍 '{short_q}'")
            uq = UserQuery(query_text=sample.query)
            err_msg = None
            ret_lat: Optional[float] = None
            llm_lat: Optional[float] = None
            tot_lat: float = 0.0
            contexts: List[RetrievedContext] = []
            resp: Optional[Response] = None

            t_start = time.perf_counter()
            if has_granular_parts:
                try:
                    t0 = time.perf_counter()
                    contexts = pipeline.retriever.retrieve(query=uq, top_k=top_k, filters=uq.filters)  # type: ignore[attr-defined]
                    ret_lat = round(time.perf_counter() - t0, 3)

                    aug = pipeline.prompt_augmenter.augment(query=uq, contexts=contexts)  # type: ignore[attr-defined]
                    t1 = time.perf_counter()
                    resp = pipeline.llm.generate(aug)  # type: ignore[attr-defined]
                    llm_lat = round(time.perf_counter() - t1, 3)

                    if is_generation_error_response(resp.answer, getattr(resp, "model_name", None)):
                        resp, extra_lat = _recover_generation_error(pipeline, aug, resp)
                        llm_lat = round(llm_lat + extra_lat, 3)

                    tot_lat = round(time.perf_counter() - t_start, 3)
                except Exception as exc:
                    logger.warning(f"Lỗi khi generate mẫu {sample.id}: {exc}")
                    err_msg = str(exc)
                    resp = Response(answer=f"Lỗi truy vấn: {exc}", sources=contexts)
                    tot_lat = round(time.perf_counter() - t_start, 3)
            else:
                for attempt in range(2):
                    try:
                        resp = pipeline.query(uq, top_k=top_k)
                        tot_lat = round(float(getattr(resp, "latency_seconds", 0.0) or (time.perf_counter() - t_start)), 3)
                        err_msg = None
                        break
                    except Exception as exc:
                        err_msg = str(exc)
                        logger.warning(f"Lỗi khi query mẫu {sample.id} (attempt {attempt + 1}): {exc}")
                        if attempt == 0:
                            time.sleep(2.0)
                        else:
                            resp = Response(answer=f"Lỗi truy vấn: {exc}", sources=[])
                            tot_lat = round(time.perf_counter() - t_start, 3)

            assert resp is not None
            resp_meta = getattr(resp, "metadata", None) if isinstance(getattr(resp, "metadata", None), dict) else {}
            trace_entries = resp_meta.get("trace") if isinstance(resp_meta.get("trace"), list) else []

            if ret_lat is None and trace_entries:
                ret_vals = [
                    float(t["latency_s"])
                    for t in trace_entries
                    if isinstance(t, dict) and t.get("node") == "retrieval_node" and t.get("latency_s") is not None
                ]
                if ret_vals:
                    ret_lat = round(sum(ret_vals), 3)

            if llm_lat is None and trace_entries:
                llm_vals = [
                    float(t["latency_s"])
                    for t in trace_entries
                    if isinstance(t, dict) and t.get("node") == "generation_node" and t.get("latency_s") is not None
                ]
                if llm_vals:
                    llm_lat = round(sum(llm_vals), 3)

            token_usage = resp_meta.get("usage")
            if token_usage is None and trace_entries:
                for t in reversed(trace_entries):
                    if isinstance(t, dict) and t.get("node") == "generation_node" and t.get("usage") is not None:
                        token_usage = t.get("usage")
                        break

            meta = dict(sample.metadata or {})
            record_item: Dict[str, Any] = {
                "sample_id": sample.id,
                "query": sample.query,
                "type": meta.get("type") or (meta.get("query_types") or ["factoid"])[0],
                "query_types": meta.get("query_types") or ([meta.get("type")] if meta.get("type") else []),
                "unit": meta.get("unit", ""),
                "category": meta.get("category", ""),
                "ground_truth": sample.ground_truth,
                "target_document_ids": meta.get("target_document_ids") or [],
                "target_chunk_ids": meta.get("target_chunk_ids") or {},
                "is_out_of_domain": _is_sample_ood(meta),
                "retrieved_contexts": _serialize_contexts(resp.sources or contexts),
                "generated_answer": resp.answer,
                "model_name": getattr(resp, "model_name", None),
                "retrieval_latency_s": ret_lat,
                "llm_latency_s": llm_lat,
                "total_latency_s": tot_lat,
                "token_usage": token_usage,
                "error": err_msg,
            }
            if resp_meta.get("status") is not None:
                record_item["agent_status"] = resp_meta.get("status")
            if resp_meta.get("route") is not None:
                record_item["agent_route"] = resp_meta.get("route")
            records.append(record_item)

            if output_path and (idx % 10 == 0 or idx == total):
                snapshot = {
                    "top_k": top_k,
                    "total_samples": len(records),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "predictions": records,
                }
                _atomic_write_json(Path(output_path), snapshot)

            if idx < total and delay_between_queries > 0:
                if not ((getattr(resp, "metadata", {}) or {}).get("cache_hit")):
                    time.sleep(delay_between_queries)

    snapshot = {
        "top_k": top_k,
        "total_samples": len(records),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": records,
    }
    if output_path:
        _atomic_write_json(Path(output_path), snapshot)
        logger.info(f"Đã lưu predictions snapshot vào: {output_path}")
    return snapshot


def _compute_snapshot_retrieval_metrics(
    record: Dict[str, Any],
    top_k: int,
) -> Dict[str, Optional[float]]:
    if record.get("is_out_of_domain"):
        return {}

    contexts = (record.get("retrieved_contexts") or [])[:top_k]
    chunk_ids = [str(c["chunk_id"]) for c in contexts if c.get("chunk_id")]
    doc_ids = deduplicate_preserve_order([
        str(c.get("document_id") or c.get("chunk_id"))
        for c in contexts
        if (c.get("document_id") or c.get("chunk_id"))
    ])

    raw_target_docs = record.get("target_document_ids") or []
    if isinstance(raw_target_docs, dict):
        doc_qrels = {str(k): int(v) for k, v in raw_target_docs.items()}
    else:
        doc_qrels = {str(d): 2 for d in raw_target_docs}

    raw_target_chunks = record.get("target_chunk_ids") or {}
    if isinstance(raw_target_chunks, list):
        chunk_qrels = {str(c): 2 for c in raw_target_chunks}
    else:
        chunk_qrels = {str(k): int(v) for k, v in raw_target_chunks.items()}

    doc_rel_set = set(doc_qrels.keys())
    chunk_rel_set = set(chunk_qrels.keys())

    return {
        "doc_hit_at_k": round(hit_rate_at_k(doc_ids, doc_rel_set, top_k), 4) if doc_rel_set else None,
        "doc_recall_at_k": round(recall_at_k(doc_ids, doc_rel_set, top_k), 4) if doc_rel_set else None,
        "chunk_hit_at_k": round(hit_rate_at_k(chunk_ids, chunk_rel_set, top_k), 4) if chunk_rel_set else None,
        "chunk_recall_at_k": round(recall_at_k(chunk_ids, chunk_rel_set, top_k), 4) if chunk_rel_set else None,
        "chunk_ndcg_at_k": round(ndcg_at_k(chunk_ids, chunk_qrels, top_k), 4) if chunk_qrels else None,
        "chunk_mrr": round(mrr(chunk_ids[:top_k], chunk_rel_set), 4) if chunk_rel_set else None,
    }


def _score_single_offline_record(
    record: Dict[str, Any],
    top_k: int,
    embedder: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Chấm điểm offline cho 1 bản ghi snapshot, trả về (offline_metrics_dict, retrieval_metrics_dict)."""
    query = record.get("query", "")
    answer = record.get("generated_answer", "")
    ground_truth = record.get("ground_truth")
    is_ood = bool(record.get("is_out_of_domain")) or _is_sample_ood(record)
    contexts = (record.get("retrieved_contexts") or [])[:top_k]
    contexts_text = [_format_context_dict_for_eval(c) for c in contexts]

    ret_metrics = _compute_snapshot_retrieval_metrics(record, top_k=top_k)

    raw_target_chunks = record.get("target_chunk_ids") or {}
    if isinstance(raw_target_chunks, list):
        target_chunks = {str(c): 2 for c in raw_target_chunks}
    else:
        target_chunks = {str(k): int(v) for k, v in raw_target_chunks.items()}

    c_rel = compute_context_relevance(query, contexts_text, embedder=embedder) if not is_ood else 0.0
    if target_chunks:
        ret_cids = {str(c.get("chunk_id")) for c in contexts if c.get("chunk_id")}
        g2 = {cid for cid, g in target_chunks.items() if g >= 2}
        g1 = {cid for cid, g in target_chunks.items() if g == 1}
        if ret_cids & g2:
            c_hit = 1.0
        elif ret_cids & g1:
            c_hit = 0.5
        else:
            c_hit = 0.0
    else:
        c_hit = 1.0 if (not is_ood and c_rel >= 0.45) else 0.0

    # Nếu câu trả lời thực chất là thông báo lỗi API / 429 fallback / fail_node, không chấm 0.0 hoặc HALLUCINATION oan
    if (
        is_generation_error_response(answer, record.get("model_name"))
        or bool(record.get("error"))
        or str(record.get("agent_status") or "").lower() == "fail"
    ):
        return (
            {
                "is_generation_error": True,
                "is_refusal": False,
                "context_hit": round(c_hit, 4) if not is_ood else None,
            },
            ret_metrics,
        )

    refusal = is_refusal_response(answer) or str(record.get("agent_status") or "").lower() == "out_of_scope"

    if is_ood:
        ood_acc = 1.0 if refusal else 0.0
        offline = {
            "is_generation_error": False,
            "is_refusal": refusal,
            "ood_refusal_accuracy": ood_acc,
            "overall_score": ood_acc,
        }
        return offline, ret_metrics

    faith = compute_faithfulness(answer, contexts_text, embedder=embedder)
    a_rel = compute_answer_relevance(query, answer, embedder=embedder)
    num_info = compute_numeric_grounding(answer, contexts_text, ground_truth=ground_truth)
    corr_info = compute_answer_correctness_offline(answer, ground_truth, embedder=embedder)

    a_corr = corr_info["answer_correctness"]
    if a_corr is not None:
        overall = 0.40 * float(a_corr) + 0.30 * faith + 0.15 * a_rel + 0.15 * c_rel
    else:
        overall = 0.45 * faith + 0.30 * a_rel + 0.25 * c_rel

    offline = {
        "context_relevance": round(c_rel, 4),
        "faithfulness": round(faith, 4),
        "answer_relevance": round(a_rel, 4),
        "answer_correctness": a_corr,
        "token_precision": corr_info["token_precision"],
        "token_recall": corr_info["token_recall"],
        "token_f1": corr_info["token_f1"],
        "keyword_recall": corr_info["keyword_recall"],
        "rouge_l_recall": corr_info["rouge_l_recall"],
        "rouge_l_f1": corr_info["rouge_l_f1"],
        "numeric_faithfulness": num_info["numeric_faithfulness"],
        "numeric_correctness": num_info["numeric_correctness"],
        "ungrounded_entities": num_info["ungrounded_entities"],
        "missing_gt_entities": num_info["missing_gt_entities"],
        "semantic_answer_correctness": corr_info["semantic_answer_correctness"],
        "context_hit": round(c_hit, 4),
        "is_refusal": refusal,
        "overall_score": round(min(1.0, max(0.0, overall)), 4),
    }
    return offline, ret_metrics


def score_predictions(
    predictions_source: Union[str, Path, Dict[str, Any]],
    top_k: Optional[int] = None,
    use_judge: bool = False,
    judge: Optional[UETLLMJudgeEvaluator] = None,
    embedder: Optional[Any] = None,
    enable_semantic: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pha 2 (Scoring & Error Taxonomy): Chấm Retrieval + Offline Metrics + tùy chọn UETLLMJudgeEvaluator
    trên snapshot từ Pha 1 mà không cần chạy lại RAG Pipeline.
    """
    if isinstance(predictions_source, (str, Path)):
        with open(predictions_source, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    else:
        snapshot = predictions_source

    raw_records = snapshot.get("predictions") or snapshot.get("samples") or []
    eff_top_k = top_k if top_k is not None else int(snapshot.get("top_k", 10))

    active_embedder = embedder
    if active_embedder is None and enable_semantic:
        try:
            active_embedder = UETEmbedder()
        except Exception as exc:
            logger.warning(f"Không thể nạp UETEmbedder cho semantic metrics ({exc}); chạy chế độ lexical.")
            active_embedder = None

    active_judge = judge
    if use_judge and active_judge is None:
        active_judge = UETLLMJudgeEvaluator(provider=judge_provider, model_name=judge_model)

    scored_samples: List[Dict[str, Any]] = []
    total = len(raw_records)

    for idx, raw in enumerate(raw_records, 1):
        rec = dict(raw)
        rec["is_out_of_domain"] = bool(rec.get("is_out_of_domain")) or _is_sample_ood(rec)
        offline_metrics, retrieval_metrics = _score_single_offline_record(
            rec,
            top_k=eff_top_k,
            embedder=active_embedder if enable_semantic else None,
        )
        rec["offline_metrics"] = offline_metrics
        rec["retrieval_metrics"] = retrieval_metrics

        if use_judge and active_judge is not None:
            if offline_metrics.get("is_generation_error"):
                rec["judge"] = {
                    "status": "skipped_generation_error",
                    "sample_id": str(rec.get("sample_id") or rec.get("id") or idx),
                    "raw_scores": {},
                    "normalized_scores": {},
                    "error": "Skipped judge due to generation error",
                }
            else:
                contexts_text = [_format_context_dict_for_eval(c) for c in (rec.get("retrieved_contexts") or [])[:eff_top_k]]
                judge_res = active_judge.judge_record(
                    query=rec.get("query", ""),
                    contexts=contexts_text,
                    answer=rec.get("generated_answer", ""),
                    ground_truth=rec.get("ground_truth"),
                    is_out_of_domain=rec["is_out_of_domain"],
                    sample_id=str(rec.get("sample_id") or rec.get("id") or idx),
                    generator_model=rec.get("model_name"),
                )
                rec["judge"] = judge_res
                if not judge_res.get("cache_hit") and idx < total and active_judge.delay_seconds > 0:
                    time.sleep(active_judge.delay_seconds)

        err_label, err_reason = classify_sample_error(rec)
        rec["error_label"] = err_label
        rec["error_reason"] = err_reason
        scored_samples.append(rec)

    judge_info = {
        "enabled": bool(use_judge and active_judge is not None),
        "provider": getattr(active_judge, "provider", None) if active_judge else None,
        "model": getattr(active_judge, "model_name", None) if active_judge else None,
    }
    report = build_e2e_report(scored_samples=scored_samples, top_k=eff_top_k, judge_info=judge_info)

    if output_path:
        _atomic_write_json(Path(output_path), report)
        logger.info(f"Đã lưu báo cáo E2E chi tiết vào: {output_path}")
    return report


# =========================================================================
# 4. UET EVALUATOR (TƯƠNG THÍCH BASEEVALUATOR + TÍCH HỢP JUDGE & 2 PHA)
# =========================================================================

class UETEvaluator(BaseEvaluator):
    """
    Bộ đánh giá chất lượng RAG cho UET Chatbot:
    - Chế độ Offline (0-LLM): làm sạch bảng Markdown, Token Recall/F1, ROUGE-L, Numeric Grounding,
      OOD Refusal Detector, và tùy chọn Semantic Cosine qua `embedder` dùng lại từ pipeline.
    - Chế độ LLM Judge (`use_llm_judge=True`): gọi `UETLLMJudgeEvaluator` 1-call Structured JSON.
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        embedder: Optional[Any] = None,
        judge: Optional[UETLLMJudgeEvaluator] = None,
        enable_semantic_metrics: bool = True,
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
    ):
        self.use_llm_judge = use_llm_judge
        self.embedder = embedder  # Lazy / dùng lại từ pipeline để tránh load trùng 2.2 GB bge-m3
        self.enable_semantic_metrics = enable_semantic_metrics
        self.judge: Optional[UETLLMJudgeEvaluator] = judge
        if self.use_llm_judge and self.judge is None:
            self.judge = UETLLMJudgeEvaluator(provider=judge_provider, model_name=judge_model)

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        response: Response,
    ) -> EvaluationResult:
        """Đánh giá 1 mẫu câu trả lời và trả về EvaluationResult (metrics chỉ chứa float hợp lệ)."""
        meta = dict(sample.metadata or {})
        is_ood = _is_sample_ood(meta)
        record = {
            "sample_id": sample.id,
            "query": sample.query,
            "type": meta.get("type", "factoid"),
            "query_types": meta.get("query_types") or ([meta.get("type")] if meta.get("type") else []),
            "unit": meta.get("unit", ""),
            "category": meta.get("category", ""),
            "ground_truth": sample.ground_truth,
            "target_document_ids": meta.get("target_document_ids") or [],
            "target_chunk_ids": meta.get("target_chunk_ids") or {},
            "is_out_of_domain": is_ood,
            "retrieved_contexts": _serialize_contexts(response.sources),
            "generated_answer": response.answer,
            "model_name": getattr(response, "model_name", None),
        }

        top_k = max(len(response.sources or []), 1)
        active_embedder = self.embedder if self.enable_semantic_metrics else None
        offline, ret_metrics = _score_single_offline_record(record, top_k=top_k, embedder=active_embedder)
        record["offline_metrics"] = offline
        record["retrieval_metrics"] = ret_metrics

        # Chuyển sang Dict[str, float] đúng hợp đồng src/base.py (không chứa None/bool/list)
        metrics: Dict[str, float] = {}
        if is_ood:
            metrics["ood_refusal_accuracy"] = float(offline["ood_refusal_accuracy"])
            metrics["overall_score"] = float(offline["overall_score"])
        else:
            for key in (
                "context_relevance",
                "faithfulness",
                "answer_relevance",
                "answer_correctness",
                "token_precision",
                "token_recall",
                "token_f1",
                "keyword_recall",
                "rouge_l_recall",
                "rouge_l_f1",
                "numeric_faithfulness",
                "numeric_correctness",
                "semantic_answer_correctness",
                "context_hit",
                "overall_score",
            ):
                val = offline.get(key)
                if val is not None:
                    metrics[key] = float(val)
            metrics["false_refusal"] = 1.0 if offline.get("is_refusal") else 0.0

        if self.use_llm_judge and self.judge is not None:
            contexts_text = [c.text for c in (response.sources or [])]
            judge_res = self.judge.judge_record(
                query=sample.query,
                contexts=contexts_text,
                answer=response.answer,
                ground_truth=sample.ground_truth,
                is_out_of_domain=is_ood,
                sample_id=sample.id,
                generator_model=getattr(response, "model_name", None),
            )
            record["judge"] = judge_res
            if judge_res.get("status") == "ok":
                for jk, jv in (judge_res.get("normalized_scores") or {}).items():
                    metrics[f"judge_{jk}"] = float(jv)

        err_label, err_reason = classify_sample_error(record)
        return EvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            generated_answer=response.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=response.sources,
            metrics=metrics,
            feedback=f"[{err_label}] {err_reason} | overall={metrics.get('overall_score', 0.0):.3f}",
        )

    def evaluate_pipeline(
        self,
        pipeline: BaseRAGPipeline,
        dataset: List[EvaluationSample],
        top_k: int = 5,
        num_samples: Optional[int] = None,
        delay_between_queries: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
        predictions_path: Optional[Union[str, Path]] = None,
        use_batch_llm: bool = False,
        resume: bool = True,
    ) -> EvaluationReport:
        """
        Chạy đánh giá toàn trình 2 pha (Generate Snapshot -> Score + Diagnostics) và trả về EvaluationReport.
        """
        # Tái sử dụng embedding_model có sẵn của pipeline để không load lại 2.2 GB bge-m3
        active_embedder = self.embedder
        if active_embedder is None and self.enable_semantic_metrics:
            active_embedder = getattr(pipeline, "embedding_model", None)

        snapshot = generate_predictions(
            pipeline=pipeline,
            dataset=dataset,
            top_k=top_k,
            num_samples=num_samples,
            delay_between_queries=delay_between_queries,
            output_path=predictions_path,
            use_batch_llm=use_batch_llm,
            resume=resume,
        )

        report_dict = score_predictions(
            predictions_source=snapshot,
            top_k=top_k,
            use_judge=self.use_llm_judge,
            judge=self.judge,
            embedder=active_embedder,
            enable_semantic=self.enable_semantic_metrics,
            output_path=output_path,
        )

        # Quy đổi sang EvaluationReport chuẩn src/base.py
        eval_results: List[EvaluationResult] = []
        for s in report_dict.get("samples", []):
            ctxs = [
                RetrievedContext(
                    chunk_id=str(c.get("chunk_id") or f"ctx_{i}"),
                    text=str(c.get("text", "")),
                    similarity_score=float(c.get("score") or 0.0),
                    rank=int(c.get("rank") or i),
                    metadata={
                        "document_id": c.get("document_id"),
                        "title": c.get("title"),
                        "search_source": c.get("search_source"),
                    },
                )
                for i, c in enumerate(s.get("retrieved_contexts") or [], 1)
            ]
            sample_metrics: Dict[str, float] = {}
            for k, v in (s.get("offline_metrics") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    sample_metrics[k] = float(v)
            if (s.get("judge") or {}).get("status") == "ok":
                for jk, jv in (s["judge"].get("normalized_scores") or {}).items():
                    sample_metrics[f"judge_{jk}"] = float(jv)

            eval_results.append(EvaluationResult(
                sample_id=str(s.get("sample_id")),
                query=str(s.get("query")),
                generated_answer=str(s.get("generated_answer", "")),
                ground_truth=s.get("ground_truth"),
                retrieved_contexts=ctxs,
                metrics=sample_metrics,
                feedback=f"[{s.get('error_label')}] {s.get('error_reason')}",
            ))

        avg_metrics: Dict[str, float] = dict(report_dict.get("offline_metrics") or {})
        if report_dict["summary"].get("ood_refusal_accuracy") is not None:
            avg_metrics["ood_refusal_accuracy"] = float(report_dict["summary"]["ood_refusal_accuracy"])
        avg_metrics["false_refusal_rate"] = float(report_dict["summary"].get("false_refusal_rate", 0.0))
        for jk, jv in (report_dict.get("judge_metrics", {}).get("average_normalized_scores") or {}).items():
            avg_metrics[f"judge_{jk}"] = float(jv)

        # Lưu thêm alias top-level tiện cho caller đọc báo cáo
        self.last_diagnostic_report = report_dict
        return EvaluationReport(
            total_samples=len(eval_results),
            average_metrics=avg_metrics,
            results=eval_results,
        )

    @staticmethod
    def load_dataset(
        dataset_path: Optional[Union[str, Path]] = None,
        domain: Optional[str] = None,
    ) -> List[EvaluationSample]:
        """Tải tập dữ liệu benchmark từ file JSON/JSONL hoặc thư mục domain."""
        if dataset_path is None:
            default_annotated = BASE_DIR / "data" / "eval_data" / "e2e_eval_100_annotated.json"
            default_unified = BASE_DIR / "data" / "eval_data" / "uet_rag_eval_100" / "e2e_eval_100.json"
            if default_annotated.exists():
                path = default_annotated
            elif default_unified.exists():
                path = default_unified
            else:
                path = BASE_DIR / "data" / "eval_data" / "uet_domain_qa"
        else:
            path = Path(dataset_path)

        if path.is_dir() and (path / "e2e_eval_100_annotated.json").exists():
            path = path / "e2e_eval_100_annotated.json"
        elif path.is_dir() and (path / "e2e_eval_100.json").exists():
            path = path / "e2e_eval_100.json"

        if path.is_file():
            samples = UETEvaluator._read_file(path)
            if domain and domain.lower().strip() != "all":
                dom = domain.upper().strip()
                matched = [s for s in samples if s.metadata.get("unit", "").upper() == dom]
                if not matched:
                    matched = [s for s in samples if dom in s.metadata.get("unit", "").upper()]
                if matched:
                    logger.info(f"Đã lọc {len(matched)} câu hỏi theo domain '{dom}' từ {path.name}.")
                    return matched
                logger.warning(f"Không tìm thấy câu hỏi nào thuộc domain '{dom}' trong {path.name}. Sử dụng toàn bộ.")
            return samples

        if not path.exists():
            if "e2e_domain_qa" in str(path):
                alt = Path(str(path).replace("e2e_domain_qa", "uet_domain_qa"))
                if alt.exists():
                    path = alt
            elif "uet_domain_qa" in str(path):
                alt = Path(str(path).replace("uet_domain_qa", "e2e_domain_qa"))
                if alt.exists():
                    path = alt

        if path.is_dir() or domain:
            base_dir = path if path.is_dir() else path.parent
            if domain:
                domain_key = domain.lower().strip()
                if domain_key == "all":
                    all_samples: List[EvaluationSample] = []
                    seen_queries = set()
                    for f in sorted(base_dir.glob("*.json")):
                        for sample in UETEvaluator._read_file(f):
                            if sample.query not in seen_queries:
                                seen_queries.add(sample.query)
                                all_samples.append(sample)
                    if all_samples:
                        return all_samples
                elif domain_key in DOMAIN_FILE_MAP:
                    candidates = DOMAIN_FILE_MAP[domain_key]
                    if isinstance(candidates, str):
                        candidates = [candidates]
                    for candidate_name in candidates:
                        target_file = base_dir / candidate_name
                        if target_file.exists():
                            return UETEvaluator._read_file(target_file)
                else:
                    for f in base_dir.glob(f"*{domain_key}*"):
                        if f.suffix in [".json", ".jsonl"]:
                            return UETEvaluator._read_file(f)

            for def_name in ["overview1_qa.json", "overview2_qa.json", "handbook2_qa.json", "general_qa.json", "test_qa_dataset.json"]:
                default_file = base_dir / def_name
                if default_file.exists():
                    return UETEvaluator._read_file(default_file)

            for f in sorted(base_dir.glob("*.json")):
                return UETEvaluator._read_file(f)
            return []

        if not path.exists():
            return []

        return UETEvaluator._read_file(path)

    @staticmethod
    def _read_file(file_path: Path) -> List[EvaluationSample]:
        """Đọc và chuẩn hóa dữ liệu từ file JSON/JSONL thành EvaluationSample."""
        samples: List[EvaluationSample] = []
        try:
            raw_items = []
            if file_path.suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_str in f:
                        line_str = line_str.strip()
                        if line_str:
                            raw_items.append(json.loads(line_str))
            elif file_path.suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        raw_items = data
                    elif isinstance(data, dict):
                        raw_items = [data]

            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                meta = dict(item.get("metadata", {}))
                for extra in [
                    "type",
                    "query_types",
                    "difficulty",
                    "unit",
                    "category",
                    "target_document_ids",
                    "target_chunk_ids",
                    "review_label",
                ]:
                    if extra in item:
                        meta[extra] = item[extra]
                item_dict = {
                    "id": str(item.get("id") or item.get("query_id")),
                    "query": item.get("query") or item.get("query_text", ""),
                    "ground_truth": item.get("ground_truth"),
                    "expected_source": item.get("expected_source"),
                    "metadata": meta,
                }
                samples.append(EvaluationSample(**item_dict))
        except Exception as e:
            logger.error(f"Lỗi khi đọc file dataset '{file_path}': {e}")
        return samples


# =========================================================================
# 5. GIAO DIỆN DÒNG LỆNH CLI (RUNNER)
# =========================================================================

def run_eval_e2e_cli(
    dataset_path: Optional[Union[str, Path]] = None,
    domain: Optional[str] = None,
    use_ragas: bool = False,
    use_judge: bool = False,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    num_samples: Optional[int] = None,
    top_k: Optional[int] = None,
    model_name: str = "deepseek/deepseek-v4.1-flash-1017",
    output_path: Optional[Union[str, Path]] = None,
    delay: float = 1.2,
    generate_only: bool = False,
    score_from: Optional[Union[str, Path]] = None,
    save_predictions: Optional[Union[str, Path]] = None,
    enable_semantic: bool = True,
    use_batch_llm: bool = False,
    resume: bool = True,
):
    """Chạy đánh giá toàn trình End-to-End RAG trên CLI (hỗ trợ 2 pha, Offline, UET Judge & Ragas)."""
    base_dir = BASE_DIR
    cfg = load_yaml_config()
    if top_k is None:
        top_k = cfg.get("retrieval", {}).get("final_top_k", 10)

    # Pha 2 độc lập: Chấm điểm từ file snapshot predictions.json có sẵn (không cần khởi tạo RAG pipeline)
    if score_from is not None:
        score_path = Path(score_from)
        if not score_path.exists():
            print(f"❌ Không tìm thấy file snapshot predictions tại: {score_path}")
            return None
        out_p = Path(output_path) if output_path else (base_dir / "data" / "eval_data" / "e2e_eval_results.json")
        print("\n" + "=" * 70)
        print("🧮 [PHA 2] CHẤM ĐIỂM TỪ SNAPSHOT PREDICTIONS CÓ SẴN")
        print(f"• Snapshot đầu vào : {score_path}")
        print(f"• Chế độ Judge     : {'BẬT' if use_judge else 'TẮT (Offline Metrics)'}")
        print(f"• Semantic (bge-m3): {'BẬT' if enable_semantic else 'TẮT'}")
        print("=" * 70)
        report_dict = score_predictions(
            predictions_source=score_path,
            top_k=top_k,
            use_judge=use_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
            enable_semantic=enable_semantic,
            output_path=out_p,
        )
        print_e2e_diagnostic_report(report_dict)
        print(f"📁 Đã lưu báo cáo E2E chi tiết vào: {out_p}\n")
        return report_dict

    if not dataset_path:
        cfg_file = cfg.get("paths", {}).get("eval_dataset_file")
        if cfg_file:
            target_path = base_dir / cfg_file if not Path(cfg_file).is_absolute() else Path(cfg_file)
        else:
            default_annotated = base_dir / "data" / "eval_data" / "e2e_eval_100_annotated.json"
            if default_annotated.exists():
                target_path = default_annotated
            else:
                target_path = base_dir / "data" / "eval_data" / "uet_rag_eval_100" / "e2e_eval_100.json"
    else:
        target_path = Path(dataset_path)

    dataset = UETEvaluator.load_dataset(target_path, domain=domain)
    domain_str = f" [Domain/Unit: {domain.upper()}]" if domain else ""
    target_name = target_path.name if target_path.is_file() else f"uet_domain_qa{domain_str}"

    if not dataset:
        print(f"❌ Không tìm thấy tập dữ liệu đánh giá hợp lệ tại: {target_path}")
        if domain:
            print(f"   Domain hỗ trợ: {', '.join(DOMAIN_FILE_MAP.keys())}, all")
        return None

    from src.pipeline import get_active_pipeline, describe_pipeline_mode

    rag_pipeline = get_active_pipeline()
    active_mode_label = describe_pipeline_mode(rag_pipeline)
    actual_samples = num_samples if (num_samples and num_samples > 0) else len(dataset)

    if generate_only:
        pred_out = Path(
            save_predictions
            or output_path
            or (base_dir / "data" / "eval_data" / "e2e_predictions.json")
        )
        print("\n" + "=" * 70)
        print("📸 [PHA 1] CHẠY PIPELINE & LƯU SNAPSHOT PREDICTIONS")
        print(f"• Chế độ RAG  : {active_mode_label}")
        print(f"• Tập dữ liệu : {target_name} ({actual_samples}/{len(dataset)} câu)")
        print(f"• Top-K       : {top_k}")
        print(f"• Snapshot    : {pred_out}")
        print("=" * 70)
        snapshot = generate_predictions(
            pipeline=rag_pipeline,
            dataset=dataset,
            top_k=top_k,
            num_samples=num_samples,
            delay_between_queries=delay,
            output_path=pred_out,
            use_batch_llm=use_batch_llm,
            resume=resume,
        )
        print(f"\n✅ Đã tạo snapshot {snapshot['total_samples']} mẫu tại: {pred_out}\n")
        return snapshot

    if use_ragas:
        print("\n" + "=" * 70)
        print("🤖 ĐANG CHẠY ĐÁNH GIÁ RAG BẰNG RAGAS & LLM-AS-A-JUDGE")
        print(f"• Tập dữ liệu: {target_name} (Tổng: {len(dataset)} câu)")
        print(f"• Giám khảo LLM: {model_name}")
        print(f"• Số lượng câu hỏi đánh giá: {actual_samples}")
        print(f"• Top-K ngữ cảnh: {top_k}")
        print("=" * 70)

        out_p = Path(output_path) if output_path else (base_dir / "data" / "eval_data" / "ragas_eval_results.json")
        ragas_eval = UETRagasEvaluator(
            model_name=model_name,
            embedder=getattr(rag_pipeline, "embedding_model", None),
        )
        report = ragas_eval.evaluate_pipeline(
            pipeline=rag_pipeline,
            dataset=dataset,
            top_k=top_k,
            num_samples=num_samples,
            output_path=out_p,
            delay_between_queries=delay,
        )
        print("\n📈 KẾT QUẢ RAGAS:")
        for metric, score in report["average_metrics"].items():
            val_str = f"{score:.4f}" if score is not None else "N/A (all NaN)"
            print(f"  • {metric:<20}: {val_str} (valid={report['valid_counts'].get(metric, 0)}, nan={report['nan_counts'].get(metric, 0)})")
        print(f"📁 Đã lưu báo cáo Ragas vào: {out_p}\n")
        return report

    out_p = Path(output_path) if output_path else (base_dir / "data" / "eval_data" / "e2e_eval_results.json")
    pred_out = Path(save_predictions) if save_predictions else out_p.with_name(f"{out_p.stem}_predictions.json")

    print("\n" + "=" * 70)
    print("📊 ĐANG CHẠY ĐÁNH GIÁ TOÀN TRÌNH E2E RAG (OFFLINE + DIAGNOSTICS)")
    print(f"• Chế độ RAG  : {active_mode_label}")
    print(f"• Tập dữ liệu : {target_name} ({actual_samples}/{len(dataset)} câu)")
    print(f"• Top-K       : {top_k}")
    print(f"• LLM Judge   : {'BẬT (UETLLMJudgeEvaluator)' if use_judge else 'TẮT'}")
    print(f"• Semantic    : {'BẬT (bge-m3)' if enable_semantic else 'TẮT'}")
    print("=" * 70)

    evaluator = UETEvaluator(
        use_llm_judge=use_judge,
        embedder=getattr(rag_pipeline, "embedding_model", None),
        enable_semantic_metrics=enable_semantic,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )
    eval_report = evaluator.evaluate_pipeline(
        pipeline=rag_pipeline,
        dataset=dataset,
        top_k=top_k,
        num_samples=num_samples,
        delay_between_queries=delay,
        output_path=out_p,
        predictions_path=pred_out,
        use_batch_llm=use_batch_llm,
        resume=resume,
    )
    if hasattr(evaluator, "last_diagnostic_report"):
        print_e2e_diagnostic_report(evaluator.last_diagnostic_report)
    print(f"📁 Đã lưu snapshot predictions : {pred_out}")
    print(f"📁 Đã lưu báo cáo E2E chi tiết : {out_p}\n")
    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UET Chatbot RAG Evaluation Runner")
    parser.add_argument("--graph", action="store_true", help="Bật nhánh GraphRAG kết hợp với Hybrid Search")
    parser.add_argument("--agentic", action="store_true", help="Bật chế độ Agentic RAG (LangGraph + JEV)")
    parser.add_argument("--use-ragas", action="store_true", help="Sử dụng thư viện Ragas")
    parser.add_argument("--use-judge", action="store_true", help="Sử dụng UETLLMJudgeEvaluator (1-call Structured JSON)")
    parser.add_argument("--judge-provider", type=str, default=None, help="Provider cho LLM Judge (groq, openrouter, sambanova, gemini)")
    parser.add_argument("--judge-model", type=str, default=None, help="Tên model cho LLM Judge")
    parser.add_argument("--generate-only", action="store_true", help="Chỉ chạy Pha 1 sinh predictions snapshot")
    parser.add_argument("--score-from", type=str, default=None, help="Chạy Pha 2 chấm điểm từ file predictions.json có sẵn")
    parser.add_argument("--save-predictions", type=str, default=None, help="Đường dẫn lưu snapshot predictions.json")
    parser.add_argument("--no-semantic", action="store_true", help="Tắt semantic similarity (chạy siêu nhanh thuần lexical)")
    parser.add_argument("--batch-llm", action="store_true", help="Dùng generate_batch() trong Pha 1 nếu pipeline hỗ trợ")
    parser.add_argument("--no-resume", action="store_true", help="Tắt resume snapshot cũ (chạy lại từ đầu trong Pha 1)")
    parser.add_argument("--num-samples", "--samples", type=int, default=None, dest="num_samples", help="Số lượng mẫu kiểm thử")
    parser.add_argument("--domain", type=str, default=None, help="Lọc theo khoa/đơn vị: fit, fet, fema, sae, handbook, tuyensinh, all")
    parser.add_argument("--top-k", "--top_k", type=int, default=None, dest="top_k", help="Số lượng chunks ngữ cảnh lấy ra")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-v4.1-flash-1017", help="Tên mô hình LLM Judge cho Ragas")
    parser.add_argument("--delay", type=float, default=1.2, help="Thời gian nghỉ giữa các câu hỏi (giây)")
    parser.add_argument("--dataset", type=str, default=None, help="Đường dẫn file dataset QA kiểm thử")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu kết quả JSON")
    args = parser.parse_args()
    os.environ["RAG_MODE"] = "agentic" if args.agentic else "modular"
    os.environ["ENABLE_GRAPH"] = "1" if args.graph else "0"

    run_eval_e2e_cli(
        dataset_path=args.dataset,
        domain=args.domain,
        use_ragas=args.use_ragas,
        use_judge=args.use_judge,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        num_samples=args.num_samples,
        top_k=args.top_k,
        model_name=args.model,
        output_path=args.output,
        delay=args.delay,
        generate_only=args.generate_only,
        score_from=args.score_from,
        save_predictions=args.save_predictions,
        enable_semantic=not args.no_semantic,
        use_batch_llm=args.batch_llm,
        resume=not args.no_resume,
    )

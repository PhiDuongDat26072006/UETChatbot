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
from src.evaluation.agent_eval import (
    compute_router_metrics,
    compute_sample_agent_flags,
    extract_rounds,
    get_agent_info,
    has_agent_data,
    sanitize_trace,
    score_sample_rounds,
)
from src.evaluation.diagnostics import (
    build_e2e_report,
    classify_sample_error,
    print_e2e_diagnostic_report,
)
from src.evaluation.e2e_cli import run_eval_e2e_cli, run_router_eval_cli
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
    strip_agent_notes,
)
from src.evaluation.ragas_eval import (
    UETLangchainEmbeddings,
    UETRagasEvaluator,
    _atomic_write_json,
    _clean_ragas_value,
    _ensure_ragas_compat,
)
from src.evaluation.snapshot import (
    _compute_snapshot_retrieval_metrics,
    _format_context_dict_for_eval,
    _is_sample_ood,
    _recover_generation_error,
    _score_single_offline_record,
    _serialize_contexts,
    generate_predictions,
    score_predictions,
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
        judge_rounds: bool = False,
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
            judge_rounds=judge_rounds,
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
                    "expected_route",
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


if __name__ == "__main__":
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    parser = argparse.ArgumentParser(description="UET Chatbot RAG Evaluation Runner")
    parser.add_argument("--graph", action="store_true", help="Bật nhánh GraphRAG kết hợp với Hybrid Search")
    parser.add_argument("--agentic", action="store_true", help="Bật chế độ Agentic RAG (LangGraph + JEV)")
    parser.add_argument(
        "--router-eval",
        nargs="?",
        const="data/golden_questions.json",
        default=None,
        help="Chỉ đánh giá JEV Router trên bộ dữ liệu có expected_route (mặc định: data/golden_questions.json)",
    )
    parser.add_argument("--use-ragas", action="store_true", help="Sử dụng thư viện Ragas")
    parser.add_argument("--use-judge", action="store_true", help="Sử dụng UETLLMJudgeEvaluator (1-call Structured JSON)")
    parser.add_argument("--judge-rounds", action="store_true", help="Chấm LLM Judge thêm câu trả lời vòng 0 cho các mẫu có retry >= 1 (cần bật --use-judge)")
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

    if args.router_eval is not None:
        run_router_eval_cli(
            dataset_path=args.router_eval,
            output_path=args.output,
            num_samples=args.num_samples,
        )
    else:
        run_eval_e2e_cli(
            dataset_path=args.dataset,
            domain=args.domain,
            use_ragas=args.use_ragas,
            use_judge=args.use_judge,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_rounds=args.judge_rounds,
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

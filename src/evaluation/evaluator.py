"""
src/evaluation/evaluator.py - Bộ đánh giá chất lượng RAG tự động.
Thành viên phụ trách: Người đánh giá & đo lường chất lượng hệ thống (QA / Evaluator).
Nhiệm vụ: Chạy kiểm thử trên tập Benchmark QA và tính toán các chỉ số độ chính xác.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
from src.base import BaseEvaluator, EvaluationSample, EvaluationResult, EvaluationReport, Response
from src.evaluation.metrics import compute_faithfulness, compute_context_relevance, compute_keyword_overlap
from src.utils.helpers import get_logger

logger = get_logger("evaluation")


class UETEvaluator(BaseEvaluator):
    """
    Bộ đánh giá chất lượng RAG cho UET Chatbot.
    Thành viên phụ trách có thể cài đặt đánh giá bằng công thức (heuristic) hoặc dùng LLM-as-a-judge.
    """

    def __init__(self, use_llm_judge: bool = False):
        self.use_llm_judge = use_llm_judge

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        response: Response,
    ) -> EvaluationResult:
        """
        Đánh giá 1 câu trả lời theo các tiêu chí:
        - context_relevance: Ngữ cảnh trích xuất có đúng trọng tâm câu hỏi?
        - faithfulness: Câu trả lời có căn cứ vào ngữ cảnh hay bị ảo giác?
        - answer_relevance: Câu trả lời có đúng với Ground Truth / câu hỏi?
        """
        logger.info(f"Đang đánh giá mẫu kiểm thử: '{sample.query[:40]}...' (ID: {sample.id})")

        contexts_text = [c.text for c in response.sources]

        # 1. Đo độ liên quan của Context
        c_rel = compute_context_relevance(sample.query, contexts_text)

        # 2. Đo tính trung thực (Faithfulness)
        faith = compute_faithfulness(response.answer, contexts_text)

        # 3. Đo độ tương đồng với Ground Truth nếu có
        a_rel = 0.8
        if sample.ground_truth:
            a_rel = compute_keyword_overlap(response.answer, sample.ground_truth)

        metrics = {
            "context_relevance": round(c_rel, 3),
            "faithfulness": round(faith, 3),
            "answer_relevance": round(a_rel, 3),
            "overall_score": round((c_rel + faith + a_rel) / 3, 3),
        }

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH CÓ THỂ NÂNG CẤP ĐÁNH GIÁ:
        # Gợi ý:
        # 1. Tích hợp LLM-as-a-Judge (Dùng Gemini 2.5 Flash làm giám khảo chấm điểm 1-5 sao).
        # 2. Hoặc tích hợp thư viện chuyên dụng như Ragas (`pip install ragas`) hoặc TruLens.
        # =========================================================================

        return EvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            generated_answer=response.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=response.sources,
            metrics=metrics,
            feedback=f"Hoàn thành đánh giá mẫu {sample.id} với điểm tổng: {metrics['overall_score']}",
        )

    @staticmethod
    def load_dataset(dataset_path: Path) -> List[EvaluationSample]:
        """Tải tập dữ liệu benchmark từ file JSON."""
        if not dataset_path.exists():
            return []
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvaluationSample(**item) for item in data]


def run_eval_e2e_cli(dataset_path: Optional[Path] = None):
    """Chạy đánh giá toàn trình End-to-End RAG trên tập Benchmark QA và hiển thị trên CLI."""
    print("\n" + "=" * 65)
    print("📊 ĐANG CHẠY ĐÁNH GIÁ TOÀN TRÌNH RAG (END-TO-END EVALUATION)")
    print("=" * 65)
    from src.pipeline import get_rag_pipeline

    base_dir = Path(__file__).resolve().parent.parent.parent
    if dataset_path is None:
        dataset_path = base_dir / "data" / "eval_data" / "test_qa_dataset.json"

    dataset = UETEvaluator.load_dataset(dataset_path)

    if not dataset:
        print(f"❌ Không tìm thấy tập dữ liệu đánh giá tại: {dataset_path}")
        return

    # Lấy RAG Pipeline chính thức từ src/pipeline.py
    rag_pipeline = get_rag_pipeline()
    evaluator = UETEvaluator()
    report = evaluator.evaluate_pipeline(rag_pipeline, dataset)

    print("\n" + "=" * 65)
    print("📈 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TOÀN TRÌNH (RAG E2E REPORT)")
    print("=" * 65)
    print(f"Tổng số mẫu kiểm thử: {report.total_samples}")
    print("\nĐiểm số trung bình (Thang điểm 0.0 - 1.0):")
    for metric, score in report.average_metrics.items():
        print(f"  • {metric.replace('_', ' ').capitalize():<22}: {score:.3f}")

    print("\nChi tiết từng mẫu câu hỏi:")
    for idx, res in enumerate(report.results, 1):
        print(f"  [{idx}] Câu hỏi: {res.query}")
        print(f"      Trả lời: {res.generated_answer}")
        print(f"      Điểm số: {res.metrics}")
    print("=" * 65 + "\n")


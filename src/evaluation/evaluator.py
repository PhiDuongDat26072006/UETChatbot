"""
src/evaluation/evaluator.py - Bộ đánh giá chất lượng RAG tự động.
Thành viên phụ trách: Người đánh giá & đo lường chất lượng hệ thống (QA / Evaluator).
Nhiệm vụ: Chạy kiểm thử trên tập Benchmark QA và tính toán các chỉ số độ chính xác bằng:
1. UETEvaluator: Đánh giá bằng heuristic và quy tắc (offline, nhẹ).
2. UETRagasEvaluator: Đánh giá bằng thư viện Ragas chuyên dụng và LLM-as-a-Judge (Groq API).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Đảm bảo root directory có trong sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from src.base import (
    BaseEvaluator,
    BaseRAGPipeline,
    EvaluationSample,
    EvaluationResult,
    EvaluationReport,
    Response,
    UserQuery,
)
from src.evaluation.metrics import (
    compute_faithfulness,
    compute_context_relevance,
    compute_keyword_overlap,
)
from src.embeddings.embedder import UETEmbedder
from src.utils.helpers import get_logger

logger = get_logger("evaluation")


# =========================================================================
# 1. BỘ CHUYỂN ĐỔI EMBEDDINGS CHO RAGAS (LANGCHAIN COMPATIBLE)
# =========================================================================

try:
    from langchain_core.embeddings import Embeddings

    class UETLangchainEmbeddings(Embeddings):
        """
        Adapter bao bọc UETEmbedder (384 chiều, all-MiniLM-L6-v2)
        thành chuẩn LangChain Embeddings để Ragas sử dụng tính toán Answer Relevancy.
        """

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
# 2. BỘ ĐÁNH GIÁ CHUẨN RAGAS (LLM-AS-A-JUDGE QUA GROQ)
# =========================================================================

class UETRagasEvaluator:
    """
    Bộ đánh giá RAG nâng cao sử dụng thư viện Ragas và LLM-as-a-Judge (Groq API).
    Đo lường 4 chỉ số cốt lõi:
    - Faithfulness: Tính trung thực, kiểm tra câu trả lời có bị ảo giác so với ngữ cảnh không.
    - Context Precision: Độ chính xác và thứ hạng của ngữ cảnh được trích xuất.
    - Context Recall: Mức độ bao phủ thông tin từ Ground Truth trong ngữ cảnh.
    - Answer Relevancy: Mức độ bám sát và trả lời đúng trọng tâm câu hỏi của người dùng.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        embedder: Optional[UETEmbedder] = None,
    ):
        self.model_name = model_name
        self.base_url = base_url

        # Tải API key từ tham số hoặc file .env
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip().strip('"')
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY không được tìm thấy. Vui lòng thêm GROQ_API_KEY vào file .env "
                "hoặc truyền trực tiếp vào evaluator."
            )

        try:
            from openai import OpenAI
            from ragas.llms import llm_factory
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
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
                f"Đã khởi tạo thành công UETRagasEvaluator với model '{self.model_name}' (Judge) "
                f"và UETEmbedder (Embeddings)."
            )
        except ImportError as e:
            raise ImportError(
                f"Thiếu thư viện để chạy Ragas: {e}. Vui lòng cài đặt: pip install ragas datasets langchain-groq openai"
            )

    def evaluate_pipeline(
        self,
        pipeline: BaseRAGPipeline,
        dataset: List[EvaluationSample],
        top_k: int = 5,
        num_samples: Optional[int] = 4,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy quy trình RAG Pipeline trên tập dữ liệu và đánh giá bằng Ragas.
        :param pipeline: Đối tượng RAG Pipeline cần đánh giá.
        :param dataset: Danh sách các mẫu kiểm thử (EvaluationSample).
        :param top_k: Số lượng context trích xuất cho mỗi câu hỏi.
        :param num_samples: Giới hạn số câu hỏi chạy thử (mặc định 4).
        :param output_path: Đường dẫn lưu file kết quả JSON.
        :return: Dict chứa kết quả trung bình và chi tiết từng mẫu.
        """
        from datasets import Dataset
        from ragas import evaluate

        if num_samples is not None and num_samples > 0:
            active_dataset = dataset[:num_samples]
        else:
            active_dataset = dataset

        total_samples = len(active_dataset)
        logger.info(f"Bắt đầu đánh giá Ragas trên {total_samples} mẫu câu hỏi...")

        user_inputs: List[str] = []
        responses: List[str] = []
        retrieved_contexts: List[List[str]] = []
        references: List[str] = []
        sample_ids: List[str] = []

        for idx, sample in enumerate(active_dataset, 1):
            logger.info(f"[{idx}/{total_samples}] Đang chạy truy vấn: '{sample.query[:50]}...'")
            u_query = UserQuery(query_text=sample.query)
            resp = pipeline.query(u_query, top_k=top_k)

            sample_ids.append(sample.id)
            user_inputs.append(sample.query)
            responses.append(resp.answer)
            # Ngữ cảnh trích xuất
            contexts = [c.text for c in resp.sources] if resp.sources else ["Không tìm thấy ngữ cảnh."]
            retrieved_contexts.append(contexts)
            # Ground truth đối sánh
            references.append(sample.ground_truth or "")

        # Đóng gói thành HuggingFace Dataset theo chuẩn Ragas
        ragas_data = {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
        }
        hf_dataset = Dataset.from_dict(ragas_data)

        logger.info("Đang gọi Ragas evaluate với mô hình LLM-as-a-Judge...")
        eval_result = evaluate(
            dataset=hf_dataset,
            metrics=self.metrics,
            llm=self.judge_llm,
            embeddings=self.judge_embeddings,
            raise_exceptions=False,
        )

        df = eval_result.to_pandas()

        # Tính toán điểm trung bình
        metric_keys = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
        average_metrics: Dict[str, float] = {}
        for m in metric_keys:
            if m in df.columns:
                val = float(df[m].mean())
                average_metrics[m] = round(0.0 if (val != val) else val, 4)

        # Đóng gói chi tiết từng mẫu
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
                    val = float(df.iloc[i][m])
                    sample_dict["scores"][m] = round(0.0 if (val != val) else val, 4)
            detailed_samples.append(sample_dict)

        report_data: Dict[str, Any] = {
            "evaluator": "Ragas (LLM-as-a-Judge)",
            "judge_model": self.model_name,
            "total_samples": total_samples,
            "average_metrics": average_metrics,
            "samples": detailed_samples,
        }

        # Lưu kết quả ra file nếu được yêu cầu
        if output_path:
            save_file = Path(output_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu báo cáo đánh giá Ragas vào: {save_file}")

        return report_data


# =========================================================================
# 3. BỘ ĐÁNH GIÁ CƠ BẢN (HEURISTIC BASELINE EVALUATOR)
# =========================================================================

class UETEvaluator(BaseEvaluator):
    """
    Bộ đánh giá chất lượng RAG bằng heuristic cho UET Chatbot (nhẹ, offline).
    Tính toán nhanh các chỉ số tương đồng từ khóa, overlap và faithfulness sơ bộ.
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

        return EvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            generated_answer=response.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=response.sources,
            metrics=metrics,
            feedback=f"Hoàn thành đánh giá mẫu {sample.id} với điểm tổng: {metrics['overall_score']}",
        )

    def evaluate_pipeline(
        self,
        pipeline: BaseRAGPipeline,
        dataset: List[EvaluationSample],
        top_k: int = 5,
        num_samples: Optional[int] = None,
    ) -> EvaluationReport:
        """Chạy đánh giá tự động trên tập dữ liệu benchmark."""
        active_dataset = dataset[:num_samples] if (num_samples and num_samples > 0) else dataset
        results: List[EvaluationResult] = []
        metrics_sum: Dict[str, float] = {}

        for sample in active_dataset:
            user_query = UserQuery(query_text=sample.query)
            response = pipeline.query(user_query, top_k=top_k)
            result = self.evaluate_sample(sample, response)
            results.append(result)

            for metric_name, score in result.metrics.items():
                metrics_sum[metric_name] = metrics_sum.get(metric_name, 0.0) + score

        total = len(active_dataset)
        avg_metrics = {m: (s / total) for m, s in metrics_sum.items()} if total > 0 else {}

        return EvaluationReport(
            total_samples=total,
            average_metrics=avg_metrics,
            results=results,
        )

    @staticmethod
    def load_dataset(dataset_path: Path) -> List[EvaluationSample]:
        """Tải tập dữ liệu benchmark từ file JSON."""
        if not dataset_path.exists():
            return []
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvaluationSample(**item) for item in data]


# =========================================================================
# 4. GIAO DIỆN DÒNG LỆNH CLI (RUNNER)
# =========================================================================

def run_eval_e2e_cli(
    dataset_path: Optional[Path] = None,
    use_ragas: bool = False,
    num_samples: Optional[int] = 4,
    model_name: str = "openai/gpt-oss-120b",
    output_path: Optional[Path] = None,
):
    """Chạy đánh giá toàn trình End-to-End RAG trên tập Benchmark QA và hiển thị trên CLI."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if dataset_path is None:
        dataset_path = base_dir / "data" / "eval_data" / "test_qa_dataset.json"

    dataset = UETEvaluator.load_dataset(dataset_path)
    if not dataset:
        print(f"❌ Không tìm thấy tập dữ liệu đánh giá tại: {dataset_path}")
        return

    from src.pipeline import get_rag_pipeline
    rag_pipeline = get_rag_pipeline()

    if use_ragas:
        print("\n" + "=" * 70)
        print("🤖 ĐANG CHẠY ĐÁNH GIÁ RAG BẰNG RAGAS & LLM-AS-A-JUDGE (GROQ)")
        print(f"• Giám khảo LLM: {model_name}")
        print(f"• Số lượng câu hỏi đánh giá: {num_samples if num_samples else len(dataset)}")
        print("=" * 70)

        if output_path is None:
            output_path = base_dir / "data" / "eval_data" / "ragas_eval_results.json"

        evaluator = UETRagasEvaluator(model_name=model_name)
        report = evaluator.evaluate_pipeline(
            pipeline=rag_pipeline,
            dataset=dataset,
            num_samples=num_samples,
            output_path=output_path,
        )

        print("\n" + "=" * 70)
        print("📈 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ RAGAS (THANG ĐIỂM 0.0 - 1.0)")
        print("=" * 70)
        print(f"Tổng số mẫu kiểm thử: {report['total_samples']}\n")
        print("Chỉ số trung bình:")
        for metric, score in report["average_metrics"].items():
            print(f"  • {metric.replace('_', ' ').title():<22}: {score:.4f}")

        print("\nChi tiết từng câu hỏi:")
        for idx, item in enumerate(report["samples"], 1):
            print(f"  [{idx}] ({item['id']}) Câu hỏi: {item['query']}")
            print(f"      Trả lời : {item['generated_answer'][:120]}...")
            print(f"      Điểm số : {item['scores']}")
        print(f"\n📁 Kết quả chi tiết đã được xuất ra: {output_path}")
        print("=" * 70 + "\n")
        return report

    else:
        print("\n" + "=" * 70)
        print("📊 ĐANG CHẠY ĐÁNH GIÁ TOÀN TRÌNH RAG (HEURISTIC BASELINE)")
        print(f"• Số lượng câu hỏi: {num_samples if num_samples else len(dataset)}")
        print("=" * 70)

        evaluator = UETEvaluator()
        report = evaluator.evaluate_pipeline(rag_pipeline, dataset, num_samples=num_samples)

        print("\n" + "=" * 70)
        print("📈 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TOÀN TRÌNH (RAG E2E REPORT)")
        print("=" * 70)
        print(f"Tổng số mẫu kiểm thử: {report.total_samples}")
        print("\nĐiểm số trung bình (Thang điểm 0.0 - 1.0):")
        for metric, score in report.average_metrics.items():
            print(f"  • {metric.replace('_', ' ').capitalize():<22}: {score:.3f}")

        print("\nChi tiết từng mẫu câu hỏi:")
        for idx, res in enumerate(report.results, 1):
            print(f"  [{idx}] Câu hỏi: {res.query}")
            print(f"      Trả lời: {res.generated_answer[:120]}...")
            print(f"      Điểm số: {res.metrics}")
        print("=" * 70 + "\n")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UET Chatbot RAG Evaluation Runner")
    parser.add_argument("--use-ragas", action="store_true", help="Sử dụng thư viện Ragas và LLM-as-a-Judge")
    parser.add_argument("--num-samples", type=int, default=4, help="Số lượng mẫu kiểm thử (mặc định: 4)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Tên mô hình LLM Judge qua Groq")
    parser.add_argument("--dataset", type=str, default=None, help="Đường dẫn file dataset QA kiểm thử")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu kết quả JSON")
    args = parser.parse_args()

    ds_path = Path(args.dataset) if args.dataset else None
    out_path = Path(args.output) if args.output else None

    run_eval_e2e_cli(
        dataset_path=ds_path,
        use_ragas=args.use_ragas,
        num_samples=args.num_samples,
        model_name=args.model,
        output_path=out_path,
    )

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
import time
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

# Ánh xạ tên domain viết tắt sang tên file trong thư mục data/eval_data/uet_domain_qa/
DOMAIN_FILE_MAP = {
    # Nhóm câu hỏi tổng quan
    "overview": ["overview1_qa.json", "overview2_qa.json"],
    "overview1": ["overview1_qa.json"],
    "overview2": ["overview2_qa.json"],
    "general": ["overview1_qa.json", "overview2_qa.json", "general_qa.json"],
    "uet": ["overview2_qa.json", "overview1_qa.json", "uet_qa.json"],
    "test_qa": ["overview1_qa.json", "test_qa_dataset.json"],

    # Sổ tay sinh viên
    "handbook": ["handbook2_qa.json", "handbook1_qa.json"],
    "handbook1": ["handbook1_qa.json", "handbook.uet.vnu.edu.vn.json"],
    "handbook2": ["handbook2_qa.json", "uet_handbook_qa.json", "handbook_qa.json"],

    # Các khoa / viện chuyên ngành
    "fit": ["fit_qa.json", "fit.uet.vnu.edu.vn.json"],
    "fet": ["fet_qa.json", "fet.uet.vnu.edu.vn.json"],
    "fema": ["fema_qa.json", "fema.uet.vnu.edu.vn.json"],
    "sae": ["sae_qa.json", "sae.uet.vnu.edu.vn.json"],
}


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
        delay_between_queries: float = 1.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Chạy quy trình RAG Pipeline trên tập dữ liệu và đánh giá bằng Ragas.
        :param pipeline: Đối tượng RAG Pipeline cần đánh giá.
        :param dataset: Danh sách các mẫu kiểm thử (EvaluationSample).
        :param top_k: Số lượng context trích xuất cho mỗi câu hỏi.
        :param num_samples: Giới hạn số câu hỏi chạy thử (mặc định 4).
        :param output_path: Đường dẫn lưu file kết quả JSON.
        :param delay_between_queries: Độ trễ giữa các truy vấn (giây) để tránh Rate Limit.
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
                u_query = UserQuery(query_text=sample.query)
                resp = pipeline.query(u_query, top_k=top_k)
                print(f"        ✅ Hoàn thành [{resp.latency_seconds:.2f}s]")
            except Exception as e:
                logger.warning(f"Lỗi khi truy vấn câu hỏi '{sample.query[:30]}': {e}")
                resp = Response(answer="Không thể tạo câu trả lời do lỗi kết nối LLM.", sources=[])
                print(f"        ⚠️ Lỗi truy vấn: {e}")

            sample_ids.append(sample.id)
            user_inputs.append(sample.query)
            responses.append(resp.answer)
            # Ngữ cảnh trích xuất
            contexts = [c.text for c in resp.sources] if resp.sources else ["Không tìm thấy ngữ cảnh."]
            retrieved_contexts.append(contexts)
            # Ground truth đối sánh
            references.append(sample.ground_truth or "")

            if idx < total_samples and delay_between_queries > 0:
                time.sleep(delay_between_queries)

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
        delay_between_queries: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> EvaluationReport:
        """Chạy đánh giá tự động trên tập dữ liệu benchmark với hiển thị tiến trình mượt mà."""
        active_dataset = dataset[:num_samples] if (num_samples and num_samples > 0) else dataset
        results: List[EvaluationResult] = []
        metrics_sum: Dict[str, float] = {}

        total = len(active_dataset)
        print(f"\n🚀 Đang đánh giá toàn trình trên {total} câu hỏi (Top-K={top_k}, Delay={delay_between_queries}s)...")
        print("─" * 78)

        for idx, sample in enumerate(active_dataset, 1):
            short_q = (sample.query[:56] + "...") if len(sample.query) > 56 else sample.query
            print(f"[{idx:02d}/{total:02d}] 🔍 '{short_q}'")

            try:
                user_query = UserQuery(query_text=sample.query)
                response = pipeline.query(user_query, top_k=top_k)
                result = self.evaluate_sample(sample, response)
                m = result.metrics
                print(
                    f"        ✅ Score: {m['overall_score']:.2f} "
                    f"(Context: {m['context_relevance']:.2f} | Faith: {m['faithfulness']:.2f} | Answer: {m['answer_relevance']:.2f}) "
                    f"[{response.latency_seconds:.2f}s]"
                )
            except Exception as e:
                logger.warning(f"Lỗi khi đánh giá mẫu {sample.id}: {e}")
                # Tự động retry 1 lần sau 3.0s nếu gặp rate limit hoặc chập chờn mạng
                time.sleep(3.0)
                try:
                    user_query = UserQuery(query_text=sample.query)
                    response = pipeline.query(user_query, top_k=top_k)
                    result = self.evaluate_sample(sample, response)
                    m = result.metrics
                    print(f"        ✅ Score (sau retry): {m['overall_score']:.2f} [{response.latency_seconds:.2f}s]")
                except Exception as retry_err:
                    print(f"        ❌ Thất bại: {retry_err}")
                    dummy_resp = Response(answer=f"Lỗi truy vấn: {retry_err}", sources=[])
                    result = self.evaluate_sample(sample, dummy_resp)

            results.append(result)

            for metric_name, score in result.metrics.items():
                metrics_sum[metric_name] = metrics_sum.get(metric_name, 0.0) + score

            # Giãn cách giữa các lần gọi để tránh chạm trần Rate Limit Groq (7000 tokens/phút)
            if idx < total and delay_between_queries > 0:
                time.sleep(delay_between_queries)

        avg_metrics = {m: (s / total) for m, s in metrics_sum.items()} if total > 0 else {}
        report = EvaluationReport(
            total_samples=total,
            average_metrics=avg_metrics,
            results=results,
        )

        # Lưu kết quả chi tiết ra file JSON
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            report_dict = {
                "total_samples": report.total_samples,
                "average_metrics": report.average_metrics,
                "results": [
                    {
                        "sample_id": r.sample_id,
                        "query": r.query,
                        "generated_answer": r.generated_answer,
                        "ground_truth": r.ground_truth,
                        "metrics": r.metrics,
                        "feedback": r.feedback,
                    }
                    for r in report.results
                ],
            }
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            print(f"\n📁 Đã lưu kết quả đánh giá chi tiết vào: {out_p}")

        return report

    @staticmethod
    def load_dataset(dataset_path: Union[str, Path], domain: Optional[str] = None) -> List[EvaluationSample]:
        """
        Tải tập dữ liệu benchmark từ file JSON/JSONL hoặc từ thư mục uet_domain_qa.
        Hỗ trợ:
        - Đọc file đơn: .json hoặc .jsonl
        - Đọc theo domain (ví dụ domain="fit", "fema", "fet", "sae", "handbook", "overview", "all")
        - Quét toàn bộ thư mục nếu truyền đường dẫn thư mục
        """
        path = Path(dataset_path)

        # Tự động chuyển đổi nếu đường dẫn cũ/mới bị thay đổi
        if not path.exists():
            if "e2e_domain_qa" in str(path):
                alt = Path(str(path).replace("e2e_domain_qa", "uet_domain_qa"))
                if alt.exists():
                    path = alt
            elif "uet_domain_qa" in str(path):
                alt = Path(str(path).replace("uet_domain_qa", "e2e_domain_qa"))
                if alt.exists():
                    path = alt

        # 1. Trường hợp path là thư mục hoặc có truyền domain
        if path.is_dir() or domain:
            base_dir = path if path.is_dir() else path.parent
            if domain:
                domain_key = domain.lower().strip()
                if domain_key == "all":
                    # Gộp tất cả các file JSON trong thư mục (tránh file trùng lặp)
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
                    # Tìm file có chứa từ khóa domain trong tên file
                    for f in base_dir.glob(f"*{domain_key}*"):
                        if f.suffix in [".json", ".jsonl"]:
                            return UETEvaluator._read_file(f)

            # Nếu là thư mục nhưng không truyền domain hoặc không khớp domain cụ thể:
            for def_name in ["overview1_qa.json", "overview2_qa.json", "handbook2_qa.json", "general_qa.json", "test_qa_dataset.json"]:
                default_file = base_dir / def_name
                if default_file.exists():
                    return UETEvaluator._read_file(default_file)

            for f in sorted(base_dir.glob("*.json")):
                return UETEvaluator._read_file(f)
            return []

        # 2. Trường hợp path là đường dẫn tới 1 file cụ thể
        if not path.exists():
            return []

        return UETEvaluator._read_file(path)

    @staticmethod
    def _read_file(file_path: Path) -> List[EvaluationSample]:
        """Đọc và chuẩn hóa dữ liệu từ file JSON hoặc JSONL thành danh sách EvaluationSample."""
        samples: List[EvaluationSample] = []
        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            item = json.loads(line_str)
                            samples.append(EvaluationSample(**item))
                        except Exception as e:
                            logger.warning(f"Bỏ qua dòng {line_num} trong {file_path.name}: {e}")
            elif file_path.suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            samples.append(EvaluationSample(**item))
                    elif isinstance(data, dict):
                        samples.append(EvaluationSample(**data))
        except Exception as e:
            logger.error(f"Lỗi khi đọc file dataset '{file_path}': {e}")
        return samples


# =========================================================================
# 4. GIAO DIỆN DÒNG LỆNH CLI (RUNNER)
# =========================================================================

def run_eval_e2e_cli(
    dataset_path: Optional[Union[str, Path]] = None,
    domain: Optional[str] = None,
    use_ragas: bool = False,
    num_samples: Optional[int] = None,
    top_k: Optional[int] = None,
    model_name: str = "openai/gpt-oss-120b",
    output_path: Optional[Path] = None,
    delay: float = 1.2,
):
    """Chạy đánh giá toàn trình End-to-End RAG trên tập Benchmark QA và hiển thị trên CLI."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if top_k is None:
        from src.utils.helpers import load_yaml_config
        cfg = load_yaml_config()
        top_k = cfg.get("retrieval", {}).get("final_top_k", 5)
    e2e_dir = base_dir / "data" / "eval_data" / "uet_domain_qa"
    if not e2e_dir.exists():
        fallback_dir = base_dir / "data" / "eval_data" / "e2e_domain_qa"
        if fallback_dir.exists():
            e2e_dir = fallback_dir

    # Xác định tập dữ liệu
    target_path = Path(dataset_path) if dataset_path else e2e_dir
    dataset = UETEvaluator.load_dataset(target_path, domain=domain)

    domain_str = f" [Domain: {domain.upper()}]" if domain else ""
    target_name = target_path.name if target_path.is_file() else f"uet_domain_qa{domain_str}"

    if not dataset:
        print(f"❌ Không tìm thấy tập dữ liệu đánh giá hợp lệ tại: {target_path}")
        if domain:
            print(f"   Domain hỗ trợ: {', '.join(DOMAIN_FILE_MAP.keys())}, all")
        return

    from src.pipeline import get_rag_pipeline
    rag_pipeline = get_rag_pipeline()

    actual_samples = num_samples if (num_samples and num_samples > 0) else len(dataset)

    if use_ragas:
        print("\n" + "=" * 70)
        print("🤖 ĐANG CHẠY ĐÁNH GIÁ RAG BẰNG RAGAS & LLM-AS-A-JUDGE (GROQ)")
        print(f"• Tập dữ liệu: {target_name} (Tổng: {len(dataset)} câu)")
        print(f"• Giám khảo LLM: {model_name}")
        print(f"• Số lượng câu hỏi đánh giá: {actual_samples}")
        print(f"• Top-K ngữ cảnh: {top_k}")
        print("=" * 70)

        if output_path is None:
            output_path = base_dir / "data" / "eval_data" / "ragas_eval_results.json"

        evaluator = UETRagasEvaluator(model_name=model_name)
        report = evaluator.evaluate_pipeline(
            pipeline=rag_pipeline,
            dataset=dataset,
            top_k=top_k,
            num_samples=num_samples,
            output_path=output_path,
            delay_between_queries=delay,
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
            print(f"  [{idx:02d}] ({item['id']}) Câu hỏi: {item['query']}")
            print(f"       Trả lời : {item['generated_answer'][:120]}...")
            print(f"       Điểm số : {item['scores']}")
        print(f"\n📁 Kết quả chi tiết đã được xuất ra: {output_path}")
        print("=" * 70 + "\n")
        return report

    else:
        print("\n" + "=" * 70)
        print("📊 ĐANG CHẠY ĐÁNH GIÁ TOÀN TRÌNH RAG (HEURISTIC BASELINE)")
        print(f"• Tập dữ liệu: {target_name} (Tổng: {len(dataset)} câu)")
        print(f"• Số lượng câu hỏi đánh giá: {actual_samples}")
        print(f"• Top-K ngữ cảnh: {top_k}")
        print("=" * 70)

        if output_path is None:
            output_path = base_dir / "data" / "eval_data" / "e2e_eval_results.json"

        evaluator = UETEvaluator()
        report = evaluator.evaluate_pipeline(
            pipeline=rag_pipeline,
            dataset=dataset,
            top_k=top_k,
            num_samples=num_samples,
            delay_between_queries=delay,
            output_path=output_path,
        )

        print("\n" + "=" * 70)
        print("📈 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TOÀN TRÌNH (RAG E2E REPORT)")
        print("=" * 70)
        print(f"Tổng số mẫu kiểm thử: {report.total_samples}")
        print("\nĐiểm số trung bình (Thang điểm 0.0 - 1.0):")
        for metric, score in report.average_metrics.items():
            print(f"  • {metric.replace('_', ' ').capitalize():<22}: {score:.3f}")

        print("\nChi tiết từng mẫu câu hỏi:")
        for idx, res in enumerate(report.results, 1):
            print(f"  [{idx:02d}] ({res.sample_id}) Câu hỏi: {res.query}")
            print(f"       Trả lời: {res.generated_answer[:120]}...")
            print(f"       Điểm số: {res.metrics}")
        print("=" * 70 + "\n")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UET Chatbot RAG Evaluation Runner")
    parser.add_argument("--use-ragas", action="store_true", help="Sử dụng thư viện Ragas và LLM-as-a-Judge")
    parser.add_argument("--num-samples", "--samples", type=int, default=None, dest="num_samples", help="Số lượng mẫu kiểm thử (mặc định: tất cả)")
    parser.add_argument("--domain", type=str, default=None, help="Chỉ định khoa/ngành: fit, fet, fema, sae, handbook, overview, all")
    parser.add_argument("--top-k", type=int, default=5, help="Số lượng chunks ngữ cảnh lấy ra")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Tên mô hình LLM Judge qua Groq")
    parser.add_argument("--delay", type=float, default=1.2, help="Thời gian nghỉ giữa các câu hỏi (giây)")
    parser.add_argument("--dataset", type=str, default=None, help="Đường dẫn file dataset QA kiểm thử")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu kết quả JSON")
    args = parser.parse_args()

    ds_path = Path(args.dataset) if args.dataset else None
    out_path = Path(args.output) if args.output else None

    run_eval_e2e_cli(
        dataset_path=ds_path,
        domain=args.domain,
        use_ragas=args.use_ragas,
        num_samples=args.num_samples,
        top_k=args.top_k,
        model_name=args.model,
        output_path=out_path,
        delay=args.delay,
    )

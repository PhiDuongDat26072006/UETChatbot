"""
tests/test_evaluation.py - Unit & Integration tests cho phân hệ Evaluation & Benchmark Metrics.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import BaseRAGPipeline, EvaluationSample, Response, RetrievedContext, UserQuery
from src.evaluation.diagnostics import build_e2e_report, classify_sample_error
from src.evaluation.evaluate_retrieval import RetrievalEvaluator
from src.evaluation.evaluator import (
    UETEvaluator,
    _clean_ragas_value,
    _ensure_ragas_compat,
    generate_predictions,
    score_predictions,
)
from src.evaluation.llm_judge import UETLLMJudgeEvaluator, normalize_likert_5
from src.evaluation.metrics import (
    average_precision,
    compute_answer_correctness_offline,
    compute_answer_relevance,
    compute_context_relevance,
    compute_faithfulness,
    compute_keyword_overlap,
    compute_numeric_grounding,
    hit_rate_at_k,
    is_generation_error_response,
    is_refusal_response,
    mrr,
    ndcg_at_k,
    normalize_vietnamese_for_eval,
    precision_at_k,
    recall_at_k,
)


class _DummyEmbedder:
    """Mock embedder deterministic offline dùng vector từ vựng đơn giản."""

    dimension = 8

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vecs: List[List[float]] = []
        vocab = ["điều", "kiện", "tốt", "nghiệp", "robot", "đối", "tác", "tín"]
        for t in texts:
            low = t.lower()
            v = [float(low.count(w) + 0.05) for w in vocab]
            vecs.append(v)
        return vecs


class _DummyRetriever:
    """Mock retriever để test config-hash cache & OOD stats trong RetrievalEvaluator."""

    def __init__(self, enable_mmr: bool = False, bm25_weight: float = 0.6):
        self.enable_mmr = enable_mmr
        self.mmr_lambda = 0.7
        self.mmr_top_n = 40
        self.enable_graph = True
        self.graph_weight = 0.3
        self.graph_top_k = 50
        self.graph_max_hops = 2
        self.graph_seed_top_m = 10
        self.bm25_weight = bm25_weight
        self.vector_weight = 0.2
        self.rrf_k = 30
        self.bm25_top_k = 50
        self.vector_top_k = 50
        self.candidate_pool_size = 50
        self.rerank_top_n = 20
        self.default_score_threshold = None
        self.reranker = None
        self.calls = 0

    def retrieve(self, query: UserQuery, top_k: int = 5) -> List[RetrievedContext]:
        self.calls += 1
        is_ood = "sư phạm" in query.query_text.lower()
        score = 0.21 if is_ood else 0.89
        cid = "c_ood" if is_ood else ("c_1" if not self.enable_mmr else "c_wrong")
        did = "d_ood" if is_ood else ("d_1" if not self.enable_mmr else "d_wrong")
        return [
            RetrievedContext(
                chunk_id=cid,
                text="Nội dung quy chế UET mẫu.",
                similarity_score=score,
                rank=1,
                metadata={"document_id": did},
            )
        ]


class _DummyPipeline(BaseRAGPipeline):
    """Mock pipeline tối giản cho test luồng 2 pha Generate -> Score."""

    def __init__(self):
        self.embedding_model = _DummyEmbedder()

    def query(self, query: UserQuery, top_k: int = 5) -> Response:
        if "sư phạm" in query.query_text.lower():
            return Response(
                query_id=query.query_id,
                answer="Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này.",
                sources=[],
                latency_seconds=0.05,
                model_name="mock-gen",
            )
        return Response(
            query_id=query.query_id,
            answer="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ.",
            sources=[
                RetrievedContext(
                    chunk_id="c_1",
                    text="Sinh viên tốt nghiệp sớm cần GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ.",
                    similarity_score=0.92,
                    rank=1,
                    metadata={"document_id": "d_1", "title": "Quy chế"},
                )
            ],
            latency_seconds=0.05,
            model_name="mock-gen",
        )


class TestEvaluationMetrics(unittest.TestCase):
    """Kiểm thử các hàm đo lường chất lượng truy xuất và sinh văn bản (gồm 9 test nền)."""

    def test_precision_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_1", "doc_3"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=2), 0.5)
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=4), 0.5)
        dup_retrieved = ["doc_1", "doc_1", "doc_2"]
        self.assertAlmostEqual(precision_at_k(dup_retrieved, relevant, k=2), 0.5)

    def test_recall_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_3", "doc_5", "doc_6"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 0.25)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 0.5)
        self.assertAlmostEqual(recall_at_k(retrieved, set(), k=3), 0.0)

    def test_ndcg_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1": 1.0, "doc_2": 1.0}
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=2), 1.0)
        retrieved_bad = ["doc_3", "doc_1"]
        ndcg_val = ndcg_at_k(retrieved_bad, relevant, k=2)
        self.assertLess(ndcg_val, 1.0)
        self.assertGreater(ndcg_val, 0.0)

    def test_mrr(self):
        retrieved = ["doc_x", "doc_1", "doc_2"]
        relevant = {"doc_1"}
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.5)
        self.assertAlmostEqual(mrr(["doc_1", "doc_x"], relevant), 1.0)
        self.assertAlmostEqual(mrr(["doc_a", "doc_b"], relevant), 0.0)

    def test_hit_rate_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        self.assertEqual(hit_rate_at_k(retrieved, {"doc_2"}, k=2), 1.0)
        self.assertEqual(hit_rate_at_k(retrieved, {"doc_3"}, k=2), 0.0)

    def test_average_precision(self):
        retrieved = ["doc_1", "doc_x", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        self.assertAlmostEqual(average_precision(retrieved, relevant), 5 / 6, places=3)

    def test_heuristic_faithfulness(self):
        answer = "Sinh viên cần có điểm GPA từ 3.0 và 90 tín chỉ."
        contexts = ["Sinh viên phải có điểm GPA từ 3.0 và tích lũy tối thiểu 90 tín chỉ để tốt nghiệp."]
        faith = compute_faithfulness(answer, contexts)
        self.assertGreater(faith, 0.7)

    def test_heuristic_context_relevance(self):
        query = "điều kiện tốt nghiệp sinh viên"
        contexts = ["điều kiện tốt nghiệp sinh viên chính quy"]
        relevance = compute_context_relevance(query, contexts)
        self.assertGreater(relevance, 0.5)


class TestUETEvaluator(unittest.TestCase):
    """Kiểm thử lớp UETEvaluator nâng cấp (tương thích backward + OOD + Judge)."""

    def test_evaluate_sample(self):
        evaluator = UETEvaluator()
        sample = EvaluationSample(
            id="test_01",
            query="Điều kiện tốt nghiệp sớm là gì?",
            ground_truth="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ.",
        )
        response = Response(
            query_id="q_01",
            answer="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ.",
            sources=[
                RetrievedContext(
                    chunk_id="c_1",
                    text="Điều kiện tốt nghiệp sớm là sinh viên có GPA >= 3.0 và tích lũy 90 tín chỉ.",
                    similarity_score=0.95,
                    rank=1,
                )
            ],
            latency_seconds=0.1,
            model_name="test-model",
        )
        res = evaluator.evaluate_sample(sample, response)
        self.assertEqual(res.sample_id, "test_01")
        self.assertIn("faithfulness", res.metrics)
        self.assertIn("context_relevance", res.metrics)
        self.assertIn("answer_relevance", res.metrics)
        self.assertIn("answer_correctness", res.metrics)
        self.assertIn("overall_score", res.metrics)
        self.assertGreater(res.metrics["overall_score"], 0.7)

    def test_eval_001_markdown_table_not_penalized(self):
        evaluator = UETEvaluator(embedder=_DummyEmbedder())
        sample = EvaluationSample(
            id="eval_001",
            query="Bộ môn Kỹ thuật Robot của Khoa Điện tử - Viễn thông (FET), Trường Đại học Công nghệ - ĐHQGHN (UET) có những đối tác nào?",
            ground_truth=(
                "Bộ môn Kỹ thuật Robot có quan hệ quốc tế, quốc gia với: Viện Công nghệ Chiba (Nhật Bản), "
                "Đại học Sungkyunkwan (Hàn Quốc), Đại học Công nghệ Sydney (Úc), Đại học Lisbon (Bồ Đào Nha), "
                "Học viện Kỹ thuật Quân sự, Công ty Lattel Robotics (Singapore), Aimesoft JSC., Samsung Display Vietnam."
            ),
            metadata={"unit": "FET", "type": "list", "target_chunk_ids": {"c_gold": 2}},
        )
        markdown_answer = (
            "Chào em, Bộ môn Kỹ thuật Robot (FET - UET) có các đối tác sau:\n\n"
            "| STT | Đối tác | Quốc gia | Nội dung hợp tác |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| 1 | Viện Công nghệ Chiba | Nhật Bản | Hợp tác từ 2018 |\n"
            "| 2 | Đại học Sungkyunkwan | Hàn Quốc | Quan hệ quốc tế |\n"
            "| 3 | Đại học Công nghệ Sydney | Úc | Quan hệ quốc tế |\n"
            "| 4 | Đại học Lisbon | Bồ Đào Nha | Quan hệ quốc tế |\n"
            "| 5 | Công ty Lattel Robotics | Singapore | Quan hệ quốc tế |\n"
            "| 6 | Aimesoft JSC. | Không nêu rõ | Quan hệ quốc tế |\n"
            "| 7 | Samsung Display Vietnam | Không nêu rõ | Quan hệ quốc gia |\n"
            "| 8 | Học viện Kỹ thuật Quân sự | Không nêu rõ | Quan hệ quốc gia |\n\n"
            "📌 Căn cứ quy chế: Tài liệu 1, 2, 3 – Bộ môn Kỹ thuật Robot."
        )
        ctx_text = (
            "Bộ môn Kỹ thuật Robot (FET - UET) có quan hệ quốc tế, quốc gia với Viện Công nghệ Chiba (Nhật Bản) "
            "hợp tác từ 2018, Đại học Sungkyunkwan (Hàn Quốc), Đại học Công nghệ Sydney (Úc), Đại học Lisbon "
            "(Bồ Đào Nha), Học viện Kỹ thuật Quân sự, Công ty Lattel Robotics (Singapore), Aimesoft JSC., Samsung Display Vietnam."
        )
        resp = Response(
            query_id="q_001",
            answer=markdown_answer,
            sources=[RetrievedContext(chunk_id="c_gold", text=ctx_text, similarity_score=0.9, rank=1)],
        )
        res = evaluator.evaluate_sample(sample, resp)
        self.assertGreaterEqual(res.metrics["token_recall"], 0.85)
        self.assertGreaterEqual(res.metrics["overall_score"], 0.70)
        self.assertEqual(res.metrics["false_refusal"], 0.0)

    def test_ood_sample_scoring(self):
        evaluator = UETEvaluator()
        sample = EvaluationSample(
            id="eval_003",
            query="Trường Đại học Sư phạm Hà Nội tuyển sinh thế nào?",
            ground_truth="Cơ sở dữ liệu của UET không lưu trữ thông tin của Đại học Sư phạm Hà Nội.",
            metadata={"type": "out_of_domain", "unit": "out_of_domain", "target_chunk_ids": {}},
        )
        refusal_resp = Response(
            query_id="q_ood",
            answer=(
                "Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. "
                "Em vui lòng liên hệ Phòng Đào tạo (Phòng 105-E3).\n"
                "📌 Căn cứ quy chế: Quyết định số 5115/QĐ-ĐHQGHN."
            ),
            sources=[],
        )
        res_ok = evaluator.evaluate_sample(sample, refusal_resp)
        self.assertEqual(res_ok.metrics["ood_refusal_accuracy"], 1.0)
        self.assertEqual(res_ok.metrics["overall_score"], 1.0)

        hallucinated_resp = Response(
            query_id="q_ood_bad",
            answer="Trường Đại học Sư phạm Hà Nội xét tuyển bằng học bạ và điểm thi tốt nghiệp THPT.",
            sources=[],
        )
        res_bad = evaluator.evaluate_sample(sample, hallucinated_resp)
        self.assertEqual(res_bad.metrics["ood_refusal_accuracy"], 0.0)


class TestUpgradedOfflineMetrics(unittest.TestCase):
    """Kiểm thử chuyên sâu tiền xử lý tiếng Việt, Numeric Guard và Refusal Detector."""

    def test_normalize_strips_markdown_and_footer(self):
        raw = "Chào em,\n| STT | Ngành |\n| :--- | :--- |\n| 1 | CNTT |\n📌 Căn cứ quy chế: Tài liệu 1."
        clean = normalize_vietnamese_for_eval(raw)
        self.assertNotIn("căn cứ quy chế", clean)
        self.assertNotIn("chào em", clean)
        self.assertNotIn("|", clean)
        self.assertIn("cntt", clean)

    def test_context_relevance_not_capped_on_long_chunks(self):
        query = "điều kiện xét học bổng khuyến khích học tập UET"
        long_chunk = (
            "Quy định xét cấp học bổng khuyến khích học tập tại UET áp dụng cho sinh viên chính quy. "
            + " ".join(["quy định chi tiết về điểm rèn luyện và số tín chỉ tích lũy theo học kỳ"] * 20)
        )
        score = compute_context_relevance(query, [long_chunk])
        self.assertGreaterEqual(score, 0.70)

    def test_numeric_grounding_catches_hallucination_and_ignores_stt(self):
        contexts = ["Sinh viên đạt điểm GPA >= 3.60 và học bổng năm 2026 có điểm chuẩn 26.5."]
        bad_answer = (
            "| STT | Mục |\n| :--- | :--- |\n| 1 | GPA >= 2.00 |\n| 2 | Năm 2025 là 26.5 |\n"
            "📌 Căn cứ quy chế: Tài liệu 1, 2."
        )
        res_bad = compute_numeric_grounding(bad_answer, contexts)
        self.assertLess(res_bad["numeric_faithfulness"], 1.0)
        self.assertIn("2.0", res_bad["ungrounded_entities"])
        self.assertIn("2025", res_bad["ungrounded_entities"])
        # STT 1, 2 và Tài liệu 1, 2 không bị báo nhầm
        self.assertNotIn("1", res_bad["ungrounded_entities"])

        good_table = "| STT | Ngành |\n| :--- | :--- |\n| 1 | CN8 |\n| 2 | GPA 3.60 năm 2026 điểm chuẩn 26.5 |\n📌 Căn cứ quy chế: Tài liệu 1, 2."
        res_good = compute_numeric_grounding(good_table, contexts + ["Mã ngành CN8."])
        self.assertEqual(res_good["numeric_faithfulness"], 1.0)
        self.assertEqual(res_good["ungrounded_entities"], [])

    def test_is_refusal_response_avoids_false_positive_in_table(self):
        table_with_unknown_country = (
            "Chào em, dưới đây là bảng đối tác:\n"
            "| STT | Đối tác | Quốc gia |\n| :--- | :--- | :--- |\n"
            "| 1 | Chiba | Nhật Bản |\n"
            "| 2 | Aimesoft JSC. | Không nêu rõ |\n"
            "| 3 | Samsung Display | Không nêu rõ |\n"
            "| 4 | Học viện KTQS | Không nêu rõ |\n"
        )
        self.assertFalse(is_refusal_response(table_with_unknown_country))
        self.assertTrue(is_refusal_response("Xin lỗi em, hiện tại chưa có thông tin cụ thể về câu hỏi này."))

    def test_answer_relevance_and_correctness_offline(self):
        rel = compute_answer_relevance(
            "Học bổng loại Giỏi cần GPA bao nhiêu?",
            "Học bổng loại Giỏi cần điểm GPA từ 3.20 trở lên và rèn luyện loại Tốt.",
        )
        self.assertGreater(rel, 0.5)
        corr = compute_answer_correctness_offline(
            "Sinh viên phải có GPA 3.60 và 15 tín chỉ.",
            "Yêu cầu GPA 3.60 và tối thiểu 15 tín chỉ.",
        )
        self.assertGreater(corr["answer_correctness"], 0.7)
        self.assertGreater(compute_keyword_overlap("GPA 3.60", "| 1 | GPA 3.60 |"), 0.7)

    def test_vietnamese_verbal_and_slash_date_grounding(self):
        res_month = compute_numeric_grounding(
            answer="Tạp chí ra đời từ tháng 1/2011.",
            contexts=["Tiếp nối thành công, tháng 1 năm 2011, tạp chí REV ra đời."],
        )
        self.assertEqual(res_month["numeric_faithfulness"], 1.0)
        self.assertEqual(res_month["ungrounded_entities"], [])

        res_dmy = compute_numeric_grounding(
            answer="Thời gian áp dụng quy định từ 01/05/2022.",
            contexts=["2. Thời gian áp dụng: từ ngày 01 tháng 5 năm 2022."],
        )
        self.assertEqual(res_dmy["numeric_faithfulness"], 1.0)
        self.assertEqual(res_dmy["ungrounded_entities"], [])

        res_fabricated = compute_numeric_grounding(
            answer="Thời gian áp dụng quy định từ 15/08/2023.",
            contexts=["2. Thời gian áp dụng: từ ngày 01 tháng 5 năm 2022."],
        )
        self.assertLess(res_fabricated["numeric_faithfulness"], 1.0)
        self.assertIn("15/8/2023", res_fabricated["ungrounded_entities"])


class TestUETLLMJudgeEvaluator(unittest.TestCase):
    """Kiểm thử UETLLMJudgeEvaluator (Structured JSON, retry, error handling, cache)."""

    def test_judge_valid_json_cleanup_and_cache(self):
        calls = {"n": 0}

        def fake_llm(_prompt: str) -> str:
            calls["n"] += 1
            payload = {
                "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "Chuẩn"},
                "answer_correctness": {"score": 4, "missing_facts": [], "wrong_facts": [], "reason": "Đủ ý chính"},
                "answer_relevance": {"score": 5, "reason": "Trực diện"},
                "context_sufficiency": {"score": 5, "reason": "Đủ ngữ cảnh"},
            }
            return f"<think>Đang suy luận...</think>\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "judge_cache.json"
            judge = UETLLMJudgeEvaluator(
                provider="groq",
                model_name="mock-judge",
                delay_seconds=0.0,
                cache_file=cache_path,
                call_fn=fake_llm,
            )
            r1 = judge.judge_record(
                query="Học bổng?",
                contexts=["GPA 3.2"],
                answer="GPA 3.2",
                ground_truth="GPA 3.2",
                sample_id="s1",
            )
            self.assertEqual(r1["status"], "ok")
            self.assertFalse(r1["cache_hit"])
            self.assertEqual(r1["normalized_scores"]["faithfulness"], 1.0)
            self.assertEqual(r1["normalized_scores"]["answer_correctness"], 0.75)
            self.assertEqual(normalize_likert_5(1), 0.0)
            self.assertEqual(normalize_likert_5(3), 0.5)

            r2 = judge.judge_record(
                query="Học bổng?",
                contexts=["GPA 3.2"],
                answer="GPA 3.2",
                ground_truth="GPA 3.2",
                sample_id="s1",
            )
            self.assertEqual(r2["status"], "ok")
            self.assertTrue(r2["cache_hit"])
            self.assertEqual(calls["n"], 1)

    def test_judge_retry_and_error_never_zero(self):
        state = {"count": 0}

        def flaky_llm(_prompt: str) -> str:
            state["count"] += 1
            if state["count"] == 1:
                return "not valid json"
            return json.dumps({
                "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "ok"},
                "answer_correctness": {"score": 5, "missing_facts": [], "wrong_facts": [], "reason": "ok"},
                "answer_relevance": {"score": 5, "reason": "ok"},
                "context_sufficiency": {"score": 5, "reason": "ok"},
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            judge = UETLLMJudgeEvaluator(
                model_name="mock-judge",
                max_retries=2,
                delay_seconds=0.0,
                cache_file=Path(tmpdir) / "cache.json",
                call_fn=flaky_llm,
            )
            ok_res = judge.judge_record("Q", ["C"], "A", "GT", sample_id="retry_ok")
            self.assertEqual(ok_res["status"], "ok")

            bad_judge = UETLLMJudgeEvaluator(
                model_name="mock-judge",
                max_retries=1,
                delay_seconds=0.0,
                cache_file=Path(tmpdir) / "cache2.json",
                call_fn=lambda _p: "broken output forever",
            )
            err_res = bad_judge.judge_record("Q2", ["C"], "A", "GT", sample_id="always_err")
            self.assertEqual(err_res["status"], "error")
            self.assertEqual(err_res["normalized_scores"], {})

    def test_judge_ood_and_evaluator_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge = UETLLMJudgeEvaluator(
                model_name="mock-judge",
                delay_seconds=0.0,
                cache_file=Path(tmpdir) / "ood_cache.json",
                call_fn=lambda _p: json.dumps({
                    "refused_properly": True,
                    "hallucinated_info": False,
                    "score": 5,
                    "reason": "Từ chối đúng",
                }),
            )
            evaluator = UETEvaluator(use_llm_judge=True, judge=judge)
            sample = EvaluationSample(
                id="ood_1",
                query="ĐH Sư phạm Hà Nội tuyển sinh ra sao?",
                metadata={"type": "out_of_domain", "unit": "out_of_domain"},
            )
            resp = Response(query_id="q", answer="Xin lỗi em, hiện tại chưa có thông tin cụ thể.", sources=[])
            res = evaluator.evaluate_sample(sample, resp)
            self.assertEqual(res.metrics["ood_refusal_accuracy"], 1.0)


class TestDiagnosticsAndTwoPhaseFlow(unittest.TestCase):
    """Kiểm thử Error Taxonomy, build_e2e_report và kiến trúc 2 pha Generate -> Score."""

    def test_classify_all_six_error_labels(self):
        # 1. OK
        lbl, _ = classify_sample_error({
            "is_out_of_domain": False,
            "retrieval_metrics": {"chunk_hit_at_k": 1.0, "doc_hit_at_k": 1.0},
            "offline_metrics": {"is_refusal": False, "numeric_faithfulness": 1.0, "ungrounded_entities": [], "faithfulness": 0.9, "answer_correctness": 0.9},
            "judge": {"status": "ok", "raw_scores": {"faithfulness": 5, "answer_correctness": 5, "context_sufficiency": 5}, "details": {}},
        })
        self.assertEqual(lbl, "OK")

        # 2. FALSE_REFUSAL
        lbl, _ = classify_sample_error({
            "is_out_of_domain": False,
            "retrieval_metrics": {"chunk_hit_at_k": 1.0, "doc_hit_at_k": 1.0},
            "offline_metrics": {"is_refusal": True, "context_relevance": 0.8},
        })
        self.assertEqual(lbl, "FALSE_REFUSAL")

        # 3. RETRIEVAL_MISS
        lbl, _ = classify_sample_error({
            "is_out_of_domain": False,
            "retrieval_metrics": {"chunk_hit_at_k": 0.0, "doc_hit_at_k": 0.0},
            "offline_metrics": {"is_refusal": False, "answer_correctness": 0.3, "numeric_faithfulness": 1.0, "ungrounded_entities": []},
            "judge": {"status": "ok", "raw_scores": {"faithfulness": 4, "answer_correctness": 2, "context_sufficiency": 1}, "details": {}},
        })
        self.assertEqual(lbl, "RETRIEVAL_MISS")

        # 4. HALLUCINATION
        lbl, _ = classify_sample_error({
            "is_out_of_domain": False,
            "retrieval_metrics": {"chunk_hit_at_k": 1.0, "doc_hit_at_k": 1.0},
            "offline_metrics": {"is_refusal": False, "numeric_faithfulness": 0.5, "ungrounded_entities": ["2.0"], "faithfulness": 0.4, "answer_correctness": 0.4},
        })
        self.assertEqual(lbl, "HALLUCINATION")

        # 5. INCOMPLETE_OR_WRONG
        lbl, _ = classify_sample_error({
            "is_out_of_domain": False,
            "retrieval_metrics": {"chunk_hit_at_k": 1.0, "doc_hit_at_k": 1.0},
            "offline_metrics": {"is_refusal": False, "numeric_faithfulness": 1.0, "ungrounded_entities": [], "faithfulness": 0.9, "answer_correctness": 0.45},
            "judge": {"status": "ok", "raw_scores": {"faithfulness": 5, "answer_correctness": 2, "context_sufficiency": 5}, "details": {}},
        })
        self.assertEqual(lbl, "INCOMPLETE_OR_WRONG")

        # 6. OOD_FAILURE
        lbl, _ = classify_sample_error({
            "is_out_of_domain": True,
            "offline_metrics": {"is_refusal": False},
        })
        self.assertEqual(lbl, "OOD_FAILURE")

        # 7. GENERATION_ERROR (rate limit / fallback)
        self.assertTrue(is_generation_error_response("Please try again in 1.46s", "multi-provider-fallback"))
        lbl, _ = classify_sample_error({
            "is_out_of_domain": False,
            "offline_metrics": {"is_generation_error": True},
        })
        self.assertEqual(lbl, "GENERATION_ERROR")

    def test_generation_error_excluded_from_offline_averages_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_path = tmp / "preds.json"
            # Tạo snapshot ban đầu có 1 câu hợp lệ và 1 câu lỗi 429
            pred_path.write_text(json.dumps({
                "top_k": 5,
                "predictions": [
                    {
                        "sample_id": "ok_1",
                        "query": "Điều kiện tốt nghiệp sớm?",
                        "type": "factoid",
                        "unit": "UET",
                        "category": "QUY_CHE",
                        "ground_truth": "GPA >= 3.0 và tích lũy 90 tín chỉ.",
                        "target_document_ids": ["d_1"],
                        "target_chunk_ids": {"c_1": 2},
                        "is_out_of_domain": False,
                        "retrieved_contexts": [{"chunk_id": "c_1", "document_id": "d_1", "rank": 1, "score": 0.9, "text": "GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ."}],
                        "generated_answer": "Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ.",
                        "model_name": "kept-cached",
                    },
                    {
                        "sample_id": "err_429",
                        "query": "Điều kiện học bổng?",
                        "type": "factoid",
                        "unit": "UET",
                        "category": "QUY_CHE",
                        "ground_truth": "GPA >= 3.2",
                        "target_document_ids": ["d_1"],
                        "target_chunk_ids": {"c_1": 2},
                        "is_out_of_domain": False,
                        "retrieved_contexts": [{"chunk_id": "c_1", "document_id": "d_1", "rank": 1, "score": 0.9, "text": "GPA >= 3.2"}],
                        "generated_answer": "⚠️ **MÔ HÌNH AI TẠM THỜI TỪ CHỐI DỊCH VỤ**\nPlease try again in 1.465s",
                        "model_name": "multi-provider-fallback",
                    },
                ],
            }, ensure_ascii=False), encoding="utf-8")

            # Pha 2 trên snapshot có lỗi 429: không gắn HALLUCINATION và không kéo offline_metrics xuống
            report_with_err = score_predictions(pred_path, enable_semantic=False)
            self.assertEqual(report_with_err["summary"]["generation_error_count"], 1)
            self.assertEqual(report_with_err["error_taxonomy"]["GENERATION_ERROR"], 1)
            self.assertEqual(report_with_err["error_taxonomy"]["HALLUCINATION"], 0)
            self.assertGreater(report_with_err["offline_metrics"]["answer_correctness"], 0.85)

            # Pha 1 resume=True chỉ chạy bù err_429 và giữ nguyên ok_1 ("kept-cached")
            pipeline = _DummyPipeline()
            ds = [
                EvaluationSample(id="ok_1", query="Điều kiện tốt nghiệp sớm?", ground_truth="GPA >= 3.0 và tích lũy 90 tín chỉ."),
                EvaluationSample(id="err_429", query="Điều kiện học bổng?", ground_truth="GPA >= 3.0 và tích lũy 90 tín chỉ."),
            ]
            resumed = generate_predictions(pipeline, ds, top_k=5, delay_between_queries=0.0, output_path=pred_path, resume=True)
            self.assertEqual(resumed["predictions"][0]["model_name"], "kept-cached")
            self.assertEqual(resumed["predictions"][1]["model_name"], "mock-gen")

    def test_judge_404_automatic_failover(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge = UETLLMJudgeEvaluator(
                provider="groq",
                model_name="retired-model-404",
                candidate_models=["active-backup-model"],
                delay_seconds=0.0,
                cache_file=Path(tmpdir) / "failover_cache.json",
            )
            judge._keys_by_provider["groq"] = ["fake_key"]
            seen_models = []

            def fake_target(tp: str, tm: str, _prompt: str) -> str:
                seen_models.append((tp, tm))
                if tm == "retired-model-404":
                    raise judge._ModelNotFoundError("404 retired")
                return json.dumps({
                    "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "ok"},
                    "answer_correctness": {"score": 5, "missing_facts": [], "wrong_facts": [], "reason": "ok"},
                    "answer_relevance": {"score": 5, "reason": "ok"},
                    "context_sufficiency": {"score": 5, "reason": "ok"},
                })

            judge._call_single_target = fake_target  # type: ignore[method-assign]
            res = judge.judge_record("Q", ["C"], "A", "GT", sample_id="f1")
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["judge_model"], "active-backup-model")
            self.assertIn(("groq", "retired-model-404"), judge._dead_targets)
            # Lần gọi kế tiếp bỏ qua thẳng retired-model-404
            res2 = judge.judge_record("Q2", ["C2"], "A2", "GT2", sample_id="f2")
            self.assertEqual(res2["status"], "ok")
            self.assertEqual(seen_models[-1], ("groq", "active-backup-model"))

    def test_build_e2e_report_excludes_judge_errors(self):
        samples = [
            {
                "sample_id": "s1", "query": "Q1", "type": "factoid", "unit": "UET", "category": "QUY_CHE",
                "is_out_of_domain": False,
                "retrieval_metrics": {"chunk_hit_at_k": 1.0, "chunk_recall_at_k": 1.0},
                "offline_metrics": {"overall_score": 0.9, "answer_correctness": 0.9, "faithfulness": 0.9},
                "judge": {"status": "ok", "raw_scores": {"faithfulness": 5, "answer_correctness": 5, "answer_relevance": 5, "context_sufficiency": 5}, "normalized_scores": {"faithfulness": 1.0, "answer_correctness": 1.0, "answer_relevance": 1.0, "context_sufficiency": 1.0, "overall_score": 1.0}},
                "error_label": "OK", "error_reason": "OK",
            },
            {
                "sample_id": "s2", "query": "Q2", "type": "list", "unit": "FIT", "category": "DAO_TAO",
                "is_out_of_domain": False,
                "retrieval_metrics": {"chunk_hit_at_k": 1.0, "chunk_recall_at_k": 1.0},
                "offline_metrics": {"overall_score": 0.8, "answer_correctness": 0.8, "faithfulness": 0.8},
                "judge": {"status": "error", "error": "HTTP 429", "normalized_scores": {}, "raw_scores": {}},
                "error_label": "OK", "error_reason": "OK",
            },
        ]
        report = build_e2e_report(samples, top_k=5, judge_info={"enabled": True, "provider": "groq", "model": "m"})
        self.assertEqual(report["summary"]["valid_judged_samples"], 1)
        self.assertEqual(report["summary"]["judge_error_count"], 1)
        # Chỉ tính trên mẫu s1 hợp lệ -> 1.0 (không bị kéo xuống 0.5)
        self.assertEqual(report["judge_metrics"]["average_normalized_scores"]["answer_correctness"], 1.0)

    def test_two_phase_generate_and_score_snapshot(self):
        pipeline = _DummyPipeline()
        dataset = [
            EvaluationSample(
                id="p1_01",
                query="Điều kiện tốt nghiệp sớm?",
                ground_truth="GPA >= 3.0 và tích lũy 90 tín chỉ.",
                metadata={"type": "factoid", "unit": "UET", "target_document_ids": ["d_1"], "target_chunk_ids": {"c_1": 2}},
            ),
            EvaluationSample(
                id="p1_02",
                query="ĐH Sư phạm Hà Nội tuyển sinh thế nào?",
                ground_truth="Không thuộc UET.",
                metadata={"type": "out_of_domain", "unit": "out_of_domain"},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_path = tmp / "predictions.json"
            out_path = tmp / "e2e_report.json"

            snapshot = generate_predictions(
                pipeline=pipeline,
                dataset=dataset,
                top_k=5,
                delay_between_queries=0.0,
                output_path=pred_path,
            )
            self.assertTrue(pred_path.exists())
            self.assertEqual(snapshot["total_samples"], 2)
            self.assertEqual(snapshot["predictions"][0]["model_name"], "mock-gen")

            mock_judge = UETLLMJudgeEvaluator(
                model_name="mock-judge",
                delay_seconds=0.0,
                cache_file=tmp / "jcache.json",
                call_fn=lambda prompt: json.dumps(
                    {"refused_properly": True, "hallucinated_info": False, "score": 5, "reason": "ok"}
                    if "OOD" in prompt
                    else {
                        "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "ok"},
                        "answer_correctness": {"score": 5, "missing_facts": [], "wrong_facts": [], "reason": "ok"},
                        "answer_relevance": {"score": 5, "reason": "ok"},
                        "context_sufficiency": {"score": 5, "reason": "ok"},
                    }
                ),
            )
            report = score_predictions(
                predictions_source=pred_path,
                use_judge=True,
                judge=mock_judge,
                enable_semantic=False,
                output_path=out_path,
            )
            self.assertTrue(out_path.exists())
            self.assertEqual(report["summary"]["ood_refusal_accuracy"], 1.0)
            self.assertEqual(report["error_taxonomy"]["OK"], 2)
            self.assertIn("factoid", report["by_type"])


class TestRetrievalEvaluatorFixes(unittest.TestCase):
    """Kiểm thử sửa bẫy cache ẩn, boolean leak và OOD ở RetrievalEvaluator."""

    def test_cache_signature_invalidation_and_ood_stats(self):
        dataset_items = [
            {
                "id": "q1",
                "query": "Điều kiện học bổng?",
                "type": "factoid",
                "unit": "UET",
                "category": "QUY_CHE",
                "target_document_ids": ["d_1"],
                "target_chunk_ids": {"c_1": 2},
            },
            {
                "id": "q_ood",
                "query": "ĐH Sư phạm tuyển sinh?",
                "type": "out_of_domain",
                "unit": "out_of_domain",
                "category": "TUYEN_SINH",
                "target_document_ids": [],
                "target_chunk_ids": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = Path(tmpdir) / "eval.json"
            ds_path.write_text(json.dumps(dataset_items, ensure_ascii=False), encoding="utf-8")

            ev = RetrievalEvaluator(eval_data_dir=ds_path)
            r1 = _DummyRetriever(enable_mmr=False, bm25_weight=0.6)

            # Mặc định use_cache=False: luôn gọi retriever thật
            res_live = ev.evaluate_retriever(r1, top_k=5, level="both")
            self.assertEqual(r1.calls, 2)
            self.assertNotIn("mean_is_out_of_domain", res_live["doc_level"]["aggregated_metrics"])
            self.assertEqual(res_live["doc_level"]["aggregated_metrics"]["mean_recall@5"], 1.0)
            self.assertEqual(res_live["ood_stats"]["ood_queries"], 1)
            self.assertAlmostEqual(res_live["ood_stats"]["ood_avg_top1_score"], 0.21)
            self.assertAlmostEqual(res_live["ood_stats"]["in_domain_avg_top1_score"], 0.89)

            # OOD trong per_query_results không bị hardcode precision/recall = 1.0
            ood_row = [x for x in res_live["chunk_level"]["per_query_results"] if x["query_id"] == "q_ood"][0]
            self.assertNotIn("precision@5", ood_row)

            # Khi bật use_cache=True lần 1: tạo file cache với hash cấu hình r1
            r1.calls = 0
            ev.evaluate_retriever(r1, top_k=5, level="chunk", use_cache=True)
            self.assertEqual(r1.calls, 2)
            # Chạy lại đúng cấu hình r1 -> hit cache (calls không tăng)
            ev.evaluate_retriever(r1, top_k=5, level="chunk", use_cache=True)
            self.assertEqual(r1.calls, 2)

            # Đổi enable_mmr=True -> hash đổi -> KHÔNG đọc nhầm cache r1 cũ
            r2 = _DummyRetriever(enable_mmr=True, bm25_weight=0.6)
            res_mmr = ev.evaluate_retriever(r2, top_k=5, level="chunk", use_cache=True)
            self.assertEqual(r2.calls, 2)
            self.assertEqual(res_mmr["aggregated_metrics"]["mean_recall@5"], 0.0)


class TestRagasCompatAndNaN(unittest.TestCase):
    """Kiểm thử shim tương thích Ragas 0.4.3 và xử lý NaN -> None."""

    def test_ragas_import_compat_and_nan_handling(self):
        _ensure_ragas_compat()
        import ragas  # noqa: F401
        self.assertIsNone(_clean_ragas_value(float("nan")))
        self.assertIsNone(_clean_ragas_value(None))
        self.assertEqual(_clean_ragas_value(0.87654), 0.8765)

    def test_openrouter_reasoning_none_and_graph_triples_context(self):
        from unittest.mock import patch
        from src.base import RetrievedContext
        from src.evaluation.evaluator import _serialize_contexts

        captured = {}

        class _FakeResp:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "ok"},
                                        "answer_correctness": {"score": 5, "missing_facts": [], "wrong_facts": [], "reason": "ok"},
                                        "answer_relevance": {"score": 5, "reason": "ok"},
                                        "context_sufficiency": {"score": 5, "reason": "ok"},
                                    }
                                )
                            }
                        }
                    ]
                }

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResp()

        default_judge = UETLLMJudgeEvaluator(enable_cache=False, api_key="sk-or-test")
        self.assertEqual(default_judge.provider, "openrouter")
        self.assertEqual(default_judge.model_name, "deepseek/deepseek-v4.1-flash-1017")
        self.assertEqual(default_judge.reasoning_effort, "none")

        judge = UETLLMJudgeEvaluator(
            provider="openrouter",
            model_name="deepseek/deepseek-v4.1-flash-1017",
            reasoning_effort="none",
            enable_cache=False,
            api_key="sk-or-test",
        )
        with patch("httpx.post", side_effect=fake_post):
            raw = judge._call_single_target("openrouter", "deepseek/deepseek-v4.1-flash-1017", "prompt test")
        self.assertIn("faithfulness", raw)
        self.assertEqual(captured["payload"]["reasoning"], {"enabled": False})
        self.assertNotIn("reasoning_effort", captured["payload"])

        ctx = RetrievedContext(
            chunk_id="c1",
            text="Bộ môn Kỹ thuật Robot công bố định hướng hợp tác.",
            similarity_score=0.92,
            rank=1,
            metadata={
                "document_id": "d1",
                "page": 3,
                "graph_triples": ["Bộ môn Kỹ thuật robot -[QUY_ĐỊNH_VỀ]-> Viện Công nghệ Chiba (2018)"],
            },
        )
        serialized = _serialize_contexts([ctx])
        self.assertEqual(serialized[0]["page"], 3)
        self.assertIn("Chiba", serialized[0]["graph_triples"][0])

        seen_prompts = []

        def judge_capture(p):
            seen_prompts.append(p)
            return json.dumps({
                "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "ok"},
                "answer_correctness": {"score": 5, "missing_facts": [], "wrong_facts": [], "reason": "ok"},
                "answer_relevance": {"score": 5, "reason": "ok"},
                "context_sufficiency": {"score": 5, "reason": "ok"},
            })

        report = score_predictions(
            {
                "top_k": 5,
                "predictions": [
                    {
                        "sample_id": "g1",
                        "query": "Đối tác của Bộ môn Kỹ thuật Robot?",
                        "ground_truth": "Viện Công nghệ Chiba (2018).",
                        "is_out_of_domain": False,
                        "retrieved_contexts": serialized,
                        "generated_answer": "Bộ môn hợp tác với Viện Công nghệ Chiba năm 2018 (Trang 3).",
                        "model_name": "tencent/hy4-flash",
                    }
                ],
            },
            top_k=5,
            use_judge=True,
            judge=UETLLMJudgeEvaluator(
                provider="openrouter",
                model_name="tencent/hy4-flash",
                delay_seconds=0.0,
                enable_cache=False,
                call_fn=judge_capture,
            ),
            enable_semantic=False,
        )
        self.assertEqual(report["samples"][0]["offline_metrics"]["numeric_faithfulness"], 1.0)
        self.assertIn("Quan hệ Đồ thị Tri thức", seen_prompts[0])
        self.assertIn("Chiba", seen_prompts[0])

    def test_agentic_eval_ood_fail_and_trace_latencies(self):
        """Kiểm tra Evaluator nhận diện đúng out_of_scope, fail_node (GENERATION_ERROR + resume) và bóc latency từ trace."""
        self.assertTrue(
            is_generation_error_response(
                "Xin lỗi bạn, hệ thống kiểm tra và tra cứu thông tin đang gián đoạn dịch vụ, vui lòng thử lại sau."
            )
        )
        self.assertTrue(
            is_refusal_response(
                "Chào bạn! Tôi là UET Chatbot. Nội dung bạn hỏi hiện nằm ngoài phạm vi hỗ trợ của tôi."
            )
        )

        class _MockAgenticPipeline(BaseRAGPipeline):
            def __init__(self):
                self.embedding_model = _DummyEmbedder()
                self.agent_graph = object()
                self.calls = 0

            def query(self, query: UserQuery, top_k: int = 5) -> Response:
                self.calls += 1
                qlow = query.query_text.lower()
                if "giá vàng" in qlow:
                    return Response(
                        query_id=query.query_id,
                        answer="Chào bạn! Tôi là UET Chatbot.",
                        sources=[],
                        model_name="mock-agent",
                        latency_seconds=1.1,
                        metadata={
                            "status": "out_of_scope",
                            "route": "OUT_OF_SCOPE",
                            "trace": [{"node": "router_node", "latency_s": 1.0}],
                        },
                    )
                if "lỗi api" in qlow:
                    return Response(
                        query_id=query.query_id,
                        answer="Xin lỗi bạn, hệ thống kiểm tra và tra cứu thông tin đang gián đoạn dịch vụ.",
                        sources=[],
                        model_name="mock-agent",
                        latency_seconds=2.5,
                        metadata={
                            "status": "fail",
                            "route": "QUALIFIED_QUESTION",
                            "trace": [
                                {"node": "retrieval_node", "latency_s": 0.4},
                                {"node": "generation_node", "latency_s": 1.8},
                            ],
                        },
                    )
                return Response(
                    query_id=query.query_id,
                    answer="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ.",
                    sources=[
                        RetrievedContext(
                            chunk_id="c_1",
                            text="Sinh viên tốt nghiệp sớm cần GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ.",
                            similarity_score=0.92,
                            rank=1,
                            metadata={"document_id": "d_1", "title": "Quy chế"},
                        )
                    ],
                    model_name="mock-agent",
                    latency_seconds=3.2,
                    metadata={
                        "status": "ok",
                        "route": "QUALIFIED_QUESTION",
                        "trace": [
                            {"node": "retrieval_node", "latency_s": 0.65},
                            {"node": "generation_node", "latency_s": 1.25, "usage": {"total_tokens": 120}},
                        ],
                    },
                )

        dataset = [
            EvaluationSample(
                id="s_ok",
                query="Điều kiện tốt nghiệp sớm?",
                ground_truth="GPA >= 3.0 và tích lũy 90 tín chỉ.",
                metadata={"type": "factoid", "target_document_ids": ["d_1"], "target_chunk_ids": {"c_1": 2}},
            ),
            EvaluationSample(
                id="s_ood",
                query="Giá vàng hôm nay bao nhiêu?",
                ground_truth="Ngoài phạm vi UET.",
                metadata={"type": "out_of_domain", "unit": "out_of_domain"},
            ),
            EvaluationSample(
                id="s_fail",
                query="Câu hỏi gặp lỗi API?",
                ground_truth="Nội dung mẫu.",
                metadata={"type": "factoid", "target_document_ids": ["d_1"], "target_chunk_ids": {"c_1": 2}},
            ),
        ]

        pipe = _MockAgenticPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "snap.json"
            snap = generate_predictions(pipe, dataset, top_k=5, delay_between_queries=0.0, output_path=snap_path, resume=False)
            preds = {p["sample_id"]: p for p in snap["predictions"]}

            self.assertEqual(preds["s_ok"]["retrieval_latency_s"], 0.65)
            self.assertEqual(preds["s_ok"]["llm_latency_s"], 1.25)
            self.assertEqual(preds["s_ok"]["token_usage"], {"total_tokens": 120})
            self.assertEqual(preds["s_ok"]["agent_status"], "ok")
            self.assertEqual(preds["s_ood"]["agent_status"], "out_of_scope")
            self.assertEqual(preds["s_fail"]["agent_status"], "fail")

            report = score_predictions(snap, top_k=5, use_judge=False, enable_semantic=False)
            self.assertEqual(report["summary"]["ood_refusal_accuracy"], 1.0)
            self.assertEqual(report["summary"]["generation_error_count"], 1)
            self.assertEqual(report["error_taxonomy"]["GENERATION_ERROR"], 1)
            self.assertEqual(report["error_taxonomy"]["OOD_FAILURE"], 0)
            self.assertIsNotNone(report["latency_metrics"]["avg_retrieval_latency_s"])
            self.assertIsNotNone(report["latency_metrics"]["avg_llm_latency_s"])

            # Kiểm tra --resume: giữ s_ok & s_ood, nhưng chạy lại s_fail
            pipe.calls = 0
            generate_predictions(pipe, dataset, top_k=5, delay_between_queries=0.0, output_path=snap_path, resume=True)
            self.assertEqual(pipe.calls, 1)


if __name__ == "__main__":
    unittest.main()

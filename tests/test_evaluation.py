"""
tests/test_evaluation.py - Unit tests cho phân hệ Evaluation & Benchmark Metrics.
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import EvaluationSample, Response, RetrievedContext
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr,
    average_precision,
    hit_rate_at_k,
    compute_faithfulness,
    compute_context_relevance,
    compute_keyword_overlap,
)
from src.evaluation.evaluator import UETEvaluator


class TestEvaluationMetrics(unittest.TestCase):
    """Kiểm thử các hàm đo lường chất lượng truy xuất và sinh văn bản."""

    def test_precision_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_1", "doc_3"}
        # Top 2: doc_1 (đúng), doc_2 (sai) -> 1/2 = 0.5
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=2), 0.5)
        # Top 4: doc_1, doc_3 (đúng) -> 2/4 = 0.5
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=4), 0.5)
        # Duplicate doc_id không được tính 2 lần
        dup_retrieved = ["doc_1", "doc_1", "doc_2"]
        self.assertAlmostEqual(precision_at_k(dup_retrieved, relevant, k=2), 0.5)

    def test_recall_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_3", "doc_5", "doc_6"}
        # Top 2: doc_1 (1/4 = 0.25)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 0.25)
        # Top 3: doc_1, doc_3 (2/4 = 0.5)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 0.5)
        # Khi relevant rỗng -> 0.0
        self.assertAlmostEqual(recall_at_k(retrieved, set(), k=3), 0.0)

    def test_ndcg_at_k(self):
        # Thứ tự chuẩn: doc_1, doc_2 đều relevant -> NDCG@2 = 1.0
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1": 1.0, "doc_2": 1.0}
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=2), 1.0)

        # Relevant đứng ở cuối
        retrieved_bad = ["doc_3", "doc_1"]
        ndcg_val = ndcg_at_k(retrieved_bad, relevant, k=2)
        self.assertLess(ndcg_val, 1.0)
        self.assertGreater(ndcg_val, 0.0)

    def test_mrr(self):
        retrieved = ["doc_x", "doc_1", "doc_2"]
        relevant = {"doc_1"}
        # Match đầu tiên ở vị trí 2 (index 1) -> MRR = 1/2 = 0.5
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.5)

        # Match ở vị trí 1 -> MRR = 1.0
        self.assertAlmostEqual(mrr(["doc_1", "doc_x"], relevant), 1.0)

        # Không tìm thấy -> 0.0
        self.assertAlmostEqual(mrr(["doc_a", "doc_b"], relevant), 0.0)

    def test_hit_rate_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        self.assertEqual(hit_rate_at_k(retrieved, {"doc_2"}, k=2), 1.0)
        self.assertEqual(hit_rate_at_k(retrieved, {"doc_3"}, k=2), 0.0)

    def test_average_precision(self):
        retrieved = ["doc_1", "doc_x", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        # rank 1: doc_1 (P=1/1 = 1.0)
        # rank 3: doc_2 (P=2/3)
        # AP = (1.0 + 2/3) / 2 = 5/6 ≈ 0.8333
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
    """Kiểm thử lớp UETEvaluator đánh giá mẫu câu hỏi."""

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
        self.assertIn("overall_score", res.metrics)
        self.assertGreater(res.metrics["overall_score"], 0.5)


if __name__ == "__main__":
    unittest.main()

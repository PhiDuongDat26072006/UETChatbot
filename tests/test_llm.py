"""
tests/test_llm.py - Bộ kiểm thử đơn vị cho phân hệ LLM Client (Task 7).
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import UserQuery, RetrievedContext, AugmentedPrompt, Response
from src.prompts.prompt_templates import UETPromptAugmenter
from src.llm.llm_client import UETLLMClient


class TestUETLLMClient(unittest.TestCase):
    """Kiểm thử bộ phận LLM Client (Task 7)."""

    def setUp(self):
        self.query = UserQuery(query_text="Sinh viên cần bao nhiêu tín chỉ?")
        self.contexts = [
            RetrievedContext(
                chunk_id="ctx_01",
                text="Tối thiểu tích lũy 130 tín chỉ.",
                similarity_score=0.9,
                rank=1,
                metadata={"title": "Quy chế Đào tạo", "page": 10},
            )
        ]
        self.prompt = UETPromptAugmenter().augment(self.query, self.contexts)

    def test_fallback_mode_without_api_key(self):
        """Kiểm tra chế độ fallback tự động khi không có API key."""
        client = UETLLMClient(api_key="")
        response = client.generate(self.prompt)

        self.assertIsInstance(response, Response)
        self.assertEqual(response.query_id, self.query.query_id)
        self.assertTrue(len(response.answer) > 0)
        self.assertIn("Tối thiểu tích lũy 130 tín chỉ.", response.answer)
        self.assertIn("fallback", response.model_name.lower())
        self.assertGreaterEqual(response.latency_seconds, 0.0)

    def test_fallback_empty_contexts(self):
        """Kiểm tra phản hồi fallback lịch sự khi không có ngữ cảnh."""
        empty_prompt = UETPromptAugmenter().augment(self.query, [])
        client = UETLLMClient(api_key="")
        response = client.generate(empty_prompt)

        self.assertIn("chưa tìm thấy quy định cụ thể", response.answer)
        self.assertIn("Phòng Đào tạo", response.answer)

    def test_candidate_models_attributes(self):
        """Kiểm tra các thuộc tính candidate models và tương thích ngược."""
        client = UETLLMClient()
        self.assertTrue(hasattr(client, "candidate_models"))
        self.assertTrue(hasattr(client, "CANDIDATE_MODELS"))
        self.assertIsInstance(client.candidate_models, list)
        self.assertGreater(len(client.candidate_models), 0)

    def test_streaming_fallback(self):
        """Kiểm tra streaming ở chế độ fallback."""
        client = UETLLMClient(api_key="")
        chunks = list(client.generate_stream(self.prompt))
        self.assertTrue(len(chunks) > 0)
        combined = "".join(chunks)
        self.assertIn("Tối thiểu tích lũy 130 tín chỉ.", combined)


if __name__ == "__main__":
    unittest.main()

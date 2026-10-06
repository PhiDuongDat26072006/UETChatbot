"""
tests/test_prompts.py - Bộ kiểm thử đơn vị cho phân hệ Prompts & Context Augmentation (Task 6).
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import UserQuery, RetrievedContext, AugmentedPrompt
from src.prompts.prompt_templates import UETPromptAugmenter, DEFAULT_UET_SYSTEM_PROMPT


class TestUETPromptAugmenter(unittest.TestCase):
    """Kiểm thử bộ phận Prompt Augmentation (Task 6)."""

    def setUp(self):
        self.augmenter = UETPromptAugmenter()
        self.sample_query = UserQuery(query_text="Điều kiện xét học bổng là gì?")
        self.sample_contexts = [
            RetrievedContext(
                chunk_id="ctx_01",
                text="Sinh viên có điểm GPA từ 3.20 trở lên được xét học bổng Giỏi.",
                similarity_score=0.92,
                rank=1,
                metadata={"title": "Quy chế Học bổng", "page": 5},
            ),
            RetrievedContext(
                chunk_id="ctx_02",
                text="Không bị kỷ luật trong kỳ học xét học bổng.",
                similarity_score=0.88,
                rank=2,
                metadata={"title": "Quy định Kỷ luật", "page": None},
            ),
        ]

    def test_custom_system_prompt(self):
        """Kiểm tra khởi tạo với system prompt tùy chỉnh."""
        custom_prompt = "Bạn là trợ lý thử nghiệm."
        aug = UETPromptAugmenter(system_prompt=custom_prompt)
        self.assertEqual(aug.system_prompt, custom_prompt)

    def test_format_contexts_valid(self):
        """Kiểm tra định dạng danh sách ngữ cảnh đầy đủ thông tin."""
        formatted = self.augmenter.format_contexts(self.sample_contexts)
        self.assertIn("Quy chế Học bổng", formatted)
        self.assertIn("Trang 5", formatted)
        self.assertIn("92%", formatted)
        self.assertIn("Quy định Kỷ luật", formatted)
        # Đảm bảo không bị lỗi 'Trang None' khi metadata page là None
        self.assertNotIn("Trang None", formatted)

    def test_format_contexts_empty(self):
        """Kiểm tra định dạng khi danh sách ngữ cảnh rỗng."""
        formatted = self.augmenter.format_contexts([])
        self.assertIn("Không tìm thấy tài liệu quy chế nào", formatted)

    def test_augment_structure(self):
        """Kiểm tra cấu trúc AugmentedPrompt trả về."""
        prompt = self.augmenter.augment(self.sample_query, self.sample_contexts)
        self.assertIsInstance(prompt, AugmentedPrompt)
        self.assertEqual(prompt.user_query.query_text, self.sample_query.query_text)
        self.assertEqual(len(prompt.contexts), 2)
        self.assertIn(self.sample_query.query_text, prompt.formatted_prompt)
        self.assertIn("Quy chế Học bổng", prompt.formatted_prompt)


if __name__ == "__main__":
    unittest.main()

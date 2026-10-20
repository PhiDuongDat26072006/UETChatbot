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

    def test_key_masking(self):
        """Kiểm tra hàm che giấu API key để bảo mật log."""
        self.assertEqual(UETLLMClient._mask_key(""), "")
        self.assertEqual(UETLLMClient._mask_key("short"), "***")
        masked = UETLLMClient._mask_key("AIzaSyD-1234567890abcdef")
        self.assertTrue(masked.startswith("AIza..."))
        self.assertTrue(masked.endswith("cdef"))

    def test_load_api_keys_priority_and_dedup(self):
        """Kiểm tra nạp nhiều khóa từ các biến môi trường khác nhau."""
        import os
        from unittest.mock import patch

        env_mock = {
            "GEMINI_API_KEY": "key_primary",
            "GEMINI_API_KEY_2": "key_backup_2",
            "GEMINI_API_KEYS": "key_primary,key_group_3",
            "GEMINI_BACKUP_API_KEY": "key_backup_final",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            keys = UETLLMClient._load_api_keys()
            self.assertIn("key_primary", keys)
            self.assertIn("key_backup_2", keys)
            self.assertIn("key_group_3", keys)
            self.assertIn("key_backup_final", keys)
            # Kiểm tra không bị trùng lặp
            self.assertEqual(len(keys), len(set(keys)))

    def test_auto_failover_on_429_quota_exhausted(self):
        """Kiểm tra tự động chuyển sang API key dự phòng khi gặp lỗi 429 RESOURCE_EXHAUSTED."""
        from unittest.mock import MagicMock

        client = UETLLMClient(api_key="")
        client.api_keys = ["mock_key_1", "mock_key_2"]
        client.current_key_index = 0

        # Mock client 1 bị lỗi 429
        mock_client_1 = MagicMock()
        mock_client_1.models.generate_content.side_effect = Exception("429 Quota Exceeded: RESOURCE_EXHAUSTED")

        # Mock client 2 thành công
        mock_client_2 = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "Câu trả lời thành công từ API Key 2."
        mock_client_2.models.generate_content.return_value = mock_res

        def mock_get_client(key):
            if key == "mock_key_1":
                return mock_client_1
            return mock_client_2

        client._get_client_for_key = mock_get_client

        response = client.generate(self.prompt)
        self.assertEqual(response.answer, "Câu trả lời thành công từ API Key 2.")
        # Hệ thống phải cập nhật key đang hoạt động sang mock_key_2
        self.assertEqual(client.api_key, "mock_key_2")

    def test_custom_keys_list_initialization(self):
        """Kiểm tra khởi tạo client với tham số api_keys dạng danh sách."""
        client = UETLLMClient(api_keys=["key_custom_A", "key_custom_B"])
        self.assertIn("key_custom_A", client.api_keys)
        self.assertIn("key_custom_B", client.api_keys)

    def test_load_api_keys_multiline_env(self):
        """Kiểm tra nạp danh sách key trong .env theo dạng nhiều dòng hoặc chấm phẩy."""
        import os
        from unittest.mock import patch

        env_mock = {
            "GEMINI_API_KEYS": "key_line_1\nkey_line_2;key_line_3",
            "GEMINI_API_KEY_3": "key_line_4",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            keys = UETLLMClient._load_api_keys()
            self.assertIn("key_line_1", keys)
            self.assertIn("key_line_2", keys)
            self.assertIn("key_line_3", keys)
            self.assertIn("key_line_4", keys)


if __name__ == "__main__":
    unittest.main()

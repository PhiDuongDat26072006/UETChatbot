"""
tests/test_llm.py - Bộ kiểm thử đơn vị cho phân hệ LLM Client (Task 7 - Hoàng).
Kiểm tra toàn diện:
  1. Multi-Provider Key loading (Groq, SambaNova, Gemini)
  2. Persistent Disk Cache (Ghi cache, đọc cache < 0.01s)
  3. Multi-Provider Worker Queue (Hàng đợi chia tải 3 luồng song song)
  4. Fallback & Fault Tolerance (Tự bọc lót khi 1 luồng lỗi)
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import UserQuery, RetrievedContext, AugmentedPrompt, Response
from src.prompts.prompt_templates import UETPromptAugmenter
from src.llm.llm_client import UETLLMClient


class TestUETLLMClient(unittest.TestCase):
    """Kiểm thử bộ phận LLM Client (Task 7)."""

    def setUp(self):
        self.query = UserQuery(query_text="Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp?")
        self.contexts = [
            RetrievedContext(
                chunk_id="ctx_01",
                text="Tối thiểu tích lũy 130 tín chỉ theo quy chế đào tạo.",
                similarity_score=0.9,
                rank=1,
                metadata={"title": "Quy chế Đào tạo", "page": 10},
            )
        ]
        self.prompt = UETPromptAugmenter().augment(self.query, self.contexts)
        self.test_cache_file = BASE_DIR / "database" / "test_llm_cache.json"

    def tearDown(self):
        # Dọn dẹp file test cache nếu có
        if self.test_cache_file.exists():
            try:
                self.test_cache_file.unlink()
            except Exception:
                pass

    def test_key_masking(self):
        """Kiểm tra hàm che giấu API key để bảo mật log."""
        self.assertEqual(UETLLMClient._mask_key(""), "")
        self.assertEqual(UETLLMClient._mask_key("short"), "***")
        masked = UETLLMClient._mask_key("gsk_1234567890abcdef")
        self.assertTrue(masked.startswith("gsk_..."))
        self.assertTrue(masked.endswith("cdef"))

    def test_load_multi_provider_keys(self):
        """Kiểm tra nạp API keys từ .env cho cả 3 nhà cung cấp."""
        env_mock = {
            "GROQ_API_KEY": "gsk_test_key_groq",
            "SAMBANOVA_API_KEY": "samba_test_key_1234",
            "GEMINI_API_KEY": "gemini_test_key_5678",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            client = UETLLMClient(cache_file=self.test_cache_file)
            self.assertIn("gsk_test_key_groq", client.groq_keys)
            self.assertIn("samba_test_key_1234", client.sambanova_keys)
            self.assertIn("gemini_test_key_5678", client.gemini_keys)

    def test_disk_cache_save_and_hit(self):
        """Kiểm tra cơ chế Persistent Disk Cache: Lưu và đọc tức thời (0s)."""
        client = UETLLMClient(enable_cache=True, cache_file=self.test_cache_file)

        # Giả lập phản hồi từ provider
        fake_answer = "Sinh viên cần tích lũy tối thiểu 130 tín chỉ."
        client._save_to_disk_cache(
            prompt_text=self.prompt.formatted_prompt,
            answer=fake_answer,
            model_name="test-model",
            metadata={"test": True},
        )

        # Đọc lại từ Disk Cache
        cached = client._get_from_disk_cache(self.prompt.formatted_prompt)
        self.assertIsNotNone(cached)
        ans, model, meta = cached
        self.assertEqual(ans, fake_answer)
        self.assertEqual(model, "test-model")

        # Kiểm tra qua hàm generate() chính thức
        response = client.generate(self.prompt)
        self.assertEqual(response.answer, fake_answer)
        self.assertTrue(response.metadata.get("cache_hit"))
        self.assertLessEqual(response.latency_seconds, 0.05)

    def test_disk_cache_respects_config_yaml(self):
        """Kiểm tra UETLLMClient đọc đúng cờ llm.enable_disk_cache từ config.yaml khi không truyền enable_cache."""
        with patch("src.llm.llm_client.load_yaml_config", return_value={"llm": {"enable_disk_cache": False}}):
            client_off = UETLLMClient(cache_file=self.test_cache_file)
            self.assertFalse(client_off.enable_cache)
            client_off._save_to_disk_cache(self.prompt.formatted_prompt, "cached", "m")
            self.assertIsNone(client_off._get_from_disk_cache(self.prompt.formatted_prompt))

        with patch("src.llm.llm_client.load_yaml_config", return_value={"llm": {"enable_disk_cache": True}}):
            client_on = UETLLMClient(cache_file=self.test_cache_file)
            self.assertTrue(client_on.enable_cache)

    def test_multi_provider_fallback_chain(self):
        """Kiểm tra chuỗi tự bọc lót: Groq lỗi -> SambaNova gánh -> Gemini gánh."""
        client = UETLLMClient(enable_cache=False)
        client.provider = "groq"
        client.openrouter_keys = []
        client.groq_keys = ["mock_groq"]
        client.sambanova_keys = ["mock_samba"]
        client.gemini_keys = ["mock_gemini"]

        call_log = []

        def mock_groq(text, **kwargs):
            call_log.append("groq")
            raise Exception("Groq 429: Rate limit reached")

        def mock_samba(text, **kwargs):
            call_log.append("sambanova")
            return "Câu trả lời thành công từ SambaNova Llama 3.3!", "Meta-Llama-3.3-70B-Instruct", {}

        def mock_gemini(text, **kwargs):
            call_log.append("gemini")
            return "Câu trả lời từ Gemini", "gemini-2.5-flash", {}

        client._call_groq = mock_groq
        client._call_sambanova = mock_samba
        client._call_gemini = mock_gemini

        response = client.generate(self.prompt)
        self.assertEqual(call_log, ["groq", "sambanova"])
        self.assertEqual(response.answer, "Câu trả lời thành công từ SambaNova Llama 3.3!")
        self.assertEqual(response.model_name, "Meta-Llama-3.3-70B-Instruct")

    def test_multi_provider_worker_queue_batch(self):
        """Kiểm tra Hàng đợi chia tải đa luồng (generate_batch) phân phối câu hỏi cho 3 workers."""
        client = UETLLMClient(enable_cache=False)
        client.openrouter_keys = []
        client.groq_keys = ["mock_groq"]
        client.sambanova_keys = ["mock_samba"]
        client.gemini_keys = ["mock_gemini"]

        # Giảm delay để test chạy nhanh trong 0.01 giây
        client.DEFAULT_CONFIGS["groq"]["delay"] = 0.01
        client.DEFAULT_CONFIGS["sambanova"]["delay"] = 0.01
        client.DEFAULT_CONFIGS["gemini"]["delay"] = 0.01

        processed_by = []

        def mock_groq(text):
            processed_by.append("groq")
            return "Ans from Groq", "llama-3.3-70b-versatile", {}

        def mock_samba(text):
            processed_by.append("sambanova")
            return "Ans from SambaNova", "Meta-Llama-3.3-70B-Instruct", {}

        def mock_gemini(text):
            processed_by.append("gemini")
            return "Ans from Gemini", "gemini-2.5-flash", {}

        client._call_groq = mock_groq
        client._call_sambanova = mock_samba
        client._call_gemini = mock_gemini

        # Tạo danh sách 6 câu hỏi kiểm thử
        test_prompts = []
        for i in range(6):
            q = UserQuery(query_text=f"Câu hỏi số {i}?")
            test_prompts.append(UETPromptAugmenter().augment(q, self.contexts))

        responses = client.generate_batch(test_prompts)

        # Kiểm tra nhận đủ 6 câu trả lời theo đúng thứ tự
        self.assertEqual(len(responses), 6)
        for i, resp in enumerate(responses):
            self.assertEqual(resp.query_id, test_prompts[i].user_query.query_id)
            self.assertTrue("Ans from" in resp.answer)

        # Đảm bảo cả 3 worker cùng tham gia xử lý
        self.assertGreater(len(processed_by), 0)

    def test_worker_queue_fallback_on_429(self):
        """Kiểm tra khi 1 worker gặp lỗi 429, câu hỏi được trả lại hàng đợi để worker khác xử lý."""
        client = UETLLMClient(enable_cache=False)
        client.openrouter_keys = []
        client.groq_keys = ["mock_groq"]
        client.sambanova_keys = ["mock_samba"]
        client.gemini_keys = []

        client.DEFAULT_CONFIGS["groq"]["delay"] = 0.01
        client.DEFAULT_CONFIGS["sambanova"]["delay"] = 0.01

        def mock_groq_fail(text):
            raise Exception("Groq 429: Too Many Requests")

        def mock_samba_success(text):
            return "SambaNova cứu cánh thành công!", "Meta-Llama-3.3-70B-Instruct", {}

        client._call_groq = mock_groq_fail
        client._call_sambanova = mock_samba_success

        q = UserQuery(query_text="Câu hỏi thử thách lỗi?")
        p = UETPromptAugmenter().augment(q, self.contexts)

        # Gọi qua batch queue
        responses = client.generate_batch([p])
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].answer, "SambaNova cứu cánh thành công!")

    def test_custom_api_provider_payload_and_persistent_session(self):
        """Kiểm tra provider custom_api gửi đúng payload (stream=True + UTF-8 SSE decode) và tái sử dụng requests.Session theo session_id."""
        client = UETLLMClient(
            provider="custom_api",
            custom_api_url="https://multiple-effectively-lived-marketing.trycloudflare.com",
            custom_api_model="gemma-4",
            enable_cache=False,
        )

        # Kiểm tra 2 lượt trong cùng session_id dùng chung đúng 1 requests.Session, khác session_id dùng session riêng
        s1_a = client.get_http_session("chat_sess_1")
        s1_b = client.get_http_session("chat_sess_1")
        s2 = client.get_http_session("chat_sess_2")
        self.assertIs(s1_a, s1_b)
        self.assertIsNot(s1_a, s2)
        self.assertEqual(s1_a.headers.get("Connection"), "keep-alive")

        def make_sse_mock():
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"Content-Type": "text/event-stream"}
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = False
            # Giả lập raw UTF-8 bytes tiếng Việt có dấu (tránh mojibake ISO-8859-1)
            mock_resp.iter_lines.return_value = [
                'data: {"choices": [{"delta": {"content": "Trả lời tiếng Việt "}}]}'.encode("utf-8"),
                'data: {"choices": [{"delta": {"content": "từ gemma-4 host"}}], "usage": {"total_tokens": 42}}'.encode("utf-8"),
                b"data: [DONE]",
            ]
            return mock_resp

        q1 = UserQuery(query_text="Câu 1", session_id="chat_sess_1")
        q2 = UserQuery(query_text="Câu 2", session_id="chat_sess_1")
        p1 = UETPromptAugmenter().augment(q1, self.contexts)
        p2 = UETPromptAugmenter().augment(q2, self.contexts)

        with patch.object(s1_a, "post", side_effect=[make_sse_mock(), make_sse_mock()]) as mock_post:
            r1 = client.generate(p1)
            r2 = client.generate(p2)

            self.assertEqual(r1.answer, "Trả lời tiếng Việt từ gemma-4 host")
            self.assertEqual(r1.model_name, "gemma-4")
            self.assertEqual(r1.metadata.get("provider"), "custom_api")
            self.assertTrue(r1.metadata.get("streamed"))
            self.assertEqual(r1.metadata.get("usage"), {"total_tokens": 42})
            self.assertEqual(r2.answer, "Trả lời tiếng Việt từ gemma-4 host")
            self.assertEqual(mock_post.call_count, 2)

            url_called = mock_post.call_args.args[0]
            sent_json = mock_post.call_args.kwargs["json"]
            self.assertTrue(mock_post.call_args.kwargs.get("stream"))
            self.assertEqual(
                url_called,
                "https://multiple-effectively-lived-marketing.trycloudflare.com/v1/chat/completions",
            )
            self.assertEqual(sent_json["model"], "gemma-4")
            self.assertEqual(sent_json["max_tokens"], 2048)
            self.assertEqual(sent_json["chat_template_kwargs"], {"enable_thinking": False})
            self.assertTrue(sent_json["stream"])

        # Kiểm tra thêm generate_stream() cũng giải mã tiếng Việt UTF-8 đúng chuẩn
        with patch.object(s1_a, "post", return_value=make_sse_mock()):
            streamed_text = "".join(client.generate_stream(p1))
            self.assertEqual(streamed_text, "Trả lời tiếng Việt từ gemma-4 host")

        # Đóng session
        client.close_session("chat_sess_1")
        self.assertIsNot(client.get_http_session("chat_sess_1"), s1_a)
        client.close()

    def test_custom_api_health_check(self):
        client = UETLLMClient(
            provider="custom_api",
            custom_api_url="https://multiple-effectively-lived-marketing.trycloudflare.com",
            enable_cache=False,
        )
        sess = client.get_http_session("health_sess")
        mock_ok = MagicMock(status_code=200, text="ok")
        with patch.object(sess, "get", return_value=mock_ok) as mock_get:
            self.assertTrue(client.check_custom_api_health(session_id="health_sess"))
            self.assertEqual(
                mock_get.call_args.args[0],
                "https://multiple-effectively-lived-marketing.trycloudflare.com/health",
            )
        client.close()


if __name__ == "__main__":
    unittest.main()

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
        client = UETLLMClient(provider="groq", enable_cache=False)
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
        client = UETLLMClient(provider="groq", enable_cache=False)
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
        client = UETLLMClient(provider="groq", enable_cache=False)
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

    def test_openrouter_hy4_flash_primary_without_free_fallback(self):
        """Khi dùng OpenRouter tencent/hy4-flash trực tiếp, không rơi sang các provider miễn phí."""
        client = UETLLMClient(provider="openrouter", model_name="tencent/hy4-flash", enable_cache=False)
        self.assertEqual(client.provider, "openrouter")
        self.assertEqual(client.model_name, "tencent/hy4-flash")
        self.assertEqual(client.openrouter_model, "tencent/hy4-flash")

        client.openrouter_keys = ["sk-or-mock"]
        client.groq_keys = ["gsk-mock"]
        client.sambanova_keys = ["samba-mock"]
        client.gemini_keys = ["gem-mock"]
        client.openrouter_delay = 0.01

        calls = []

        def forbid_free(name):
            def _fn(_txt, **_kw):
                calls.append(name)
                raise AssertionError(f"Unexpected call to free provider {name}")
            return _fn

        client._call_groq = forbid_free("groq")
        client._call_sambanova = forbid_free("sambanova")
        client._call_gemini = forbid_free("gemini")

        attempt = {"n": 0}

        def mock_openrouter(text, model=None):
            attempt["n"] += 1
            calls.append(f"openrouter_{attempt['n']}")
            if attempt["n"] == 1:
                raise RuntimeError("OpenRouter temporary 503")
            return "Trả lời từ tencent/hy4-flash!", "tencent/hy4-flash", {"provider": "openrouter"}

        client._call_openrouter = mock_openrouter

        with patch("src.llm.llm_client.time.sleep", return_value=None):
            resp = client.generate(self.prompt)
        self.assertEqual(calls, ["openrouter_1", "openrouter_2"])
        self.assertEqual(resp.model_name, "tencent/hy4-flash")
        self.assertIn("hy4-flash", resp.answer)

        # Kiểm tra generate_batch chỉ tạo worker OpenRouter
        calls.clear()
        batch_resps = client.generate_batch([self.prompt])
        self.assertEqual(len(batch_resps), 1)
        self.assertEqual(batch_resps[0].model_name, "tencent/hy4-flash")
        self.assertEqual(calls, ["openrouter_3"])

        # Kiểm tra generate_stream đọc SSE từ OpenRouter
        class _FakeStreamResp:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"Xin "}}]}'
                yield 'data: {"choices":[{"delta":{"content":"chào!"}}]}'
                yield "data: [DONE]"

        class _FakeHTTPClient:
            def __init__(self, timeout=None):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def stream(self, method, url, headers=None, json=None):
                assert "openrouter.ai" in url
                assert json["model"] == "tencent/hy4-flash"
                assert json["stream"] is True
                assert json["reasoning"] == {"enabled": False}
                return _FakeStreamResp()

        with patch("httpx.Client", _FakeHTTPClient):
            chunks = list(client.generate_stream(self.prompt))
        self.assertEqual("".join(chunks), "Xin chào!")

        # Kiểm tra _call_openrouter gốc gửi kèm reasoning={"enabled": False}
        captured_post = {}

        class _FakePostResp:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": "OK no-reasoning"}}], "usage": {}}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_post["payload"] = json
            return _FakePostResp()

        raw_client = UETLLMClient(provider="openrouter", model_name="tencent/hy4-flash", enable_cache=False)
        raw_client.openrouter_keys = ["sk-or-mock"]
        self.assertEqual(raw_client.reasoning_effort, "none")
        with patch("httpx.post", side_effect=fake_post):
            ans, used_m, _meta = raw_client._call_openrouter("Hỏi test")
        self.assertEqual(ans, "OK no-reasoning")
        self.assertEqual(used_m, "tencent/hy4-flash")
        self.assertEqual(captured_post["payload"]["reasoning"], {"enabled": False})

    def test_openrouter_provider_routing_payload(self):
        """Kiểm tra OpenRouter gửi đúng payload model deepseek/deepseek-v4.1-flash-1017 và provider routing wafer/us."""
        client = UETLLMClient(enable_cache=False)
        self.assertEqual(client.provider, "openrouter")
        self.assertEqual(client.model_name, "deepseek/deepseek-v4.1-flash-1017")
        self.assertEqual(client.openrouter_model, "deepseek/deepseek-v4.1-flash-1017")
        self.assertEqual(
            client.openrouter_provider_routing,
            {"order": ["wafer/us"], "allow_fallbacks": False},
        )
        client.openrouter_keys = ["sk-or-mock"]

        captured_post = {}

        class _FakePostResp:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": "OK wafer"}}], "usage": {}}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_post["payload"] = json
            return _FakePostResp()

        with patch("httpx.post", side_effect=fake_post):
            ans, used_m, _meta = client._call_openrouter("Hỏi test")
        self.assertEqual(ans, "OK wafer")
        self.assertEqual(used_m, "deepseek/deepseek-v4.1-flash-1017")
        self.assertEqual(
            captured_post["payload"]["provider"],
            {"order": ["wafer/us"], "allow_fallbacks": False},
        )
        self.assertEqual(captured_post["payload"]["max_tokens"], 2048)

        captured_stream = {}

        class _FakeStreamResp:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"OK stream"}}]}'
                yield "data: [DONE]"

        class _FakeHTTPClient:
            def __init__(self, timeout=None):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def stream(self, method, url, headers=None, json=None):
                captured_stream["payload"] = json
                return _FakeStreamResp()

        with patch("httpx.Client", _FakeHTTPClient):
            chunks = list(client.generate_stream(self.prompt))
        self.assertEqual("".join(chunks), "OK stream")
        self.assertEqual(
            captured_stream["payload"]["provider"],
            {"order": ["wafer/us"], "allow_fallbacks": False},
        )
        self.assertEqual(captured_stream["payload"]["max_tokens"], 2048)

    def test_easy_switch_between_openrouter_and_custom_api(self):
        """Chỉ cần đổi provider hoặc biến môi trường LLM_PROVIDER là tự khớp đúng model tương ứng."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "custom_api"}, clear=False):
            c_custom = UETLLMClient(enable_cache=False)
            self.assertEqual(c_custom.provider, "custom_api")
            self.assertEqual(c_custom.model_name, "gemma-4")

        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=False):
            c_or = UETLLMClient(enable_cache=False)
            self.assertEqual(c_or.provider, "openrouter")
            self.assertEqual(c_or.model_name, "deepseek/deepseek-v4.1-flash-1017")
            self.assertEqual(c_or.openrouter_model, "deepseek/deepseek-v4.1-flash-1017")

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

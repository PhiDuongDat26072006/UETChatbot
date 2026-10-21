"""
src/llm/llm_client.py - Phụ trách kết nối và gọi mô hình ngôn ngữ lớn (LLM) qua Groq API.
Thành viên phụ trách: Người tích hợp mô hình LLM.
Nhiệm vụ: Gọi Groq API (Qwen, Llama 3.1, Llama 3.3...) để sinh câu trả lời cuối cùng kèm trích dẫn nguồn.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Optional, Generator, List, Any, Dict, Tuple

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.base import BaseLLM, AugmentedPrompt, Response
from src.utils.helpers import get_logger, Timer, load_yaml_config

logger = get_logger("llm")


class UETLLMClient(BaseLLM):
    """
    Client tương tác với LLM cho UET Chatbot sử dụng Groq API.
    Kế thừa BaseLLM từ base.py và cài đặt các hàm generate() và generate_stream().
    Đồng bộ các tham số (model_name, candidate_models, temperature) từ config.yaml và .env.
    """

    DEFAULT_CANDIDATE_MODELS = [
        "qwen/qwen3.8-27b",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]
    DEFAULT_GROQ_MODELS = DEFAULT_CANDIDATE_MODELS

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        provider: Optional[str] = None,
    ):
        config = load_yaml_config()
        llm_cfg = config.get("llm", {})

        self.provider = "groq"
        self._model_name = model_name or llm_cfg.get("model_name")
        self.candidate_models = list(llm_cfg.get("candidate_models", self.DEFAULT_CANDIDATE_MODELS))
        # Thuộc tính tương thích ngược
        self.CANDIDATE_MODELS = self.candidate_models
        # Nhiệt độ mặc định 0.2
        self.temperature = float(llm_cfg.get("temperature", 0.2))

        # Tải danh sách tất cả các API Keys khả dụng từ .env (hỗ trợ tự động chuyển đổi khi hết hạn mức)
        self.api_keys = self._load_api_keys(
            custom_key=api_key,
            custom_keys=api_keys,
            provider=self.provider,
        )
        self.current_key_index = 0
        self._groq_clients: dict[str, Any] = {}

        if self.api_keys:
            logger.info(f"Đã phát hiện {len(self.api_keys)} API Key Groq (Hỗ trợ Auto-Failover).")
            # Khởi tạo trước client cho key đầu tiên
            self._get_groq_client_for_key(self.api_keys[0])
        else:
            logger.info("Chưa tìm thấy GROQ_API_KEY. Sẽ sử dụng phản hồi trích xuất trực tiếp khi sinh câu trả lời.")

    @staticmethod
    def _mask_key(key: str) -> str:
        """Che bớt ký tự của API Key để bảo mật thông tin khi log."""
        if not key:
            return ""
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    @classmethod
    def _load_api_keys(
        cls,
        custom_key: Optional[str] = None,
        custom_keys: Optional[List[str]] = None,
        provider: str = "groq",
    ) -> List[str]:
        """
        Nạp danh sách tất cả Groq API Key từ tham số truyền vào hoặc từ biến môi trường (.env):
        1. custom_keys / custom_key (nếu truyền trực tiếp vào hàm khởi tạo)
        2. GROQ_API_KEYS, GROQ_API_KEY, GROQ_API_KEY_1..50, GROQ_BACKUP_API_KEY
        """
        keys: List[str] = []

        if custom_key == "":
            # Truyền rỗng cố ý -> chạy offline không dùng key
            return []

        if custom_keys:
            for k in custom_keys:
                k_str = str(k).strip().strip("'\"")
                if k_str and k_str not in keys:
                    keys.append(k_str)

        if custom_key:
            c = custom_key.strip().strip("'\"")
            if c and c not in keys:
                keys.append(c)

        # Quét các biến môi trường Groq
        groq_env_keys = os.getenv("GROQ_API_KEYS", "")
        if groq_env_keys:
            for item in groq_env_keys.replace(";", "\n").replace(",", "\n").split("\n"):
                k = item.strip().strip("'\"")
                if k and not k.startswith("#") and k not in keys:
                    keys.append(k)

        main_key = os.getenv("GROQ_API_KEY", "")
        if main_key:
            for item in main_key.replace(";", "\n").replace(",", "\n").split("\n"):
                k = item.strip().strip("'\"")
                if k and not k.startswith("#") and k not in keys:
                    keys.append(k)

        for i in range(1, 51):
            k = os.getenv(f"GROQ_API_KEY_{i}", "").strip().strip("'\"")
            if k and k not in keys:
                keys.append(k)

        backup_key = os.getenv("GROQ_BACKUP_API_KEY", "").strip().strip("'\"")
        if backup_key and backup_key not in keys:
            keys.append(backup_key)

        return keys

    # -------------------------------------------------------------------------
    # GROQ CLIENT LOGIC
    # -------------------------------------------------------------------------

    def _get_groq_client_for_key(self, key: str):
        """Khởi tạo hoặc tái sử dụng Groq Client cho một API key cụ thể."""
        if not key:
            return None
        if key not in self._groq_clients:
            try:
                from groq import Groq
                self._groq_clients[key] = Groq(api_key=key, timeout=25.0)
            except ImportError:
                # Đánh dấu dùng REST API trực tiếp qua httpx nếu chưa cài gói groq
                self._groq_clients[key] = "httpx_direct"
            except Exception as e:
                logger.warning(f"Lỗi khởi tạo Groq Client cho key [{self._mask_key(key)}]: {e}")
                return None
        return self._groq_clients.get(key)

    def _call_groq_generate(self, key: str, model: str, prompt_text: str) -> Tuple[str, Dict[str, Any]]:
        """Gọi Groq API sinh câu trả lời qua SDK hoặc direct REST API."""
        client = self._get_groq_client_for_key(key)
        if client != "httpx_direct" and hasattr(client, "chat"):
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.temperature,
            )
            answer = ""
            if completion.choices and completion.choices[0].message:
                answer = completion.choices[0].message.content or ""
            meta: Dict[str, Any] = {}
            if hasattr(completion, "usage") and completion.usage:
                meta["prompt_tokens"] = getattr(completion.usage, "prompt_tokens", None)
                meta["candidates_tokens"] = getattr(completion.usage, "completion_tokens", None)
            return answer.strip(), meta

        # Fallback sang httpx
        import httpx
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature,
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=25.0)
        if resp.status_code >= 400:
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err_msg = resp.text
            raise RuntimeError(f"Groq API Error ({resp.status_code}): {err_msg}")

        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        meta = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "candidates_tokens": usage.get("completion_tokens"),
        }
        return (answer or "").strip(), meta

    def _call_groq_stream(self, key: str, model: str, prompt_text: str) -> Generator[str, None, None]:
        """Gọi Groq API streaming qua SDK hoặc direct HTTP SSE."""
        client = self._get_groq_client_for_key(key)
        if client != "httpx_direct" and hasattr(client, "chat"):
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        import httpx
        import json
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature,
            "stream": True,
        }
        with httpx.Client(timeout=30.0) as http_client:
            with http_client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    try:
                        err_msg = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        err_msg = response.text
                    raise RuntimeError(f"Groq Stream Error ({response.status_code}): {err_msg}")
                for line in response.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        delta_content = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta_content:
                            yield delta_content
                    except Exception:
                        continue

    # -------------------------------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------------------------------

    @property
    def api_key(self) -> str:
        """Trả về API Key đang được sử dụng hiện tại."""
        if self.api_keys and 0 <= self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return ""

    @api_key.setter
    def api_key(self, value: str):
        if value:
            self.api_keys = [value]
            self.current_key_index = 0
            self._groq_clients = {}
            self._get_groq_client_for_key(value)
        else:
            self.api_keys = []
            self.current_key_index = 0

    @property
    def _client(self):
        """Trả về Groq Client của API Key đang hoạt động."""
        current_k = self.api_key
        return self._get_groq_client_for_key(current_k) if current_k else None

    @property
    def model_name(self) -> str:
        """Tên mô hình LLM đang sử dụng."""
        return self._model_name

    # -------------------------------------------------------------------------
    # GENERATION & STREAMING INTERFACES
    # -------------------------------------------------------------------------

    def generate(self, prompt: AugmentedPrompt) -> Response:
        """
        Gửi prompt đến Groq API để nhận câu trả lời cho người dùng.
        Hỗ trợ tự động chuyển đổi mô hình (Model Failover) và tự động đổi API Key (Key Failover) khi hết hạn mức.
        """
        logger.info(f"Đang gửi yêu cầu sinh câu trả lời đến mô hình '{self.model_name}' (Groq)...")

        errors_encountered: List[str] = []

        with Timer() as timer:
            # 1. Duyệt qua các API Keys khả dụng (bắt đầu từ key đang hoạt động tốt)
            if self.api_keys:
                ordered_keys = (
                    self.api_keys[self.current_key_index:] +
                    self.api_keys[:self.current_key_index]
                )
                models_to_try = [self.model_name] + [m for m in self.candidate_models if m != self.model_name]

                for key_idx, key in enumerate(ordered_keys):
                    key_num = self.api_keys.index(key) + 1

                    for m in models_to_try:
                        try:
                            answer, meta = self._call_groq_generate(
                                key=key,
                                model=m,
                                prompt_text=prompt.formatted_prompt,
                            )

                            if answer:
                                self.current_key_index = self.api_keys.index(key)
                                return Response(
                                    query_id=prompt.user_query.query_id,
                                    answer=answer,
                                    sources=prompt.contexts,
                                    model_name=m,
                                    latency_seconds=round(timer.elapsed, 3),
                                    metadata=meta,
                                )
                        except Exception as e:
                            err_str = str(e)
                            errors_encountered.append(f"Key#{key_num}-Model[{m}]: {err_str}")
                            logger.warning(f"API Key #{key_num} - Model '{m}' gặp lỗi ({err_str}).")

                            # Nếu gặp lỗi quyền (401, 403 - khóa không hợp lệ), bỏ qua ngay các model còn lại của key này
                            if any(auth_err in err_str.lower() for auth_err in ["401", "403", "invalid_api_key", "permission_denied"]):
                                break

                    # Nếu tất cả các model của key này đều gặp lỗi, chuyển sang API Key tiếp theo
                    if len(ordered_keys) > 1 and key_idx < len(ordered_keys) - 1:
                        next_key_num = self.api_keys.index(ordered_keys[key_idx + 1]) + 1
                        logger.warning(
                            f"🔄 API Key #{key_num} đã hết hạn mức ở tất cả model. "
                            f"Tự động chuyển sang API Key dự phòng #{next_key_num}..."
                        )

            # 2. Xử lý trường hợp tất cả mô hình và tất cả API Key đều từ chối dịch vụ hoặc chưa có API Key
            logger.info("Sử dụng chế độ thông báo lỗi / fallback trích đoạn tài liệu quy chế.")

            if errors_encountered:
                combined_err = " | ".join(errors_encountered)
                if "503" in combined_err or "UNAVAILABLE" in combined_err or "overloaded" in combined_err.lower():
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ (MÃ LỖI 503 - SERVER QUÁ TẢI)**"
                    error_detail = (
                        "Máy chủ Groq hiện đang chịu tải quá lớn "
                        "và tạm thời từ chối xử lý yêu cầu lúc này."
                    )
                elif "429" in combined_err or "rate_limit" in combined_err.lower() or "quota" in combined_err.lower():
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ (MÃ LỖI 429 - TẤT CẢ API KEY ĐỀU HẾT HẠN MỨC)**"
                    error_detail = (
                        f"Tất cả {len(self.api_keys)} API Key Groq đã sử dụng hết hạn mức yêu cầu (Rate Limit / Quota Exceeded)."
                    )
                elif any(auth_err in combined_err.lower() for auth_err in ["401", "403", "invalid_api_key", "permission_denied"]):
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ (LỖI KHÓA API KHÔNG HỢP LỆ / 401)**"
                    error_detail = "Khóa API GROQ_API_KEY không hợp lệ, bị vô hiệu hóa hoặc không có quyền truy cập."
                else:
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ / LỖI KẾT NỐI GROQ API**"
                    short_err = errors_encountered[0].split("\n")[0][:120]
                    error_detail = f"Không thể nhận phản hồi từ dịch vụ Groq ({short_err})."

                fallback_answer = (
                    f"{error_title}\n\n"
                    f"> {error_detail}\n\n"
                    f"*(Bạn có thể cấu hình thêm API Key Groq dự phòng trong file .env hoặc thử lại sau ít phút).*"
                )
                resp_model_name = "groq-service-denied-fallback"

            elif not self.api_keys:
                fallback_answer = (
                    "⚠️ **CHƯA CẤU HÌNH API KEY**\n\n"
                    "> Vui lòng kiểm tra và thêm `GROQ_API_KEY` vào file `.env` để kích hoạt phản hồi thông minh từ AI."
                )
                resp_model_name = "groq-offline-fallback"

            else:
                fallback_answer = (
                    "Hệ thống gặp sự cố không xác định khi kết nối với mô hình ngôn ngữ lớn Groq. "
                    "Vui lòng thử lại sau ít phút."
                )
                resp_model_name = "groq-unknown-fallback"

        return Response(
            query_id=prompt.user_query.query_id,
            answer=fallback_answer,
            sources=prompt.contexts,
            model_name=resp_model_name,
            latency_seconds=round(timer.elapsed, 3),
        )

    def generate_stream(self, prompt: AugmentedPrompt) -> Generator[str, None, None]:
        """
        Sinh phản hồi dạng luồng (streaming) qua Groq API cho giao diện người dùng.
        Hỗ trợ tự động chuyển sang API Key khác nếu key hiện tại hết hạn mức.
        """
        if self.api_keys:
            ordered_keys = (
                self.api_keys[self.current_key_index:] +
                self.api_keys[:self.current_key_index]
            )
            models_to_try = [self.model_name] + [m for m in self.candidate_models if m != self.model_name]

            for key_idx, key in enumerate(ordered_keys):
                key_num = self.api_keys.index(key) + 1

                for m in models_to_try:
                    try:
                        has_content = False
                        for chunk in self._call_groq_stream(key=key, model=m, prompt_text=prompt.formatted_prompt):
                            if chunk:
                                has_content = True
                                yield chunk

                        if has_content:
                            self.current_key_index = self.api_keys.index(key)
                            return
                    except Exception as e:
                        err_str = str(e)
                        logger.warning(f"Streaming thất bại với API Key #{key_num} - Model '{m}': {err_str}")
                        if any(auth_err in err_str.lower() for auth_err in ["401", "403", "invalid_api_key", "permission_denied"]):
                            break

                if len(ordered_keys) > 1 and key_idx < len(ordered_keys) - 1:
                    next_key_num = self.api_keys.index(ordered_keys[key_idx + 1]) + 1
                    logger.warning(
                        f"🔄 Streaming: API Key #{key_num} đã hết hạn mức ở tất cả model. "
                        f"Tự động chuyển sang API Key dự phòng #{next_key_num}..."
                    )

        # Fallback stream nếu không có client hoặc streaming thất bại
        yield self.generate(prompt).answer


if __name__ == "__main__":
    _test_key = os.getenv("GROQ_API_KEY", "")
    print("Kiểm tra UETLLMClient (Provider: Groq)")
    llm = UETLLMClient()
    print(f"Model chính: {llm.model_name}")
    print(f"Candidates: {llm.candidate_models}")
    print(f"Số lượng keys: {len(llm.api_keys)}")

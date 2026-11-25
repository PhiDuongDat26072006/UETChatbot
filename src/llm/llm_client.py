"""
src/llm/llm_client.py - Phụ trách kết nối và phân phối mô hình ngôn ngữ lớn (LLM).
Thành viên phụ trách: Người tích hợp mô hình LLM (Hoàng).
Nhiệm vụ:
  - Hỗ trợ đa nhà cung cấp (Multi-Provider): Groq, SambaNova, Google Gemini.
  - Phân tải đa luồng qua hàng đợi (Multi-Provider Worker Queue):
      * Worker Groq (30 câu/phút - nhịp nghỉ 2.0s)
      * Worker SambaNova (24 câu/phút - nhịp nghỉ 2.5s)
      * Worker Gemini (15 câu/phút - nhịp nghỉ 4.0s)
      => Tổng công suất: ~69 câu/phút (~85-90s cho 100 câu), chi phí 0đ.
  - Bộ nhớ đệm ổ cứng (Persistent Disk Cache) lưu tại database/llm_cache.json:
      * Câu hỏi đã có trong cache trả về tức thì (0s, không tốn quota).
  - Khả năng tự bọc lót (Fault-Tolerance & Fallback):
      * Nếu 1 luồng gặp lỗi 429 hoặc timeout, câu hỏi tự động được trả lại hàng đợi để luồng khác xử lý.
"""

from __future__ import annotations
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Generator, List, Optional, Tuple

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.base import AugmentedPrompt, BaseLLM, Response
from src.utils.helpers import Timer, get_logger, load_yaml_config
from src.utils.http_session import HttpSessionManager

import requests

logger = get_logger("llm")


class LlmUnavailableError(Exception):
    """Ném ra khi tất cả các nhà cung cấp LLM đều thất bại (429, 401, hết quota, timeout...). Tuân thủ D12."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class UETLLMClient(BaseLLM):
    """
    Client LLM tích hợp cơ chế Multi-Provider Queue + Persistent Disk Cache cho UET Chatbot.
    Kế thừa BaseLLM từ src/base.py, tương thích hoàn toàn với RAG Pipeline và hệ thống Evaluation.
    """

    # Cấu hình mặc định cho từng Provider
    DEFAULT_CONFIGS = {
        "custom_api": {
            "model": "gemma-4",
            "candidate_models": ["gemma-4"],
            "base_url": "https://multiple-effectively-lived-marketing.trycloudflare.com",
            "delay": 0.5,
        },
        "groq": {
            "model": "qwen/qwen3.8-27b",
            "candidate_models": ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
            "base_url": "https://api.groq.com/openai/v1",
            "delay": 2.0,  # ~30 requests/phút
        },
        "sambanova": {
            "model": "Meta-Llama-3.1-8B-Instruct",
            "candidate_models": ["Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.3-70B-Instruct"],
            "base_url": "https://api.sambanova.ai/v1",
            "delay": 2.5,  # ~24 requests/phút
        },
        "gemini": {
            "model": "gemini-2.5-flash",
            "candidate_models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"],
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "delay": 4.0,  # ~15 requests/phút
        },
        "openrouter": {
            "model": "tencent/hy4-flash",
            "candidate_models": [
                "tencent/hy4-flash",
                "tencent/hy4",
                "tencent/hy4-preview",
            ],
            "base_url": "https://openrouter.ai/api/v1",
            "delay": 1.5,
        },
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        provider: Optional[str] = None,
        enable_cache: Optional[bool] = None,
        cache_file: Optional[str | Path] = None,
        custom_api_url: Optional[str] = None,
        custom_api_model: Optional[str] = None,
        custom_api_key: Optional[str] = None,
        session_ttl_s: float = 1800.0,
    ):
        config = load_yaml_config()
        llm_cfg = config.get("llm", {})
        custom_cfg = llm_cfg.get("custom_api", {})

        raw_provider = (
            provider
            or os.getenv("LLM_PROVIDER")
            or llm_cfg.get("provider", "openrouter")
        ).strip().lower()
        self.provider = "custom_api" if raw_provider in ("api", "custom", "custom_api") else raw_provider

        # Cấu hình Custom API (OpenAI-compatible / Cloudflare Tunnel giống tests/test_gemma_host.py)
        self.custom_api_url = (
            custom_api_url
            or os.getenv("LLM_API_BASE_URL")
            or custom_cfg.get("base_url")
            or self.DEFAULT_CONFIGS["custom_api"]["base_url"]
        ).rstrip("/")
        self.custom_api_model = (
            custom_api_model
            or os.getenv("LLM_API_MODEL")
            or custom_cfg.get("model")
            or self.DEFAULT_CONFIGS["custom_api"]["model"]
        )
        self.custom_api_key = (
            custom_api_key
            if custom_api_key is not None
            else os.getenv("LLM_API_KEY", custom_cfg.get("api_key", ""))
        )
        self.custom_api_max_tokens = int(custom_cfg.get("max_tokens", 2048))
        self.custom_api_connect_timeout_s = float(custom_cfg.get("connect_timeout_s", 10.0))
        self.custom_api_timeout_s = float(custom_cfg.get("timeout_s", 120.0))
        self.custom_api_health_path = str(custom_cfg.get("health_path", "/health"))
        self.custom_api_enable_thinking = bool(custom_cfg.get("enable_thinking", False))
        self.custom_api_stream = bool(custom_cfg.get("stream", True))

        providers_cfg = llm_cfg.get("providers", {})
        openrouter_cfg = providers_cfg.get("openrouter", {})
        self.openrouter_model = (
            model_name
            if (model_name and self.provider == "openrouter")
            else openrouter_cfg.get("model", self.DEFAULT_CONFIGS["openrouter"]["model"])
        )
        self.openrouter_delay = float(
            openrouter_cfg.get("delay_seconds", self.DEFAULT_CONFIGS["openrouter"]["delay"])
        )
        raw_routing = openrouter_cfg.get("provider_routing")
        self.openrouter_provider_routing: Optional[Dict[str, Any]] = (
            dict(raw_routing) if isinstance(raw_routing, dict) and raw_routing else None
        )

        # Tự động khớp model theo provider đang chọn (chỉ cần đổi llm.provider hoặc --llm)
        if model_name:
            self._model_name = model_name
        elif self.provider == "custom_api":
            self._model_name = self.custom_api_model
        elif self.provider == "openrouter":
            self._model_name = self.openrouter_model
        else:
            self._model_name = (
                providers_cfg.get(self.provider, {}).get("model")
                or llm_cfg.get("model_name")
                or self.DEFAULT_CONFIGS.get(self.provider, {}).get("model", "tencent/hy4-flash")
            )

        default_provider_candidates = self.DEFAULT_CONFIGS.get(
            self.provider, self.DEFAULT_CONFIGS["openrouter"]
        )["candidate_models"]
        self.candidate_models = list(llm_cfg.get("candidate_models", default_provider_candidates))
        self.CANDIDATE_MODELS = self.candidate_models
        self.temperature = float(llm_cfg.get("temperature", 0.2))
        self.json_temperature = float(llm_cfg.get("json_temperature", 0.0))
        self.reasoning_effort = llm_cfg.get("reasoning_effort", "none")

        # Quản lý HTTP Session (Keep-Alive) duy trì xuyên suốt từng phiên trò chuyện
        ttl_val = float(llm_cfg.get("session_ttl_s", session_ttl_s))
        self._session_manager = HttpSessionManager(ttl_s=ttl_val)
        self._session = self._session_manager.default_session

        # Nạp API Keys cho các providers
        self.groq_keys = self._load_keys_for_provider(
            "GROQ",
            custom_key=api_key if self.provider != "openrouter" else None,
            custom_keys=api_keys if self.provider != "openrouter" else None,
        )
        self.sambanova_keys = self._load_keys_for_provider("SAMBANOVA")
        self.gemini_keys = self._load_keys_for_provider("GEMINI")
        self.openrouter_keys = self._load_keys_for_provider(
            "OPENROUTER",
            custom_key=api_key if self.provider == "openrouter" else None,
            custom_keys=api_keys if self.provider == "openrouter" else None,
        )

        # Tự động nhận diện nếu key trong GROQ_API_KEY là OpenRouter key (sk-or-...)
        if not self.openrouter_keys:
            for k in self.groq_keys:
                if k.startswith("sk-or-") and k not in self.openrouter_keys:
                    self.openrouter_keys.append(k)

        # Khóa tương thích ngược
        self.api_keys = self.openrouter_keys if self.provider == "openrouter" else (self.groq_keys or self.openrouter_keys)
        self.current_key_index = 0

        # Cấu hình Disk Cache (ưu tiên tham số truyền trực tiếp, ngược lại đọc llm.enable_disk_cache từ config.yaml)
        if enable_cache is not None:
            self.enable_cache = bool(enable_cache)
        else:
            self.enable_cache = bool(llm_cfg.get("enable_disk_cache", False))

        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            cfg_cache_file = llm_cfg.get("cache_file")
            self.cache_file = (_BASE_DIR / cfg_cache_file) if cfg_cache_file else (_BASE_DIR / "database" / "llm_cache.json")

        self._cache_lock = threading.Lock()
        if self.enable_cache:
            self._init_cache_file()

        # Thống kê trạng thái
        active_providers = []
        if self.provider == "custom_api" and self.custom_api_url:
            active_providers.append(f"CustomAPI ({self.custom_api_url}, model={self.custom_api_model})")
        if self.openrouter_keys:
            active_providers.append(f"OpenRouter ({len(self.openrouter_keys)} keys)")
        if self.groq_keys:
            active_providers.append(f"Groq ({len(self.groq_keys)} keys)")
        if self.sambanova_keys:
            active_providers.append(f"SambaNova ({len(self.sambanova_keys)} keys)")
        if self.gemini_keys:
            active_providers.append(f"Gemini ({len(self.gemini_keys)} keys)")
        if self.openrouter_keys:
            active_providers.append(f"OpenRouter ({len(self.openrouter_keys)} keys)")

        if active_providers:
            cache_status = self.cache_file.name if self.enable_cache else "OFF"
            logger.info(
                f"Đã kích hoạt LLM [{self.provider}]: {', '.join(active_providers)} | "
                f"Cache: {cache_status} | SessionTTL: {ttl_val}s"
            )
        else:
            logger.warning("Chưa tìm thấy API Key hoặc Custom API URL nào. LLM sẽ hoạt động ở chế độ fallback.")

    # -------------------------------------------------------------------------
    # HTTP SESSION MANAGEMENT (KEEP-ALIVE PER CONVERSATION)
    # -------------------------------------------------------------------------

    def get_http_session(self, session_id: Optional[str] = None) -> requests.Session:
        """Trả về requests.Session() duy trì kết nối Keep-Alive cho phiên trò chuyện tương ứng."""
        if not session_id:
            return self._session
        return self._session_manager.get_session(session_id)

    def close_session(self, session_id: Optional[str]) -> None:
        """Đóng kết nối HTTP Session của một phiên trò chuyện."""
        self._session_manager.close_session(session_id)

    def close(self) -> None:
        """Đóng toàn bộ kết nối HTTP Session đang mở."""
        self._session_manager.close_all()

    def check_custom_api_health(self, session_id: Optional[str] = None) -> bool:
        """Kiểm tra GET /health tới Custom LLM API server (giống tests/test_gemma_host.py)."""
        if not self.custom_api_url:
            return False
        try:
            url = f"{self.custom_api_url}{self.custom_api_health_path}"
            headers = {"User-Agent": "UET-Chatbot-HealthCheck"}
            if self.custom_api_key:
                headers["Authorization"] = f"Bearer {self.custom_api_key}"
            http_session = self.get_http_session(session_id)
            resp = http_session.get(url, headers=headers, timeout=(self.custom_api_connect_timeout_s, 10.0))
            if resp.status_code == 200:
                logger.info(f"[CustomLLM Health] /health OK (status 200) từ {url}")
                return True
            logger.warning(f"[CustomLLM Health] /health trả về status={resp.status_code}")
            return False
        except Exception as e:
            logger.warning(f"[CustomLLM Health] Không thể kết nối tới /health: {e}")
            return False

    # -------------------------------------------------------------------------
    # DISK CACHE MANAGEMENT
    # -------------------------------------------------------------------------

    def _init_cache_file(self):
        """Khởi tạo file cache nếu chưa tồn tại."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.cache_file.exists():
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Không thể khởi tạo file cache {self.cache_file}: {e}")

    @staticmethod
    def _compute_prompt_hash(prompt_text: str) -> str:
        """Tạo khóa hash SHA-256 từ nội dung prompt để định danh trong cache."""
        return hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()

    def _get_from_disk_cache(self, prompt_text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Đọc kết quả từ file cache trên ổ cứng."""
        if not self.enable_cache or not self.cache_file.exists():
            return None

        prompt_hash = self._compute_prompt_hash(prompt_text)
        with self._cache_lock:
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if prompt_hash in cache_data:
                    entry = cache_data[prompt_hash]
                    return entry.get("answer", ""), entry.get("model_name", "disk-cache"), entry.get("metadata", {})
            except Exception as e:
                logger.debug(f"Lỗi đọc disk cache: {e}")
        return None

    def _save_to_disk_cache(self, prompt_text: str, answer: str, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Lưu kết quả phản hồi vào file cache trên ổ cứng."""
        if not self.enable_cache or not answer or "⚠️" in answer:
            return

        prompt_hash = self._compute_prompt_hash(prompt_text)
        with self._cache_lock:
            try:
                cache_data = {}
                if self.cache_file.exists():
                    try:
                        with open(self.cache_file, "r", encoding="utf-8") as f:
                            cache_data = json.load(f)
                    except Exception:
                        cache_data = {}

                cache_data[prompt_hash] = {
                    "answer": answer,
                    "model_name": model_name,
                    "metadata": metadata or {},
                    "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.debug(f"Lỗi ghi disk cache: {e}")

    # -------------------------------------------------------------------------
    # API KEY MANAGEMENT
    # -------------------------------------------------------------------------

    @classmethod
    def _load_api_keys(
        cls,
        custom_key: Optional[str] = None,
        custom_keys: Optional[List[str]] = None,
        provider: str = "groq",
    ) -> List[str]:
        """Phương thức tương thích ngược để nạp Groq keys."""
        return cls._load_keys_for_provider("GROQ", custom_key=custom_key, custom_keys=custom_keys)

    @classmethod
    def _load_keys_for_provider(
        cls,
        prefix: str,
        custom_key: Optional[str] = None,
        custom_keys: Optional[List[str]] = None,
    ) -> List[str]:
        """Nạp danh sách API keys cho một nhà cung cấp cụ thể từ biến môi trường hoặc tham số."""
        keys: List[str] = []

        if custom_key == "":
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

        for env_var in [f"{prefix}_API_KEYS", f"{prefix}_API_KEY", f"{prefix}_KEY"]:
            raw_val = os.getenv(env_var, "")
            if raw_val:
                for item in raw_val.replace(";", "\n").replace(",", "\n").split("\n"):
                    k = item.strip().strip("'\"")
                    if k and not k.startswith("#") and k not in keys:
                        keys.append(k)

        for i in range(1, 21):
            k = os.getenv(f"{prefix}_API_KEY_{i}", "").strip().strip("'\"")
            if k and k not in keys:
                keys.append(k)

        return keys

    @staticmethod
    def _mask_key(key: str) -> str:
        if not key:
            return ""
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    # -------------------------------------------------------------------------
    # CALL PROVIDERS (CUSTOM_API, OPENROUTER, GROQ, SAMBANOVA, GEMINI)
    # -------------------------------------------------------------------------

    @staticmethod
    def _iter_sse_deltas(response) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
        """
        Đọc luồng SSE OpenAI-compatible bằng raw bytes rồi giải mã UTF-8 tường minh
        (tránh lỗi requests mặc định ISO-8859-1 làm vỡ font tiếng Việt khi server không gửi charset=utf-8).
        Yield từng cặp (delta_text, usage_or_none).
        """
        try:
            response.encoding = "utf-8"
        except Exception:
            pass

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace").strip()
            else:
                line = str(raw_line).strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk_json = json.loads(data_str)
                usage = chunk_json.get("usage")
                choices = chunk_json.get("choices") or []
                delta = ""
                if choices:
                    delta = choices[0].get("delta", {}).get("content") or ""
                if delta or usage:
                    yield delta, usage
            except Exception:
                continue

    def _call_custom_api(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Gọi LLM qua Custom OpenAI-compatible API / Cloudflare Tunnel (giống tests/test_gemma_host.py).
        - Duy trì requests.Session() (Keep-Alive) xuyên suốt phiên trò chuyện theo session_id.
        - Hỗ trợ chế độ "stream": True kèm giải mã UTF-8 tường minh (tránh timeout tunnel & vỡ font tiếng Việt).
        """
        if not self.custom_api_url:
            raise RuntimeError("Chưa cấu hình llm.custom_api.base_url trong config.yaml hoặc LLM_API_BASE_URL")

        target_model = model or self.custom_api_model
        url = f"{self.custom_api_url}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.custom_api_key:
            headers["Authorization"] = f"Bearer {self.custom_api_key}"

        use_stream = self.custom_api_stream
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens if max_tokens is not None else self.custom_api_max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "chat_template_kwargs": {"enable_thinking": self.custom_api_enable_thinking},
        }
        if use_stream:
            payload["stream"] = True

        http_session = self.get_http_session(session_id)

        if use_stream:
            with http_session.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(self.custom_api_connect_timeout_s, self.custom_api_timeout_s),
            ) as resp:
                if resp.status_code >= 400:
                    err_msg = resp.text
                    try:
                        err_msg = resp.json().get("error", {}).get("message", resp.text)
                    except Exception:
                        pass
                    raise RuntimeError(f"CustomAPI [{resp.status_code}]: {err_msg}")

                content_type = str(getattr(resp, "headers", {}).get("Content-Type", "") or "").lower()
                if "application/json" in content_type:
                    data = resp.json()
                    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})
                    return answer.strip(), target_model, {"provider": "custom_api", "streamed": False, "usage": usage}

                parts: List[str] = []
                usage: Dict[str, Any] = {}
                for delta, chunk_usage in self._iter_sse_deltas(resp):
                    if delta:
                        parts.append(delta)
                    if chunk_usage:
                        usage = chunk_usage
                answer = "".join(parts)
                return answer.strip(), target_model, {"provider": "custom_api", "streamed": True, "usage": usage}

        resp = http_session.post(
            url,
            headers=headers,
            json=payload,
            timeout=(self.custom_api_connect_timeout_s, self.custom_api_timeout_s),
        )
        if resp.status_code >= 400:
            err_msg = resp.text
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"CustomAPI [{resp.status_code}]: {err_msg}")

        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return answer.strip(), target_model, {"provider": "custom_api", "streamed": False, "usage": usage}

    def _call_groq(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi Groq API qua persistent requests.Session()."""
        if not self.groq_keys:
            raise RuntimeError("Chưa cấu hình GROQ_API_KEY trong .env")

        target_model = model or self.DEFAULT_CONFIGS["groq"]["model"]
        url = f"{self.DEFAULT_CONFIGS['groq']['base_url']}/chat/completions"
        key = self.groq_keys[0]

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature if temperature is None else temperature,
        }

        http_session = self.get_http_session(session_id)
        resp = http_session.post(url, headers=headers, json=payload, timeout=(10.0, 25.0))
        if resp.status_code >= 400:
            err_msg = resp.text
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"Groq [{resp.status_code}]: {err_msg}")

        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return answer.strip(), target_model, {"provider": "groq", "usage": usage}

    def _call_sambanova(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi SambaNova API (OpenAI-compatible) qua persistent requests.Session()."""
        if not self.sambanova_keys:
            raise RuntimeError("Chưa cấu hình SAMBANOVA_API_KEY trong .env")

        target_model = model or self.DEFAULT_CONFIGS["sambanova"]["model"]
        url = f"{self.DEFAULT_CONFIGS['sambanova']['base_url']}/chat/completions"
        key = self.sambanova_keys[0]

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature if temperature is None else temperature,
        }

        http_session = self.get_http_session(session_id)
        resp = http_session.post(url, headers=headers, json=payload, timeout=(10.0, 30.0))
        if resp.status_code >= 400:
            err_msg = resp.text
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"SambaNova [{resp.status_code}]: {err_msg}")

        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return answer.strip(), target_model, {"provider": "sambanova", "usage": usage}

    def _call_gemini(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi Google Gemini API qua persistent requests.Session()."""
        if not self.gemini_keys:
            raise RuntimeError("Chưa cấu hình GEMINI_API_KEY trong .env")

        candidate_models = [model] if model else self.DEFAULT_CONFIGS["gemini"].get("candidate_models", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"])
        key = self.gemini_keys[0]
        last_error = ""
        http_session = self.get_http_session(session_id)
        eff_temp = self.temperature if temperature is None else temperature

        for target_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": eff_temp},
            }

            try:
                resp = http_session.post(url, headers=headers, json=payload, timeout=(10.0, 30.0))
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    answer = ""
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        answer = "".join(p.get("text", "") for p in parts)
                    usage = data.get("usageMetadata", {})
                    return answer.strip(), target_model, {"provider": "gemini", "usage": usage}

                err_msg = resp.text
                try:
                    err_msg = resp.json().get("error", {}).get("message", resp.text)
                except Exception:
                    pass
                last_error = f"Gemini [{resp.status_code}]: {err_msg}"
                if resp.status_code == 404:
                    continue
                else:
                    break
            except Exception as e:
                last_error = f"Gemini: {e}"

        raise RuntimeError(last_error or "Gemini API không phản hồi")

    def _apply_openrouter_reasoning(self, payload: Dict[str, Any]) -> None:
        """Gắn cấu hình reasoning cho OpenRouter (mặc định tắt để tránh tốn reasoning_tokens)."""
        raw_eff = getattr(self, "reasoning_effort", None)
        if not raw_eff:
            return
        eff_low = str(raw_eff).strip().lower()
        if eff_low in ("none", "off", "false", "disabled"):
            payload["reasoning"] = {"enabled": False}
        elif eff_low in ("minimal", "low", "medium", "high"):
            payload["reasoning"] = {"effort": eff_low}

    def _apply_openrouter_provider_routing(self, payload: Dict[str, Any]) -> None:
        """Gắn cấu hình định tuyến provider (vd: order=['wafer/us']) cho OpenRouter nếu có."""
        routing = getattr(self, "openrouter_provider_routing", None)
        if isinstance(routing, dict) and routing:
            payload["provider"] = dict(routing)

    def _call_openrouter(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi OpenRouter API (OpenAI-compatible, mặc định Tencent Hy4)."""
        if not self.openrouter_keys:
            raise RuntimeError("Chưa cấu hình OPENROUTER_API_KEY trong .env")

        import httpx

        target_model = (
            model
            or getattr(self, "openrouter_model", None)
            or self.DEFAULT_CONFIGS["openrouter"]["model"]
        )
        url = f"{self.DEFAULT_CONFIGS['openrouter']['base_url']}/chat/completions"
        key = self.openrouter_keys[0]

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://uet.vnu.edu.vn",
            "X-Title": "UETChatbot",
        }
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature if temperature is None else temperature,
        }
        self._apply_openrouter_reasoning(payload)
        self._apply_openrouter_provider_routing(payload)

        resp = httpx.post(url, headers=headers, json=payload, timeout=150.0)
        if resp.status_code == 400 and "reasoning" in payload and "reasoning" in resp.text.lower():
            payload.pop("reasoning", None)
            resp = httpx.post(url, headers=headers, json=payload, timeout=150.0)
        if resp.status_code >= 400:
            err_msg = resp.text
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"OpenRouter [{resp.status_code}]: {err_msg}")

        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        return answer.strip(), target_model, {"provider": "openrouter", "usage": usage}

    def _legacy_gemini_snippet(self):
        pass
        """
        key = self.gemini_keys[0]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": self.temperature},
        }

        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        if resp.status_code >= 400:
            err_msg = resp.text
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"Gemini [{resp.status_code}]: {err_msg}")

        data = resp.json()
        candidates = data.get("candidates", [])
        answer = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            answer = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        """

    # -------------------------------------------------------------------------
    # CORE GENERATION (SINGLE & FAILOVER)
    # -------------------------------------------------------------------------

    @staticmethod
    def _invoke_provider(call_fn, prompt_text: str, model: Optional[str], session_id: Optional[str], temperature: Optional[float]):
        """Gọi hàm provider an toàn (hỗ trợ cả mock chỉ nhận prompt_text hoặc model)."""
        try:
            return call_fn(prompt_text, model=model, session_id=session_id, temperature=temperature)
        except TypeError:
            try:
                return call_fn(prompt_text, model=model)
            except TypeError:
                return call_fn(prompt_text)

    def generate(
        self,
        prompt: AugmentedPrompt,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Response:
        """
        Sinh câu trả lời cho một câu hỏi đơn lẻ:
        1. Kiểm tra Disk Cache (nếu có, trả về ngay 0s).
        2. Nếu chưa có, gọi chuỗi Multi-Provider (ưu tiên provider được chọn: custom_api / openrouter / groq / ...),
           duy trì kết nối HTTP Session theo prompt.user_query.session_id.
        3. Lưu kết quả mới vào Disk Cache.
        """
        prompt_text = prompt.formatted_prompt
        session_id = getattr(prompt.user_query, "session_id", None) if prompt.user_query else None

        # 1. Kiểm tra Disk Cache
        cached = self._get_from_disk_cache(prompt_text)
        if cached:
            cached_ans, cached_model, cached_meta = cached
            logger.info(f"⚡ [Disk Cache Hit] Trả về ngay lập tức cho query: {prompt.user_query.query_id[:8]}...")
            return Response(
                query_id=prompt.user_query.query_id,
                answer=cached_ans,
                sources=prompt.contexts,
                model_name=f"{cached_model} (cached)",
                latency_seconds=0.005,
                metadata={**cached_meta, "cache_hit": True},
            )

        # 2. Chuỗi gọi Provider (OpenRouter dùng trực tiếp không fallback về các model miễn phí)
        errors: List[str] = []
        if self.provider == "openrouter":
            providers_to_try = [
                ("openrouter", self._call_openrouter),
            ]
        else:
            providers_to_try = []
            if self.provider in ("custom_api", "api", "custom") and self.custom_api_url:
                providers_to_try.append(("custom_api", self._call_custom_api))
            if self.openrouter_keys and self.provider in ("custom_api", "api", "custom"):
                providers_to_try.append(("openrouter", self._call_openrouter))
            if self.groq_keys:
                providers_to_try.append(("groq", self._call_groq))
            if self.sambanova_keys:
                providers_to_try.append(("sambanova", self._call_sambanova))
            if self.gemini_keys:
                providers_to_try.append(("gemini", self._call_gemini))

            if not providers_to_try:
                providers_to_try = [
                    ("groq", self._call_groq),
                    ("sambanova", self._call_sambanova),
                    ("gemini", self._call_gemini),
                ]

        with Timer() as timer:
            for name, call_fn in providers_to_try:
                try:
                    logger.info(f"Đang gọi mô hình qua {name.upper()} (session={session_id or 'default'})...")
                    answer, used_model, meta = self._invoke_provider(
                        call_fn,
                        prompt_text,
                        model=model,
                        session_id=session_id,
                        temperature=temperature,
                    )
                    if answer:
                        self._save_to_disk_cache(prompt_text, answer, used_model, meta)
                        return Response(
                            query_id=prompt.user_query.query_id,
                            answer=answer,
                            sources=prompt.contexts,
                            model_name=used_model,
                            latency_seconds=round(timer.elapsed, 3),
                            metadata=meta,
                        )
                except Exception as e:
                    err_msg = str(e)
                    errors.append(f"{name}: {err_msg}")
                    logger.warning(f"Provider {name} gặp sự cố: {err_msg}. Chuyển sang bước dự phòng...")

            # 3. Tự bọc lót cấp 2: Nghỉ 2.5s rồi thử lại 1 lần
            if self.provider == "openrouter" and self.openrouter_keys:
                try:
                    logger.info("Đang tự động phục hồi: Thử lại OpenRouter sau 2.5s...")
                    time.sleep(2.5)
                    answer, used_model, meta = self._invoke_provider(
                        self._call_openrouter,
                        prompt_text,
                        model=model,
                        session_id=session_id,
                        temperature=temperature,
                    )
                    if answer:
                        self._save_to_disk_cache(prompt_text, answer, used_model, meta)
                        return Response(
                            query_id=prompt.user_query.query_id,
                            answer=answer,
                            sources=prompt.contexts,
                            model_name=used_model,
                            latency_seconds=round(timer.elapsed, 3),
                            metadata=meta,
                        )
                except Exception as retry_e:
                    errors.append(f"openrouter_retry: {retry_e}")
            elif self.provider != "openrouter" and self.gemini_keys:
                try:
                    logger.info("Đang tự động phục hồi: Thử lại lần cuối với Gemini sau 2.5s...")
                    time.sleep(2.5)
                    answer, used_model, meta = self._invoke_provider(
                        self._call_gemini,
                        prompt_text,
                        model=model,
                        session_id=session_id,
                        temperature=temperature,
                    )
                    if answer:
                        self._save_to_disk_cache(prompt_text, answer, used_model, meta)
                        return Response(
                            query_id=prompt.user_query.query_id,
                            answer=answer,
                            sources=prompt.contexts,
                            model_name=used_model,
                            latency_seconds=round(timer.elapsed, 3),
                            metadata=meta,
                        )
                except Exception as retry_e:
                    errors.append(f"gemini_retry: {retry_e}")

            # 4. Khi tất cả provider đều không khả dụng -> Ném LlmUnavailableError theo D12
            logger.error(f"[UETLLMClient] Tất cả nhà cung cấp LLM đều thất bại: {errors}")
            raise LlmUnavailableError(
                f"Tất cả nhà cung cấp LLM đều thất bại: {errors}",
                errors=errors,
            )

    def generate_json(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Gọi LLM và bóc tách an toàn kết quả JSON (Structured Output).
        Hỗ trợ markdown codeblock (```json ... ```) và raw JSON object.
        """
        try:
            resp = self.generate(prompt, model=model, temperature=self.json_temperature)
            text = (resp.answer or "").strip()
            import re
            json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_block:
                return json.loads(json_block.group(1))

            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                return json.loads(text[first_brace : last_brace + 1])

            return json.loads(text)
        except Exception as e:
            logger.warning(f"[generate_json] Không thể parse JSON từ phản hồi LLM: {e}")
            return None

    # -------------------------------------------------------------------------
    # MULTI-PROVIDER WORKER QUEUE (BATCH EVALUATION)
    # -------------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[AugmentedPrompt],
        max_workers: Optional[int] = None,
    ) -> List[Response]:
        """
        Xử lý đồng thời một danh sách câu hỏi bằng hàng đợi Multi-Provider Queue:
        - 3 Workers chạy song song: Groq (delay 2s), SambaNova (delay 2.5s), Gemini (delay 4s).
        - Câu nào đã có trong Disk Cache được trả về ngay (0s, 0 request).
        - Nếu 1 luồng gặp lỗi 429, câu hỏi tự động được trả lại hàng đợi để luồng khác gánh.
        - Đảm bảo trả về đúng thứ tự gốc của prompts.
        """
        total = len(prompts)
        if total == 0:
            return []

        logger.info(f"🚀 Bắt đầu Multi-Provider Batch Queue trên {total} câu hỏi...")
        results: Dict[int, Response] = {}
        task_queue: Queue = Queue()

        # 1. Quét nhanh Disk Cache trước
        cache_hits = 0
        for idx, p in enumerate(prompts):
            cached = self._get_from_disk_cache(p.formatted_prompt)
            if cached:
                cached_ans, cached_model, cached_meta = cached
                results[idx] = Response(
                    query_id=p.user_query.query_id,
                    answer=cached_ans,
                    sources=p.contexts,
                    model_name=f"{cached_model} (cached)",
                    latency_seconds=0.005,
                    metadata={**cached_meta, "cache_hit": True},
                )
                cache_hits += 1
            else:
                task_queue.put((idx, p))

        logger.info(f"📊 Disk Cache Hit: {cache_hits}/{total} câu. Số câu cần gọi API qua Queue: {task_queue.qsize()}")

        if task_queue.empty():
            return [results[i] for i in range(total)]

        # 2. Cấu hình các Worker
        workers_cfg = []
        if self.provider == "openrouter":
            if self.openrouter_keys:
                workers_cfg.append(("OpenRouter", self._call_openrouter, self.openrouter_delay))
        elif self.provider == "custom_api" and self.custom_api_url:
            workers_cfg.append(("CustomAPI", self._call_custom_api, self.DEFAULT_CONFIGS["custom_api"]["delay"]))
        else:
            if self.groq_keys:
                workers_cfg.append(("Groq", self._call_groq, self.DEFAULT_CONFIGS["groq"]["delay"]))
            if self.sambanova_keys:
                workers_cfg.append(("SambaNova", self._call_sambanova, self.DEFAULT_CONFIGS["sambanova"]["delay"]))
            if self.gemini_keys:
                workers_cfg.append(("Gemini", self._call_gemini, self.DEFAULT_CONFIGS["gemini"]["delay"]))

        if not workers_cfg:
            logger.warning("Không có API provider nào sẵn sàng. Chuyển sang fallback tuần tự.")
            for idx, p in enumerate(prompts):
                if idx not in results:
                    results[idx] = self.generate(p)
            return [results[i] for i in range(total)]

        # 3. Định nghĩa vòng lặp của Worker Thread
        stop_event = threading.Event()

        def worker_loop(worker_name: str, call_fn, delay_sec: float):
            logger.info(f"🟢 Worker [{worker_name}] đã sẵn sàng (Nhịp nghỉ: {delay_sec}s)...")
            while not stop_event.is_set():
                try:
                    item = task_queue.get(timeout=1.0)
                except Empty:
                    if task_queue.empty():
                        break
                    continue

                idx, p = item
                try:
                    with Timer() as t:
                        ans, used_model, meta = call_fn(p.formatted_prompt)
                    self._save_to_disk_cache(p.formatted_prompt, ans, used_model, meta)
                    results[idx] = Response(
                        query_id=p.user_query.query_id,
                        answer=ans,
                        sources=p.contexts,
                        model_name=used_model,
                        latency_seconds=round(t.elapsed, 3),
                        metadata=meta,
                    )
                    logger.info(f"[{worker_name}] ✅ Đã xong câu #{idx + 1}/{total} ({t.elapsed:.2f}s). Nghỉ {delay_sec}s...")
                    task_queue.task_done()
                    # Nghỉ định mức an toàn để không bao giờ bị dính lỗi 429
                    time.sleep(delay_sec)

                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"[{worker_name}] ⚠️ Gặp lỗi tại câu #{idx + 1}: {err_str[:100]}... Đẩy lại vào hàng đợi.")
                    # Fallback: Trả câu hỏi lại hàng đợi để worker khác bốc
                    task_queue.put((idx, p))
                    task_queue.task_done()
                    # Nếu dính 429, tạm dừng luồng này 5 giây rồi mới bốc tiếp
                    time.sleep(5.0)

        # 4. Khởi chạy song song các Worker Threads
        threads: List[threading.Thread] = []
        for w_name, w_fn, w_delay in workers_cfg:
            t = threading.Thread(target=worker_loop, args=(w_name, w_fn, w_delay), daemon=True)
            threads.append(t)
            t.start()

        # Chờ hàng đợi hoàn thành
        task_queue.join()
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        logger.info(f"🎉 Hoàn thành xử lý toàn bộ {total} câu hỏi bằng Multi-Provider Queue!")
        # Sắp xếp đúng theo thứ tự ban đầu
        return [results[i] for i in range(total)]

    # -------------------------------------------------------------------------
    # STREAMING
    # -------------------------------------------------------------------------

    def generate_stream(self, prompt: AugmentedPrompt) -> Generator[str, None, None]:
        """Sinh phản hồi dạng luồng (streaming) qua Custom API, OpenRouter hoặc Groq."""
        session_id = getattr(prompt.user_query, "session_id", None) if prompt.user_query else None
        http_session = self.get_http_session(session_id)

        # 1. Thử streaming qua Custom API (nếu đang chọn custom_api)
        if self.provider in ("custom_api", "api", "custom") and self.custom_api_url:
            try:
                url = f"{self.custom_api_url}/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                if self.custom_api_key:
                    headers["Authorization"] = f"Bearer {self.custom_api_key}"
                payload: Dict[str, Any] = {
                    "model": self.custom_api_model,
                    "messages": [{"role": "user", "content": prompt.formatted_prompt}],
                    "max_tokens": self.custom_api_max_tokens,
                    "temperature": self.temperature,
                    "chat_template_kwargs": {"enable_thinking": self.custom_api_enable_thinking},
                    "stream": True,
                }
                with http_session.post(
                    url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=(self.custom_api_connect_timeout_s, self.custom_api_timeout_s),
                ) as response:
                    if response.status_code == 200:
                        for delta, _ in self._iter_sse_deltas(response):
                            if delta:
                                yield delta
                        return
            except Exception as e:
                logger.warning(f"Streaming qua CustomAPI không thành công: {e}")

        # 2. Thử streaming qua OpenRouter (hỗ trợ tắt reasoning cho hy4-flash)
        if self.openrouter_keys and (self.provider == "openrouter" or not self.groq_keys):
            try:
                import httpx
                url = f"{self.DEFAULT_CONFIGS['openrouter']['base_url']}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.openrouter_keys[0]}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://uet.vnu.edu.vn",
                    "X-Title": "UETChatbot",
                }
                payload = {
                    "model": self.openrouter_model,
                    "messages": [{"role": "user", "content": prompt.formatted_prompt}],
                    "temperature": self.temperature,
                    "stream": True,
                }
                self._apply_openrouter_reasoning(payload)
                self._apply_openrouter_provider_routing(payload)
                with httpx.Client(timeout=150.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code == 200:
                            yielded_any = False
                            for line in response.iter_lines():
                                line = line.strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    return
                                try:
                                    chunk_json = json.loads(data_str)
                                    delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        yielded_any = True
                                        yield delta
                                except Exception:
                                    continue
                            if yielded_any:
                                return
                        else:
                            logger.warning(f"Streaming qua OpenRouter trả về mã HTTP {response.status_code}")
            except Exception as e:
                logger.warning(f"Streaming qua OpenRouter không thành công: {e}")

        # 3. Thử streaming qua Groq khi provider != 'openrouter'
        if self.provider != "openrouter" and self.groq_keys:
            try:
                url = f"{self.DEFAULT_CONFIGS['groq']['base_url']}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.groq_keys[0]}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.DEFAULT_CONFIGS["groq"]["model"],
                    "messages": [{"role": "user", "content": prompt.formatted_prompt}],
                    "temperature": self.temperature,
                    "stream": True,
                }
                with http_session.post(url, headers=headers, json=payload, stream=True, timeout=(10.0, 30.0)) as response:
                    if response.status_code == 200:
                        for delta, _ in self._iter_sse_deltas(response):
                            if delta:
                                yield delta
                        return
            except Exception as e:
                logger.warning(f"Streaming qua Groq không thành công: {e}")

        # Fallback non-streaming qua generate()
        resp = self.generate(prompt)
        yield resp.answer

    def complete_text(
        self,
        prompt_text: str,
        preferred_provider: Optional[str] = "openrouter",
        model: Optional[str] = None,
    ) -> str:
        """Gọi trực tiếp LLM cho các tác vụ trích xuất offline (GraphRAG) có Disk Cache."""
        cache_key = f"[{preferred_provider}:{model or 'default'}]\n{prompt_text}"
        cached = self._get_from_disk_cache(cache_key)
        if cached and cached[0]:
            return cached[0]

        provider_map = {
            "openrouter": lambda txt: self._call_openrouter(txt, model=model),
            "groq": lambda txt: self._call_groq(txt, model=model if preferred_provider == "groq" else None),
            "sambanova": lambda txt: self._call_sambanova(txt, model=model if preferred_provider == "sambanova" else None),
            "gemini": lambda txt: self._call_gemini(txt, model=model if preferred_provider == "gemini" else None),
        }

        order: List[str] = []
        if preferred_provider and preferred_provider in provider_map:
            order.append(preferred_provider)
        for pname in ("openrouter", "groq", "sambanova", "gemini"):
            if pname not in order:
                order.append(pname)

        errors: List[str] = []
        for pname in order:
            try:
                answer, used_model, meta = provider_map[pname](prompt_text)
                if answer:
                    self._save_to_disk_cache(cache_key, answer, used_model, meta)
                    return answer
            except Exception as exc:
                errors.append(f"{pname}: {exc}")

        raise RuntimeError("Không thể gọi LLM complete_text: " + "; ".join(errors))

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def api_key(self) -> str:
        if self.provider == "openrouter" and self.openrouter_keys:
            return self.openrouter_keys[0]
        if self.groq_keys:
            return self.groq_keys[0]
        if self.sambanova_keys:
            return self.sambanova_keys[0]
        if self.gemini_keys:
            return self.gemini_keys[0]
        if self.openrouter_keys:
            return self.openrouter_keys[0]
        return ""


if __name__ == "__main__":
    from src.base import UserQuery, RetrievedContext
    from src.prompts.prompt_templates import UETPromptAugmenter

    print("=" * 65)
    print("🚀 KIỂM THỬ ĐỘC LẬP: UETLLMClient (Multi-Provider Queue + Disk Cache)")
    print("=" * 65)

    client = UETLLMClient()
    sample_query = UserQuery(query_text="Điều kiện xét học bổng khuyến khích học tập loại Giỏi tại UET là gì?")
    sample_ctx = [
        RetrievedContext(
            chunk_id="ctx_01",
            text="Học bổng loại Giỏi: Sinh viên có điểm GPA từ 3.20 đến 3.59 và điểm rèn luyện đạt loại Tốt trở lên.",
            similarity_score=0.95,
            rank=1,
            metadata={"title": "Quy chế học bổng UET"},
        )
    ]

    augmenter = UETPromptAugmenter()
    prompt = augmenter.augment(query=sample_query, contexts=sample_ctx)

    print("\n[1] Thử nghiệm hàm generate() đơn lẻ:")
    res = client.generate(prompt)
    print(f"  • Model phản hồi: {res.model_name}")
    print(f"  • Thời gian: {res.latency_seconds}s")
    print(f"  • Trả lời: {res.answer[:120]}...\n")

    print("[2] Thử nghiệm Disk Cache (gọi lại câu vừa hỏi):")
    res_cached = client.generate(prompt)
    print(f"  • Model phản hồi: {res_cached.model_name}")
    print(f"  • Thời gian: {res_cached.latency_seconds}s (Disk Cache Hit: {res_cached.metadata.get('cache_hit')})")

    print("\n" + "=" * 65)
    print("✅ HOÀN TẤT KIỂM THỬ!")

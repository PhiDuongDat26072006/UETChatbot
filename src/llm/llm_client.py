"""
src/llm/llm_client.py - Phụ trách kết nối và phân phối mô hình ngôn ngữ lớn (LLM).

Core transport, caching, key management, and batch queue helpers live in:
- ``src/llm/cache.py`` (``LLMDiskCache``)
- ``src/llm/keys.py`` (``load_keys_for_provider``, ``mask_key``)
- ``src/llm/providers.py`` (provider calls & SSE streaming)
- ``src/llm/batch.py`` (multi-provider worker queue)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import requests

from src.base import AugmentedPrompt, BaseLLM, Response
from src.llm.batch import WorkerSpec, run_batch_queue
from src.llm.cache import LLMDiskCache
from src.llm.keys import load_keys_for_provider, mask_key
from src.llm.providers import (
    DEFAULT_PROVIDER_CONFIGS,
    apply_openrouter_provider_routing,
    apply_openrouter_reasoning,
    call_custom_api,
    call_gemini,
    call_groq,
    call_openrouter,
    call_sambanova,
    iter_sse_deltas,
    stream_custom_api,
    stream_groq,
    stream_openrouter,
)
from src.utils.helpers import BASE_DIR, Timer, get_logger, load_yaml_config
from src.utils.http_session import HttpSessionManager

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

    DEFAULT_CONFIGS = DEFAULT_PROVIDER_CONFIGS

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
        self.max_tokens = int(llm_cfg.get("max_tokens", 2048))
        self.custom_api_max_tokens = int(custom_cfg.get("max_tokens", self.max_tokens))
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
        self.openrouter_max_tokens = int(openrouter_cfg.get("max_tokens", self.max_tokens))
        raw_routing = openrouter_cfg.get("provider_routing")
        self.openrouter_provider_routing: Optional[Dict[str, Any]] = (
            dict(raw_routing) if isinstance(raw_routing, dict) and raw_routing else None
        )

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

        ttl_val = float(llm_cfg.get("session_ttl_s", session_ttl_s))
        self._session_manager = HttpSessionManager(ttl_s=ttl_val)
        self._session = self._session_manager.default_session

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

        if not self.openrouter_keys:
            for key in self.groq_keys:
                if key.startswith("sk-or-") and key not in self.openrouter_keys:
                    self.openrouter_keys.append(key)

        self.api_keys = self.openrouter_keys if self.provider == "openrouter" else (self.groq_keys or self.openrouter_keys)
        self.current_key_index = 0

        if enable_cache is not None:
            self.enable_cache = bool(enable_cache)
        else:
            self.enable_cache = bool(llm_cfg.get("enable_disk_cache", False))

        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            cfg_cache_file = llm_cfg.get("cache_file")
            self.cache_file = (BASE_DIR / cfg_cache_file) if cfg_cache_file else (BASE_DIR / "database" / "llm_cache.json")

        self._disk_cache = LLMDiskCache(self.cache_file, enabled=self.enable_cache)
        self._cache_lock = self._disk_cache._lock

        active_providers: List[str] = []
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

        if active_providers:
            cache_status = self.cache_file.name if self.enable_cache else "OFF"
            logger.info(
                f"Đã kích hoạt LLM [{self.provider}]: {', '.join(active_providers)} | "
                f"Cache: {cache_status} | SessionTTL: {ttl_val}s"
            )
        else:
            logger.warning("Chưa tìm thấy API Key hoặc Custom API URL nào. LLM sẽ hoạt động ở chế độ fallback.")

    # -------------------------------------------------------------------------
    # HTTP SESSION MANAGEMENT
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
        """Kiểm tra GET /health tới Custom LLM API server."""
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
        except Exception as exc:
            logger.warning(f"[CustomLLM Health] Không thể kết nối tới /health: {exc}")
            return False

    # -------------------------------------------------------------------------
    # DISK CACHE MANAGEMENT
    # -------------------------------------------------------------------------

    def _sync_disk_cache(self) -> LLMDiskCache:
        self._disk_cache.enabled = self.enable_cache
        self._disk_cache.cache_file = Path(self.cache_file)
        return self._disk_cache

    def _init_cache_file(self) -> None:
        self._sync_disk_cache().init_file()

    @staticmethod
    def _compute_prompt_hash(prompt_text: str) -> str:
        return LLMDiskCache.compute_prompt_hash(prompt_text)

    def _get_from_disk_cache(self, prompt_text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        return self._sync_disk_cache().get(prompt_text)

    def _save_to_disk_cache(
        self,
        prompt_text: str,
        answer: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._sync_disk_cache().save(prompt_text, answer, model_name, metadata)

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
        return load_keys_for_provider(prefix, custom_key=custom_key, custom_keys=custom_keys)

    @staticmethod
    def _mask_key(key: str) -> str:
        return mask_key(key)

    # -------------------------------------------------------------------------
    # PROVIDER DELEGATES
    # -------------------------------------------------------------------------

    @staticmethod
    def _iter_sse_deltas(response) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
        yield from iter_sse_deltas(response, decode_bytes=True)

    def _apply_openrouter_reasoning(self, payload: Dict[str, Any]) -> None:
        apply_openrouter_reasoning(payload, getattr(self, "reasoning_effort", None))

    def _apply_openrouter_provider_routing(self, payload: Dict[str, Any]) -> None:
        apply_openrouter_provider_routing(payload, getattr(self, "openrouter_provider_routing", None))

    def _call_custom_api(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        return call_custom_api(
            self.get_http_session(session_id),
            base_url=self.custom_api_url,
            api_key=self.custom_api_key,
            model=model or self.custom_api_model,
            prompt_text=prompt_text,
            max_tokens=max_tokens if max_tokens is not None else self.custom_api_max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            connect_timeout_s=self.custom_api_connect_timeout_s,
            timeout_s=self.custom_api_timeout_s,
            enable_thinking=self.custom_api_enable_thinking,
            use_stream=self.custom_api_stream,
        )

    def _call_groq(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.groq_keys:
            raise RuntimeError("Chưa cấu hình GROQ_API_KEY trong .env")
        groq_cfg = self.DEFAULT_CONFIGS["groq"]
        candidate_models = (
            [model]
            if model
            else list(groq_cfg.get("candidate_models", [groq_cfg["model"]]))
        )
        return call_groq(
            self.get_http_session(session_id),
            api_key=self.groq_keys[0],
            base_url=groq_cfg["base_url"],
            candidate_models=candidate_models,
            prompt_text=prompt_text,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )

    def _call_sambanova(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.sambanova_keys:
            raise RuntimeError("Chưa cấu hình SAMBANOVA_API_KEY trong .env")
        samba_cfg = self.DEFAULT_CONFIGS["sambanova"]
        return call_sambanova(
            self.get_http_session(session_id),
            api_key=self.sambanova_keys[0],
            base_url=samba_cfg["base_url"],
            model=model or samba_cfg["model"],
            prompt_text=prompt_text,
            temperature=self.temperature if temperature is None else temperature,
        )

    def _call_gemini(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.gemini_keys:
            raise RuntimeError("Chưa cấu hình GEMINI_API_KEY trong .env")
        candidate_models = (
            [model]
            if model
            else self.DEFAULT_CONFIGS["gemini"].get("candidate_models", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"])
        )
        return call_gemini(
            self.get_http_session(session_id),
            api_key=self.gemini_keys[0],
            candidate_models=candidate_models,
            prompt_text=prompt_text,
            temperature=self.temperature if temperature is None else temperature,
        )

    def _call_openrouter(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.openrouter_keys:
            raise RuntimeError("Chưa cấu hình OPENROUTER_API_KEY trong .env")
        target_model = (
            model
            or getattr(self, "openrouter_model", None)
            or self.DEFAULT_CONFIGS["openrouter"]["model"]
        )
        return call_openrouter(
            api_key=self.openrouter_keys[0],
            base_url=self.DEFAULT_CONFIGS["openrouter"]["base_url"],
            model=target_model,
            prompt_text=prompt_text,
            max_tokens=max_tokens if max_tokens is not None else self.openrouter_max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            reasoning_effort=getattr(self, "reasoning_effort", None),
            provider_routing=getattr(self, "openrouter_provider_routing", None),
        )

    # -------------------------------------------------------------------------
    # CORE GENERATION (SINGLE & FAILOVER)
    # -------------------------------------------------------------------------

    @staticmethod
    def _invoke_provider(
        call_fn: Callable[..., Tuple[str, str, Dict[str, Any]]],
        prompt_text: str,
        model: Optional[str],
        session_id: Optional[str],
        temperature: Optional[float],
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi hàm provider an toàn (hỗ trợ cả mock chỉ nhận prompt_text hoặc model)."""
        try:
            return call_fn(prompt_text, model=model, session_id=session_id, temperature=temperature)
        except TypeError:
            try:
                return call_fn(prompt_text, model=model)
            except TypeError:
                return call_fn(prompt_text)

    def _build_provider_chain(self) -> List[Tuple[str, Callable[..., Tuple[str, str, Dict[str, Any]]]]]:
        if self.provider == "openrouter":
            return [("openrouter", self._call_openrouter)]

        chain: List[Tuple[str, Callable[..., Tuple[str, str, Dict[str, Any]]]]] = []
        if self.provider in ("custom_api", "api", "custom") and self.custom_api_url:
            chain.append(("custom_api", self._call_custom_api))
        if self.openrouter_keys and self.provider in ("custom_api", "api", "custom"):
            chain.append(("openrouter", self._call_openrouter))
        if self.groq_keys:
            chain.append(("groq", self._call_groq))
        if self.sambanova_keys:
            chain.append(("sambanova", self._call_sambanova))
        if self.gemini_keys:
            chain.append(("gemini", self._call_gemini))

        if not chain:
            return [
                ("groq", self._call_groq),
                ("sambanova", self._call_sambanova),
                ("gemini", self._call_gemini),
            ]
        return chain

    def _record_and_build_response(
        self,
        prompt: AugmentedPrompt,
        answer: str,
        used_model: str,
        meta: Dict[str, Any],
        elapsed: float,
    ) -> Response:
        self._save_to_disk_cache(prompt.formatted_prompt, answer, used_model, meta)
        return Response(
            query_id=prompt.user_query.query_id,
            answer=answer,
            sources=prompt.contexts,
            model_name=used_model,
            latency_seconds=round(elapsed, 3),
            metadata=meta,
        )

    def generate(
        self,
        prompt: AugmentedPrompt,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Response:
        """
        Sinh câu trả lời cho một câu hỏi đơn lẻ (Disk Cache -> Provider Chain -> Retry -> LlmUnavailableError).
        """
        prompt_text = prompt.formatted_prompt
        session_id = getattr(prompt.user_query, "session_id", None) if prompt.user_query else None

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

        errors: List[str] = []
        providers_to_try = self._build_provider_chain()

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
                        return self._record_and_build_response(prompt, answer, used_model, meta, timer.elapsed)
                except Exception as exc:
                    err_msg = str(exc)
                    errors.append(f"{name}: {err_msg}")
                    logger.warning(f"Provider {name} gặp sự cố: {err_msg}. Chuyển sang bước dự phòng...")

            # Tự bọc lót cấp 2: Nghỉ 2.5s rồi thử lại 1 lần
            retry_target: Optional[Tuple[str, Callable[..., Tuple[str, str, Dict[str, Any]]], str]] = None
            if self.provider == "openrouter" and self.openrouter_keys:
                retry_target = ("openrouter_retry", self._call_openrouter, "Đang tự động phục hồi: Thử lại OpenRouter sau 2.5s...")
            elif self.provider != "openrouter" and self.gemini_keys:
                retry_target = ("gemini_retry", self._call_gemini, "Đang tự động phục hồi: Thử lại lần cuối với Gemini sau 2.5s...")

            if retry_target is not None:
                retry_name, retry_fn, retry_log = retry_target
                try:
                    logger.info(retry_log)
                    time.sleep(2.5)
                    answer, used_model, meta = self._invoke_provider(
                        retry_fn,
                        prompt_text,
                        model=model,
                        session_id=session_id,
                        temperature=temperature,
                    )
                    if answer:
                        return self._record_and_build_response(prompt, answer, used_model, meta, timer.elapsed)
                except Exception as retry_exc:
                    errors.append(f"{retry_name}: {retry_exc}")

            logger.error(f"[UETLLMClient] Tất cả nhà cung cấp LLM đều thất bại: {errors}")
            raise LlmUnavailableError(
                f"Tất cả nhà cung cấp LLM đều thất bại: {errors}",
                errors=errors,
            )

    def generate_json(self, prompt: AugmentedPrompt, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Gọi LLM và bóc tách an toàn kết quả JSON (Structured Output)."""
        try:
            resp = self.generate(prompt, model=model, temperature=self.json_temperature)
            text = (resp.answer or "").strip()
            json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_block:
                return json.loads(json_block.group(1))

            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                return json.loads(text[first_brace : last_brace + 1])

            return json.loads(text)
        except Exception as exc:
            logger.warning(f"[generate_json] Không thể parse JSON từ phản hồi LLM: {exc}")
            return None

    # -------------------------------------------------------------------------
    # MULTI-PROVIDER WORKER QUEUE & STREAMING
    # -------------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[AugmentedPrompt],
        max_workers: Optional[int] = None,
    ) -> List[Response]:
        """Xử lý đồng thời một danh sách câu hỏi bằng hàng đợi Multi-Provider Queue."""
        workers_cfg: List[WorkerSpec] = []
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

        return run_batch_queue(
            prompts,
            get_cached=self._get_from_disk_cache,
            save_cached=self._save_to_disk_cache,
            workers_cfg=workers_cfg,
            fallback_generate=self.generate,
        )

    def generate_stream(self, prompt: AugmentedPrompt) -> Generator[str, None, None]:
        """Sinh phản hồi dạng luồng (streaming) qua Custom API, OpenRouter hoặc Groq."""
        session_id = getattr(prompt.user_query, "session_id", None) if prompt.user_query else None
        http_session = self.get_http_session(session_id)

        if self.provider in ("custom_api", "api", "custom") and self.custom_api_url:
            try:
                succeeded = yield from stream_custom_api(
                    http_session,
                    base_url=self.custom_api_url,
                    api_key=self.custom_api_key,
                    model=self.custom_api_model,
                    prompt_text=prompt.formatted_prompt,
                    max_tokens=self.custom_api_max_tokens,
                    temperature=self.temperature,
                    connect_timeout_s=self.custom_api_connect_timeout_s,
                    timeout_s=self.custom_api_timeout_s,
                    enable_thinking=self.custom_api_enable_thinking,
                )
                if succeeded:
                    return
            except Exception as exc:
                logger.warning(f"Streaming qua CustomAPI không thành công: {exc}")

        if self.openrouter_keys and (self.provider == "openrouter" or not self.groq_keys):
            try:
                succeeded = yield from stream_openrouter(
                    api_key=self.openrouter_keys[0],
                    base_url=self.DEFAULT_CONFIGS["openrouter"]["base_url"],
                    model=self.openrouter_model,
                    prompt_text=prompt.formatted_prompt,
                    max_tokens=self.openrouter_max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=getattr(self, "reasoning_effort", None),
                    provider_routing=getattr(self, "openrouter_provider_routing", None),
                )
                if succeeded:
                    return
            except Exception as exc:
                logger.warning(f"Streaming qua OpenRouter không thành công: {exc}")

        if self.provider != "openrouter" and self.groq_keys:
            try:
                succeeded = yield from stream_groq(
                    http_session,
                    api_key=self.groq_keys[0],
                    base_url=self.DEFAULT_CONFIGS["groq"]["base_url"],
                    model=self.DEFAULT_CONFIGS["groq"]["model"],
                    prompt_text=prompt.formatted_prompt,
                    temperature=self.temperature,
                )
                if succeeded:
                    return
            except Exception as exc:
                logger.warning(f"Streaming qua Groq không thành công: {exc}")

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

        provider_map: Dict[str, Callable[[str], Tuple[str, str, Dict[str, Any]]]] = {
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
    base_dir = Path(__file__).resolve().parent.parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from src.base import RetrievedContext, UserQuery
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

    prompt = UETPromptAugmenter().augment(query=sample_query, contexts=sample_ctx)

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

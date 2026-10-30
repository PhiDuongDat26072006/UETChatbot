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

logger = get_logger("llm")


class UETLLMClient(BaseLLM):
    """
    Client LLM tích hợp cơ chế Multi-Provider Queue + Persistent Disk Cache cho UET Chatbot.
    Kế thừa BaseLLM từ src/base.py, tương thích hoàn toàn với RAG Pipeline và hệ thống Evaluation.
    """

    # Cấu hình mặc định cho từng Provider
    DEFAULT_CONFIGS = {
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
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        provider: Optional[str] = None,
        enable_cache: bool = True,
        cache_file: Optional[str | Path] = None,
    ):
        config = load_yaml_config()
        llm_cfg = config.get("llm", {})

        self.provider = provider or llm_cfg.get("provider", "groq")
        self._model_name = model_name or llm_cfg.get("model_name", "llama-3.3-70b-versatile")
        self.candidate_models = list(llm_cfg.get("candidate_models", self.DEFAULT_CONFIGS["groq"]["candidate_models"]))
        self.CANDIDATE_MODELS = self.candidate_models
        self.temperature = float(llm_cfg.get("temperature", 0.2))

        # Nạp API Keys cho cả 3 providers
        self.groq_keys = self._load_keys_for_provider("GROQ", custom_key=api_key, custom_keys=api_keys)
        self.sambanova_keys = self._load_keys_for_provider("SAMBANOVA")
        self.gemini_keys = self._load_keys_for_provider("GEMINI")

        # Khóa tương thích ngược
        self.api_keys = self.groq_keys
        self.current_key_index = 0

        # Cấu hình Disk Cache
        self.enable_cache = enable_cache
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            self.cache_file = _BASE_DIR / "database" / "llm_cache.json"

        self._cache_lock = threading.Lock()
        self._init_cache_file()

        # Thống kê trạng thái
        active_providers = []
        if self.groq_keys:
            active_providers.append(f"Groq ({len(self.groq_keys)} keys)")
        if self.sambanova_keys:
            active_providers.append(f"SambaNova ({len(self.sambanova_keys)} keys)")
        if self.gemini_keys:
            active_providers.append(f"Gemini ({len(self.gemini_keys)} keys)")

        if active_providers:
            logger.info(f"Đã kích hoạt Multi-Provider: {', '.join(active_providers)} | Cache: {self.cache_file.name}")
        else:
            logger.warning("Chưa tìm thấy API Key nào trong .env. LLM sẽ hoạt động ở chế độ fallback trích đoạn.")

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
    # CALL PROVIDERS (GROQ, SAMBANOVA, GEMINI)
    # -------------------------------------------------------------------------

    def _call_groq(self, prompt_text: str, model: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi Groq API."""
        if not self.groq_keys:
            raise RuntimeError("Chưa cấu hình GROQ_API_KEY trong .env")

        import httpx

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
            "temperature": self.temperature,
        }

        resp = httpx.post(url, headers=headers, json=payload, timeout=25.0)
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

    def _call_sambanova(self, prompt_text: str, model: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi SambaNova API (OpenAI-compatible)."""
        if not self.sambanova_keys:
            raise RuntimeError("Chưa cấu hình SAMBANOVA_API_KEY trong .env")

        import httpx

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
            "temperature": self.temperature,
        }

        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
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

    def _call_gemini(self, prompt_text: str, model: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """Gọi Google Gemini API qua REST endpoint chính thức."""
        if not self.gemini_keys:
            raise RuntimeError("Chưa cấu hình GEMINI_API_KEY trong .env")

        import httpx

        candidate_models = [model] if model else self.DEFAULT_CONFIGS["gemini"].get("candidate_models", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"])
        key = self.gemini_keys[0]
        last_error = ""

        for target_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": self.temperature},
            }

            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
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

    def generate(self, prompt: AugmentedPrompt) -> Response:
        """
        Sinh câu trả lời cho một câu hỏi đơn lẻ:
        1. Kiểm tra Disk Cache (nếu có, trả về ngay 0s).
        2. Nếu chưa có, gọi chuỗi Multi-Provider: Groq -> SambaNova -> Gemini.
        3. Lưu kết quả mới vào Disk Cache.
        """
        prompt_text = prompt.formatted_prompt

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

        # 2. Chuỗi gọi Multi-Provider tự bọc lót
        errors: List[str] = []
        providers_to_try = [
            ("groq", self._call_groq),
            ("sambanova", self._call_sambanova),
            ("gemini", self._call_gemini),
        ]

        with Timer() as timer:
            for name, call_fn in providers_to_try:
                try:
                    logger.info(f"Đang gọi mô hình qua {name.upper()}...")
                    answer, used_model, meta = call_fn(prompt_text)
                    if answer:
                        # Lưu vào Disk Cache
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
                    logger.warning(f"Provider {name} gặp sự cố: {err_msg}. Chuyển sang provider dự phòng...")

            # 3. Tự bọc lót cấp 2: Nếu cả 3 đều bận tức thời, nghỉ 2.5s rồi thử lại lần cuối với Gemini
            if self.gemini_keys:
                try:
                    logger.info("Đang tự động phục hồi: Thử lại lần cuối với Gemini sau 2.5s...")
                    time.sleep(2.5)
                    answer, used_model, meta = self._call_gemini(prompt_text)
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

            # 4. Fallback khi tất cả provider đều không khả dụng
            fallback_text = (
                "⚠️ **MÔ HÌNH AI TẠM THỜI TỪ CHỐI DỊCH VỤ**\n\n"
                f"> Hệ thống đã thử qua các nhà cung cấp (Groq, SambaNova, Gemini) nhưng đều gặp giới hạn hạn mức hoặc sự cố mạng.\n"
                f"> Chi tiết: {errors[0] if errors else 'Chưa cấu hình API Key trong file .env'}\n\n"
                "Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau ít phút."
            )
            return Response(
                query_id=prompt.user_query.query_id,
                answer=fallback_text,
                sources=prompt.contexts,
                model_name="multi-provider-fallback",
                latency_seconds=round(timer.elapsed, 3),
                metadata={"errors": errors},
            )

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
        """Sinh phản hồi dạng luồng (streaming) qua Groq hoặc SambaNova."""
        # Thử streaming qua Groq
        if self.groq_keys:
            try:
                import httpx
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
                with httpx.Client(timeout=30.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code == 200:
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
                                        yield delta
                                except Exception:
                                    continue
                            return
            except Exception as e:
                logger.warning(f"Streaming qua Groq không thành công: {e}")

        # Fallback non-streaming qua generate()
        resp = self.generate(prompt)
        yield resp.answer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def api_key(self) -> str:
        if self.groq_keys:
            return self.groq_keys[0]
        if self.sambanova_keys:
            return self.sambanova_keys[0]
        if self.gemini_keys:
            return self.gemini_keys[0]
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

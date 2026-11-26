"""
src/evaluation/llm_judge.py - Bộ giám khảo LLM-as-a-Judge chuẩn hóa cho UET Chatbot.

Đặc điểm thiết kế:
- 1 lời gọi LLM duy nhất / câu hỏi với Structured JSON Output (Pydantic v2 validation).
- Rubric tiếng Việt thang 1-5 (chuẩn hóa về 0.0 - 1.0) cho 4 trục:
  Faithfulness, Answer Correctness, Answer Relevance, Context Sufficiency.
- Luồng đánh giá riêng cho câu hỏi ngoài phạm vi (out_of_domain).
- Không bao giờ ép lỗi API/JSON thành điểm 0.0 (ghi nhận status='error').
- Disk Cache SHA-256 riêng tại database/eval_judge_cache.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, ValidationError

from src.base import (
    BaseEvaluator,
    BaseRAGPipeline,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    Response,
    UserQuery,
)
from src.llm.llm_client import UETLLMClient
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("llm_judge")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RUBRIC_VERSION = "v1.0"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# ============================================================================
# Pydantic Schemas cho Structured JSON Output
# ============================================================================

class FaithfulnessJudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    unsupported_claims: List[str] = Field(default_factory=list)
    reason: str = ""


class CorrectnessJudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    missing_facts: List[str] = Field(default_factory=list)
    wrong_facts: List[str] = Field(default_factory=list)
    reason: str = ""


class SimpleJudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    reason: str = ""


class InDomainJudgeOutput(BaseModel):
    faithfulness: FaithfulnessJudgeScore
    answer_correctness: CorrectnessJudgeScore
    answer_relevance: SimpleJudgeScore
    context_sufficiency: SimpleJudgeScore


class OODJudgeOutput(BaseModel):
    refused_properly: bool
    hallucinated_info: bool
    score: int = Field(..., ge=1, le=5)
    reason: str = ""


def normalize_likert_5(score: int) -> float:
    """Quy đổi điểm Likert 1..5 về thang [0.0, 1.0]: 1->0.0, 2->0.25, 3->0.5, 4->0.75, 5->1.0."""
    clamped = max(1, min(5, int(score)))
    return round((clamped - 1) / 4.0, 4)


def _strip_reasoning_and_extract_json(raw_text: str) -> Dict[str, Any]:
    """Làm sạch khối <think>, code fence Markdown và phân tích JSON đầu ra."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("LLM trả về chuỗi rỗng.")

    text = _THINK_BLOCK_RE.sub("", text).strip()
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Không trích xuất được JSON hợp lệ từ phản hồi: {text[:180]}")


DEFAULT_MAX_CHARS_PER_CONTEXT = 4000
DEFAULT_MAX_TOTAL_CHARS = 24000


def _truncate_contexts(
    contexts: List[str],
    max_chars_per_context: int = DEFAULT_MAX_CHARS_PER_CONTEXT,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Định dạng các ngữ cảnh truy xuất theo ngưỡng độ dài an toàn."""
    if not contexts:
        return "(Không có ngữ cảnh nào được truy xuất)"

    blocks: List[str] = []
    total = 0
    for idx, ctx in enumerate(contexts, 1):
        text = str(ctx or "").strip()
        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + " ...[cắt bớt]"
        block = f"[Tài liệu {idx}]\n{text}"
        if total + len(block) > max_total_chars and blocks:
            blocks.append(f"...(Đã lược bớt {len(contexts) - idx + 1} tài liệu còn lại do giới hạn độ dài)")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _build_indomain_prompt(
    query: str,
    contexts: List[str],
    answer: str,
    ground_truth: Optional[str],
) -> str:
    """Tạo prompt chấm điểm tiếng Việt theo Rubric 1-5 cho câu hỏi in-domain."""
    formatted_contexts = _truncate_contexts(contexts)
    gt_block = ground_truth.strip() if ground_truth else "(Không có Ground Truth tham chiếu)"

    return (
        "Bạn là Giám khảo Đánh giá Chất lượng Hệ thống RAG Học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET).\n"
        "Hãy đánh giá khách quan CÂU TRẢ LỜI CỦA BOT dựa trên CÂU HỎI, NGỮ CẢNH TRUY XUẤT và ĐÁP ÁN CHUẨN (Ground Truth).\n\n"
        "QUY TẮC CHẤM ĐIỂM BẮT BUỘC:\n"
        "- KHÔNG trừ điểm chỉ vì bot trình bày bằng bảng Markdown (`| ... |`), lời chào lịch sự, khối lưu ý hoặc dòng `📌 Căn cứ quy chế:`.\n"
        "- Chỉ xét nội dung sự thật: số liệu (GPA, tín chỉ, điểm chuẩn, năm 2025 vs 2026), mã học phần/mã ngành, số quyết định, tên đơn vị/đối tác, điều kiện phủ định (`không`, `chỉ`, `tối thiểu`, `tối đa`).\n"
        "- Nếu câu trả lời có chi tiết bổ sung hợp lý có thật trong NGỮ CẢNH (ngoài phần tóm tắt của Ground Truth) thì KHÔNG trừ điểm `answer_correctness`, miễn không sai lệch hay mâu thuẫn với Ground Truth.\n"
        "- Nếu bot bịa số liệu, sai năm học, sai đơn vị hoặc kết luận trái ngược ngữ cảnh -> chấm điểm thấp và liệt kê rõ vào `unsupported_claims` / `wrong_facts`.\n\n"
        "THANG ĐIỂM 1 ĐẾN 5 CHO TỪNG TIÊU CHÍ:\n"
        "1. `faithfulness` (Tính trung thực với Ngữ cảnh):\n"
        "   - 5: Mọi khẳng định/số liệu đều có căn cứ trực tiếp trong Ngữ cảnh.\n"
        "   - 4: Phần lớn bám Ngữ cảnh, có diễn đạt mở rộng rất nhỏ không sai lệch.\n"
        "   - 3: Đúng ý chính nhưng có chi tiết thiếu căn cứ rõ trong Ngữ cảnh.\n"
        "   - 2: Có thông tin/số liệu suy diễn hoặc sai lệch so với Ngữ cảnh.\n"
        "   - 1: Bịa đặt nghiêm trọng hoặc mâu thuẫn hoàn toàn Ngữ cảnh.\n"
        "2. `answer_correctness` (Độ chính xác & đầy đủ so với Ground Truth):\n"
        "   - 5: Bao phủ đủ mọi ý/số liệu cốt lõi trong Ground Truth, không sai.\n"
        "   - 4: Đúng hầu hết ý chính, chỉ thiếu chi tiết phụ.\n"
        "   - 3: Đúng một phần (~50%), thiếu ý quan trọng hoặc lẫn tiểu tiết chưa chuẩn.\n"
        "   - 2: Thiếu phần lớn ý chính hoặc sai số liệu/điều kiện quan trọng.\n"
        "   - 1: Sai hoàn toàn hoặc từ chối trả lời sai (false refusal) dù có Ground Truth.\n"
        "3. `answer_relevance` (Độ bám sát Câu hỏi):\n"
        "   - 5: Trả lời trực diện, đúng trọng tâm câu hỏi, dễ theo dõi.\n"
        "   - 3: Có chạm ý câu hỏi nhưng lan man hoặc thiếu trọng tâm.\n"
        "   - 1: Lạc đề hoặc không giải quyết câu hỏi.\n"
        "4. `context_sufficiency` (Ngữ cảnh truy xuất có đủ để trả lời Ground Truth không?):\n"
        "   - 5: Ngữ cảnh chứa đầy đủ 100% dữ kiện cần thiết.\n"
        "   - 3: Ngữ cảnh có một phần thông tin, thiếu một số dữ kiện phụ.\n"
        "   - 1: Ngữ cảnh hoàn toàn thiếu thông tin cần thiết để trả lời.\n\n"
        f"=== CÂU HỎI ===\n{query}\n\n"
        f"=== NGỮ CẢNH TRUY XUẤT ===\n{formatted_contexts}\n\n"
        f"=== ĐÁP ÁN CHUẨN (GROUND TRUTH) ===\n{gt_block}\n\n"
        f"=== CÂU TRẢ LỜI CỦA BOT ===\n{answer}\n\n"
        "CHỈ TRẢ VỀ DUY NHẤT 1 OBJECT JSON HỢP LỆ THEO ĐÚNG CẤU TRÚC SAU (không markdown thừa, không giải thích ngoài JSON):\n"
        '{\n'
        '  "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "..."},\n'
        '  "answer_correctness": {"score": 5, "missing_facts": [], "wrong_facts": [], "reason": "..."},\n'
        '  "answer_relevance": {"score": 5, "reason": "..."},\n'
        '  "context_sufficiency": {"score": 5, "reason": "..."}\n'
        '}'
    )


def _build_ood_prompt(
    query: str,
    answer: str,
    ground_truth: Optional[str],
) -> str:
    """Tạo prompt đánh giá câu hỏi ngoài phạm vi UET (out_of_domain)."""
    gt_block = ground_truth.strip() if ground_truth else "Câu hỏi nằm ngoài phạm vi dữ liệu lưu trữ của UET."
    return (
        "Bạn là Giám khảo Đánh giá An toàn Ngoài Phạm Vi (Out-Of-Domain) cho Chatbot Trường ĐH Công nghệ - ĐHQGHN (UET).\n"
        "Câu hỏi dưới đây thuộc nhóm NGOÀI PHẠM VI UET (hệ thống không lưu trữ dữ liệu về đối tượng được hỏi).\n"
        "Hãy kiểm tra xem bot có từ chối lịch sự đúng nguyên tắc hay cố tình bịa đặt/trả lời nhầm.\n\n"
        "Lưu ý: Dòng `📌 Căn cứ quy chế:` liệt kê tên tài liệu tham khảo là định dạng mặc định cuối câu, KHÔNG tính là bịa nếu nội dung chính đã từ chối rõ ràng và không cung cấp thông tin sai về trường/đối tượng ngoài UET.\n\n"
        "THANG ĐIỂM `score` (1-5):\n"
        "- 5: Từ chối rõ ràng, lịch sự và không bịa bất kỳ thông tin nào về đối tượng ngoài UET.\n"
        "- 3: Có từ chối nhưng câu chữ mơ hồ hoặc đưa thêm chi tiết gây hiểu nhầm.\n"
        "- 1: Không từ chối mà tự bịa câu trả lời hoặc trả lời nhầm bằng thông tin của UET.\n\n"
        f"=== CÂU HỎI (OOD) ===\n{query}\n\n"
        f"=== GHI CHÚ CHUẨN ===\n{gt_block}\n\n"
        f"=== CÂU TRẢ LỜI CỦA BOT ===\n{answer}\n\n"
        "CHỈ TRẢ VỀ DUY NHẤT 1 OBJECT JSON HỢP LỆ:\n"
        '{\n'
        '  "refused_properly": true,\n'
        '  "hallucinated_info": false,\n'
        '  "score": 5,\n'
        '  "reason": "..."\n'
        '}'
    )


# ============================================================================
# UETLLMJudgeEvaluator
# ============================================================================

class UETLLMJudgeEvaluator(BaseEvaluator):
    """
    Bộ đánh giá LLM-as-a-Judge 1-call Structured JSON bằng tiếng Việt.
    Hỗ trợ OpenRouter, Groq, SambaNova, Gemini, Disk Cache SHA-256 và tiêm `call_fn` để test offline.
    """

    PROVIDER_URLS = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "sambanova": "https://api.sambanova.ai/v1/chat/completions",
    }

    DEFAULT_MODELS = {
        "groq": "openai/gpt-oss-120b",
        "openrouter": "deepseek/deepseek-v4.1-flash-1017",
        "sambanova": "DeepSeek-V3.1",
        "gemini": "gemini-2.5-flash",
    }

    class _ModelNotFoundError(RuntimeError):
        """Lỗi model không tồn tại / đã ngừng hoạt động (HTTP 404) để failover ngay."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        delay_seconds: Optional[float] = None,
        enable_cache: Optional[bool] = None,
        cache_file: Optional[Union[str, Path]] = None,
        call_fn: Optional[Callable[[str], str]] = None,
        config: Optional[Dict[str, Any]] = None,
        candidate_models: Optional[List[str]] = None,
        reasoning_effort: Optional[str] = None,
    ):
        raw_cfg = config if config is not None else load_yaml_config()
        judge_cfg = raw_cfg.get("evaluation", {}).get("judge", {}) if isinstance(raw_cfg, dict) else {}

        self.provider = (provider or judge_cfg.get("provider") or "openrouter").lower().strip()
        self.model_name = model_name or judge_cfg.get("model") or self.DEFAULT_MODELS.get(self.provider, "deepseek/deepseek-v4.1-flash-1017")
        self.candidate_models: List[str] = list(
            candidate_models if candidate_models is not None else (judge_cfg.get("candidate_models") or [])
        )
        self.fallback_provider = (judge_cfg.get("fallback_provider") or "groq").lower().strip()
        self.fallback_model = judge_cfg.get("fallback_model") or self.DEFAULT_MODELS.get(self.fallback_provider, "openai/gpt-oss-120b")
        self.base_url = base_url

        self.temperature = float(temperature if temperature is not None else judge_cfg.get("temperature", 0.0))
        self.max_tokens = int(max_tokens if max_tokens is not None else judge_cfg.get("max_tokens", 2048))
        self.reasoning_effort: Optional[str] = (
            reasoning_effort if reasoning_effort is not None else judge_cfg.get("reasoning_effort")
        )
        self.max_retries = int(max_retries if max_retries is not None else judge_cfg.get("max_retries", 2))
        self.delay_seconds = float(delay_seconds if delay_seconds is not None else judge_cfg.get("delay_seconds", 1.0))

        self.enable_cache = bool(enable_cache if enable_cache is not None else judge_cfg.get("enable_cache", True))
        cfg_cache = cache_file or judge_cfg.get("cache_file") or "database/eval_judge_cache.json"
        self.cache_file = Path(cfg_cache) if Path(cfg_cache).is_absolute() else (BASE_DIR / cfg_cache)

        self._call_fn = call_fn
        self._custom_api_key = api_key
        self._cache_lock = threading.Lock()
        self._warned_same_models: Set[Tuple[str, str]] = set()
        self._supports_json_mode = True
        self._dead_targets: Set[Tuple[str, str]] = set()
        self._no_reasoning_models: Set[str] = set()

        self._keys_by_provider: Dict[str, List[str]] = {
            "groq": UETLLMClient._load_keys_for_provider("GROQ", custom_key=api_key if self.provider == "groq" else None),
            "openrouter": UETLLMClient._load_keys_for_provider("OPENROUTER", custom_key=api_key if self.provider == "openrouter" else None),
            "sambanova": UETLLMClient._load_keys_for_provider("SAMBANOVA", custom_key=api_key if self.provider == "sambanova" else None),
            "gemini": UETLLMClient._load_keys_for_provider("GEMINI", custom_key=api_key if self.provider == "gemini" else None),
        }

        if self._call_fn is None and not self._keys_by_provider.get(self.provider):
            # Tự động fallback nếu provider chính thiếu key (chỉ khi người dùng không ép api_key)
            if not api_key and self.fallback_provider and self._keys_by_provider.get(self.fallback_provider):
                logger.warning(
                    "Thiếu API key cho Judge provider '%s'; tự động chuyển sang fallback '%s' (%s).",
                    self.provider,
                    self.fallback_provider,
                    self.fallback_model,
                )
                self.provider = self.fallback_provider
                if model_name is None:
                    self.model_name = self.fallback_model

        if self.enable_cache:
            self._init_cache_file()

    @property
    def has_api_key(self) -> bool:
        """Kiểm tra Judge đã có hàm gọi tùy chỉnh hoặc API key khả dụng chưa."""
        return self._call_fn is not None or bool(self._keys_by_provider.get(self.provider))

    def _init_cache_file(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.cache_file.exists():
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Judge cache {self.cache_file}: {e}")

    def _compute_cache_key(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str],
        is_out_of_domain: bool,
    ) -> str:
        payload = {
            "rubric_version": RUBRIC_VERSION,
            "provider": self.provider,
            "model": self.model_name,
            "is_out_of_domain": is_out_of_domain,
            "query": query,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth or "",
            "max_chars_per_context": DEFAULT_MAX_CHARS_PER_CONTEXT,
            "max_total_chars": DEFAULT_MAX_TOTAL_CHARS,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.enable_cache or not self.cache_file.exists():
            return None
        with self._cache_lock:
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = data.get(cache_key)
                if isinstance(entry, dict) and entry.get("status") == "ok":
                    copied = dict(entry)
                    copied["cache_hit"] = True
                    return copied
            except Exception as e:
                logger.debug(f"Lỗi đọc Judge cache: {e}")
        return None

    def _save_cached(self, cache_key: str, record: Dict[str, Any]) -> None:
        if not self.enable_cache or record.get("status") != "ok":
            return
        with self._cache_lock:
            try:
                data: Dict[str, Any] = {}
                if self.cache_file.exists():
                    try:
                        with open(self.cache_file, "r", encoding="utf-8") as f:
                            loaded = json.load(f)
                            if isinstance(loaded, dict):
                                data = loaded
                    except Exception:
                        data = {}

                to_store = dict(record)
                to_store["cache_hit"] = False
                to_store["cached_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                data[cache_key] = to_store

                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.cache_file.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.cache_file)
            except Exception as e:
                logger.warning(f"Lỗi ghi Judge cache: {e}")

    def _warn_self_preference_if_needed(self, generator_model: Optional[str]) -> None:
        if not generator_model:
            return
        gen_clean = generator_model.replace(" (cached)", "").strip().lower()
        judge_clean = self.model_name.strip().lower()
        pair = (gen_clean, judge_clean)
        if gen_clean and gen_clean == judge_clean and pair not in self._warned_same_models:
            self._warned_same_models.add(pair)
            logger.warning(
                "Self-Preference Bias warning: Judge model '%s' trùng với Generator model '%s'. "
                "Nên cấu hình model Judge khác để đánh giá khách quan hơn.",
                self.model_name,
                generator_model,
            )

    def _build_target_chain(self) -> List[Tuple[str, str]]:
        """Tạo danh sách (provider, model) theo thứ tự ưu tiên, bỏ qua target đã 404."""
        if self._call_fn is not None:
            return [(self.provider, self.model_name)]

        candidates: List[Tuple[str, str]] = []
        if self._keys_by_provider.get(self.provider):
            candidates.append((self.provider, self.model_name))
            for cm in self.candidate_models:
                if cm and (self.provider, cm) not in candidates:
                    candidates.append((self.provider, cm))

        if self.fallback_provider and self._keys_by_provider.get(self.fallback_provider):
            fb_pair = (self.fallback_provider, self.fallback_model)
            if fb_pair not in candidates:
                candidates.append(fb_pair)

        # Thêm các provider còn lại có key nếu chưa nằm trong danh sách
        for extra_p in ("openrouter", "groq", "sambanova", "gemini"):
            if self._keys_by_provider.get(extra_p):
                extra_pair = (extra_p, self.DEFAULT_MODELS[extra_p])
                if extra_pair not in candidates:
                    candidates.append(extra_pair)

        alive = [pair for pair in candidates if pair not in self._dead_targets]
        return alive

    def _call_single_target(self, target_provider: str, target_model: str, prompt_text: str) -> str:
        if self._call_fn is not None:
            return self._call_fn(prompt_text)

        keys = self._keys_by_provider.get(target_provider, [])
        if not keys:
            raise RuntimeError(f"Chưa cấu hình API Key cho Judge provider '{target_provider}'.")

        import httpx

        api_key = keys[0]
        if target_provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                    "responseMimeType": "application/json",
                },
            }
            resp = httpx.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=45.0)
            if resp.status_code == 404:
                raise self._ModelNotFoundError(f"Gemini Judge [404]: {resp.text[:300]}")
            if resp.status_code >= 400:
                raise RuntimeError(f"Gemini Judge [{resp.status_code}]: {resp.text[:300]}")
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini Judge không trả về candidates.")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()

        url = self.base_url if (target_provider == self.provider and self.base_url) else self.PROVIDER_URLS.get(target_provider)
        if not url:
            raise ValueError(f"Provider '{target_provider}' chưa được hỗ trợ.")
        if self.base_url and target_provider == self.provider and not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if target_provider == "openrouter":
            headers["HTTP-Referer"] = "https://uet.vnu.edu.vn"
            headers["X-Title"] = "UETChatbot-EvalJudge"

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._supports_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if (
            self.reasoning_effort
            and "gpt-oss" in target_model.lower()
            and target_model not in self._no_reasoning_models
        ):
            payload["reasoning_effort"] = self.reasoning_effort
        elif (
            target_provider == "openrouter"
            and self.reasoning_effort
            and target_model not in self._no_reasoning_models
        ):
            eff_low = str(self.reasoning_effort).strip().lower()
            if eff_low in ("none", "off", "false", "disabled"):
                payload["reasoning"] = {"enabled": False}
            else:
                payload["reasoning"] = {"effort": self.reasoning_effort}

        resp = httpx.post(url, headers=headers, json=payload, timeout=120.0)
        if resp.status_code == 400 and (
            "response_format" in payload or "reasoning_effort" in payload or "reasoning" in payload
        ):
            err_low = resp.text.lower()
            if "reasoning_effort" in err_low:
                self._no_reasoning_models.add(target_model)
                payload.pop("reasoning_effort", None)
            if "reasoning" in err_low:
                self._no_reasoning_models.add(target_model)
                payload.pop("reasoning", None)
            if "response_format" in err_low or "json_object" in err_low:
                self._supports_json_mode = False
                payload.pop("response_format", None)
            elif "reasoning_effort" not in err_low and "reasoning" not in err_low:
                # Fallback an toàn nếu thông báo 400 không nêu rõ trường
                self._supports_json_mode = False
                payload.pop("response_format", None)
            resp = httpx.post(url, headers=headers, json=payload, timeout=120.0)

        if resp.status_code >= 400:
            err_msg = resp.text
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            if resp.status_code == 404 or "does not exist" in err_msg.lower() or "model_not_found" in err_msg.lower():
                raise self._ModelNotFoundError(f"{target_provider} Judge [{resp.status_code}]: {err_msg}")
            raise RuntimeError(f"{target_provider} Judge [{resp.status_code}]: {err_msg}")

        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    def _call_provider(self, prompt_text: str) -> str:
        """Giữ tương thích ngược cho caller gọi trực tiếp _call_provider."""
        return self._call_single_target(self.provider, self.model_name, prompt_text)

    def judge_record(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
        is_out_of_domain: bool = False,
        sample_id: str = "",
        generator_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Chấm điểm 1 mẫu câu hỏi bằng Structured JSON LLM-as-a-Judge.
        Tự động failover sang candidate_models / fallback_provider khi model trả 404 hoặc 429.
        Thất bại sau max_retries trả về status='error' (KHÔNG BAO GIỜ ép thành 0.0).
        """
        self._warn_self_preference_if_needed(generator_model)
        cache_key = self._compute_cache_key(
            query=query,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
            is_out_of_domain=is_out_of_domain,
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            cached["sample_id"] = sample_id
            return cached

        prompt_text = (
            _build_ood_prompt(query=query, answer=answer, ground_truth=ground_truth)
            if is_out_of_domain
            else _build_indomain_prompt(query=query, contexts=contexts, answer=answer, ground_truth=ground_truth)
        )

        last_err: Optional[Exception] = None
        last_raw = ""
        last_provider = self.provider
        last_model = self.model_name

        for attempt in range(self.max_retries + 1):
            targets = self._build_target_chain()
            if not targets:
                last_err = RuntimeError("Không còn Judge provider/model nào khả dụng (thiếu API key hoặc tất cả đều 404).")
                break

            for target_provider, target_model in targets:
                last_provider = target_provider
                last_model = target_model
                try:
                    raw_response = self._call_single_target(target_provider, target_model, prompt_text)
                    last_raw = raw_response
                    raw_json = _strip_reasoning_and_extract_json(raw_response)

                    if is_out_of_domain:
                        parsed_ood = OODJudgeOutput.model_validate(raw_json)
                        ood_acc = 1.0 if (parsed_ood.refused_properly and not parsed_ood.hallucinated_info) else 0.0
                        record: Dict[str, Any] = {
                            "status": "ok",
                            "sample_id": sample_id,
                            "is_out_of_domain": True,
                            "judge_provider": target_provider,
                            "judge_model": target_model,
                            "rubric_version": RUBRIC_VERSION,
                            "cache_hit": False,
                            "raw_scores": {"ood_score": parsed_ood.score},
                            "normalized_scores": {
                                "ood_refusal_accuracy": ood_acc,
                                "ood_score": normalize_likert_5(parsed_ood.score),
                            },
                            "details": parsed_ood.model_dump(),
                            "error": None,
                        }
                    else:
                        parsed_in = InDomainJudgeOutput.model_validate(raw_json)
                        raw_scores = {
                            "faithfulness": parsed_in.faithfulness.score,
                            "answer_correctness": parsed_in.answer_correctness.score,
                            "answer_relevance": parsed_in.answer_relevance.score,
                            "context_sufficiency": parsed_in.context_sufficiency.score,
                        }
                        norm_scores = {k: normalize_likert_5(v) for k, v in raw_scores.items()}
                        norm_scores["overall_score"] = round(
                            0.30 * norm_scores["faithfulness"]
                            + 0.40 * norm_scores["answer_correctness"]
                            + 0.15 * norm_scores["answer_relevance"]
                            + 0.15 * norm_scores["context_sufficiency"],
                            4,
                        )
                        record = {
                            "status": "ok",
                            "sample_id": sample_id,
                            "is_out_of_domain": False,
                            "judge_provider": target_provider,
                            "judge_model": target_model,
                            "rubric_version": RUBRIC_VERSION,
                            "cache_hit": False,
                            "raw_scores": raw_scores,
                            "normalized_scores": norm_scores,
                            "details": parsed_in.model_dump(),
                            "error": None,
                        }

                    self._save_cached(cache_key, record)
                    return record

                except self._ModelNotFoundError as exc:
                    last_err = exc
                    self._dead_targets.add((target_provider, target_model))
                    logger.warning(
                        "Judge model '%s:%s' trả 404 (ngừng sử dụng) -> chuyển sang target dự phòng: %s",
                        target_provider,
                        target_model,
                        exc,
                    )
                    continue
                except (ValidationError, ValueError, json.JSONDecodeError, Exception) as exc:
                    last_err = exc
                    logger.warning(
                        "Judge thất bại (sample=%s, target=%s:%s, attempt=%d/%d): %s",
                        sample_id or "?",
                        target_provider,
                        target_model,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    continue

            if attempt < self.max_retries:
                backoff = max(0.2, self.delay_seconds) * (attempt + 1)
                time.sleep(backoff)

        return {
            "status": "error",
            "sample_id": sample_id,
            "is_out_of_domain": is_out_of_domain,
            "judge_provider": last_provider,
            "judge_model": last_model,
            "rubric_version": RUBRIC_VERSION,
            "cache_hit": False,
            "raw_scores": {},
            "normalized_scores": {},
            "details": {},
            "error": str(last_err) if last_err else "Unknown judge error",
            "raw_response": last_raw[:500] if last_raw else "",
        }

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        response: Response,
    ) -> EvaluationResult:
        """Triển khai BaseEvaluator.evaluate_sample dùng UETLLMJudgeEvaluator."""
        meta = sample.metadata or {}
        is_ood = (
            meta.get("type") == "out_of_domain"
            or meta.get("unit") == "out_of_domain"
            or "out_of_domain" in (meta.get("query_types") or [])
        )
        contexts = [c.text for c in (response.sources or [])]
        res = self.judge_record(
            query=sample.query,
            contexts=contexts,
            answer=response.answer,
            ground_truth=sample.ground_truth,
            is_out_of_domain=is_ood,
            sample_id=sample.id,
            generator_model=getattr(response, "model_name", None),
        )
        if res["status"] != "ok":
            return EvaluationResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer=response.answer,
                ground_truth=sample.ground_truth,
                retrieved_contexts=response.sources,
                metrics={},
                feedback=f"JUDGE_ERROR: {res.get('error')}",
            )

        return EvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            generated_answer=response.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=response.sources,
            metrics=res["normalized_scores"],
            feedback=json.dumps(res["details"], ensure_ascii=False),
        )

    def evaluate_pipeline(
        self,
        pipeline: BaseRAGPipeline,
        dataset: List[EvaluationSample],
        top_k: int = 5,
    ) -> EvaluationReport:
        """Đánh giá pipeline theo giao diện BaseEvaluator (trung bình chỉ trên mẫu chấm thành công)."""
        results: List[EvaluationResult] = []
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for idx, sample in enumerate(dataset):
            response = pipeline.query(UserQuery(query_text=sample.query), top_k=top_k)
            result = self.evaluate_sample(sample, response)
            results.append(result)

            if result.metrics:
                for k, v in result.metrics.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1

            if idx + 1 < len(dataset) and self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        avg_metrics = {k: round(sums[k] / counts[k], 4) for k in sums if counts.get(k, 0) > 0}
        return EvaluationReport(
            total_samples=len(dataset),
            average_metrics=avg_metrics,
            results=results,
        )

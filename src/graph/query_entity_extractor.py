"""
src/graph/query_entity_extractor.py - Bộ bóc tách thực thể & chủ đề truy vấn bằng LLM (Dual-Level LightRAG).

Chức năng:
  - Phân tách câu hỏi của người dùng thành 2 tầng theo chuẩn LightRAG:
      1. low_level_entities: Thực thể học vụ cụ thể (ngành, học bổng, chứng chỉ, trạng thái học vụ, mã học phần...).
      2. high_level_themes: Chủ đề quy chế / quy trình tổng quát.
  - Tự động dịch khẩu ngữ/tiếng lóng sinh viên sang thuật ngữ học vụ chuẩn UET.
  - Lọc sạch các thực thể chung chung gây nhiễu đồ thị (FORBIDDEN_ENTITIES).
  - Tích hợp In-memory LRU Cache an toàn đa luồng + Timeout nghiêm ngặt + Graceful Fallback.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

import httpx

from src.graph.thesaurus_utils import is_forbidden_entity
from src.llm.llm_client import UETLLMClient
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("graph.query_extractor")

_EXTRACTION_SYSTEM_PROMPT = """Bạn là bộ bóc tách thực thể học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET).
Nhiệm vụ: Phân tích câu hỏi của sinh viên và xuất ra DUY NHẤT 1 đối tượng JSON hợp lệ (không giải thích, không markdown thừa):
{"low_level_entities": ["..."], "high_level_themes": ["..."]}

Quy tắc bắt buộc:
1. low_level_entities (tối đa 5 mục): Các thực thể cụ thể như tên ngành/khoa, mã ngành (CN1, CN8...), học bổng, chứng chỉ ngoại ngữ, chỉ số (GPA, CPA, Điểm rèn luyện), trạng thái/thủ tục học vụ cụ thể.
2. high_level_themes (tối đa 3 mục): Chủ đề quy chế hoặc quy trình đào tạo bao quát của câu hỏi.
3. Chuyển khẩu ngữ/tiếng lóng sinh viên sang thuật ngữ học vụ chuẩn UET:
   - "tạch môn / rớt môn / trượt môn" -> "Học lại"
   - "đuổi học / bị đuổi" -> "Buộc thôi học", "Cảnh báo học vụ"
   - "học 2 ngành / bằng kép / song bằng" -> "Học cùng lúc hai chương trình đào tạo"
   - "điểm sàn" -> "Ngưỡng bảo đảm chất lượng đầu vào"
   - "điểm chuẩn" -> "Điểm trúng tuyển", "Tuyển sinh đại học chính quy"
   - "ra trường" -> "Xét tốt nghiệp", "Chuẩn đầu ra"
4. Giữ nguyên cụm danh từ trọn vẹn (ví dụ: "Ngành Công nghệ thông tin"; TUYỆT ĐỐI KHÔNG cắt rời thành "Thông tin" hay "Công nghệ").
5. KHÔNG đưa các từ chung chung (sinh viên, nhà trường, UET, quy định, văn bản, học kỳ, năm học) vào danh sách."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class QueryEntityExtractor:
    """
    Bộ bóc tách thực thể truy vấn bằng LLM có Cache & Timeout an toàn cho nhánh LightRAG (PA1).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_callable: Optional[Callable[[str], str]] = None,
        timeout: Optional[float] = None,
        cache_size: int = 512,
    ):
        raw_cfg = config or load_yaml_config()
        llm_cfg = raw_cfg.get("llm", {})
        lightrag_cfg = raw_cfg.get("lightrag", {})

        self.provider: str = llm_cfg.get("provider", "openrouter")
        self.model_name: str = llm_cfg.get("model_name", "tencent/hy4-flash")
        self.groq_fallback_model: str = (
            llm_cfg.get("providers", {}).get("groq", {}).get("model")
            or "llama-3.1-8b-instant"
        )
        self.timeout: float = float(
            timeout if timeout is not None else lightrag_cfg.get("llm_extraction_timeout", 3.5)
        )
        self._llm_callable = llm_callable

        self.openrouter_keys = UETLLMClient._load_keys_for_provider("OPENROUTER")
        self.groq_keys = UETLLMClient._load_keys_for_provider("GROQ")

        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_size = max(16, int(cache_size))
        self._lock = threading.Lock()
        self._http_client: Optional[httpx.Client] = None

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.Client(timeout=self.timeout)
            return self._http_client

    @staticmethod
    def _sanitize_list(items: Any, max_items: int = 5) -> List[str]:
        if not isinstance(items, list):
            return []
        cleaned: List[str] = []
        seen_lower: Set[str] = set()
        for raw in items:
            if not isinstance(raw, str):
                continue
            val = re.sub(r"\s+", " ", raw.strip().strip("\"'`•-*")).strip()
            if not val or is_forbidden_entity(val):
                continue
            low = val.lower()
            if low not in seen_lower:
                seen_lower.add(low)
                cleaned.append(val)
                if len(cleaned) >= max_items:
                    break
        return cleaned

    @classmethod
    def _parse_json_response(cls, raw_text: str) -> Optional[Dict[str, List[str]]]:
        """Phân tích phản hồi JSON từ LLM một cách chịu lỗi (chấp nhận cả markdown code fence)."""
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            match = _JSON_BLOCK_RE.search(text)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None

        if not isinstance(parsed, dict):
            return None

        low_level = cls._sanitize_list(
            parsed.get("low_level_entities") or parsed.get("entities") or parsed.get("low_level") or [],
            max_items=6,
        )
        high_level = cls._sanitize_list(
            parsed.get("high_level_themes") or parsed.get("themes") or parsed.get("high_level") or [],
            max_items=4,
        )
        return {
            "low_level_entities": low_level,
            "high_level_themes": high_level,
        }

    def _call_llm_api(self, query_text: str) -> str:
        """Gọi nhanh LLM API với kết nối tái sử dụng và tổng thời gian chờ không vượt quá self.timeout."""
        if self._llm_callable is not None:
            return self._llm_callable(query_text)

        deadline = time.monotonic() + max(0.5, self.timeout)
        client = self._get_http_client()

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Câu hỏi: {query_text}"},
        ]

        # 1. Ưu tiên OpenRouter (nhanh, ổn định với tencent/hy4-flash)
        if self.openrouter_keys:
            rem = deadline - time.monotonic()
            if rem >= 0.3:
                try:
                    resp = client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.openrouter_keys[0]}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "http://localhost:8000",
                            "X-Title": "UET Chatbot Entity Extractor",
                        },
                        json={
                            "model": self.model_name if self.provider == "openrouter" else "tencent/hy4-flash",
                            "messages": messages,
                            "temperature": 0.0,
                            "max_tokens": 200,
                            "reasoning": {"effort": "none"},
                        },
                        timeout=rem,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                except Exception as e:
                    logger.debug(f"[QueryEntityExtractor] OpenRouter lỗi/timeout ({e}), thử fallback...")

        # 2. Fallback sang Groq nếu vẫn còn thời gian trong ngân sách timeout
        rem_groq = deadline - time.monotonic()
        if self.groq_keys and rem_groq >= 0.4:
            try:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_keys[0]}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_fallback_model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 200,
                    },
                    timeout=min(2.5, rem_groq),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.debug(f"[QueryEntityExtractor] Groq fallback lỗi/timeout ({e}).")

        raise RuntimeError("Không thể gọi LLM API để bóc tách thực thể truy vấn.")

    def extract(self, query_text: str) -> Dict[str, Any]:
        """
        Trích xuất thực thể & chủ đề từ câu hỏi.
        Không bao giờ ném ngoại lệ ra ngoài; nếu lỗi sẽ trả về used_llm=False để kích hoạt Fallback PA2.
        """
        clean_q = (query_text or "").strip()
        if not clean_q:
            return {"low_level_entities": [], "high_level_themes": [], "used_llm": False}

        cache_key = re.sub(r"\s+", " ", clean_q.lower())
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return dict(self._cache[cache_key])

        try:
            raw_output = self._call_llm_api(clean_q)
            parsed = self._parse_json_response(raw_output)
            if parsed is not None:
                result = {
                    "low_level_entities": parsed["low_level_entities"],
                    "high_level_themes": parsed["high_level_themes"],
                    "used_llm": True,
                }
                with self._lock:
                    self._cache[cache_key] = result
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                logger.info(
                    f"[QueryEntityExtractor] LLM bóc tách thành công: "
                    f"low={result['low_level_entities']}, high={result['high_level_themes']}"
                )
                return dict(result)
        except Exception as err:
            logger.warning(f"[QueryEntityExtractor] Chuyển sang Fallback PA2 do LLM không khả dụng: {err}")

        return {"low_level_entities": [], "high_level_themes": [], "used_llm": False}

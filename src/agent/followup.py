"""
src/agent/followup.py - Sinh truy vấn tìm kiếm bổ sung cho phần thông tin còn thiếu (followup_query_node, v3).
Đầu vào: câu hỏi gốc, câu trả lời vừa sinh, danh sách used_queries (KHÔNG gửi chunk để tiết kiệm token).
Đầu ra: (missing_info, followup_queries) với 1..max_queries truy vấn không trùng used_queries.
"""

from __future__ import annotations
import json
from typing import List, Optional, Tuple

from src.base import AugmentedPrompt, UserQuery
from src.agent.rewriter import UETQueryRewriter
from src.prompts.prompts import LLM_FOLLOWUP_SYSTEM, LLM_FOLLOWUP_TEMPLATE
from src.utils.helpers import get_logger

logger = get_logger("followup")


class UETFollowupQueryGenerator:
    """
    Bộ sinh truy vấn bổ sung khi answer_eval_node đánh giá câu trả lời còn THIẾU.
    Tái dùng quy tắc chuẩn hóa thuật ngữ quy chế và mở rộng viết tắt từ UETQueryRewriter.
    """

    def __init__(
        self,
        llm_client=None,
        enable_llm: bool = True,
        model: Optional[str] = None,
        max_queries: int = 2,
    ):
        self.llm_client = llm_client
        self.enable_llm = enable_llm
        self.model = model
        self.max_queries = max(1, int(max_queries))

    @staticmethod
    def _normalize_query(q: str) -> str:
        return " ".join(q.strip().lower().split())

    def _filter_novel_queries(
        self,
        candidates: List[str],
        used_queries: List[str],
    ) -> List[str]:
        """Loại bỏ truy vấn rỗng, trùng lặp nội bộ hoặc trùng với used_queries."""
        used_norm = {self._normalize_query(u) for u in used_queries if u and u.strip()}
        seen_norm = set()
        filtered: List[str] = []

        for cand in candidates:
            cleaned = str(cand).strip()
            if not cleaned:
                continue
            norm = self._normalize_query(cleaned)
            if norm in used_norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            filtered.append(cleaned)
            if len(filtered) >= self.max_queries:
                break
        return filtered

    def _heuristic_fallback(
        self,
        query: str,
        used_queries: List[str],
    ) -> Tuple[Optional[str], List[str]]:
        """Fallback tất định đảm bảo luôn trả về ít nhất 1 truy vấn chưa xuất hiện trong used_queries."""
        expanded_q = UETQueryRewriter.expand_acronyms(query.strip())
        used_norm = set()
        for u in used_queries:
            if u and u.strip():
                used_norm.add(self._normalize_query(u))
                used_norm.add(self._normalize_query(UETQueryRewriter.expand_for_retrieval(u)))

        suffixes = [
            "quy định điều kiện tiêu chuẩn chính thức UET",
            "quy chế đào tạo đại học chính quy điều khoản chi tiết",
            "hướng dẫn học vụ thông báo tuyển sinh điểm chuẩn UET",
            "quy định bổ sung chương trình đào tạo Chuẩn chất lượng cao UET",
        ]

        candidates: List[str] = []
        if expanded_q and self._normalize_query(expanded_q) not in used_norm:
            candidates.append(expanded_q)

        for suffix in suffixes:
            cand = f"{expanded_q} {suffix}".strip()
            if self._normalize_query(cand) not in used_norm:
                candidates.append(cand)
                break

        novel = self._filter_novel_queries(candidates, used_queries)
        if not novel:
            round_idx = len(used_queries) + 1
            novel = [f"{expanded_q} quy định chi tiết vòng {round_idx}".strip()]

        return "Chưa đủ thông tin chi tiết để giải đáp trọn vẹn câu hỏi", novel[: self.max_queries]

    def generate(
        self,
        query: str,
        answer: str,
        used_queries: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[str], List[str], str]:
        """
        Xác định phần thông tin còn thiếu trong câu trả lời và sinh 1..max_queries truy vấn tìm kiếm mới.
        Trả về: (missing_info, followup_queries, source) với source in {"llm", "heuristic"}.
        """
        used = list(used_queries or [])

        if self.enable_llm and self.llm_client is not None:
            try:
                used_str = "\n".join(f"- {u}" for u in used) if used else "(Chưa có)"
                prompt_text = LLM_FOLLOWUP_TEMPLATE.format(
                    query=query,
                    answer=answer,
                    used_str=used_str,
                    max_queries=self.max_queries,
                )
                uq = UserQuery(query_text=query, session_id=session_id)
                ap = AugmentedPrompt(
                    user_query=uq,
                    contexts=[],
                    system_instruction=LLM_FOLLOWUP_SYSTEM,
                    formatted_prompt=prompt_text,
                )

                data = None
                if hasattr(self.llm_client, "generate_json"):
                    data = self.llm_client.generate_json(ap, model=self.model)
                else:
                    try:
                        resp = self.llm_client.generate(ap, model=self.model)
                    except TypeError:
                        resp = self.llm_client.generate(ap)
                    text = (resp.answer or "").strip()
                    first_b = text.find("{")
                    last_b = text.rfind("}")
                    if first_b != -1 and last_b > first_b:
                        data = json.loads(text[first_b : last_b + 1])

                if isinstance(data, dict):
                    raw_queries = data.get("followup_queries")
                    if isinstance(raw_queries, str):
                        raw_queries = [raw_queries]
                    missing_raw = data.get("missing_info")
                    missing_info = str(missing_raw).strip() if missing_raw else None

                    if isinstance(raw_queries, list):
                        novel = self._filter_novel_queries(raw_queries, used)
                        if novel:
                            logger.info(
                                f"[FollowupQueryGenerator] LLM sinh {len(novel)} truy vấn mới "
                                f"(missing_info='{missing_info}')"
                            )
                            return missing_info, novel, "llm"
            except Exception as e:
                logger.warning(f"[FollowupQueryGenerator] LLM lỗi ({e}). Chuyển sang heuristic fallback.")

        missing_info, novel = self._heuristic_fallback(query, used)
        logger.info(f"[FollowupQueryGenerator] Heuristic fallback sinh {len(novel)} truy vấn mới: {novel}")
        return missing_info, novel, "heuristic"

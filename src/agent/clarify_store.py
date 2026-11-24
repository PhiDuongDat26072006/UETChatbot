"""
src/agent/clarify_store.py - Quản lý phiên chờ bổ sung thông tin cho nhánh AMBIGUOUS (v3).
Lưu trữ trạng thái trong bộ nhớ kèm thời gian sống (TTL), bổ sung known_info và missing_info.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from src.utils.helpers import get_logger

logger = get_logger("clarify_store")


class ClarifySessionStore:
    """
    Kho lưu trữ phiên chờ bổ sung cho nhánh AMBIGUOUS.
    Chỉ lưu dữ liệu của một câu hỏi đang cần làm rõ, không phải cache hỏi đáp.
    """

    def __init__(self, ttl_s: float = 900.0):
        self.ttl_s = ttl_s
        self._store: Dict[str, Dict[str, Any]] = {}

    def save_session(
        self,
        session_id: str,
        query: str,
        clarify_asks: List[str],
        clarify_supplements: List[str],
        retried3: int = 0,
        known_info: Optional[List[str]] = None,
        missing_info: Optional[List[str]] = None,
    ) -> None:
        """Lưu hoặc cập nhật phiên chờ bổ sung (gồm known_info và missing_info)."""
        self._store[session_id] = {
            "query": query,
            "asks": list(clarify_asks),
            "supplements": list(clarify_supplements),
            "known_info": list(known_info or []),
            "missing_info": list(missing_info or []),
            "retried3": retried3,
            "updated_at": time.time(),
        }
        logger.info(
            f"[ClarifySessionStore] Đã lưu phiên chờ cho session_id='{session_id}' "
            f"(retried3={retried3}, missing={missing_info or []})"
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Lấy phiên chờ nếu còn hiệu lực TTL. Tự động dọn dẹp nếu hết hạn."""
        data = self._store.get(session_id)
        if not data:
            return None

        elapsed = time.time() - data.get("updated_at", 0)
        if elapsed > self.ttl_s:
            logger.info(
                f"[ClarifySessionStore] Phiên '{session_id}' đã hết hạn TTL ({elapsed:.1f}s > {self.ttl_s}s). Xóa bỏ."
            )
            self.clear_session(session_id)
            return None

        return data

    def clear_session(self, session_id: str) -> None:
        """Xóa phiên chờ bổ sung khi đã ĐỦ thông tin, quá số lần, hoặc kết thúc."""
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"[ClarifySessionStore] Đã xóa phiên '{session_id}'.")

    def has_pending_session(self, session_id: Optional[str]) -> bool:
        """Kiểm tra session_id có đang trong trạng thái chờ bổ sung không."""
        if not session_id:
            return False
        return self.get_session(session_id) is not None

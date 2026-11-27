"""
src/utils/http_session.py - Quản lý requests.Session() duy trì kết nối Keep-Alive theo từng phiên trò chuyện.
Giúp tái sử dụng kết nối TCP/TLS với Cloudflare Tunnel cho cả JEV và LLM trong suốt phiên hội thoại.
"""

from __future__ import annotations
import threading
import time
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter


class HttpSessionManager:
    """
    Quản lý pool các requests.Session() theo session_id của cuộc trò chuyện:
    - Cùng một session_id sẽ dùng chung 1 requests.Session() (HTTP Keep-Alive + connection pooling),
      tránh tốn chi phí bắt tay TCP/TLS lại trên mỗi lượt gọi JEV / LLM.
    - Khi session_id là None: trả về default_session dùng chung.
    - Tự động đóng và dọn dẹp các phiên nhàn rỗi quá ttl_s để không rò rỉ socket.
    """

    def __init__(
        self,
        ttl_s: float = 1800.0,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
    ):
        self.ttl_s = max(1.0, float(ttl_s))
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize
        self._lock = threading.Lock()
        self._sessions: Dict[str, Tuple[requests.Session, float]] = {}
        self.default_session: requests.Session = self._create_session()

    def _create_session(self) -> requests.Session:
        sess = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
        )
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        sess.headers.update({"Connection": "keep-alive"})
        return sess

    def _prune_expired_locked(self, now: float) -> None:
        expired_ids = [
            sid
            for sid, (_, last_used) in self._sessions.items()
            if (now - last_used) > self.ttl_s
        ]
        for sid in expired_ids:
            sess, _ = self._sessions.pop(sid)
            try:
                sess.close()
            except Exception:
                pass

    def get_session(self, session_id: Optional[str] = None) -> requests.Session:
        """Lấy requests.Session() ứng với session_id (tạo mới nếu chưa có, tái sử dụng nếu đã tồn tại)."""
        if not session_id:
            return self.default_session

        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            entry = self._sessions.get(session_id)
            if entry is not None:
                sess, _ = entry
                self._sessions[session_id] = (sess, now)
                return sess

            sess = self._create_session()
            self._sessions[session_id] = (sess, now)
            return sess

    def has_session(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        with self._lock:
            return session_id in self._sessions

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def close_session(self, session_id: Optional[str]) -> None:
        """Đóng và giải phóng kết nối HTTP của một phiên trò chuyện cụ thể."""
        if not session_id:
            return
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is not None:
            sess, _ = entry
            try:
                sess.close()
            except Exception:
                pass

    def close_all(self) -> None:
        """Đóng toàn bộ các phiên HTTP đang mở."""
        with self._lock:
            entries = list(self._sessions.values())
            self._sessions.clear()
        for sess, _ in entries:
            try:
                sess.close()
            except Exception:
                pass
        try:
            self.default_session.close()
        except Exception:
            pass

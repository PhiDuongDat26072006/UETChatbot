"""
src/decision/jev_client.py - Triển khai DecisionClient Protocol, JevDecisionClient và FakeDecisionClient (v3).
Sử dụng mô hình System One JEV (strands-decider / typesafe/jev) cho 3 quyết định vi mô:
  1. route(query) -> Route (3 nhãn: OUT_OF_SCOPE, AMBIGUOUS, QUALIFIED_QUESTION)
  2. check_supplement(query, last_ask, supplements, missing_info) -> Verdict
  3. evaluate_answer(query, answer) -> Verdict (chỉ đọc câu hỏi + câu trả lời, không gửi chunk)
Giữ nguyên cơ chế bảo vệ: _call_lock tuần tự, CircuitBreaker, tách connect/read timeout, nhận diện lỗi CUDA (D10).
"""

from __future__ import annotations
import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import urlparse

import requests

from src.decision.jev_enums import Route, Verdict
from src.prompts.prompts import (
    JEV_ROUTE_INSTRUCTIONS,
    JEV_ROUTE_CRITERIA,
    JEV_CHECK_SUPPLEMENT_INSTRUCTIONS,
    JEV_CHECK_SUPPLEMENT_CRITERIA,
    JEV_EVALUATE_ANSWER_INSTRUCTIONS,
    JEV_EVALUATE_ANSWER_CRITERIA,
)
from src.utils.helpers import get_logger, load_yaml_config
from src.utils.http_session import HttpSessionManager

logger = get_logger("jev_client")


# =============================================================================
# CÁC NGOẠI LỆ TÙY BIẾN CHO JEV CLIENT
# =============================================================================

class JevUnavailableError(Exception):
    """Ném ra khi JEV server không khả dụng (circuit breaker mở, hết retry, lỗi 5xx lặp, hoặc server down)."""
    pass


class JevTimeoutError(JevUnavailableError):
    """Ném ra khi gọi JEV bị timeout (connect hoặc read timeout)."""
    pass


# =============================================================================
# CIRCUIT BREAKER CHO JEV CLIENT
# =============================================================================

class CircuitBreaker:
    """
    Bộ ngắt mạch Circuit Breaker:
    - Trạng thái: CLOSED (bình thường), OPEN (ngắt mạch, ném lỗi ngay), HALF_OPEN (thử nghiệm qua /health).
    - Mở sau failure_threshold (mặc định 2) lỗi liên tiếp.
    - Giữ mở open_duration_s (mặc định 20s).
    - Hỗ trợ cờ server_down (lỗi CUDA / 500 lặp) yêu cầu /health OK mới cho đóng lại.
    """

    def __init__(self, failure_threshold: int = 2, open_duration_s: float = 20.0):
        self.failure_threshold = failure_threshold
        self.open_duration_s = open_duration_s
        self.state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self.failure_count: int = 0
        self.opened_at: float = 0.0
        self.is_server_down: bool = False
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"
            self.is_server_down = False

    def record_failure(self, is_server_down: bool = False):
        with self._lock:
            self.failure_count += 1
            if is_server_down:
                self.is_server_down = True
            if self.failure_count >= self.failure_threshold or self.is_server_down:
                self.state = "OPEN"
                self.opened_at = time.time()
                logger.warning(
                    f"[CircuitBreaker] Breaker CHUYỂN SANG OPEN (failures={self.failure_count}, "
                    f"server_down={self.is_server_down}). Mọi request sẽ ném lỗi ngay trong {self.open_duration_s}s."
                )

    def can_execute(self) -> Tuple[bool, str]:
        with self._lock:
            if self.state == "CLOSED":
                return True, "CLOSED"
            if self.state == "OPEN":
                elapsed = time.time() - self.opened_at
                if elapsed >= self.open_duration_s:
                    self.state = "HALF_OPEN"
                    logger.info("[CircuitBreaker] Breaker chuyển sang HALF_OPEN. Đang thử nghiệm phục hồi...")
                    return True, "HALF_OPEN"
                return False, "OPEN"
            return True, "HALF_OPEN"


# =============================================================================
# GIAO THỨC DECISION CLIENT (PROTOCOL)
# =============================================================================

class DecisionClient(Protocol):
    """Giao thức trừu tượng cho các quyết định vi mô (micro-decisions) trong luồng v3."""

    def check_health(self, session_id: Optional[str] = None) -> bool:
        """Kiểm tra sức khỏe JEV Server qua endpoint GET /health."""
        ...

    def route(self, query: str, session_id: Optional[str] = None) -> Route:
        """Phân loại 3 nhánh ý định: OUT_OF_SCOPE, AMBIGUOUS, QUALIFIED_QUESTION."""
        ...

    def check_supplement(
        self,
        query: str,
        last_ask: str,
        supplements: List[str],
        missing_info: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Verdict:
        """Kiểm tra thông tin sinh viên vừa bổ sung đã ĐỦ từng mục cụ thể trong missing_info chưa."""
        ...

    def evaluate_answer(self, query: str, answer: str, session_id: Optional[str] = None) -> Verdict:
        """Đánh giá câu trả lời đã giải đáp đầy đủ mọi ý của câu hỏi hay còn THIẾU (chỉ đọc query + answer)."""
        ...


# Ánh xạ tương thích ngược cho nhãn cũ của JEV Router
_LEGACY_ROUTE_MAP: Dict[str, Route] = {
    "SINGLE_HOP": Route.QUALIFIED_QUESTION,
    "MULTI_HOP": Route.QUALIFIED_QUESTION,
}


# =============================================================================
# JEV DECISION CLIENT TRIỂN KHAI THỰC TẾ
# =============================================================================

class JevDecisionClient:
    """
    Client kết nối tới JEV Decision Model / System One API (v3).
    Bảo vệ hạ tầng:
      - Khóa tuần tự 1 request tại một thời điểm (_call_lock)
      - Kết nối keep-alive qua requests.Session()
      - Timeout tách biệt connect (5s) và read (30s)
      - Không retry khi read timeout; không retry khi gặp lỗi 500 lặp / CUDA
      - Circuit breaker ngắt mạch khi lỗi lặp
      - Bỏ toàn bộ silent fallback fail-open, ném JevUnavailableError / JevTimeoutError
    """

    _call_lock = threading.Lock()

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        connect_timeout_s: float = 5.0,
        read_timeout_s: Optional[float] = None,
        max_retries: int = 1,
        max_concurrency: int = 1,
        health_path: str = "/health",
        breaker_failures: int = 2,
        breaker_open_s: float = 20.0,
        answer_eval_threshold: float = 0.5,
        clarify_threshold: float = 0.5,
        session_ttl_s: float = 1800.0,
    ):
        try:
            full_cfg = load_yaml_config()
            cfg = full_cfg.get("jev", {})
            agent_cfg = full_cfg.get("agent", {})
        except Exception:
            cfg = {}
            agent_cfg = {}

        self.endpoint = (
            endpoint
            or os.getenv("JEV_ENDPOINT")
            or cfg.get("endpoint")
            or "https://openrouter.ai/api/alpha/decisions"
        )
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("JEV_API_KEY", os.getenv("OPENROUTER_API_KEY", cfg.get("api_key", "")))
        )
        self.model = model or cfg.get("model", "strands-decider-12B-gemma4-v1-2610")

        self.connect_timeout_s = float(cfg.get("connect_timeout_s", connect_timeout_s))
        if read_timeout_s is not None:
            self.read_timeout_s = float(read_timeout_s)
        elif timeout_s is not None:
            self.read_timeout_s = float(timeout_s)
        else:
            self.read_timeout_s = float(cfg.get("read_timeout_s", cfg.get("timeout_s", 30.0)))

        self.max_retries = int(cfg.get("max_retries", max_retries))
        self.max_concurrency = int(cfg.get("max_concurrency", max_concurrency))
        self.health_path = str(cfg.get("health_path", health_path))

        breaker_cfg = cfg.get("breaker", {})
        bf = breaker_cfg.get("failures", breaker_failures)
        bo = breaker_cfg.get("open_s", breaker_open_s)
        self.breaker = CircuitBreaker(failure_threshold=int(bf), open_duration_s=float(bo))

        noul_cfg = agent_cfg.get("noul_threshold", {})
        self.answer_eval_threshold = float(noul_cfg.get("answer", answer_eval_threshold))
        self.clarify_threshold = float(noul_cfg.get("clarify", noul_cfg.get("sufficiency", clarify_threshold)))

        # Quản lý HTTP Session (Keep-Alive) duy trì xuyên suốt từng phiên trò chuyện (giống tests/test_tunnel.py)
        ttl_val = float(cfg.get("session_ttl_s", session_ttl_s))
        self._session_manager = HttpSessionManager(ttl_s=ttl_val)
        self._session = self._session_manager.default_session
        self.last_trace_meta: Dict[str, Any] = {}

        logger.info(
            f"[JevDecisionClient] Khởi tạo thành công (v3): endpoint='{self.endpoint}', model='{self.model}', "
            f"connect_timeout={self.connect_timeout_s}s, read_timeout={self.read_timeout_s}s, "
            f"answer_threshold={self.answer_eval_threshold}, session_ttl={ttl_val}s"
        )

    def get_http_session(self, session_id: Optional[str] = None) -> requests.Session:
        """Trả về requests.Session() duy trì kết nối cho phiên trò chuyện tương ứng."""
        if not session_id:
            return self._session
        return self._session_manager.get_session(session_id)

    def close_session(self, session_id: Optional[str]) -> None:
        """Đóng kết nối HTTP Session của một phiên trò chuyện khi kết thúc."""
        self._session_manager.close_session(session_id)

    def close(self) -> None:
        """Đóng toàn bộ kết nối HTTP Session."""
        self._session_manager.close_all()

    def check_health(self, session_id: Optional[str] = None) -> bool:
        """Kiểm tra GET /health tới server."""
        try:
            parsed = urlparse(self.endpoint)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            url = f"{base_url}{self.health_path}"
            headers = {"User-Agent": "UET-Chatbot-HealthCheck"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            http_session = self.get_http_session(session_id)
            resp = http_session.get(url, headers=headers, timeout=(self.connect_timeout_s, 5.0))
            if resp.status_code == 200:
                logger.info(f"[JEV Health] /health OK (status 200) từ {url}")
                return True
            logger.warning(f"[JEV Health] /health trả về status={resp.status_code}")
            return False
        except Exception as e:
            logger.warning(f"[JEV Health] Không thể kết nối tới /health: {e}")
            return False

    def _call_decisions_api(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gọi API Decisions với khóa tuần tự _call_lock, Circuit Breaker, tách connect/read timeout,
        duy trì requests.Session() theo phiên trò chuyện và nhận diện lỗi CUDA (D10).
        Trả về (response_dict, trace_meta).
        """
        can_run, breaker_state = self.breaker.can_execute()
        if not can_run:
            msg = f"JEV Circuit Breaker đang OPEN ({breaker_state}). Hủy request tức thì để bảo vệ hệ thống."
            logger.error(f"[JEV API] {msg}")
            raise JevUnavailableError(msg)

        if breaker_state == "HALF_OPEN":
            is_healthy = self.check_health(session_id=session_id)
            if not is_healthy:
                self.breaker.record_failure(is_server_down=True)
                msg = "JEV Server /health không phản hồi khi Breaker ở trạng thái HALF_OPEN."
                logger.error(f"[JEV API] {msg}")
                raise JevUnavailableError(msg)

        is_openrouter = "openrouter.ai" in self.endpoint
        if is_openrouter and not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        payload: Dict[str, Any] = {
            "state": state,
            "questions": questions,
        }
        if self.model:
            payload["model"] = self.model

        payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://uet.vnu.edu.vn",
            "X-Title": "UET Chatbot Agentic RAG v3",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        http_session = self.get_http_session(session_id)
        last_error = None
        consecutive_5xx = 0

        with self._call_lock:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = http_session.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                        timeout=(self.connect_timeout_s, self.read_timeout_s),
                    )
                    latency_s = round(time.perf_counter() - t0, 3)

                    if resp.status_code == 200:
                        self.breaker.record_success()
                        resp_data = resp.json()
                        server_latency = resp_data.get("latency")
                        trace_meta = {
                            "source": "jev",
                            "latency_s": latency_s,
                            "server_latency_s": server_latency,
                            "payload_bytes": payload_bytes,
                        }
                        self.last_trace_meta = trace_meta
                        return resp_data, trace_meta

                    consecutive_5xx += 1
                    resp_text = resp.text
                    is_cuda = "CUDA" in resp_text or "AcceleratorError" in resp_text or "launch failure" in resp_text

                    if resp.status_code >= 500:
                        logger.error(
                            f"[JEV API] Lỗi Server HTTP {resp.status_code} (attempt {attempt + 1}): {resp_text[:300]}"
                        )
                        if is_cuda or consecutive_5xx >= 2 or attempt >= self.max_retries:
                            self.breaker.record_failure(is_server_down=True)
                            logger.critical(
                                "🚨 [JEV D10] Phát hiện JEV Server hỏng (lỗi CUDA / 500 lặp)! Cần restart server Kaggle."
                            )
                            raise JevUnavailableError(
                                f"JEV Server bị lỗi nghiêm trọng ({resp.status_code}). Cần restart server Kaggle: {resp_text[:200]}"
                            )
                        jitter = random.uniform(0.2, 0.8)
                        time.sleep(jitter)
                        continue

                    resp.raise_for_status()

                except requests.exceptions.ReadTimeout as r_to:
                    latency_s = round(time.perf_counter() - t0, 3)
                    logger.error(f"[JEV API] ReadTimeout ({self.read_timeout_s}s) sau {latency_s}s: {r_to}")
                    self.breaker.record_failure()
                    raise JevTimeoutError(f"JEV read timeout ({self.read_timeout_s}s) - không thử lại.")

                except requests.exceptions.ConnectTimeout as c_to:
                    latency_s = round(time.perf_counter() - t0, 3)
                    logger.warning(f"[JEV API] ConnectTimeout ({self.connect_timeout_s}s) lần {attempt + 1}: {c_to}")
                    last_error = c_to
                    if attempt < self.max_retries:
                        jitter = random.uniform(0.2, 0.8)
                        time.sleep(jitter)
                        continue
                    self.breaker.record_failure()
                    raise JevTimeoutError(f"JEV connect timeout ({self.connect_timeout_s}s) sau retry: {c_to}")

                except requests.exceptions.RequestException as req_err:
                    latency_s = round(time.perf_counter() - t0, 3)
                    logger.warning(f"[JEV API] Lỗi kết nối lần {attempt + 1}: {req_err}")
                    last_error = req_err
                    if attempt < self.max_retries:
                        jitter = random.uniform(0.2, 0.8)
                        time.sleep(jitter)
                        continue
                    self.breaker.record_failure()
                    raise JevUnavailableError(f"JEV connection error sau retry: {req_err}")

            self.breaker.record_failure()
            raise JevUnavailableError(f"JEV API thất bại sau {self.max_retries + 1} lần thử: {last_error}")

    def route(self, query: str, session_id: Optional[str] = None) -> Route:
        """
        Phân loại query thành 1 trong 3 nhãn: OUT_OF_SCOPE, AMBIGUOUS, QUALIFIED_QUESTION.
        - Tuyển sinh, điểm chuẩn, ngành học, học phí, học bổng, quy chế UET là IN-SCOPE (D7).
        - Câu so sánh / nhiều ý nếu đủ thông tin đều thuộc QUALIFIED_QUESTION.
        - Tương thích ngược: ánh xạ SINGLE_HOP / MULTI_HOP (nếu server trả về) -> QUALIFIED_QUESTION.
        """
        state = {"query": query}
        questions = {
            "route": {
                "type": "choice",
                "instructions": JEV_ROUTE_INSTRUCTIONS,
                "criteria": JEV_ROUTE_CRITERIA,
            }
        }
        resp, meta = self._call_decisions_api(state, questions, session_id=session_id)
        ans_obj = resp.get("answers", {}).get("route", {})
        answer_val = ans_obj.get("choice") or ans_obj.get("answer")

        if answer_val in _LEGACY_ROUTE_MAP:
            res_route = _LEGACY_ROUTE_MAP[answer_val]
            logger.info(
                f"[JEV Decision: route] Legacy label '{answer_val}' mapped -> {res_route.value} "
                f"(latency={meta.get('latency_s')}s)"
            )
            return res_route

        if answer_val in Route._value2member_map_:
            res_route = Route(answer_val)
            logger.info(
                f"[JEV Decision: route] Query='{query[:60]}' -> {res_route.value} "
                f"(latency={meta.get('latency_s')}s)"
            )
            return res_route

        logger.warning(
            f"[JevDecisionClient.route] Phản hồi không khớp enum: {answer_val}. Trả về QUALIFIED_QUESTION."
        )
        return Route.QUALIFIED_QUESTION

    def check_supplement(
        self,
        query: str,
        last_ask: str,
        supplements: List[str],
        missing_info: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Verdict:
        """
        Kiểm tra thông tin sinh viên vừa bổ sung đã ĐỦ từng mục cụ thể trong missing_info chưa.
        """
        missing_list = list(missing_info or [])
        state = {
            "query": query,
            "last_ask": last_ask,
            "missing_info": missing_list,
            "supplements": supplements,
        }
        missing_desc = ", ".join(missing_list) if missing_list else "các thông tin còn thiếu được yêu cầu"
        questions = {
            "sufficiency": {
                "type": "noul",
                "instructions": JEV_CHECK_SUPPLEMENT_INSTRUCTIONS.format(missing_desc=missing_desc),
                "criteria": JEV_CHECK_SUPPLEMENT_CRITERIA,
            }
        }
        resp, meta = self._call_decisions_api(state, questions, session_id=session_id)
        ans_obj = resp.get("answers", {}).get("sufficiency", {})
        noul_val = ans_obj.get("noul")
        if noul_val is not None:
            is_sufficient = float(noul_val) >= self.clarify_threshold
        else:
            is_sufficient = bool(ans_obj.get("answer", False))
        verdict = Verdict.SUFFICIENT if is_sufficient else Verdict.INSUFFICIENT
        self.last_trace_meta = {**meta, "noul_prob": noul_val}
        logger.info(
            f"[JEV Decision: check_supplement] Query='{query[:40]}' missing={missing_list} -> "
            f"{verdict.value} (noul_prob={noul_val}, latency={meta.get('latency_s')}s)"
        )
        return verdict

    def evaluate_answer(self, query: str, answer: str, session_id: Optional[str] = None) -> Verdict:
        """
        Đánh giá cặp (câu hỏi, câu trả lời) — KHÔNG gửi chunk ngữ cảnh.
        Kiểm tra câu trả lời đã giải đáp đầy đủ mọi ý của câu hỏi hay còn THIẾU.
        """
        if not answer or not answer.strip():
            return Verdict.INSUFFICIENT

        state = {
            "query": query,
            "answer": answer,
        }
        questions = {
            "answer_complete": {
                "type": "noul",
                "instructions": JEV_EVALUATE_ANSWER_INSTRUCTIONS,
                "criteria": JEV_EVALUATE_ANSWER_CRITERIA,
            }
        }
        resp, meta = self._call_decisions_api(state, questions, session_id=session_id)
        ans_obj = resp.get("answers", {}).get("answer_complete", {})
        noul_val = ans_obj.get("noul")
        if noul_val is not None:
            is_sufficient = float(noul_val) >= self.answer_eval_threshold
        else:
            is_sufficient = bool(ans_obj.get("answer", False))

        verdict = Verdict.SUFFICIENT if is_sufficient else Verdict.INSUFFICIENT
        self.last_trace_meta = {**meta, "noul_prob": noul_val}
        logger.info(
            f"[JEV Decision: evaluate_answer] Query='{query[:40]}' -> {verdict.value} "
            f"(noul_prob={noul_val}, threshold={self.answer_eval_threshold}, latency={meta.get('latency_s')}s)"
        )
        return verdict


# =============================================================================
# FAKE DECISION CLIENT CHO KIỂM THỬ ĐƠN VỊ VÀ OFFLINE
# =============================================================================

class FakeDecisionClient:
    """
    Fake Decision Client phục vụ kiểm thử đơn vị và kịch bản offline (v3).
    Hỗ trợ 3 quyết định: route (3 nhãn), check_supplement (có missing_info), evaluate_answer.
    """

    def __init__(
        self,
        routes: Optional[List[Route]] = None,
        check_supplements: Optional[List[Verdict]] = None,
        evaluate_answers: Optional[List[Verdict]] = None,
        default_route: Route = Route.QUALIFIED_QUESTION,
        default_verdict: Verdict = Verdict.SUFFICIENT,
        health_status: bool = True,
    ):
        self.routes = list(routes) if routes else []
        self.check_supplements = list(check_supplements) if check_supplements else []
        self.evaluate_answers = list(evaluate_answers) if evaluate_answers else []

        self.default_route = default_route
        self.default_verdict = default_verdict
        self.health_status = health_status

        self.history: List[Dict[str, Any]] = []
        self.last_trace_meta: Dict[str, Any] = {}

    def check_health(self, session_id: Optional[str] = None) -> bool:
        self.history.append({"func": "check_health", "session_id": session_id, "result": self.health_status})
        return self.health_status

    def route(self, query: str, session_id: Optional[str] = None) -> Route:
        res = self.routes.pop(0) if self.routes else self.default_route
        self.history.append({"func": "route", "query": query, "session_id": session_id, "result": res})
        return res

    def check_supplement(
        self,
        query: str,
        last_ask: str,
        supplements: List[str],
        missing_info: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Verdict:
        res = self.check_supplements.pop(0) if self.check_supplements else self.default_verdict
        self.history.append({
            "func": "check_supplement",
            "query": query,
            "last_ask": last_ask,
            "supplements": list(supplements),
            "missing_info": list(missing_info) if missing_info is not None else [],
            "session_id": session_id,
            "result": res,
        })
        return res

    def evaluate_answer(self, query: str, answer: str, session_id: Optional[str] = None) -> Verdict:
        res = self.evaluate_answers.pop(0) if self.evaluate_answers else self.default_verdict
        self.history.append({
            "func": "evaluate_answer",
            "query": query,
            "answer": answer,
            "session_id": session_id,
            "result": res,
        })
        return res

    def close_session(self, session_id: Optional[str]) -> None:
        self.history.append({"func": "close_session", "session_id": session_id})

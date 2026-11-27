"""
src/evaluation/agent_trace.py - Trace sanitization, route normalization,
and shared statistical helpers for Agentic RAG evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

VALID_ROUTES: Tuple[str, ...] = ("OUT_OF_SCOPE", "QUALIFIED_QUESTION")

_LEGACY_ROUTE_MAP: Dict[str, str] = {
    "SINGLE_HOP": "QUALIFIED_QUESTION",
    "MULTI_HOP": "QUALIFIED_QUESTION",
    "AMBIGUOUS": "QUALIFIED_QUESTION",
}


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _safe_ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den > 0 else None


def _to_json_safe(value: Any) -> Any:
    """Chuyển đổi giá trị về dạng JSON-serializable an toàn (hỗ trợ Enum, Path, dict/list lồng nhau)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    return str(value)


def sanitize_trace(trace: Any) -> List[Dict[str, Any]]:
    """Làm sạch danh sách trace của LangGraph trước khi ghi snapshot JSON."""
    if not isinstance(trace, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for entry in trace:
        if isinstance(entry, dict):
            cleaned.append(_to_json_safe(entry))
    return cleaned


def normalize_route_label(label: Any) -> Optional[str]:
    """Chuẩn hóa nhãn route (hỗ trợ cả nhãn cũ SINGLE_HOP/MULTI_HOP và Enum)."""
    if label is None:
        return None
    if hasattr(label, "value"):
        label = label.value
    raw = str(label).strip().upper()
    if not raw:
        return None
    mapped = _LEGACY_ROUTE_MAP.get(raw, raw)
    return mapped if mapped in VALID_ROUTES else None


def has_agent_data(record: Dict[str, Any]) -> bool:
    """Kiểm tra bản ghi có dữ liệu thực thi của Agent (tương thích cả snapshot mới và cũ)."""
    agent_block = record.get("agent")
    if isinstance(agent_block, dict) and any(
        agent_block.get(k) is not None for k in ("route", "status", "trace", "retry", "fail_reason")
    ):
        return True
    if record.get("agent_route") is not None or record.get("agent_status") is not None:
        return True
    return False


def get_agent_info(record: Dict[str, Any]) -> Dict[str, Any]:
    """Trích xuất khối thông tin agent thống nhất từ record (ưu tiên `record['agent']`, fallback field phẳng)."""
    agent_block = record.get("agent") if isinstance(record.get("agent"), dict) else {}
    trace = agent_block.get("trace")
    if not isinstance(trace, list):
        trace = []

    raw_route = agent_block.get("route")
    if raw_route is None:
        raw_route = record.get("agent_route")
    if raw_route is None:
        for t in trace:
            if isinstance(t, dict) and t.get("node") == "router_node" and t.get("result") is not None:
                raw_route = t.get("result")
                break

    raw_status = agent_block.get("status")
    if raw_status is None:
        raw_status = record.get("agent_status")

    retry_val = agent_block.get("retry")
    if retry_val is None:
        eval_entries = [
            t for t in trace
            if isinstance(t, dict) and t.get("node") == "answer_eval_node" and not t.get("error")
        ]
        insuff_count = sum(1 for t in eval_entries if str(t.get("result", "")).upper() == "INSUFFICIENT")
        retry_val = insuff_count

    fail_reason = agent_block.get("fail_reason")
    if fail_reason is None:
        for t in reversed(trace):
            if isinstance(t, dict) and t.get("fail_reason"):
                fail_reason = t.get("fail_reason")
                break

    return {
        "route": normalize_route_label(raw_route),
        "raw_route": str(raw_route) if raw_route is not None else None,
        "status": str(raw_status).strip().lower() if raw_status is not None else None,
        "retry": int(retry_val) if retry_val is not None else 0,
        "fail_reason": str(fail_reason) if fail_reason is not None else None,
        "missing_info": agent_block.get("missing_info"),
        "used_queries": list(agent_block.get("used_queries") or []),
        "trace": trace,
    }


def expected_route_of(record: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Xác định nhãn/chính sách route kỳ vọng của 1 mẫu:
    - Nếu có `expected_route` tường minh (ví dụ golden_questions.json): trả `(label, 'explicit')`.
    - Nếu là câu out_of_domain (không có nhãn tường minh): trả `('OOD_FLEX', 'inferred_ood')`
      (chấp nhận cả OUT_OF_SCOPE lẫn QUALIFIED miễn câu trả lời cuối từ chối).
    - Nếu là câu in-domain (không có nhãn tường minh): trả `('IN_DOMAIN', 'inferred_in_domain')`
      (chỉ tính lỗi nếu router chặn nhầm thành OUT_OF_SCOPE).
    """
    explicit = normalize_route_label(record.get("expected_route"))
    if explicit is not None:
        return explicit, "explicit"
    if bool(record.get("is_out_of_domain")):
        return "OOD_FLEX", "inferred_ood"
    return "IN_DOMAIN", "inferred_in_domain"

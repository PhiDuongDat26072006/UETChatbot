"""
src/evaluation/agent_rounds.py - Follow-up loop round extraction and scoring
for Agentic RAG evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.evaluation.agent_trace import (
    _safe_mean,
    _safe_ratio,
    get_agent_info,
    has_agent_data,
)
from src.evaluation.metrics import (
    compute_answer_correctness_offline,
    strip_agent_notes,
)

DEFAULT_OFFLINE_WIN_THRESHOLD: float = 0.05
DEFAULT_JUDGE_WIN_THRESHOLD: float = 1.0


def _extract_target_chunk_set(record: Dict[str, Any]) -> Set[str]:
    """Lấy tập ID chunk vàng (grade >= 1 nếu dạng dict) từ record."""
    raw = record.get("target_chunk_ids")
    if isinstance(raw, dict):
        return {str(k) for k, v in raw.items() if int(v) >= 1}
    if isinstance(raw, list):
        return {str(c) for c in raw if c}
    return set()


def extract_rounds(trace: Any) -> List[Dict[str, Any]]:
    """
    Gom các entry trong `agent.trace` theo từng vòng (round 0, 1, 2, ...).
    Hỗ trợ trace mới có sẵn trường `"round"` và fallback suy luận tuần tự theo `retrieval_node`.
    """
    if not isinstance(trace, list) or not trace:
        return []

    rounds_by_idx: Dict[int, Dict[str, Any]] = {}
    inferred_round = -1

    def _ensure_round(r_idx: int) -> Dict[str, Any]:
        if r_idx not in rounds_by_idx:
            rounds_by_idx[r_idx] = {
                "round": r_idx,
                "queries": [],
                "retrieved_chunk_ids": [],
                "selected_chunk_ids": [],
                "answer_text": None,
                "verdict": None,
                "noul_prob": None,
                "missing_info": None,
                "followup_queries": [],
            }
        return rounds_by_idx[r_idx]

    for entry in trace:
        if not isinstance(entry, dict):
            continue
        node = entry.get("node")
        if node not in ("retrieval_node", "generation_node", "answer_eval_node", "followup_query_node"):
            continue

        raw_r = entry.get("round")
        if raw_r is not None:
            try:
                r_idx = int(raw_r)
            except (TypeError, ValueError):
                r_idx = max(inferred_round, 0)
            inferred_round = max(inferred_round, r_idx)
        else:
            if node == "retrieval_node":
                inferred_round += 1
                r_idx = inferred_round
            elif node == "followup_query_node":
                r_idx = inferred_round + 1
            else:
                r_idx = max(inferred_round, 0)

        slot = _ensure_round(r_idx)

        if node == "retrieval_node":
            if isinstance(entry.get("queries"), list):
                slot["queries"] = [str(q) for q in entry["queries"] if q is not None]
            if isinstance(entry.get("chunk_ids"), list):
                slot["retrieved_chunk_ids"] = [str(cid) for cid in entry["chunk_ids"] if cid is not None]
            if isinstance(entry.get("selected_chunk_ids"), list):
                slot["selected_chunk_ids"] = [str(cid) for cid in entry["selected_chunk_ids"] if cid is not None]
        elif node == "generation_node" and not entry.get("error"):
            if entry.get("answer_text") is not None:
                slot["answer_text"] = str(entry["answer_text"])
        elif node == "answer_eval_node" and not entry.get("error"):
            res = str(entry.get("result") or "").strip().upper()
            if res in ("SUFFICIENT", "INSUFFICIENT"):
                slot["verdict"] = res
            if entry.get("noul_prob") is not None:
                try:
                    slot["noul_prob"] = float(entry["noul_prob"])
                except (TypeError, ValueError):
                    pass
        elif node == "followup_query_node":
            if entry.get("missing_info") is not None:
                slot["missing_info"] = str(entry["missing_info"])
            if isinstance(entry.get("followup_queries"), list):
                slot["followup_queries"] = [str(q) for q in entry["followup_queries"] if q is not None]

    return [rounds_by_idx[k] for k in sorted(rounds_by_idx.keys())]


def _determine_chosen_round(rounds: List[Dict[str, Any]], status: Optional[str]) -> int:
    """Xác định vòng (0-based) tương ứng câu trả lời cuối (ok: vòng cuối; partial: noul_prob cao nhất, hòa lấy sau)."""
    if not rounds:
        return 0
    if status == "partial":
        best_idx = len(rounds) - 1
        best_prob: Optional[float] = None
        for idx, r in enumerate(rounds):
            p = r.get("noul_prob")
            if p is not None:
                pf = float(p)
                if best_prob is None or pf >= best_prob:
                    best_prob = pf
                    best_idx = idx
        return best_idx
    return len(rounds) - 1


def score_sample_rounds(
    record: Dict[str, Any],
    embedder: Optional[Any] = None,
    offline_win_threshold: float = DEFAULT_OFFLINE_WIN_THRESHOLD,
    judge_win_threshold: float = DEFAULT_JUDGE_WIN_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """
    Chấm chi tiết vòng lặp follow-up cho 1 mẫu in-domain có >= 2 vòng.
    Trả về `None` nếu mẫu không có retry hoặc snapshot thiếu dữ liệu theo vòng.
    """
    if not has_agent_data(record) or bool(record.get("is_out_of_domain")):
        return None
    offline = record.get("offline_metrics") or {}
    if offline.get("is_generation_error"):
        return None

    info = get_agent_info(record)
    if info["status"] in ("out_of_scope", "fail"):
        return None

    rounds = extract_rounds(info["trace"])
    if len(rounds) < 2:
        return None

    has_chunk_trace = any(r["retrieved_chunk_ids"] or r["selected_chunk_ids"] for r in rounds)
    has_answer_trace = any(r["answer_text"] is not None for r in rounds)
    if not has_chunk_trace and not has_answer_trace:
        return None

    chosen_idx = _determine_chosen_round(rounds, info["status"])
    target_gold = _extract_target_chunk_set(record)

    # 1 & 2: Gold recall (r0 -> final) và Gold evicted (do giới hạn max_chunks)
    gold_recall_r0: Optional[float] = None
    gold_recall_final: Optional[float] = None
    gold_evicted: Optional[bool] = None

    if target_gold and has_chunk_trace:
        seen_gold: Set[str] = set()
        recalls: List[float] = []
        for r in rounds:
            ret_set = set(r["retrieved_chunk_ids"])
            sel_set = set(r["selected_chunk_ids"] or r["retrieved_chunk_ids"])
            recalls.append(round(len(sel_set & target_gold) / len(target_gold), 4))
            seen_gold.update((ret_set | sel_set) & target_gold)
        gold_recall_r0 = recalls[0]
        gold_recall_final = recalls[chosen_idx]
        chosen_sel = set(rounds[chosen_idx]["selected_chunk_ids"] or rounds[chosen_idx]["retrieved_chunk_ids"])
        gold_evicted = bool(seen_gold - chosen_sel)

    # 3 & 4: Correctness theo vòng, Win/Harm/Tie và chọn best_answer ở nhóm partial
    ground_truth = record.get("ground_truth")
    correctness_by_round: List[Optional[float]] = []
    if ground_truth and has_answer_trace:
        for r in rounds:
            ans_t = r.get("answer_text")
            if ans_t is not None:
                clean_t = strip_agent_notes(ans_t)
                corr_val = compute_answer_correctness_offline(
                    clean_t, ground_truth, embedder=embedder
                ).get("answer_correctness")
                correctness_by_round.append(round(float(corr_val), 4) if corr_val is not None else None)
            else:
                correctness_by_round.append(None)

    score_source: Optional[str] = None
    delta: Optional[float] = None
    outcome: Optional[str] = None

    r0_judge = (record.get("agent_round_judge") or {}).get("r0") or {}
    final_judge = record.get("judge") or {}
    r0_j_corr = (
        (r0_judge.get("raw_scores") or {}).get("answer_correctness")
        if r0_judge.get("status") == "ok"
        else None
    )
    fin_j_corr = (
        (final_judge.get("raw_scores") or {}).get("answer_correctness")
        if final_judge.get("status") == "ok"
        else None
    )

    if r0_j_corr is not None and fin_j_corr is not None:
        score_source = "judge"
        delta = round(float(fin_j_corr) - float(r0_j_corr), 4)
        if delta >= judge_win_threshold:
            outcome = "improved"
        elif delta <= -judge_win_threshold:
            outcome = "harmed"
        else:
            outcome = "tie"
    elif (
        len(correctness_by_round) == len(rounds)
        and correctness_by_round[0] is not None
        and correctness_by_round[chosen_idx] is not None
    ):
        score_source = "offline"
        delta = round(float(correctness_by_round[chosen_idx]) - float(correctness_by_round[0]), 4)
        if delta >= offline_win_threshold:
            outcome = "improved"
        elif delta <= -offline_win_threshold:
            outcome = "harmed"
        else:
            outcome = "tie"

    partial_best_selected: Optional[bool] = None
    if (
        info["status"] == "partial"
        and correctness_by_round
        and all(v is not None for v in correctness_by_round)
    ):
        float_scores = [float(v) for v in correctness_by_round]  # type: ignore[arg-type]
        partial_best_selected = float_scores[chosen_idx] >= max(float_scores)

    out: Dict[str, Any] = {
        "rounds": len(rounds),
        "status": info["status"],
        "score_source": score_source,
        "correctness_by_round": correctness_by_round if any(v is not None for v in correctness_by_round) else [],
        "chosen_round": chosen_idx,
        "delta": delta,
        "outcome": outcome,
        "gold_recall_round0": gold_recall_r0,
        "gold_recall_final": gold_recall_final,
        "gold_evicted": gold_evicted,
    }
    if partial_best_selected is not None:
        out["partial_best_selected"] = partial_best_selected
    return out


def _get_or_compute_rounds_eval(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = record.get("agent_rounds_eval")
    if isinstance(existing, dict):
        return existing
    return score_sample_rounds(record, embedder=None)


def compute_loop_metrics(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Tổng hợp 4 metric cốt lõi của vòng lặp follow-up vào `agent_metrics.loop`:
    1. `gold_recall_round0` -> `gold_recall_final`
    2. `gold_evicted_rate`
    3. `win_rate`, `harm_rate`, `avg_delta`
    4. `correctness_by_round` (r0, r1, ...) & `partial_best_answer_accuracy`
    Trả về `None` nếu snapshot không có mẫu retry có dữ liệu theo vòng.
    """
    evals: List[Dict[str, Any]] = []
    for rec in records:
        reval = _get_or_compute_rounds_eval(rec)
        if reval is not None:
            evals.append(reval)

    if not evals:
        return None

    # Metric 1 & 2: Retrieval vòng lặp
    gold_items = [e for e in evals if e.get("gold_recall_round0") is not None]
    r0_recs = [float(e["gold_recall_round0"]) for e in gold_items]
    fin_recs = [float(e["gold_recall_final"]) for e in gold_items if e.get("gold_recall_final") is not None]
    evicted_cnt = sum(1 for e in gold_items if e.get("gold_evicted") is True)

    # Metric 3: Win / Harm / Avg delta
    scored = [e for e in evals if e.get("outcome") is not None]
    win_cnt = sum(1 for e in scored if e["outcome"] == "improved")
    harm_cnt = sum(1 for e in scored if e["outcome"] == "harmed")
    tie_cnt = sum(1 for e in scored if e["outcome"] == "tie")
    deltas = [float(e["delta"]) for e in scored if e.get("delta") is not None]

    src_counts: Dict[str, int] = {}
    for e in scored:
        s = str(e.get("score_source") or "offline")
        src_counts[s] = src_counts.get(s, 0) + 1
    if not src_counts:
        dominant_source = None
    elif len(src_counts) == 1:
        dominant_source = next(iter(src_counts.keys()))
    else:
        dominant_source = "mixed"

    # Metric 4: Correctness theo vòng + chọn best_answer ở nhóm partial
    round_buckets: Dict[int, List[float]] = {}
    for e in evals:
        for r_idx, val in enumerate(e.get("correctness_by_round") or []):
            if val is not None:
                round_buckets.setdefault(r_idx, []).append(float(val))
    correctness_curve = {
        f"r{r_idx}": _safe_mean(vals)
        for r_idx, vals in sorted(round_buckets.items())
        if vals
    }

    partial_items = [e for e in evals if e.get("partial_best_selected") is not None]
    partial_ok = sum(1 for e in partial_items if e["partial_best_selected"] is True)

    return {
        "samples_with_retry": len(evals),
        "samples_with_gold": len(gold_items),
        "gold_recall_round0": _safe_mean(r0_recs),
        "gold_recall_final": _safe_mean(fin_recs),
        "gold_evicted_count": evicted_cnt,
        "gold_evicted_rate": _safe_ratio(evicted_cnt, len(gold_items)),
        "evaluated_samples": len(scored),
        "score_source": dominant_source,
        "win_count": win_cnt,
        "harm_count": harm_cnt,
        "tie_count": tie_cnt,
        "win_rate": _safe_ratio(win_cnt, len(scored)),
        "harm_rate": _safe_ratio(harm_cnt, len(scored)),
        "avg_delta": _safe_mean(deltas),
        "correctness_by_round": correctness_curve,
        "partial_samples": len(partial_items),
        "partial_best_answer_accuracy": _safe_ratio(partial_ok, len(partial_items)),
    }

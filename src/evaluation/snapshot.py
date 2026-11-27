"""
src/evaluation/snapshot.py - Two-phase E2E evaluation:
Phase 1 (`generate_predictions`) and Phase 2 (`score_predictions`) plus context/snapshot helpers.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.base import (
    BaseRAGPipeline,
    EvaluationSample,
    Response,
    RetrievedContext,
    UserQuery,
)
from src.embeddings.embedder import UETEmbedder
from src.evaluation.agent_eval import (
    compute_sample_agent_flags,
    extract_rounds,
    get_agent_info,
    has_agent_data,
    sanitize_trace,
    score_sample_rounds,
)
from src.evaluation.diagnostics import (
    build_e2e_report,
    classify_sample_error,
)
from src.evaluation.llm_judge import UETLLMJudgeEvaluator
from src.evaluation.metrics import (
    compute_answer_correctness_offline,
    compute_answer_relevance,
    compute_context_relevance,
    compute_faithfulness,
    compute_numeric_grounding,
    deduplicate_preserve_order,
    hit_rate_at_k,
    is_generation_error_response,
    is_refusal_response,
    mrr,
    ndcg_at_k,
    recall_at_k,
    strip_agent_notes,
)
from src.evaluation.ragas_eval import _atomic_write_json
from src.utils.helpers import get_logger

logger = get_logger("evaluation")


def _is_sample_ood(meta: Dict[str, Any]) -> bool:
    return (
        str(meta.get("type", "")).lower() == "out_of_domain"
        or str(meta.get("unit", "")).lower() == "out_of_domain"
        or "out_of_domain" in [str(x).lower() for x in (meta.get("query_types") or [])]
    )


def _serialize_contexts(sources: List[RetrievedContext]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for idx, c in enumerate(sources or [], 1):
        meta = c.metadata if isinstance(getattr(c, "metadata", None), dict) else {}
        entry: Dict[str, Any] = {
            "chunk_id": getattr(c, "chunk_id", None),
            "document_id": meta.get("document_id") or meta.get("doc_id"),
            "rank": getattr(c, "rank", idx),
            "score": round(float(c.similarity_score), 6) if getattr(c, "similarity_score", None) is not None else None,
            "search_source": meta.get("search_source"),
            "title": meta.get("title"),
            "text": getattr(c, "text", ""),
        }
        if meta.get("page") is not None:
            entry["page"] = meta.get("page")
        graph_triples = meta.get("graph_triples") or []
        if graph_triples:
            entry["graph_triples"] = [str(t) for t in graph_triples[:5]]
        lightrag_overview = meta.get("lightrag_overview")
        if isinstance(lightrag_overview, str) and lightrag_overview.strip():
            entry["lightrag_overview"] = lightrag_overview.strip()
        serialized.append(entry)
    return serialized


def _format_context_dict_for_eval(c: Dict[str, Any]) -> str:
    """Bổ sung thông tin Trang, Bản đồ LightRAG 2 tầng và Quan hệ Đồ thị để Offline/Judge nhìn thấy đủ ngữ cảnh như Generator."""
    prefixes: List[str] = []
    overview = c.get("lightrag_overview")
    if isinstance(overview, str) and overview.strip():
        prefixes.append(f"[BẢN ĐỒ LIÊN KẾT QUY CHẾ (LIGHTRAG 2-TẦNG)]\n{overview.strip()}")
    if c.get("page") is not None:
        prefixes.append(f"[Trang {c['page']}]")
    triples = c.get("graph_triples") or []
    if triples:
        prefixes.append(f"[Quan hệ Đồ thị Tri thức: {'; '.join(str(t) for t in triples[:5])}]")
    if not prefixes:
        return str(c.get("text", ""))
    prefix_block = "\n".join(prefixes)
    return f"{prefix_block}\n{str(c.get('text', '') or '').strip()}"


def _recover_generation_error(
    pipeline: BaseRAGPipeline,
    aug_prompt: Any,
    resp: Response,
) -> Tuple[Response, float]:
    """
    Khi UETLLMClient trả về fallback 429/lỗi dịch vụ:
    1. Thử ngay OpenRouter nếu có OPENROUTER_API_KEY (không tốn thời gian chờ Groq).
    2. Nếu không, đọc thời gian chờ từ thông báo lỗi Groq (tối đa 25s) và retry 1 lần.
    """
    llm = getattr(pipeline, "llm", None)
    if llm is None:
        return resp, 0.0

    if getattr(llm, "openrouter_keys", None) and hasattr(llm, "_call_openrouter"):
        try:
            t_or = time.perf_counter()
            or_ans, or_model, or_meta = llm._call_openrouter(aug_prompt.formatted_prompt)
            elapsed = round(time.perf_counter() - t_or, 3)
            if or_ans and not is_generation_error_response(or_ans, or_model):
                if hasattr(llm, "_save_to_disk_cache"):
                    llm._save_to_disk_cache(aug_prompt.formatted_prompt, or_ans, or_model, or_meta)
                return (
                    Response(
                        query_id=aug_prompt.user_query.query_id,
                        answer=or_ans,
                        sources=aug_prompt.contexts,
                        model_name=or_model,
                        latency_seconds=elapsed,
                        metadata=or_meta,
                    ),
                    elapsed,
                )
        except Exception as or_exc:
            logger.warning(f"OpenRouter failover khi generate thất bại: {or_exc}")

    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", str(resp.answer or ""), re.IGNORECASE)
    wait_s = min(25.0, float(match.group(1)) + 1.0) if match else 6.0
    logger.info(f"Đợi {wait_s:.1f}s rồi thử lại generate...")
    time.sleep(wait_s)
    t_retry = time.perf_counter()
    retry_resp = llm.generate(aug_prompt)
    return retry_resp, round(time.perf_counter() - t_retry, 3)


def generate_predictions(
    pipeline: BaseRAGPipeline,
    dataset: List[EvaluationSample],
    top_k: int = 10,
    num_samples: Optional[int] = None,
    delay_between_queries: float = 1.0,
    output_path: Optional[Union[str, Path]] = None,
    use_batch_llm: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Pha 1 (Inference / Snapshot): Chạy Retrieval + Generation trên tập benchmark,
    đo riêng retrieval_latency_s và llm_latency_s, lưu snapshot đầy đủ contexts để Pha 2 chấm lại 0s.
    Hỗ trợ `resume=True` để bỏ qua câu đã sinh hợp lệ và chỉ chạy bù câu thiếu/lỗi 429.
    """
    active_dataset = dataset[:num_samples] if (num_samples and num_samples > 0) else dataset
    total = len(active_dataset)
    has_granular_parts = (
        not hasattr(pipeline, "agent_graph")
        and hasattr(pipeline, "retriever")
        and hasattr(pipeline, "prompt_augmenter")
        and hasattr(pipeline, "llm")
        and getattr(pipeline, "retriever", None) is not None
        and getattr(pipeline, "prompt_augmenter", None) is not None
        and getattr(pipeline, "llm", None) is not None
    )

    existing_valid: Dict[str, Dict[str, Any]] = {}
    if resume and output_path and Path(output_path).exists():
        try:
            prev = json.loads(Path(output_path).read_text(encoding="utf-8"))
            for item in prev.get("predictions") or []:
                sid = str(item.get("sample_id") or "")
                ans = str(item.get("generated_answer") or "")
                if (
                    sid
                    and ans.strip()
                    and not item.get("error")
                    and str(item.get("agent_status") or "").lower() != "fail"
                    and not is_generation_error_response(ans, item.get("model_name"))
                ):
                    existing_valid[sid] = item
            if existing_valid:
                logger.info(f"[Resume] Tái sử dụng {len(existing_valid)}/{total} mẫu hợp lệ từ {output_path}.")
        except Exception as exc:
            logger.warning(f"Không thể đọc snapshot resume '{output_path}': {exc}")

    records: List[Dict[str, Any]] = []
    print(f"\n🚀 [Pha 1 - Snapshot] Đang sinh câu trả lời cho {total} câu hỏi (Top-K={top_k}, Resume={resume})...")

    agent_graph = getattr(pipeline, "agent_graph", None)
    prev_trace_verbose: Optional[bool] = None
    if agent_graph is not None and hasattr(agent_graph, "trace_verbose"):
        prev_trace_verbose = bool(agent_graph.trace_verbose)
        agent_graph.trace_verbose = True

    try:
        if has_granular_parts and use_batch_llm and hasattr(pipeline.llm, "generate_batch"):  # type: ignore[attr-defined]
            staged: List[Tuple[int, EvaluationSample, List[RetrievedContext], Any, float, Optional[str]]] = []
            prompts = []
            records_by_idx: Dict[int, Dict[str, Any]] = {}
            for idx, sample in enumerate(active_dataset):
                if sample.id in existing_valid:
                    records_by_idx[idx] = existing_valid[sample.id]
                    continue
                uq = UserQuery(query_text=sample.query)
                t0 = time.perf_counter()
                err_msg: Optional[str] = None
                contexts: List[RetrievedContext] = []
                try:
                    contexts = pipeline.retriever.retrieve(query=uq, top_k=top_k, filters=uq.filters)  # type: ignore[attr-defined]
                except Exception as exc:
                    err_msg = f"Retrieval error: {exc}"
                    logger.warning(f"[{sample.id}] {err_msg}")
                ret_lat = round(time.perf_counter() - t0, 3)
                aug = pipeline.prompt_augmenter.augment(query=uq, contexts=contexts)  # type: ignore[attr-defined]
                staged.append((idx, sample, contexts, aug, ret_lat, err_msg))
                prompts.append(aug)
                if (idx + 1) % 10 == 0 or (idx + 1) == total:
                    print(f"   • Đã retrieve {idx + 1}/{total} câu hỏi...")

            batch_responses = pipeline.llm.generate_batch(prompts) if prompts else []  # type: ignore[attr-defined]
            for b_i, (orig_idx, sample, contexts, aug, ret_lat, err_msg) in enumerate(staged):
                resp = batch_responses[b_i] if b_i < len(batch_responses) else Response(answer="", sources=contexts)
                llm_lat = round(float(getattr(resp, "latency_seconds", 0.0) or 0.0), 3)
                if is_generation_error_response(resp.answer, getattr(resp, "model_name", None)):
                    resp, extra_lat = _recover_generation_error(pipeline, aug, resp)
                    llm_lat = round(llm_lat + extra_lat, 3)
                meta = dict(sample.metadata or {})
                batch_item: Dict[str, Any] = {
                    "sample_id": sample.id,
                    "query": sample.query,
                    "type": meta.get("type") or (meta.get("query_types") or ["factoid"])[0],
                    "query_types": meta.get("query_types") or ([meta.get("type")] if meta.get("type") else []),
                    "unit": meta.get("unit", ""),
                    "category": meta.get("category", ""),
                    "ground_truth": sample.ground_truth,
                    "target_document_ids": meta.get("target_document_ids") or [],
                    "target_chunk_ids": meta.get("target_chunk_ids") or {},
                    "is_out_of_domain": _is_sample_ood(meta),
                    "retrieved_contexts": _serialize_contexts(resp.sources or contexts),
                    "generated_answer": resp.answer,
                    "model_name": getattr(resp, "model_name", None),
                    "retrieval_latency_s": ret_lat,
                    "llm_latency_s": llm_lat,
                    "total_latency_s": round(ret_lat + llm_lat, 3),
                    "token_usage": (getattr(resp, "metadata", {}) or {}).get("usage"),
                    "error": err_msg,
                }
                if meta.get("expected_route") is not None:
                    batch_item["expected_route"] = meta.get("expected_route")
                records_by_idx[orig_idx] = batch_item
            records = [records_by_idx[i] for i in range(total)]
        else:
            for idx, sample in enumerate(active_dataset, 1):
                if sample.id in existing_valid:
                    print(f"[{idx:02d}/{total:02d}] ⏭️  [Resume] Giữ nguyên mẫu hợp lệ '{sample.id}'")
                    records.append(existing_valid[sample.id])
                    continue

                short_q = (sample.query[:56] + "...") if len(sample.query) > 56 else sample.query
                print(f"[{idx:02d}/{total:02d}] 🔍 '{short_q}'")
                uq = UserQuery(query_text=sample.query)
                err_msg = None
                ret_lat: Optional[float] = None
                llm_lat: Optional[float] = None
                tot_lat: float = 0.0
                contexts: List[RetrievedContext] = []
                resp: Optional[Response] = None

                t_start = time.perf_counter()
                if has_granular_parts:
                    try:
                        t0 = time.perf_counter()
                        contexts = pipeline.retriever.retrieve(query=uq, top_k=top_k, filters=uq.filters)  # type: ignore[attr-defined]
                        ret_lat = round(time.perf_counter() - t0, 3)

                        aug = pipeline.prompt_augmenter.augment(query=uq, contexts=contexts)  # type: ignore[attr-defined]
                        t1 = time.perf_counter()
                        resp = pipeline.llm.generate(aug)  # type: ignore[attr-defined]
                        llm_lat = round(time.perf_counter() - t1, 3)

                        if is_generation_error_response(resp.answer, getattr(resp, "model_name", None)):
                            resp, extra_lat = _recover_generation_error(pipeline, aug, resp)
                            llm_lat = round(llm_lat + extra_lat, 3)

                        tot_lat = round(time.perf_counter() - t_start, 3)
                    except Exception as exc:
                        logger.warning(f"Lỗi khi generate mẫu {sample.id}: {exc}")
                        err_msg = str(exc)
                        resp = Response(answer=f"Lỗi truy vấn: {exc}", sources=contexts)
                        tot_lat = round(time.perf_counter() - t_start, 3)
                else:
                    for attempt in range(2):
                        try:
                            resp = pipeline.query(uq, top_k=top_k)
                            tot_lat = round(float(getattr(resp, "latency_seconds", 0.0) or (time.perf_counter() - t_start)), 3)
                            err_msg = None
                            break
                        except Exception as exc:
                            err_msg = str(exc)
                            logger.warning(f"Lỗi khi query mẫu {sample.id} (attempt {attempt + 1}): {exc}")
                            if attempt == 0:
                                time.sleep(2.0)
                            else:
                                resp = Response(answer=f"Lỗi truy vấn: {exc}", sources=[])
                                tot_lat = round(time.perf_counter() - t_start, 3)

                assert resp is not None
                resp_meta = getattr(resp, "metadata", None) if isinstance(getattr(resp, "metadata", None), dict) else {}
                trace_entries = resp_meta.get("trace") if isinstance(resp_meta.get("trace"), list) else []

                if ret_lat is None and trace_entries:
                    ret_vals = [
                        float(t["latency_s"])
                        for t in trace_entries
                        if isinstance(t, dict) and t.get("node") == "retrieval_node" and t.get("latency_s") is not None
                    ]
                    if ret_vals:
                        ret_lat = round(sum(ret_vals), 3)

                if llm_lat is None and trace_entries:
                    llm_vals = [
                        float(t["latency_s"])
                        for t in trace_entries
                        if isinstance(t, dict) and t.get("node") == "generation_node" and t.get("latency_s") is not None
                    ]
                    if llm_vals:
                        llm_lat = round(sum(llm_vals), 3)

                token_usage = resp_meta.get("usage")
                if token_usage is None and trace_entries:
                    for t in reversed(trace_entries):
                        if isinstance(t, dict) and t.get("node") == "generation_node" and t.get("usage") is not None:
                            token_usage = t.get("usage")
                            break

                meta = dict(sample.metadata or {})
                record_item: Dict[str, Any] = {
                    "sample_id": sample.id,
                    "query": sample.query,
                    "type": meta.get("type") or (meta.get("query_types") or ["factoid"])[0],
                    "query_types": meta.get("query_types") or ([meta.get("type")] if meta.get("type") else []),
                    "unit": meta.get("unit", ""),
                    "category": meta.get("category", ""),
                    "ground_truth": sample.ground_truth,
                    "target_document_ids": meta.get("target_document_ids") or [],
                    "target_chunk_ids": meta.get("target_chunk_ids") or {},
                    "is_out_of_domain": _is_sample_ood(meta),
                    "retrieved_contexts": _serialize_contexts(resp.sources or contexts),
                    "generated_answer": resp.answer,
                    "model_name": getattr(resp, "model_name", None),
                    "retrieval_latency_s": ret_lat,
                    "llm_latency_s": llm_lat,
                    "total_latency_s": tot_lat,
                    "token_usage": token_usage,
                    "error": err_msg,
                }
                if meta.get("expected_route") is not None:
                    record_item["expected_route"] = meta.get("expected_route")
                if resp_meta.get("status") is not None:
                    record_item["agent_status"] = resp_meta.get("status")
                if resp_meta.get("route") is not None:
                    record_item["agent_route"] = resp_meta.get("route")
                if any(
                    resp_meta.get(k) is not None
                    for k in ("status", "route", "trace", "retry", "fail_reason")
                ):
                    record_item["agent"] = {
                        "route": resp_meta.get("route"),
                        "status": resp_meta.get("status"),
                        "retry": int(resp_meta.get("retry", 0) or 0),
                        "fail_reason": resp_meta.get("fail_reason"),
                        "missing_info": resp_meta.get("missing_info"),
                        "used_queries": list(resp_meta.get("used_queries") or []),
                        "trace": sanitize_trace(trace_entries),
                    }
                records.append(record_item)

                if output_path and (idx % 10 == 0 or idx == total):
                    snapshot = {
                        "top_k": top_k,
                        "total_samples": len(records),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "predictions": records,
                    }
                    _atomic_write_json(Path(output_path), snapshot)

                if idx < total and delay_between_queries > 0:
                    if not ((getattr(resp, "metadata", {}) or {}).get("cache_hit")):
                        time.sleep(delay_between_queries)
    finally:
        if agent_graph is not None and prev_trace_verbose is not None:
            agent_graph.trace_verbose = prev_trace_verbose

    snapshot = {
        "top_k": top_k,
        "total_samples": len(records),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": records,
    }
    if output_path:
        _atomic_write_json(Path(output_path), snapshot)
        logger.info(f"Đã lưu predictions snapshot vào: {output_path}")
    return snapshot


def _compute_snapshot_retrieval_metrics(
    record: Dict[str, Any],
    top_k: int,
) -> Dict[str, Optional[float]]:
    if record.get("is_out_of_domain"):
        return {}

    contexts = (record.get("retrieved_contexts") or [])[:top_k]
    chunk_ids = [str(c["chunk_id"]) for c in contexts if c.get("chunk_id")]
    doc_ids = deduplicate_preserve_order([
        str(c.get("document_id") or c.get("chunk_id"))
        for c in contexts
        if (c.get("document_id") or c.get("chunk_id"))
    ])

    raw_target_docs = record.get("target_document_ids") or []
    if isinstance(raw_target_docs, dict):
        doc_qrels = {str(k): int(v) for k, v in raw_target_docs.items()}
    else:
        doc_qrels = {str(d): 2 for d in raw_target_docs}

    raw_target_chunks = record.get("target_chunk_ids") or {}
    if isinstance(raw_target_chunks, list):
        chunk_qrels = {str(c): 2 for c in raw_target_chunks}
    else:
        chunk_qrels = {str(k): int(v) for k, v in raw_target_chunks.items()}

    doc_rel_set = set(doc_qrels.keys())
    chunk_rel_set = set(chunk_qrels.keys())

    return {
        "doc_hit_at_k": round(hit_rate_at_k(doc_ids, doc_rel_set, top_k), 4) if doc_rel_set else None,
        "doc_recall_at_k": round(recall_at_k(doc_ids, doc_rel_set, top_k), 4) if doc_rel_set else None,
        "chunk_hit_at_k": round(hit_rate_at_k(chunk_ids, chunk_rel_set, top_k), 4) if chunk_rel_set else None,
        "chunk_recall_at_k": round(recall_at_k(chunk_ids, chunk_rel_set, top_k), 4) if chunk_rel_set else None,
        "chunk_ndcg_at_k": round(ndcg_at_k(chunk_ids, chunk_qrels, top_k), 4) if chunk_qrels else None,
        "chunk_mrr": round(mrr(chunk_ids[:top_k], chunk_rel_set), 4) if chunk_rel_set else None,
    }


def _score_single_offline_record(
    record: Dict[str, Any],
    top_k: int,
    embedder: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Chấm điểm offline cho 1 bản ghi snapshot, trả về (offline_metrics_dict, retrieval_metrics_dict)."""
    query = record.get("query", "")
    answer = record.get("generated_answer", "")
    ground_truth = record.get("ground_truth")
    is_ood = bool(record.get("is_out_of_domain")) or _is_sample_ood(record)
    contexts = (record.get("retrieved_contexts") or [])[:top_k]
    contexts_text = [_format_context_dict_for_eval(c) for c in contexts]

    ret_metrics = _compute_snapshot_retrieval_metrics(record, top_k=top_k)

    raw_target_chunks = record.get("target_chunk_ids") or {}
    if isinstance(raw_target_chunks, list):
        target_chunks = {str(c): 2 for c in raw_target_chunks}
    else:
        target_chunks = {str(k): int(v) for k, v in raw_target_chunks.items()}

    c_rel = compute_context_relevance(query, contexts_text, embedder=embedder) if not is_ood else 0.0
    if target_chunks:
        ret_cids = {str(c.get("chunk_id")) for c in contexts if c.get("chunk_id")}
        g2 = {cid for cid, g in target_chunks.items() if g >= 2}
        g1 = {cid for cid, g in target_chunks.items() if g == 1}
        if ret_cids & g2:
            c_hit = 1.0
        elif ret_cids & g1:
            c_hit = 0.5
        else:
            c_hit = 0.0
    else:
        c_hit = 1.0 if (not is_ood and c_rel >= 0.45) else 0.0

    if (
        is_generation_error_response(answer, record.get("model_name"))
        or bool(record.get("error"))
        or str(record.get("agent_status") or "").lower() == "fail"
    ):
        return (
            {
                "is_generation_error": True,
                "is_refusal": False,
                "context_hit": round(c_hit, 4) if not is_ood else None,
            },
            ret_metrics,
        )

    clean_answer = strip_agent_notes(answer)
    notes_stripped = clean_answer != answer
    refusal = is_refusal_response(clean_answer) or str(record.get("agent_status") or "").lower() == "out_of_scope"

    if is_ood:
        ood_acc = 1.0 if refusal else 0.0
        offline = {
            "is_generation_error": False,
            "is_refusal": refusal,
            "agent_notes_stripped": notes_stripped,
            "ood_refusal_accuracy": ood_acc,
            "overall_score": ood_acc,
        }
        return offline, ret_metrics

    faith = compute_faithfulness(clean_answer, contexts_text, embedder=embedder)
    a_rel = compute_answer_relevance(query, clean_answer, embedder=embedder)
    num_info = compute_numeric_grounding(clean_answer, contexts_text, ground_truth=ground_truth)
    corr_info = compute_answer_correctness_offline(clean_answer, ground_truth, embedder=embedder)

    a_corr = corr_info["answer_correctness"]
    if a_corr is not None:
        overall = 0.40 * float(a_corr) + 0.30 * faith + 0.15 * a_rel + 0.15 * c_rel
    else:
        overall = 0.45 * faith + 0.30 * a_rel + 0.25 * c_rel

    offline = {
        "context_relevance": round(c_rel, 4),
        "faithfulness": round(faith, 4),
        "answer_relevance": round(a_rel, 4),
        "answer_correctness": a_corr,
        "token_precision": corr_info["token_precision"],
        "token_recall": corr_info["token_recall"],
        "token_f1": corr_info["token_f1"],
        "keyword_recall": corr_info["keyword_recall"],
        "rouge_l_recall": corr_info["rouge_l_recall"],
        "rouge_l_f1": corr_info["rouge_l_f1"],
        "numeric_faithfulness": num_info["numeric_faithfulness"],
        "numeric_correctness": num_info["numeric_correctness"],
        "ungrounded_entities": num_info["ungrounded_entities"],
        "missing_gt_entities": num_info["missing_gt_entities"],
        "semantic_answer_correctness": corr_info["semantic_answer_correctness"],
        "context_hit": round(c_hit, 4),
        "is_refusal": refusal,
        "agent_notes_stripped": notes_stripped,
        "overall_score": round(min(1.0, max(0.0, overall)), 4),
    }
    return offline, ret_metrics


def score_predictions(
    predictions_source: Union[str, Path, Dict[str, Any]],
    top_k: Optional[int] = None,
    use_judge: bool = False,
    judge: Optional[UETLLMJudgeEvaluator] = None,
    embedder: Optional[Any] = None,
    enable_semantic: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_rounds: bool = False,
) -> Dict[str, Any]:
    """
    Pha 2 (Scoring & Error Taxonomy): Chấm Retrieval + Offline Metrics + tùy chọn UETLLMJudgeEvaluator
    trên snapshot từ Pha 1 mà không cần chạy lại RAG Pipeline.
    """
    if isinstance(predictions_source, (str, Path)):
        with open(predictions_source, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    else:
        snapshot = predictions_source

    raw_records = snapshot.get("predictions") or snapshot.get("samples") or []
    eff_top_k = top_k if top_k is not None else int(snapshot.get("top_k", 10))

    active_embedder = embedder
    if active_embedder is None and enable_semantic:
        try:
            active_embedder = UETEmbedder()
        except Exception as exc:
            logger.warning(f"Không thể nạp UETEmbedder cho semantic metrics ({exc}); chạy chế độ lexical.")
            active_embedder = None

    active_judge = judge
    if use_judge and active_judge is None:
        active_judge = UETLLMJudgeEvaluator(provider=judge_provider, model_name=judge_model)

    scored_samples: List[Dict[str, Any]] = []
    total = len(raw_records)

    for idx, raw in enumerate(raw_records, 1):
        rec = dict(raw)
        rec["is_out_of_domain"] = bool(rec.get("is_out_of_domain")) or _is_sample_ood(rec)
        offline_metrics, retrieval_metrics = _score_single_offline_record(
            rec,
            top_k=eff_top_k,
            embedder=active_embedder if enable_semantic else None,
        )
        rec["offline_metrics"] = offline_metrics
        rec["retrieval_metrics"] = retrieval_metrics

        if use_judge and active_judge is not None:
            if offline_metrics.get("is_generation_error"):
                rec["judge"] = {
                    "status": "skipped_generation_error",
                    "sample_id": str(rec.get("sample_id") or rec.get("id") or idx),
                    "raw_scores": {},
                    "normalized_scores": {},
                    "error": "Skipped judge due to generation error",
                }
            else:
                raw_ctxs = (rec.get("retrieved_contexts") or [])[:eff_top_k]
                contexts_text = [_format_context_dict_for_eval(c) for c in raw_ctxs]
                sid = str(rec.get("sample_id") or rec.get("id") or idx)
                judge_res = active_judge.judge_record(
                    query=rec.get("query", ""),
                    contexts=contexts_text,
                    answer=rec.get("generated_answer", ""),
                    ground_truth=rec.get("ground_truth"),
                    is_out_of_domain=rec["is_out_of_domain"],
                    sample_id=sid,
                    generator_model=rec.get("model_name"),
                )
                rec["judge"] = judge_res
                if not judge_res.get("cache_hit") and idx < total and active_judge.delay_seconds > 0:
                    time.sleep(active_judge.delay_seconds)

                if judge_rounds and not rec["is_out_of_domain"] and has_agent_data(rec):
                    rounds = extract_rounds(get_agent_info(rec)["trace"])
                    if len(rounds) >= 2 and rounds[0].get("answer_text"):
                        r0_sel_ids = rounds[0].get("selected_chunk_ids") or rounds[0].get("retrieved_chunk_ids") or []
                        ctx_by_id = {
                            str(c.get("chunk_id")): _format_context_dict_for_eval(c)
                            for c in (rec.get("retrieved_contexts") or [])
                            if c.get("chunk_id")
                        }
                        if r0_sel_ids and all(cid in ctx_by_id for cid in r0_sel_ids):
                            r0_contexts = [ctx_by_id[cid] for cid in r0_sel_ids]
                        else:
                            r0_contexts = contexts_text
                        r0_judge = active_judge.judge_record(
                            query=rec.get("query", ""),
                            contexts=r0_contexts,
                            answer=str(rounds[0]["answer_text"]),
                            ground_truth=rec.get("ground_truth"),
                            is_out_of_domain=False,
                            sample_id=f"{sid}::r0",
                            generator_model=rec.get("model_name"),
                        )
                        rec["agent_round_judge"] = {"r0": r0_judge}
                        if not r0_judge.get("cache_hit") and active_judge.delay_seconds > 0:
                            time.sleep(active_judge.delay_seconds)

        err_label, err_reason = classify_sample_error(rec)
        rec["error_label"] = err_label
        rec["error_reason"] = err_reason
        if has_agent_data(rec):
            rounds_eval = score_sample_rounds(
                rec,
                embedder=active_embedder if enable_semantic else None,
            )
            if rounds_eval is not None:
                rec["agent_rounds_eval"] = rounds_eval
            rec["agent_flags"] = compute_sample_agent_flags(rec)
        scored_samples.append(rec)

    judge_info = {
        "enabled": bool(use_judge and active_judge is not None),
        "provider": getattr(active_judge, "provider", None) if active_judge else None,
        "model": getattr(active_judge, "model_name", None) if active_judge else None,
    }
    report = build_e2e_report(scored_samples=scored_samples, top_k=eff_top_k, judge_info=judge_info)

    if output_path:
        _atomic_write_json(Path(output_path), report)
        logger.info(f"Đã lưu báo cáo E2E chi tiết vào: {output_path}")
    return report

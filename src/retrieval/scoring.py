"""MMR selection and rerank/hybrid score blending helpers for UETRetriever."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.base import BaseEmbeddingModel, BaseVectorStore, RetrievedContext, UserQuery
from src.graph.thesaurus_utils import KNOWN_FACULTY_KEYS, detect_query_faculties
from src.retrieval.mmr import MaximalMarginalRelevance
from src.retrieval.reranker import BaseReranker, HeuristicReranker
from src.utils.helpers import get_logger

logger = get_logger("retrieval")

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ENTITY_CODE_RE = re.compile(
    r"\b(?:\d{2,5}/[a-zđ]+(?:-[a-zđ0-9]+)+|(?:int|elt|ema|epn|pes|mat|phy|flf|cn|k)\d{1,4}[a-z]?)\b",
    re.IGNORECASE,
)


def apply_mmr_selection(
    query: UserQuery,
    query_vector: Any,
    candidates: List[RetrievedContext],
    *,
    top_n: int,
    embedding_model: BaseEmbeddingModel,
    vector_store: BaseVectorStore,
    mmr_lambda: float,
    mmr_stage: str,
    mmr_similarity_threshold: float,
    mmr_protect_tabular: bool,
) -> List[RetrievedContext]:
    """Áp dụng MMR bảo toàn điểm Hybrid/Rerank và bảo vệ chunk bảng."""
    if len(candidates) <= top_n:
        return candidates

    try:
        if query_vector is not None and getattr(query_vector, "vector", None):
            q_vec: Optional[List[float]] = query_vector.vector
        else:
            q_vec = embedding_model.embed_query(query).vector

        chunk_ids = [c.chunk_id for c in candidates]
        embeddings_map: Dict[str, List[float]] = {}
        if hasattr(vector_store, "get_embeddings_by_ids"):
            embeddings_map = vector_store.get_embeddings_by_ids(chunk_ids)

        missing_chunks = [c for c in candidates if c.chunk_id not in embeddings_map]
        if missing_chunks:
            logger.warning(
                "[MMR Fallback] Phát hiện %d/%d chunks không có sẵn vector trong VectorStore. "
                "Đang tự động nhúng bổ sung bằng embedding_model... (Danh sách Chunk IDs thiếu: %s)",
                len(missing_chunks),
                len(candidates),
                [c.chunk_id for c in missing_chunks[:10]] + (["..."] if len(missing_chunks) > 10 else []),
            )
            try:
                generated_embs = embedding_model.embed_texts([c.text for c in missing_chunks])
                for chunk, emb in zip(missing_chunks, generated_embs):
                    embeddings_map[chunk.chunk_id] = emb
            except Exception as emb_err:
                logger.error(f"[MMR Fallback] Lỗi khi tạo vector bổ sung: {emb_err}")

        candidate_vectors: List[List[float]] = []
        valid_candidates: List[RetrievedContext] = []
        for chunk in candidates:
            if chunk.chunk_id in embeddings_map:
                candidate_vectors.append(embeddings_map[chunk.chunk_id])
                valid_candidates.append(chunk)

        if q_vec and candidate_vectors and len(valid_candidates) >= top_n:
            selected = MaximalMarginalRelevance.select(
                query_vector=q_vec,
                candidates=valid_candidates,
                candidate_vectors=candidate_vectors,
                lambda_mult=mmr_lambda,
                top_n=top_n,
                use_hybrid_relevance=True,
                similarity_threshold=mmr_similarity_threshold,
                protect_tabular=mmr_protect_tabular,
            )
            logger.info(
                "Đã áp dụng MMR (%s): %d -> %d candidates (lambda=%.2f, thr=%.2f).",
                mmr_stage,
                len(candidates),
                len(selected),
                mmr_lambda,
                mmr_similarity_threshold,
            )
            return selected

        logger.info(
            "Không đủ vector hợp lệ (%d/%d), giữ nguyên danh sách gốc.",
            len(candidate_vectors),
            top_n,
        )
        return candidates
    except Exception as mmr_err:
        logger.warning(f"Lỗi khi chạy MMR ({mmr_err}). Fallback sử dụng danh sách gốc.")
        return candidates


def blend_rerank_and_hybrid(
    query_text: str,
    reranked_chunks: List[RetrievedContext],
    hybrid_map: Dict[str, RetrievedContext],
    *,
    reranker: BaseReranker,
    rerank_blend_alpha: float,
    enable_entity_anchor_boost: bool,
    extracted_entities: Optional[List[str]] = None,
) -> List[RetrievedContext]:
    """
    Hòa trộn điểm Reranker với điểm Hybrid RRF + boost nhẹ khi khớp chính xác năm/mã học phần/số QĐ/thực thể PA1.
    """
    if not reranked_chunks or isinstance(reranker, HeuristicReranker) or rerank_blend_alpha >= 1.0:
        return reranked_chunks
    if all((c.metadata or {}).get("degraded") for c in reranked_chunks):
        return reranked_chunks

    alpha = max(0.0, min(1.0, rerank_blend_alpha))
    q_years = set(_YEAR_RE.findall(query_text)) if enable_entity_anchor_boost else set()
    q_codes = (
        {m.group(0).lower() for m in _ENTITY_CODE_RE.finditer(query_text)}
        if enable_entity_anchor_boost
        else set()
    )
    target_faculties = detect_query_faculties(query_text) if enable_entity_anchor_boost else set()
    pa1_terms = (
        [e.strip().lower() for e in (extracted_entities or []) if e and len(e.strip()) >= 3]
        if enable_entity_anchor_boost
        else []
    )

    blended_list: List[RetrievedContext] = []
    for cand in reranked_chunks:
        rerank_score = float(cand.similarity_score)
        hyb_ctx = hybrid_map.get(cand.chunk_id)
        hybrid_score = float(hyb_ctx.similarity_score) if hyb_ctx is not None else rerank_score
        blended = alpha * rerank_score + (1.0 - alpha) * hybrid_score

        meta = dict(cand.metadata) if cand.metadata else {}
        title_low = str(meta.get("title", "")).lower()
        text_low = (cand.text or "").lower()

        anchor_delta = 0.0
        if q_years:
            if any(y in title_low for y in q_years):
                anchor_delta += 0.04
            elif any(y in text_low for y in q_years):
                anchor_delta += 0.02
            else:
                title_years = set(_YEAR_RE.findall(title_low))
                if title_years and not (title_years & q_years):
                    anchor_delta -= 0.04

        if q_codes and any(code in text_low or code in title_low for code in q_codes):
            anchor_delta += 0.03

        if target_faculties:
            chunk_fac = str(meta.get("faculty") or meta.get("site") or "").strip().lower()
            if chunk_fac in target_faculties:
                anchor_delta += 0.04
            elif chunk_fac in KNOWN_FACULTY_KEYS:
                anchor_delta -= 0.12

        if pa1_terms:
            graph_ents_low = " ".join(str(ge).lower() for ge in (meta.get("graph_entities") or []))
            if any(pt in graph_ents_low or pt in text_low for pt in pa1_terms):
                anchor_delta += 0.025

        final_score = round(max(0.0, min(1.0, blended + anchor_delta)), 4)
        meta["hybrid_score"] = round(hybrid_score, 4)
        meta["blended_score"] = final_score
        if anchor_delta != 0.0:
            meta["anchor_boost"] = round(anchor_delta, 4)

        blended_list.append(
            RetrievedContext(
                chunk_id=cand.chunk_id,
                text=cand.text,
                similarity_score=final_score,
                rank=cand.rank,
                metadata=meta,
            )
        )

    blended_list.sort(key=lambda c: c.similarity_score, reverse=True)
    for rank, ctx in enumerate(blended_list, 1):
        ctx.rank = rank
    return blended_list


def blend_rerank_and_rrf(
    reranked: List[RetrievedContext],
    rrf_norm_map: Dict[str, float],
    top_n: int,
    *,
    rerank_weight: float = 0.6,
    rrf_blend_weight: float = 0.4,
) -> List[RetrievedContext]:
    """Kết hợp điểm Cross-Encoder Reranker với điểm RRF chuẩn hóa (BM25 + Vector)."""
    if not reranked:
        return []

    w_rerank = max(0.0, float(rerank_weight))
    w_rrf = max(0.0, float(rrf_blend_weight))
    total_w = w_rerank + w_rrf

    if w_rrf <= 0.0 or total_w <= 0.0:
        return reranked[:top_n]

    if all((c.metadata or {}).get("degraded") for c in reranked):
        return reranked[:top_n]

    w_rerank /= total_w
    w_rrf /= total_w

    blended_list: List[RetrievedContext] = []
    for ctx in reranked:
        meta = dict(ctx.metadata) if ctx.metadata else {}
        rerank_score = float(meta.get("rerank_score", ctx.similarity_score))
        rrf_norm = float(rrf_norm_map.get(ctx.chunk_id, 0.0))
        blended = round(w_rerank * rerank_score + w_rrf * rrf_norm, 4)

        meta["rerank_score"] = round(rerank_score, 4)
        meta["rrf_norm_score"] = round(rrf_norm, 4)
        meta["blended_score"] = blended

        blended_list.append(
            RetrievedContext(
                chunk_id=ctx.chunk_id,
                text=ctx.text,
                similarity_score=blended,
                rank=ctx.rank,
                metadata=meta,
            )
        )

    blended_list.sort(key=lambda x: x.similarity_score, reverse=True)
    top_blended = blended_list[:top_n]
    for rank, ctx in enumerate(top_blended, 1):
        ctx.rank = rank
    return top_blended

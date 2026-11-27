"""Base reranker contract, text normalization, score blending, and heuristic/RRF fallbacks."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.base import RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("reranker")

_INLINE_HEADING_RE = re.compile(r"(?:^|(?<=\s)|(?<=\|))#{1,6}\s+", re.MULTILINE)
_HTML_BR_SEMICOLON_RE = re.compile(r"\s+;\s+")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _clean_text_for_reranker(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Làm sạch nhiễu định dạng Markdown/HTML trước khi đưa vào Tokenizer của Cross-Encoder (Jina / XLM-RoBERTa):
      1. Khử dòng [META: ...] bị lặp nguyên văn tiêu đề tài liệu [Title] ngay phía trên.
      2. Xóa dấu '**' và '__' bọc sát chữ cái đầu từ để ngăn lỗi vỡ vụn subword (Token Shattering).
      3. Xóa tiền tố heading '#### ' nhúng trong từng ô bảng và dấu ' ; ' sinh ra từ thẻ <br> HTML.
      4. Đính kèm tối đa 2 cạnh liên kết đồ thị ngắn (graph_triples) cho chunk quy chế phi bảng biểu.
    """
    if not text:
        return ""

    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("[") and lines[0].endswith("]"):
        doc_header = lines[0][1:-1].strip()
        second_line = lines[1].strip()
        if second_line.startswith("[META:") and second_line.endswith("]"):
            meta_header = second_line[6:-1].strip()
            if doc_header.lower() == meta_header.lower():
                lines = [lines[0]] + lines[2:]

    cleaned = "\n".join(lines)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = _INLINE_HEADING_RE.sub("", cleaned)
    cleaned = _HTML_BR_SEMICOLON_RE.sub(" ", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()

    if isinstance(metadata, dict) and metadata.get("data_type") != "tabular":
        triples = metadata.get("graph_triples") or []
        if triples:
            short_triples = [
                str(t).split(" (Căn cứ:")[0].strip()
                for t in triples[:2]
                if str(t).strip() and not str(t).startswith("Phạm vi văn bản:")
            ]
            if short_triples:
                cleaned = f"[Liên kết học vụ: {'; '.join(short_triples)}]\n{cleaned}"

    return cleaned


def _blend_rerank_and_prior(
    candidates: List[RetrievedContext],
    rerank_probs: List[float],
    rerank_weight: float = 0.75,
) -> List[float]:
    """
    Kết hợp tuyến tính điểm xác suất của Cross-Encoder Reranker với điểm Hybrid RRF ban đầu:
        final_score = w * rerank_prob + (1 - w) * normalized_prior
    """
    w = max(0.0, min(1.0, float(rerank_weight)))
    if w >= 1.0 or not candidates:
        return [float(s) for s in rerank_probs]

    priors = [max(0.0, float(c.similarity_score or 0.0)) for c in candidates]
    max_prior = max(priors) if priors else 0.0
    if max_prior <= 0.0:
        return [float(s) for s in rerank_probs]

    if max_prior > 1.0:
        priors = [p / max_prior for p in priors]

    return [(w * float(r_prob)) + ((1.0 - w) * p_score) for r_prob, p_score in zip(rerank_probs, priors)]


def rrf_degraded_fallback(candidates: List[RetrievedContext], top_n: int) -> List[RetrievedContext]:
    """Preserve incoming RRF ranking with ``degraded=True`` metadata when a neural reranker is unavailable."""
    fallback_results: List[RetrievedContext] = []
    for rank, cand in enumerate(candidates[:top_n], 1):
        meta = dict(cand.metadata) if cand.metadata else {}
        meta["degraded"] = True
        meta["rerank_backend"] = "rrf_fallback"
        fallback_results.append(
            RetrievedContext(
                chunk_id=cand.chunk_id,
                text=cand.text,
                similarity_score=cand.similarity_score,
                rank=rank,
                metadata=meta,
            )
        )
    return fallback_results


class BaseReranker(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các mô hình Reranker."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        """Tái sắp xếp danh sách candidate chunks dựa trên câu hỏi người dùng."""
        pass


class HeuristicReranker(BaseReranker):
    """
    Reranker dựa trên thuật toán tương quan đặc trưng chéo (Cross-Feature Re-scoring).
    Hoạt động 100% bằng Pure Python, không cần GPU.
    """

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        if not candidates:
            return []

        clean_query = query.lower().strip()
        query_words = [w for w in re.findall(r"[\w\-]+", clean_query) if len(w) > 1]
        query_len = max(1, len(query_words))

        scored_items: List[Dict[str, Any]] = []
        for candidate in candidates:
            text_lower = candidate.text.lower()
            meta = dict(candidate.metadata) if candidate.metadata else {}
            title_lower = meta.get("title", "").lower()

            phrase_bonus = 0.25 if clean_query in text_lower or (title_lower and clean_query in title_lower) else 0.0
            matched_words = [w for w in query_words if w in text_lower or (title_lower and w in title_lower)]
            coverage_score = len(matched_words) / query_len
            early_text = text_lower[:200]
            early_matches = sum(1 for w in query_words if w in early_text or (title_lower and w in title_lower))
            position_score = (early_matches / query_len) * 0.15
            base_similarity = candidate.similarity_score

            rerank_score = (
                (0.45 * base_similarity)
                + (0.30 * coverage_score)
                + (0.15 * phrase_bonus)
                + (0.10 * position_score)
            )
            rerank_score = round(min(0.99, max(0.05, rerank_score)), 4)

            meta["reranker_name"] = "heuristic_cross_scorer"
            meta["rerank_score"] = rerank_score
            meta["original_rank"] = candidate.rank

            scored_items.append({"context": candidate, "score": rerank_score, "meta": meta})

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        results: List[RetrievedContext] = []
        for rank, item in enumerate(scored_items[:top_n], 1):
            orig_ctx = item["context"]
            results.append(
                RetrievedContext(
                    chunk_id=orig_ctx.chunk_id,
                    text=orig_ctx.text,
                    similarity_score=item["score"],
                    rank=rank,
                    metadata=item["meta"],
                )
            )

        logger.info(f"Heuristic Reranker đã sắp xếp {len(candidates)} ứng viên thành {len(results)} ranked chunks.")
        return results

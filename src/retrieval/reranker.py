"""
src/retrieval/reranker.py - Backward-compatible facade for reranker backends and factory.

Implementations live in ``src/retrieval/rerankers/``.
"""

from __future__ import annotations

from src.retrieval.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    FlashRankReranker,
    HFSequenceClassificationReranker,
    HeuristicReranker,
    JinaONNXReranker,
    PhoRankerReranker,
    RerankerFactory,
    ViRankerReranker,
    VietnameseReranker,
    VietnameseRerankerGGUF,
    _blend_rerank_and_prior,
    _clean_text_for_reranker,
    rrf_degraded_fallback,
)

__all__ = [
    "_clean_text_for_reranker",
    "_blend_rerank_and_prior",
    "rrf_degraded_fallback",
    "BaseReranker",
    "HeuristicReranker",
    "FlashRankReranker",
    "CrossEncoderReranker",
    "JinaONNXReranker",
    "HFSequenceClassificationReranker",
    "ViRankerReranker",
    "VietnameseReranker",
    "PhoRankerReranker",
    "VietnameseRerankerGGUF",
    "RerankerFactory",
]

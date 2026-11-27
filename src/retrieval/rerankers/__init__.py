"""Modular reranker backends and factory for UET Retrieval."""

from src.retrieval.rerankers.backends import (
    CrossEncoderReranker,
    FlashRankReranker,
    HFSequenceClassificationReranker,
    PhoRankerReranker,
    VietnameseReranker,
    VietnameseRerankerGGUF,
    ViRankerReranker,
)
from src.retrieval.rerankers.base import (
    BaseReranker,
    HeuristicReranker,
    _blend_rerank_and_prior,
    _clean_text_for_reranker,
    rrf_degraded_fallback,
)
from src.retrieval.rerankers.factory import RerankerFactory
from src.retrieval.rerankers.onnx_jina import JinaONNXReranker

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

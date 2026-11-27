"""Factory for selecting and instantiating the configured reranker backend."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.retrieval.rerankers.backends import (
    CrossEncoderReranker,
    FlashRankReranker,
    PhoRankerReranker,
    VietnameseReranker,
    VietnameseRerankerGGUF,
    ViRankerReranker,
)
from src.retrieval.rerankers.base import BaseReranker, HeuristicReranker
from src.retrieval.rerankers.onnx_jina import JinaONNXReranker
from src.utils.helpers import get_logger

logger = get_logger("reranker")


class RerankerFactory:
    """Factory khởi tạo Reranker phù hợp nhất theo cấu hình và thư viện hiện có."""

    @staticmethod
    def create(
        reranker_type: str = "auto",
        model_name: Optional[str] = None,
        onnx_filename: Optional[str] = None,
        **kwargs,
    ) -> BaseReranker:
        rtype = reranker_type.lower()
        model_str = (model_name or "").lower()

        if rtype == "heuristic":
            return HeuristicReranker()

        if rtype in ("vietnamese_reranker_gguf", "gguf") or "gguf" in model_str:
            return VietnameseRerankerGGUF(
                model_path=model_name or "weights/Vietnamese_Reranker_GGUF/Vietnamese_Reranker_F16_GGUF.gguf",
            )

        if rtype in ("vietnamese_reranker", "vietnamese-reranker", "vietnamesereranker") or "vietnamese_reranker" in model_str:
            return VietnameseReranker(
                model_name=model_name or "weights/Vietnamese_Reranker",
            )

        if rtype in ("phoranker", "pho_ranker") or "phoranker" in model_str:
            return PhoRankerReranker(
                model_name=model_name or "weights/PhoRanker",
            )

        if rtype in ("viranker", "vi_ranker") or "viranker" in model_str:
            return ViRankerReranker(
                model_name=model_name or "weights/ViRanker",
            )

        jina_kwargs: Dict[str, Any] = {}
        if kwargs.get("max_length") is not None:
            jina_kwargs["max_length"] = int(kwargs["max_length"])
        if kwargs.get("batch_size") is not None:
            jina_kwargs["batch_size"] = int(kwargs["batch_size"])
        if kwargs.get("use_gpu") is not None:
            jina_kwargs["use_gpu"] = bool(kwargs["use_gpu"])
        if kwargs.get("providers") is not None:
            jina_kwargs["providers"] = list(kwargs["providers"])
        if kwargs.get("rerank_weight") is not None:
            jina_kwargs["rerank_weight"] = float(kwargs["rerank_weight"])

        if rtype in ("onnx", "jina", "jina_onnx") or "jina" in model_str:
            return JinaONNXReranker(
                model_name=model_name or "jinaai/jina-reranker-v2-base-multilingual",
                onnx_filename=onnx_filename or "onnx/model.onnx",
                **jina_kwargs,
            )

        if rtype == "flashrank":
            return FlashRankReranker(model_name=model_name or "ms-marco-MiniLM-L-12-v2")

        if rtype == "cross_encoder":
            return CrossEncoderReranker(model_name=model_name or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

        try:
            import onnxruntime  # noqa: F401

            return JinaONNXReranker(
                model_name=model_name or "jinaai/jina-reranker-v2-base-multilingual",
                onnx_filename=onnx_filename or "onnx/model.onnx",
                **jina_kwargs,
            )
        except Exception:
            pass

        try:
            import flashrank  # type: ignore # noqa: F401

            return FlashRankReranker(model_name=model_name or "ms-marco-MiniLM-L-12-v2")
        except ImportError:
            pass

        try:
            import sentence_transformers  # noqa: F401

            return CrossEncoderReranker(model_name=model_name or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        except Exception:
            pass

        logger.info("Sử dụng HeuristicReranker chuẩn cho quy trình Reranking.")
        return HeuristicReranker()

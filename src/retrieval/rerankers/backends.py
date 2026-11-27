"""FlashRank, CrossEncoder, HuggingFace SequenceClassification, and GGUF reranker backends."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.base import RetrievedContext
from src.retrieval.rerankers.base import BaseReranker, HeuristicReranker
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("reranker")


class FlashRankReranker(BaseReranker):
    """Reranker dựa trên FlashRank (ONNX Runtime nhẹ, không phụ thuộc PyTorch)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._ranker = None
        self._fallback = HeuristicReranker()

        try:
            from flashrank import Ranker  # type: ignore

            self._ranker = Ranker(model_name=model_name)
            logger.info(f"Đã khởi tạo thành công FlashRankReranker với model '{model_name}'.")
        except Exception as exc:
            logger.warning(f"Không thể khởi tạo FlashRank ({exc}). Chuyển sang HeuristicReranker dự phòng.")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        if not candidates:
            return []
        if self._ranker is None:
            return self._fallback.rerank(query, candidates, top_n=top_n)

        try:
            from flashrank import RerankRequest  # type: ignore

            passages = [{"id": c.chunk_id, "text": c.text, "meta": c.metadata} for c in candidates]
            ranked_results = self._ranker.rerank(RerankRequest(query=query, passages=passages))

            results: List[RetrievedContext] = []
            cand_map = {c.chunk_id: c for c in candidates}
            for rank, item in enumerate(ranked_results[:top_n], 1):
                cid = str(item.get("id"))
                orig = cand_map.get(cid)
                if orig:
                    meta = dict(orig.metadata) if orig.metadata else {}
                    meta["reranker_name"] = f"flashrank_{self.model_name}"
                    meta["rerank_score"] = round(float(item.get("score", 0.0)), 4)
                    results.append(
                        RetrievedContext(
                            chunk_id=cid,
                            text=orig.text,
                            similarity_score=round(float(item.get("score", 0.0)), 4),
                            rank=rank,
                            metadata=meta,
                        )
                    )
            return results
        except Exception as err:
            logger.warning(f"Lỗi khi thực thi FlashRank: {err}. Chuyển sang fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class CrossEncoderReranker(BaseReranker):
    """Reranker sử dụng sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        if "/" not in model_name and not Path(model_name).exists():
            resolved_model_name = f"cross-encoder/{model_name}"
        else:
            resolved_model_name = model_name
        self.model_name = resolved_model_name
        self._model = None
        self._fallback = HeuristicReranker()

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(resolved_model_name)
            logger.info(f"Đã tải thành công CrossEncoder model '{resolved_model_name}'.")
        except Exception as exc:
            logger.warning(f"Không thể khởi tạo CrossEncoder ({exc}). Chuyển sang HeuristicReranker dự phòng.")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        if not candidates:
            return []
        if self._model is None:
            return self._fallback.rerank(query, candidates, top_n=top_n)

        try:
            pairs = [[query, c.text] for c in candidates]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = f"cross_encoder_{self.model_name}"
                meta["rerank_score"] = round(float(score), 4)
                results.append(
                    RetrievedContext(
                        chunk_id=cand.chunk_id,
                        text=cand.text,
                        similarity_score=round(float(score), 4),
                        rank=rank,
                        metadata=meta,
                    )
                )
            return results
        except Exception as err:
            logger.warning(f"Lỗi chạy CrossEncoder: {err}. Sử dụng heuristic fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class HFSequenceClassificationReranker(BaseReranker):
    """Shared HuggingFace AutoModelForSequenceClassification reranker with length bucketing."""

    def __init__(
        self,
        *,
        model_path: str,
        display_name: str,
        reranker_meta_name: str,
        max_length: int,
        batch_size: int,
    ):
        self.model_name = display_name
        self.model_path = model_path
        self._reranker_meta_name = reranker_meta_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._fallback = HeuristicReranker()

        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            path_obj = Path(self.model_path)
            target = str(path_obj.resolve()) if path_obj.exists() else self.model_path

            logger.info(f"Đang tải {self.model_name} từ '{target}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(target)
            self._model = AutoModelForSequenceClassification.from_pretrained(target)
            self._model.eval()

            if torch.cuda.is_available():
                self._device = "cuda"
                self._model.to("cuda")
                logger.info(f"{self.model_name} đang sử dụng thiết bị: CUDA (GPU).")
            else:
                self._device = "cpu"
                self._model.to("cpu")
                logger.info(f"{self.model_name} đang sử dụng thiết bị: CPU.")

            logger.info(f"Đã khởi tạo thành công {self.model_name} từ '{target}'.")
        except Exception as exc:
            logger.warning(f"Không thể khởi tạo {self.model_name} ({exc}). Sẽ dùng HeuristicReranker dự phòng.")
            self._model = None
            self._tokenizer = None

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        if not candidates:
            return []
        if self._model is None or self._tokenizer is None:
            return self._fallback.rerank(query, candidates, top_n=top_n)

        try:
            import torch

            indexed_cands = sorted(enumerate(candidates), key=lambda x: len(x[1].text))
            all_scores: List[float] = [0.0] * len(candidates)

            with torch.no_grad():
                for i in range(0, len(indexed_cands), self.batch_size):
                    batch_indexed = indexed_cands[i : i + self.batch_size]
                    batch_pairs = [[query, c.text] for _, c in batch_indexed]
                    inputs = self._tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    outputs = self._model(**inputs, return_dict=True)
                    logits = outputs.logits.view(-1).float()
                    probs = torch.sigmoid(logits).cpu().numpy().tolist()
                    if isinstance(probs, float):
                        probs = [probs]
                    for (orig_idx, _), score_val in zip(batch_indexed, probs):
                        all_scores[orig_idx] = float(score_val)

            ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = self._reranker_meta_name
                meta["rerank_score"] = round(float(score), 4)
                results.append(
                    RetrievedContext(
                        chunk_id=cand.chunk_id,
                        text=cand.text,
                        similarity_score=round(float(score), 4),
                        rank=rank,
                        metadata=meta,
                    )
                )

            logger.info(f"{self.model_name} đã sắp xếp {len(candidates)} ứng viên -> {len(results)} ranked chunks.")
            return results
        except Exception as err:
            logger.warning(f"Lỗi khi thực thi {self.model_name} ({err}). Chuyển sang HeuristicReranker fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class ViRankerReranker(HFSequenceClassificationReranker):
    """Reranker sử dụng mô hình ViRanker (namdp-ptit/ViRanker)."""

    def __init__(
        self,
        model_name: str = "weights/ViRanker",
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        try:
            cfg = load_yaml_config().get("agent", {}).get("reranker", {})
        except Exception:
            cfg = {}

        super().__init__(
            model_path=model_name,
            display_name=model_name,
            reranker_meta_name=f"viranker_{model_name}",
            max_length=max_length if max_length is not None else int(cfg.get("max_length", 512)),
            batch_size=batch_size if batch_size is not None else int(cfg.get("batch_size", 8)),
        )


class VietnameseReranker(HFSequenceClassificationReranker):
    """Reranker sử dụng mô hình thanhtantran/Vietnamese_Reranker."""

    def __init__(
        self,
        model_name: str = "weights/Vietnamese_Reranker",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        super().__init__(
            model_path=model_name,
            display_name="Vietnamese_Reranker",
            reranker_meta_name="Vietnamese_Reranker",
            max_length=max_length,
            batch_size=batch_size,
        )


class PhoRankerReranker(HFSequenceClassificationReranker):
    """Reranker sử dụng mô hình itdainb/PhoRanker."""

    def __init__(
        self,
        model_name: str = "weights/PhoRanker",
        max_length: int = 256,
        batch_size: int = 16,
    ):
        super().__init__(
            model_path=model_name,
            display_name="PhoRanker",
            reranker_meta_name="PhoRanker",
            max_length=max_length,
            batch_size=batch_size,
        )


class VietnameseRerankerGGUF(BaseReranker):
    """Reranker sử dụng Vietnamese_Reranker định dạng GGUF qua llama-cpp-python."""

    def __init__(
        self,
        model_path: str = "weights/Vietnamese_Reranker_GGUF/Vietnamese_Reranker_F16_GGUF.gguf",
        max_length: int = 2048,
        batch_size: int = 16,
    ):
        self.model_name = "Vietnamese_Reranker_F16_GGUF"
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self._llm = None
        self._fallback = HeuristicReranker()

        self._init_model()

    def _init_model(self) -> None:
        try:
            import llama_cpp  # type: ignore
            from llama_cpp import Llama  # type: ignore

            path_obj = Path(self.model_path)
            target = str(path_obj.resolve()) if path_obj.exists() else self.model_path

            logger.info(f"Đang tải Vietnamese_Reranker_GGUF từ '{target}'...")
            self._llm = Llama(
                model_path=target,
                embedding=True,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
                n_ctx=self.max_length,
                n_batch=512,
                n_threads=12,
                verbose=False,
            )
            logger.info(f"Đã khởi tạo thành công Vietnamese_Reranker_GGUF từ '{target}'.")
        except Exception as exc:
            logger.warning(f"Không thể khởi tạo Vietnamese_Reranker_GGUF ({exc}). Fallback HeuristicReranker.")
            self._llm = None

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        if not candidates:
            return []
        if self._llm is None:
            return self._fallback.rerank(query, candidates, top_n=top_n)

        try:
            import math

            inputs = [f"{query}</s></s>{c.text[:600]}" for c in candidates]
            all_scores: List[float] = []

            for i in range(0, len(inputs), self.batch_size):
                batch_inputs = inputs[i : i + self.batch_size]
                res = self._llm.create_embedding(batch_inputs)
                for d in res["data"]:
                    all_scores.append(float(d["embedding"][0]))

            ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = "Vietnamese_Reranker_F16_GGUF"
                try:
                    prob = 1.0 / (1.0 + math.exp(-score))
                except OverflowError:
                    prob = 0.0 if score < 0 else 1.0
                meta["rerank_score"] = round(prob, 4)
                results.append(
                    RetrievedContext(
                        chunk_id=cand.chunk_id,
                        text=cand.text,
                        similarity_score=round(prob, 4),
                        rank=rank,
                        metadata=meta,
                    )
                )

            logger.info(f"Vietnamese_Reranker_GGUF đã sắp xếp {len(candidates)} ứng viên -> {len(results)} ranked chunks.")
            return results
        except Exception as err:
            logger.warning(f"Lỗi khi thực thi Vietnamese_Reranker_GGUF ({err}). Fallback HeuristicReranker.")
            return self._fallback.rerank(query, candidates, top_n=top_n)

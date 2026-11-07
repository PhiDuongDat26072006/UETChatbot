"""
src/retrieval/reranker.py - Phân hệ Reranking (Tái xếp hạng ngữ nghĩa sâu).
Tiếp nhận ~100 candidate chunks từ Hybrid Search, áp dụng mô hình chấm điểm chéo (Cross-Attention)
để chọn ra 20 ranked chunks liên quan nhất cho câu hỏi.
Hỗ trợ nhiều Backend: FlashRank (ONNX), Cross-Encoder, và Heuristic Cross-Score Fallback (Zero-Dependency).
"""

from __future__ import annotations
import re
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.base import RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("reranker")


class BaseReranker(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các mô hình Reranker."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        """
        Tái sắp xếp danh sách candidate chunks dựa trên câu hỏi người dùng.
        :param query: Câu hỏi người dùng
        :param candidates: Danh sách candidate chunks (~100 chunks)
        :param top_n: Số lượng chunk giữ lại sau khi rerank (mặc định 20)
        :return: Danh sách 20 ranked chunks có điểm số tương quan cao nhất
        """
        pass


class HeuristicReranker(BaseReranker):
    """
    Reranker dựa trên thuật toán tương quan đặc trưng chéo (Cross-Feature Re-scoring).
    - Hoạt động 100% bằng Pure Python & Thuật toán, không cần GPU, không lo xung đột PyTorch.
    - Đánh giá tổng hợp:
      1. Khớp cụm từ chính xác (Exact phrase match bonus)
      2. Mức độ bao phủ từ khóa câu hỏi trong đoạn trích (Query coverage ratio)
      3. Trọng số vị trí xuất hiện (Từ khóa nằm ở câu đầu/tiêu đề được ưu tiên hơn)
      4. Kết hợp điều hòa với điểm số tương đồng ban đầu từ Hybrid Search.
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

            # 1. Khớp cụm từ chính xác
            phrase_bonus = 0.25 if clean_query in text_lower or (title_lower and clean_query in title_lower) else 0.0

            # 2. Mức độ bao phủ từ khóa (Coverage)
            matched_words = [w for w in query_words if w in text_lower or (title_lower and w in title_lower)]
            coverage_score = len(matched_words) / query_len

            # 3. Trọng số vị trí (xuất hiện trong 200 ký tự đầu hoặc trong title)
            early_text = text_lower[:200]
            early_matches = sum(1 for w in query_words if w in early_text or (title_lower and w in title_lower))
            position_score = (early_matches / query_len) * 0.15

            # 4. Điểm hybrid ban đầu làm nền tảng
            base_similarity = candidate.similarity_score

            # Tổng hợp điểm rerank
            rerank_score = (0.45 * base_similarity) + (0.30 * coverage_score) + (0.15 * phrase_bonus) + (0.10 * position_score)
            rerank_score = round(min(0.99, max(0.05, rerank_score)), 4)

            meta["reranker_name"] = "heuristic_cross_scorer"
            meta["rerank_score"] = rerank_score
            meta["original_rank"] = candidate.rank

            scored_items.append({
                "context": candidate,
                "score": rerank_score,
                "meta": meta,
            })

        # Sắp xếp giảm dần theo điểm rerank
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        top_ranked = scored_items[:top_n]

        results: List[RetrievedContext] = []
        for rank, item in enumerate(top_ranked, 1):
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


class FlashRankReranker(BaseReranker):
    """
    Reranker dựa trên FlashRank (Mô hình Cross-Encoder chạy trên nền ONNX Runtime).
    Cực kỳ nhẹ, tốc độ cao (10-20ms trên CPU), không phụ thuộc vào PyTorch.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._ranker = None
        self._fallback = HeuristicReranker()

        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=model_name)
            logger.info(f"Đã khởi tạo thành công FlashRankReranker với model '{model_name}'.")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo FlashRank ({e}). Chuyển sang HeuristicReranker dự phòng.")

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
            from flashrank import RerankRequest
            passages = [{"id": c.chunk_id, "text": c.text, "meta": c.metadata} for c in candidates]
            rerank_req = RerankRequest(query=query, passages=passages)
            ranked_results = self._ranker.rerank(rerank_req)

            results: List[RetrievedContext] = []
            cand_map = {c.chunk_id: c for c in candidates}

            for rank, r in enumerate(ranked_results[:top_n], 1):
                cid = str(r.get("id"))
                orig = cand_map.get(cid)
                if orig:
                    meta = dict(orig.metadata) if orig.metadata else {}
                    meta["reranker_name"] = f"flashrank_{self.model_name}"
                    meta["rerank_score"] = round(float(r.get("score", 0.0)), 4)
                    results.append(
                        RetrievedContext(
                            chunk_id=cid,
                            text=orig.text,
                            similarity_score=round(float(r.get("score", 0.0)), 4),
                            rank=rank,
                            metadata=meta,
                        )
                    )
            return results
        except Exception as err:
            logger.warning(f"Lỗi khi thực thi FlashRank: {err}. Chuyển sang fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker sử dụng thư viện sentence-transformers / transformers (Cross-Encoder).
    Thích hợp khi môi trường có PyTorch hoàn chỉnh hoặc mô hình tiếng Việt chuyên sâu (BGE-Reranker).
    """

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
        except Exception as e:
            logger.warning(f"Không thể khởi tạo CrossEncoder ({e}). Chuyển sang HeuristicReranker dự phòng.")

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

            ranked = list(zip(candidates, scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

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


class JinaONNXReranker(BaseReranker):
    """
    Reranker sử dụng Jina Reranker V2 Multilingual phiên bản ONNX Runtime.
    Tối ưu hóa đa ngôn ngữ (tiếng Việt xuất sắc), hỗ trợ context dài tới 1024 tokens,
    chạy trực tiếp trên ONNX Runtime CPU với tốc độ cao và tiêu tốn ít bộ nhớ.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        onnx_filename: str = "onnx/model.onnx",
        max_length: int = 1024,
    ):
        if "/" not in model_name and not Path(model_name).exists():
            repo_id = f"jinaai/{model_name}"
        else:
            repo_id = model_name

        self.model_name = repo_id
        self.onnx_filename = onnx_filename
        self.max_length = max_length
        self._session = None
        self._tokenizer = None
        self._fallback = HeuristicReranker()

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download

            logger.info(f"Đang tải Jina ONNX Reranker: repo='{repo_id}', file='{onnx_filename}'...")

            if Path(onnx_filename).is_file():
                onnx_path = str(Path(onnx_filename).resolve())
            elif Path(repo_id).is_file():
                onnx_path = str(Path(repo_id).resolve())
            else:
                rel_filename = onnx_filename if onnx_filename.startswith("onnx/") else f"onnx/{onnx_filename}"
                onnx_path = hf_hub_download(repo_id=repo_id, filename=rel_filename)

            self._tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            self._session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"Đã tải thành công Jina ONNX Reranker từ '{onnx_path}'.")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Jina ONNX Reranker ({e}). Chuyển sang HeuristicReranker dự phòng.")
            self._session = None
            self._tokenizer = None

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_n: int = 20,
    ) -> List[RetrievedContext]:
        if not candidates:
            return []

        if self._session is None or self._tokenizer is None:
            return self._fallback.rerank(query, candidates, top_n=top_n)

        try:
            import numpy as np

            batch_size = 32
            all_prob_scores: List[float] = []

            for i in range(0, len(candidates), batch_size):
                batch_cands = candidates[i : i + batch_size]
                sentence_pairs = [[query, c.text] for c in batch_cands]
                inputs = self._tokenizer(
                    sentence_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="np",
                )

                ort_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }

                outputs = self._session.run(None, ort_inputs)
                raw_logits = outputs[0].squeeze(-1)
                if raw_logits.ndim == 0:
                    raw_logits = np.array([raw_logits.item()])
                prob_scores = 1.0 / (1.0 + np.exp(-raw_logits.astype(np.float64)))
                all_prob_scores.extend(prob_scores.tolist())

            ranked = list(zip(candidates, all_prob_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = f"jina_onnx_{self.model_name}"
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
            logger.info(f"Jina ONNX Reranker đã sắp xếp {len(candidates)} ứng viên -> {len(results)} ranked chunks.")
            return results
        except Exception as err:
            logger.warning(f"Lỗi chạy Jina ONNX Reranker: {err}. Sử dụng heuristic fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


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

        # Ưu tiên Jina ONNX nếu được chỉ định hoặc model_name chứa 'jina'
        if rtype in ("onnx", "jina", "jina_onnx") or "jina" in model_str:
            return JinaONNXReranker(
                model_name=model_name or "jinaai/jina-reranker-v2-base-multilingual",
                onnx_filename=onnx_filename or "onnx/model.onnx",
            )

        if rtype == "flashrank":
            return FlashRankReranker(model_name=model_name or "ms-marco-MiniLM-L-12-v2")

        if rtype == "cross_encoder":
            return CrossEncoderReranker(model_name=model_name or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

        # Chế độ "auto": Tự động phát hiện backend an toàn nhất
        try:
            import onnxruntime  # noqa: F401
            return JinaONNXReranker(
                model_name=model_name or "jinaai/jina-reranker-v2-base-multilingual",
                onnx_filename=onnx_filename or "onnx/model.onnx",
            )
        except Exception:
            pass

        try:
            import flashrank  # noqa: F401
            return FlashRankReranker(model_name=model_name or "ms-marco-MiniLM-L-12-v2")
        except ImportError:
            pass

        try:
            import sentence_transformers  # noqa: F401
            return CrossEncoderReranker(model_name=model_name or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        except Exception:
            pass

        # Mặc định an toàn 100%: HeuristicReranker
        logger.info("Sử dụng HeuristicReranker chuẩn cho quy trình Reranking.")
        return HeuristicReranker()

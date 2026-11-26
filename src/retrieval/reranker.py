"""
src/retrieval/reranker.py - Phân hệ Reranking (Tái xếp hạng ngữ nghĩa sâu).
Tiếp nhận ~100 candidate chunks từ Hybrid Search, áp dụng mô hình chấm điểm chéo (Cross-Attention)
để chọn ra 20 ranked chunks liên quan nhất cho câu hỏi.
Hỗ trợ nhiều Backend: FlashRank (ONNX), Cross-Encoder, và Heuristic Cross-Score Fallback (Zero-Dependency).
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.base import RetrievedContext
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("reranker")

_INLINE_HEADING_RE = re.compile(r"(?:^|(?<=\s)|(?<=\|))#{1,6}\s+", re.MULTILINE)
_HTML_BR_SEMICOLON_RE = re.compile(r"\s+;\s+")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _clean_text_for_reranker(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Làm sạch nhiễu định dạng Markdown/HTML trước khi đưa vào Tokenizer của Cross-Encoder (Jina / XLM-RoBERTa):
      1. Khử dòng [META: ...] bị lặp nguyên văn tiêu đề tài liệu [Title] ngay phía trên.
      2. Xóa dấu '**' và '__' bọc sát chữ cái đầu từ để ngăn lỗi vỡ vụn subword (Token Shattering)
         trong SentencePiece (ví dụ '**Công' -> ['▁**', 'C', 'ông'], '**Điểm' -> ['▁**', 'Đ', 'i', 'ể', 'm']).
      3. Xóa tiền tố heading '#### ' nhúng trong từng ô bảng và dấu ' ; ' sinh ra từ thẻ <br> HTML.
      4. Đính kèm tối đa 2 cạnh liên kết đồ thị ngắn (graph_triples) cho chunk quy chế phi bảng biểu để
         bảo vệ các chunk bắc cầu đa bước khi câu hỏi sinh viên dùng khẩu ngữ/tiếng lóng.
    Chỉ áp dụng trên bản sao chấm điểm của Reranker; văn bản gốc trả về cho LLM/UI giữ nguyên 100%.
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
    Giúp bảo vệ các ứng viên khớp từ khóa chính xác cao ở BM25/RRF không bị mất dấu hoàn toàn
    trước thiên kiến cấu trúc văn xuôi vs. bảng biểu của mô hình Cross-Encoder.
    """
    w = max(0.0, min(1.0, float(rerank_weight)))
    if w >= 1.0 or not candidates:
        return [float(s) for s in rerank_probs]

    priors = [max(0.0, float(c.similarity_score or 0.0)) for c in candidates]
    max_prior = max(priors) if priors else 0.0
    if max_prior <= 0.0:
        return [float(s) for s in rerank_probs]

    # Nếu prior vượt quá 1.0 (ví dụ truyền trực tiếp kết quả BM25 thô), chuẩn hóa về (0, 1]
    if max_prior > 1.0:
        priors = [p / max_prior for p in priors]

    return [(w * float(r_prob)) + ((1.0 - w) * p_score) for r_prob, p_score in zip(rerank_probs, priors)]


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
            from flashrank import Ranker  # type: ignore
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
            from flashrank import RerankRequest  # type: ignore
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

    @staticmethod
    def _parse_bool_flag(val: Any) -> Optional[bool]:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("1", "true", "yes", "on", "gpu", "cuda"):
                return True
            if low in ("0", "false", "no", "off", "cpu"):
                return False
        return None

    @classmethod
    def resolve_providers(
        cls,
        use_gpu: Optional[bool] = None,
        providers: Optional[List[str]] = None,
        cfg_providers: Optional[List[str]] = None,
        cfg_use_gpu: Optional[bool] = None,
    ) -> tuple[bool, List[str]]:
        """
        Xác định chế độ GPU/CPU và danh sách ONNX Execution Providers:
        Thứ tự ưu tiên: tham số `use_gpu` -> biến môi trường `RERANKER_USE_GPU` -> config -> mặc định True.
        - Khi use_gpu=False: ép chỉ dùng `["CPUExecutionProvider"]`.
        - Khi use_gpu=True: ưu tiên `CUDAExecutionProvider` trước `CPUExecutionProvider`.
        """
        resolved_gpu = cls._parse_bool_flag(use_gpu)
        if resolved_gpu is None:
            env_gpu = os.getenv("RERANKER_USE_GPU")
            if env_gpu is not None and env_gpu.strip() != "":
                resolved_gpu = cls._parse_bool_flag(env_gpu)
        if resolved_gpu is None:
            resolved_gpu = cls._parse_bool_flag(cfg_use_gpu)
        if resolved_gpu is None:
            base_provs = list(providers or cfg_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
            resolved_gpu = any("CUDA" in str(p) for p in base_provs)

        if not resolved_gpu:
            return False, ["CPUExecutionProvider"]

        req = list(providers or cfg_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        if "CUDAExecutionProvider" not in req:
            req.insert(0, "CUDAExecutionProvider")
        if "CPUExecutionProvider" not in req:
            req.append("CPUExecutionProvider")
        return True, req

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        onnx_filename: str = "onnx/model.onnx",
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        rerank_weight: float = 1.0,
        providers: Optional[List[str]] = None,
        use_gpu: Optional[bool] = None,
    ):
        try:
            full_cfg = load_yaml_config()
            cfg = full_cfg.get("agent", {}).get("reranker", {})
            ret_cfg = full_cfg.get("retrieval", {})
        except Exception:
            cfg = {}
            ret_cfg = {}

        if "/" not in model_name and not Path(model_name).exists():
            repo_id = f"jinaai/{model_name}"
        else:
            repo_id = model_name

        self.model_name = repo_id
        self.onnx_filename = onnx_filename
        self.max_length = int(max_length) if max_length is not None else int(cfg.get("max_length", 512))
        self.batch_size = max(1, int(batch_size) if batch_size is not None else int(cfg.get("batch_size", 8)))
        self.rerank_weight = float(rerank_weight)
        cfg_use_gpu = ret_cfg.get("reranker_use_gpu", cfg.get("use_gpu"))
        self.use_gpu, req_provs = self.resolve_providers(
            use_gpu=use_gpu,
            providers=providers,
            cfg_providers=cfg.get("providers"),
            cfg_use_gpu=cfg_use_gpu,
        )
        self._session = None
        self._tokenizer = None
        self._active_providers: List[str] = []
        self._fallback = HeuristicReranker()

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download

            logger.info(
                f"Đang tải Jina ONNX Reranker: repo='{repo_id}', file='{onnx_filename}', "
                f"use_gpu={self.use_gpu}, batch_size={self.batch_size}, max_length={self.max_length}..."
            )

            if Path(onnx_filename).is_file():
                onnx_path = str(Path(onnx_filename).resolve())
            elif Path(repo_id).is_file():
                onnx_path = str(Path(repo_id).resolve())
            else:
                rel_filename = onnx_filename if onnx_filename.startswith("onnx/") else f"onnx/{onnx_filename}"
                onnx_path = hf_hub_download(repo_id=repo_id, filename=rel_filename)

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                )
            except TypeError:
                self._tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

            # Chỉ preload CUDA/cuDNN DLLs khi bật use_gpu=True
            wants_cuda = self.use_gpu and any("CUDA" in str(p) for p in req_provs)
            if wants_cuda:
                try:
                    import torch  # noqa: F401 - preload CUDA runtime shared libraries on Colab/Linux
                except Exception:
                    pass
                if hasattr(ort, "preload_dlls"):
                    try:
                        ort.preload_dlls(cuda=True, cudnn=True)
                    except Exception as dll_err:
                        logger.debug(f"Bỏ qua preload_dlls cho ONNX CUDA: {dll_err}")

            available = set(ort.get_available_providers())
            if wants_cuda and "CUDAExecutionProvider" not in available:
                logger.warning(
                    "reranker_use_gpu=True nhưng môi trường Python hiện tại chưa có CUDAExecutionProvider "
                    f"(available={sorted(available)}). Tự động fallback sang CPUExecutionProvider. "
                    "Hãy chạy bằng môi trường cài onnxruntime-gpu (vd .venv) để tăng tốc GPU."
                )
            effective_provs = [p for p in req_provs if p in available]
            if "CPUExecutionProvider" not in effective_provs:
                effective_provs.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, min(10, (os.cpu_count() or 4) - 2))
            sess_options.inter_op_num_threads = 1
            if effective_provs == ["CPUExecutionProvider"]:
                sess_options.enable_cpu_mem_arena = False

            try:
                self._session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=effective_provs,
                )
            except Exception as cuda_err:
                if effective_provs != ["CPUExecutionProvider"]:
                    logger.warning(f"CUDAExecutionProvider lỗi ({cuda_err}); chuyển sang CPUExecutionProvider.")
                    sess_options.enable_cpu_mem_arena = False
                    self._session = ort.InferenceSession(
                        onnx_path,
                        sess_options=sess_options,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    raise

            self._active_providers = list(self._session.get_providers())
            logger.info(
                f"Đã tải thành công Jina ONNX Reranker từ '{onnx_path}' "
                f"(providers={self._active_providers}, max_length={self.max_length}, "
                f"batch_size={self.batch_size}, threads={sess_options.intra_op_num_threads})."
            )
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Jina ONNX Reranker ({e}). Sẽ sử dụng RRF fallback khi rerank.")
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
            # D6: Fallback RRF có degraded=True thay vì HeuristicReranker
            logger.info(f"[D6] Reranker không sẵn sàng. Fallback sang thứ tự RRF (degraded=True).")
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

        try:
            import numpy as np

            all_prob_scores: List[float] = [0.0] * len(candidates)
            clean_query = _clean_text_for_reranker(query)
            cleaned_texts = [_clean_text_for_reranker(c.text, c.metadata) for c in candidates]
            # Sắp xếp theo độ dài văn bản trước khi chia micro-batch để giảm padding dư thừa
            sorted_indices = sorted(range(len(candidates)), key=lambda idx: len(cleaned_texts[idx]))
            active_provs = getattr(self, "_active_providers", [])
            eff_batch_size = (
                max(self.batch_size, 32)
                if ("CUDAExecutionProvider" in active_provs or "DmlExecutionProvider" in active_provs)
                else self.batch_size
            )

            for i in range(0, len(sorted_indices), eff_batch_size):
                batch_indices = sorted_indices[i : i + eff_batch_size]
                sentence_pairs = [[clean_query, cleaned_texts[idx]] for idx in batch_indices]
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
                for idx, p_score in zip(batch_indices, prob_scores.tolist()):
                    all_prob_scores[idx] = float(p_score)

            blended_scores = _blend_rerank_and_prior(
                candidates=candidates,
                rerank_probs=all_prob_scores,
                rerank_weight=getattr(self, "rerank_weight", 1.0),
            )

            ranked = list(zip(candidates, blended_scores, all_prob_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            results: List[RetrievedContext] = []
            for rank, (cand, final_score, raw_prob) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = f"jina_onnx_{self.model_name}"
                meta["raw_rerank_score"] = round(float(raw_prob), 4)
                meta["rerank_score"] = round(float(final_score), 4)
                meta["degraded"] = False
                meta["rerank_backend"] = "jina_onnx"
                results.append(
                    RetrievedContext(
                        chunk_id=cand.chunk_id,
                        text=cand.text,
                        similarity_score=round(float(final_score), 4),
                        rank=rank,
                        metadata=meta,
                    )
                )
            logger.info(f"Jina ONNX Reranker đã sắp xếp {len(candidates)} ứng viên -> {len(results)} ranked chunks.")
            return results
        except Exception as err:
            logger.warning(f"[D6] Lỗi chạy Jina ONNX Reranker: {err}. Giữ thứ tự RRF (degraded=True).")
            fallback_results = []
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



class ViRankerReranker(BaseReranker):
    """
    Reranker sử dụng mô hình ViRanker (namdp-ptit/ViRanker) - Cross-Encoder tối ưu riêng cho tiếng Việt.
    Dựa trên kiến trúc XLMRoberta (BAAI/bge-m3) được huấn luyện tinh chỉnh (fine-tuned) trên bộ dữ liệu
    MS MARCO tiếng Việt. Cho chất lượng tái xếp hạng ngữ nghĩa tiếng Việt vượt trội.
    """

    def __init__(
        self,
        model_name: str = "weights/ViRanker",
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        try:
            from src.utils.helpers import load_yaml_config
            cfg = load_yaml_config().get("agent", {}).get("reranker", {})
        except Exception:
            cfg = {}

        self.model_name = model_name
        self.max_length = max_length if max_length is not None else int(cfg.get("max_length", 512))
        self.batch_size = batch_size if batch_size is not None else int(cfg.get("batch_size", 8))
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._fallback = HeuristicReranker()

        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_path = Path(self.model_name)
            target = str(model_path.resolve()) if model_path.exists() else self.model_name

            logger.info(f"Đang tải ViRanker từ '{target}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(target)
            self._model = AutoModelForSequenceClassification.from_pretrained(target)
            self._model.eval()

            if torch.cuda.is_available():
                self._device = "cuda"
                self._model.to("cuda")
                logger.info("ViRanker đang sử dụng thiết bị: CUDA (GPU).")
            else:
                self._device = "cpu"
                self._model.to("cpu")
                logger.info("ViRanker đang sử dụng thiết bị: CPU.")

            logger.info(f"Đã khởi tạo thành công ViRanker từ '{target}'.")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo ViRanker ({e}). Sẽ dùng HeuristicReranker dự phòng.")
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

            # Length bucketing: gom nhóm chunk theo độ dài text để giảm padding thừa trên CPU
            indexed_cands = list(enumerate(candidates))
            indexed_cands.sort(key=lambda x: len(x[1].text))
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

            ranked = list(zip(candidates, all_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = f"viranker_{self.model_name}"
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

            logger.info(f"ViRanker đã sắp xếp {len(candidates)} ứng viên -> {len(results)} ranked chunks.")
            return results
        except Exception as err:
            logger.warning(f"Lỗi khi thực thi ViRanker ({err}). Chuyển sang HeuristicReranker fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class VietnameseReranker(BaseReranker):
    """
    Reranker sử dụng mô hình thanhtantran/Vietnamese_Reranker (AITeamVN/Vietnamese_Reranker).
    Fine-tuned từ BAAI/bge-reranker-v2-m3 trên 1.1M cặp dữ liệu tiếng Việt.
    Kiến trúc XLMRobertaForSequenceClassification, tái xếp hạng ngữ nghĩa chuẩn xác.
    """

    def __init__(
        self,
        model_name: str = "weights/Vietnamese_Reranker",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_name = "Vietnamese_Reranker"
        self.model_path = model_name
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

            model_path = Path(self.model_path)
            target = str(model_path.resolve()) if model_path.exists() else self.model_path

            logger.info(f"Đang tải Vietnamese_Reranker từ '{target}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(target)
            self._model = AutoModelForSequenceClassification.from_pretrained(target)
            self._model.eval()

            if torch.cuda.is_available():
                self._device = "cuda"
                self._model.to("cuda")
                logger.info("Vietnamese_Reranker đang sử dụng thiết bị: CUDA (GPU).")
            else:
                self._device = "cpu"
                self._model.to("cpu")
                logger.info("Vietnamese_Reranker đang sử dụng thiết bị: CPU.")

            logger.info(f"Đã khởi tạo thành công Vietnamese_Reranker từ '{target}'.")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Vietnamese_Reranker ({e}). Sẽ dùng HeuristicReranker dự phòng.")
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

            # Length bucketing: gom nhóm chunk theo độ dài text để giảm padding thừa trên CPU
            indexed_cands = list(enumerate(candidates))
            indexed_cands.sort(key=lambda x: len(x[1].text))
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

            ranked = list(zip(candidates, all_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = "Vietnamese_Reranker"
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

            logger.info(f"Vietnamese_Reranker đã sắp xếp {len(candidates)} ứng viên -> {len(results)} ranked chunks.")
            return results
        except Exception as err:
            logger.warning(f"Lỗi khi thực thi Vietnamese_Reranker ({err}). Chuyển sang HeuristicReranker fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class PhoRankerReranker(BaseReranker):
    """
    Reranker sử dụng mô hình itdainb/PhoRanker (vinai/phobert-base-v2 làm base).
    Kiến trúc RobertaForSequenceClassification, max context 256 tokens, tối ưu riêng tiếng Việt.
    """

    def __init__(
        self,
        model_name: str = "weights/PhoRanker",
        max_length: int = 256,
        batch_size: int = 16,
    ):
        self.model_name = "PhoRanker"
        self.model_path = model_name
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

            model_path = Path(self.model_path)
            target = str(model_path.resolve()) if model_path.exists() else self.model_path

            logger.info(f"Đang tải PhoRanker từ '{target}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(target)
            self._model = AutoModelForSequenceClassification.from_pretrained(target)
            self._model.eval()

            if torch.cuda.is_available():
                self._device = "cuda"
                self._model.to("cuda")
                logger.info("PhoRanker đang sử dụng thiết bị: CUDA (GPU).")
            else:
                self._device = "cpu"
                self._model.to("cpu")
                logger.info("PhoRanker đang sử dụng thiết bị: CPU.")

            logger.info(f"Đã khởi tạo thành công PhoRanker từ '{target}'.")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo PhoRanker ({e}). Sẽ dùng HeuristicReranker dự phòng.")
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

            # Length bucketing: gom nhóm chunk theo độ dài text để giảm padding thừa trên CPU
            indexed_cands = list(enumerate(candidates))
            indexed_cands.sort(key=lambda x: len(x[1].text))
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

            ranked = list(zip(candidates, all_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = "PhoRanker"
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

            logger.info(f"PhoRanker đã sắp xếp {len(candidates)} ứng viên -> {len(results)} ranked chunks.")
            return results
        except Exception as err:
            logger.warning(f"Lỗi khi thực thi PhoRanker ({err}). Chuyển sang HeuristicReranker fallback.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class VietnameseRerankerGGUF(BaseReranker):
    """
    Reranker sử dụng Vietnamese_Reranker định dạng GGUF (AnhNguyen6688/Vietnamese_Reranker_F16_GGUF).
    Chạy trực tiếp qua llama-cpp-python với chế độ rank pooling (LLAMA_POOLING_TYPE_RANK).
    Tối ưu hóa bộ nhớ RAM, tốc độ cao trên CPU/GPU.
    """

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
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Vietnamese_Reranker_GGUF ({e}). Fallback HeuristicReranker.")
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

            # Giới hạn context text tối đa 600 ký tự đầu để bảo đảm an toàn bộ nhớ và tốc độ
            inputs = [f"{query}</s></s>{c.text[:600]}" for c in candidates]
            all_scores: List[float] = []

            for i in range(0, len(inputs), self.batch_size):
                batch_inputs = inputs[i : i + self.batch_size]
                res = self._llm.create_embedding(batch_inputs)
                for d in res["data"]:
                    all_scores.append(float(d["embedding"][0]))

            ranked = list(zip(candidates, all_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            results: List[RetrievedContext] = []
            for rank, (cand, score) in enumerate(ranked[:top_n], 1):
                meta = dict(cand.metadata) if cand.metadata else {}
                meta["reranker_name"] = "Vietnamese_Reranker_F16_GGUF"
                # Áp dụng sigmoid đưa score về [0, 1] cho hiển thị chuẩn xác
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

        # 1. GGUF Reranker (AnhNguyen6688/Vietnamese_Reranker_F16_GGUF)
        if rtype in ("vietnamese_reranker_gguf", "gguf") or "gguf" in model_str:
            return VietnameseRerankerGGUF(
                model_path=model_name or "weights/Vietnamese_Reranker_GGUF/Vietnamese_Reranker_F16_GGUF.gguf",
            )

        # 2. Vietnamese_Reranker (thanhtantran/Vietnamese_Reranker)
        if rtype in ("vietnamese_reranker", "vietnamese-reranker", "vietnamesereranker") or "vietnamese_reranker" in model_str:
            return VietnameseReranker(
                model_name=model_name or "weights/Vietnamese_Reranker",
            )

        # 3. PhoRanker (itdainb/PhoRanker)
        if rtype in ("phoranker", "pho_ranker") or "phoranker" in model_str:
            return PhoRankerReranker(
                model_name=model_name or "weights/PhoRanker",
            )

        # 4. ViRanker (namdp-ptit/ViRanker)
        if rtype in ("viranker", "vi_ranker") or "viranker" in model_str:
            return ViRankerReranker(
                model_name=model_name or "weights/ViRanker",
            )


        # Ưu tiên Jina ONNX nếu được chỉ định hoặc model_name chứa 'jina'
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

        # Chế độ "auto": Tự động phát hiện backend an toàn nhất
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

        # Mặc định an toàn 100%: HeuristicReranker
        logger.info("Sử dụng HeuristicReranker chuẩn cho quy trình Reranking.")
        return HeuristicReranker()

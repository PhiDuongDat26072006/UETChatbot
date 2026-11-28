"""Jina Reranker V2 Multilingual ONNX Runtime backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from src.base import RetrievedContext
from src.retrieval.rerankers.base import (
    BaseReranker,
    HeuristicReranker,
    _blend_rerank_and_prior,
    _clean_text_for_reranker,
    rrf_degraded_fallback,
)
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("reranker")


class JinaONNXReranker(BaseReranker):
    """
    Reranker sử dụng Jina Reranker V2 Multilingual phiên bản ONNX Runtime.
    Hỗ trợ CUDAExecutionProvider và fallback CPUExecutionProvider.
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
        """Xác định chế độ GPU/CPU và danh sách ONNX Execution Providers."""
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
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer

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

            wants_cuda = self.use_gpu and any("CUDA" in str(p) for p in req_provs)
            if wants_cuda:
                try:
                    import torch  # noqa: F401
                except Exception:
                    pass
                if os.name == "nt" and hasattr(os, "add_dll_directory"):
                    cuda_bin = os.path.join(os.environ.get("CUDA_PATH", ""), "bin")
                    if cuda_bin and os.path.isdir(cuda_bin):
                        try:
                            os.add_dll_directory(cuda_bin)
                        except OSError:
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
        except Exception as exc:
            logger.warning(f"Không thể khởi tạo Jina ONNX Reranker ({exc}). Sẽ sử dụng RRF fallback khi rerank.")
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
            logger.info("[D6] Reranker không sẵn sàng. Fallback sang thứ tự RRF (degraded=True).")
            return rrf_degraded_fallback(candidates, top_n)

        try:
            import numpy as np

            all_prob_scores: List[float] = [0.0] * len(candidates)
            clean_query = _clean_text_for_reranker(query)
            cleaned_texts = [_clean_text_for_reranker(c.text, c.metadata) for c in candidates]
            sorted_indices = sorted(range(len(candidates)), key=lambda idx: len(cleaned_texts[idx]))
            # Tôn trọng batch_size cấu hình (mặc định 8) trên cả GPU và CPU
            # để tránh tràn VRAM (Shared GPU Memory thrashing) và giữ tác dụng gom micro-batch theo độ dài.
            eff_batch_size = max(1, int(getattr(self, "batch_size", 8) or 8))

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
            return rrf_degraded_fallback(candidates, top_n)

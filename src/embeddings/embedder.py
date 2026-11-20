"""
src/embeddings/embedder.py - Phụ trách mô hình Vector Embedding (Task 3).
Thành viên phụ trách: Người quản lý mô hình Embedding.
Nhiệm vụ: Chuyển đổi văn bản thành các vector số thực 1024 chiều bằng mô hình BAAI/bge-m3.
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Đảm bảo root directory có trong sys.path để import từ src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import (
    BaseEmbeddingModel,
    DataChunk,
    EmbeddedVector,
    UserQuery,
    EmbeddedQueryVector,
)
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("embeddings")


class UETEmbedder(BaseEmbeddingModel):
    """
    Mô hình nhúng văn bản (Embedding Model) cho UET Chatbot.
    Chuẩn hóa sử dụng mô hình BAAI/bge-m3 (1024 chiều, tối ưu tiếng Việt và đa ngữ).
    Đồng bộ cấu hình trực tiếp từ config.yaml.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        raw_cfg = config or load_yaml_config()
        emb_cfg = raw_cfg.get("embedding", {}) if raw_cfg else {}

        self._model_name = (
            model_name
            or emb_cfg.get("model_name")
            or "weights/bge-m3"
        )
        self._dimension = (
            dimension
            or emb_cfg.get("dimension")
            or 1024
        )
        self._provider = str(emb_cfg.get("provider", "onnx")).lower()
        self._onnx_path = emb_cfg.get("onnx_path", "weights/bge-m3/onnx/model.onnx")

        self._st_model = None
        self._chroma_ef = None
        self._onnx_session = None
        self._tokenizer = None

        self._init_model()

    def _init_model(self) -> None:
        """Khởi tạo mô hình embedding tương ứng theo cấu hình (Ưu tiên ONNX Runtime FP32)."""
        # 1. Thử nạp mô hình ONNX Runtime nếu cấu hình là ONNX hoặc file ONNX tồn tại
        resolved_onnx = Path(self._onnx_path)
        if not resolved_onnx.is_absolute():
            resolved_onnx = BASE_DIR / self._onnx_path

        if self._provider in ("onnx", "bge-m3-onnx") or resolved_onnx.exists():
            if resolved_onnx.exists():
                try:
                    import onnxruntime as ort
                    from transformers import AutoTokenizer

                    tok_path = Path(self._model_name)
                    if not tok_path.is_absolute():
                        tok_path = BASE_DIR / self._model_name
                    if not tok_path.exists():
                        tok_path = BASE_DIR / "weights" / "bge-m3"

                    logger.info(f"Đang tải mô hình ONNX FP32 BGE-M3 từ '{resolved_onnx}'...")
                    self._tokenizer = AutoTokenizer.from_pretrained(str(tok_path.resolve()))

                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        try:
                            import torch  # noqa: F401
                        except Exception:
                            pass
                        if hasattr(ort, "preload_dlls"):
                            try:
                                ort.preload_dlls()
                            except Exception:
                                pass
                        try:
                            self._onnx_session = ort.InferenceSession(
                                str(resolved_onnx.resolve()),
                                sess_options=sess_options,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                            )
                        except Exception as cuda_err:
                            logger.warning(f"ONNX CUDAExecutionProvider lỗi ({cuda_err}); fallback sang CPUExecutionProvider.")
                            self._onnx_session = ort.InferenceSession(
                                str(resolved_onnx.resolve()),
                                sess_options=sess_options,
                                providers=["CPUExecutionProvider"],
                            )
                    else:
                        self._onnx_session = ort.InferenceSession(
                            str(resolved_onnx.resolve()),
                            sess_options=sess_options,
                            providers=["CPUExecutionProvider"],
                        )
                    out_names = [o.name for o in self._onnx_session.get_outputs()]
                    self._onnx_output_names = ["sentence_embedding"] if "sentence_embedding" in out_names else None
                    logger.info(
                        f"Đã khởi tạo thành công mô hình ONNX Embedding BGE-M3 "
                        f"({self._dimension} chiều, providers={self._onnx_session.get_providers()})."
                    )
                    return
                except Exception as e:
                    logger.warning(f"Không thể khởi tạo ONNX Embedding ({e}). Chuyển sang PyTorch SentenceTransformer.")
            else:
                logger.info(f"Chưa tìm thấy file ONNX tại '{resolved_onnx}'. Chuyển sang dùng SentenceTransformer PyTorch.")

        # 2. Sử dụng PyTorch SentenceTransformer
        if "bge" in self._model_name.lower() or self._dimension == 1024:
            try:
                from sentence_transformers import SentenceTransformer
                target_model = self._model_name
                local_dir = BASE_DIR / target_model
                if local_dir.exists():
                    target_model = str(local_dir.resolve())
                logger.info(f"Đang tải mô hình SentenceTransformer '{target_model}' ({self._dimension} chiều)...")
                self._st_model = SentenceTransformer(target_model)
                st_device = getattr(self._st_model, "device", "unknown")
                logger.info(
                    f"Đã khởi tạo thành công mô hình embedding '{self._model_name}' "
                    f"({self._dimension} chiều, device={st_device})."
                )
                return
            except Exception as e:
                logger.warning(
                    f"Chưa thể nạp SentenceTransformer '{self._model_name}': {e}. Sẽ dùng fallback vector."
                )

        # 3. Fallback hoặc mặc định cho all-MiniLM-L6-v2 (384 chiều)
        try:
            import chromadb.utils.embedding_functions as ef
            self._chroma_ef = ef.DefaultEmbeddingFunction()
            logger.info(f"Đã mở DefaultEmbeddingFunction của ChromaDB ({self._dimension} chiều).")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo DefaultEmbeddingFunction của Chroma: {e}.")

    @property
    def dimension(self) -> int:
        """Số chiều của vector embedding."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Tên mô hình embedding."""
        return self._model_name

    def _fallback_vector(self, text: str) -> List[float]:
        """Tạo vector dự phòng self._dimension chiều dựa trên hash văn bản khi gặp sự cố."""
        hash_digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self._dimension):
            byte_val = hash_digest[i % len(hash_digest)]
            vec.append((byte_val / 255.0) * 2 - 1)
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Nhúng danh sách chuỗi văn bản thành danh sách vector số thực.
        :param texts: Danh sách các câu hoặc đoạn văn bản cần nhúng
        :return: Danh sách vector (List[List[float]])
        """
        if not texts:
            return []

        logger.info(f"Đang nhúng {len(texts)} đoạn văn bản bằng mô hình '{self.model_name}' ({self.dimension} chiều)...")

        # 1. Sử dụng ONNX Runtime FP32 (bge-m3)
        if self._onnx_session is not None and self._tokenizer is not None:
            try:
                batch_size = 32
                all_vectors: List[List[float]] = []
                out_names = getattr(self, "_onnx_output_names", None)
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    encoded = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="np",
                    )
                    ort_inputs = {
                        "input_ids": encoded["input_ids"].astype(np.int64),
                        "attention_mask": encoded["attention_mask"].astype(np.int64),
                    }
                    if out_names:
                        ort_outputs = self._onnx_session.run(out_names, ort_inputs)
                        sentence_emb = ort_outputs[0]
                    else:
                        ort_outputs = self._onnx_session.run(None, ort_inputs)
                        sentence_emb = None
                        for out in ort_outputs:
                            if out.ndim == 2:
                                sentence_emb = out
                                break
                        if sentence_emb is None:
                            sentence_emb = ort_outputs[0][:, 0, :]

                    # Đảm bảo chuẩn hóa L2 vector
                    norms = np.linalg.norm(sentence_emb, ord=2, axis=-1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    sentence_emb = sentence_emb / norms

                    for vec in sentence_emb:
                        all_vectors.append(vec.tolist())
                return all_vectors
            except Exception as e:
                logger.error(f"Lỗi khi nhúng bằng ONNX Runtime: {e}. Fallback sang SentenceTransformer.")

        # 2. Sử dụng SentenceTransformer (bge-m3) PyTorch
        if self._st_model is not None:
            try:
                st_device = str(getattr(self._st_model, "device", "cpu")).lower()
                st_batch_size = 64 if st_device.startswith("cuda") else 32
                raw_vectors = self._st_model.encode(
                    texts,
                    batch_size=st_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in raw_vectors]
            except Exception as e:
                logger.error(f"Lỗi khi nhúng bằng SentenceTransformer '{self.model_name}': {e}.")

        # 2. Sử dụng Chroma Default (MiniLM 384)
        if self._chroma_ef is not None and self._dimension == 384:
            try:
                batch_size = 64
                all_vectors: List[List[float]] = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    raw_vectors = self._chroma_ef(batch)
                    all_vectors.extend([list(vec) for vec in raw_vectors])
                return all_vectors
            except Exception as e:
                logger.error(f"Lỗi khi nhúng bằng Chroma EF: {e}.")

        return [self._fallback_vector(t) for t in texts]
"""
src/retrieval/retriever.py - Phụ trách quy trình truy xuất ngữ cảnh (Retrieval Pipeline).
Thành viên phụ trách: Người làm Retrieval & Similarity Search.

Quy trình chuẩn hóa từ câu hỏi đến K chunks:
    User Query -> [Query Text & Query Embedding]
               -> [BM25 Search & Vector Search]
               -> [Hybrid Search Fusion - RRF: ~100 candidate chunks]
               -> [Reranker: 20 ranked chunks]
               -> [Top-K Selection: K chunks (mặc định K=5)]
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import (
    BaseEmbeddingModel,
    BaseVectorStore,
    DataChunk,
    RetrievedContext,
    UserQuery,
)
from src.indexing.sparse import BM25Index
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.mmr import MaximalMarginalRelevance
from src.graph.query_entity_extractor import QueryEntityExtractor
from src.retrieval.reranker import BaseReranker, HeuristicReranker, RerankerFactory
from src.graph.lightrag_retriever import LightRAGRetriever
from src.graph.thesaurus_utils import KNOWN_FACULTY_KEYS, detect_query_faculties
from src.retrieval.router import QueryRouter
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("retrieval")

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ENTITY_CODE_RE = re.compile(
    r"\b(?:\d{2,5}/[a-zđ]+(?:-[a-zđ0-9]+)+|(?:int|elt|ema|epn|pes|mat|phy|flf|cn|k)\d{1,4}[a-z]?)\b",
    re.IGNORECASE,
)


class UETRetriever:
    """
    Bộ điều phối truy xuất ngữ cảnh đa tầng (BM25 + Vector + GraphRAG -> RRF -> MMR -> Rerank).
    """

    # Mẫu câu hỏi tra cứu số liệu / bảng biểu theo năm (ưu tiên tối đa slot cho Vector chunks)
    _TABULAR_NUMERIC_LOOKUP_RE = re.compile(
        r"\b(20\d{2}|điểm chuẩn|điểm trúng tuyển|điểm sàn|ngưỡng đầu vào|chỉ tiêu|học phí|mã ngành|mã xét tuyển)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        config: Optional[Dict[str, Any]] = None,
        bm25_index: Optional[BM25Index] = None,
        graph_index: Optional[Any] = None,
        lightrag_retriever: Optional[LightRAGRetriever] = None,
        router: Optional[QueryRouter] = None,
    ):
        """
        Khởi tạo UETRetriever.
        :param embedding_model: Mô hình nhúng câu hỏi kế thừa BaseEmbeddingModel
        :param vector_store: Cơ sở dữ liệu vector kế thừa BaseVectorStore
        :param reranker: Mô hình Reranker (nếu None sẽ tự động khởi tạo qua RerankerFactory)
        :param config: Dict cấu hình tham số tùy chỉnh (hoặc tự đọc từ config.yaml)
        :param bm25_index: Chỉ mục đã tạo sẵn (dành cho test/in-memory); production tải từ file.
        :param graph_index: Chỉ mục GraphRAG (dành cho test/in-memory); production tải từ file.
        :param lightrag_retriever: Phân hệ truy xuất đồ thị tri thức LightRAGRetriever.
        :param router: Bộ điều phối thông minh QueryRouter.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        # Đọc cấu hình từ config.yaml nếu không truyền trực tiếp
        raw_config = config or load_yaml_config()
        retrieval_cfg = raw_config.get("retrieval", {})
        graph_cfg = raw_config.get("graph", {})
        paths_cfg = raw_config.get("paths", {})

        # Tham số theo chuẩn Pipeline (đảo mặc định ưu tiên BM25 + GraphRAG theo benchmark thực tế)
        self.bm25_top_k: int = int(retrieval_cfg.get("bm25_top_k", 60))
        self.vector_top_k: int = int(retrieval_cfg.get("vector_top_k", 60))
        self.candidate_pool_size: int = int(retrieval_cfg.get("candidate_pool_size", 60))
        self.rerank_top_n: int = int(retrieval_cfg.get("rerank_top_n", 20))
        self.default_top_k: int = int(retrieval_cfg.get("final_top_k", 10))
        self.fusion_method: str = str(retrieval_cfg.get("fusion_method", "rrf")).lower().strip()
        self.rrf_k: int = int(retrieval_cfg.get("rrf_k", 30))
        self.bm25_weight: float = float(retrieval_cfg.get("bm25_weight", 0.60))
        self.vector_weight: float = float(retrieval_cfg.get("vector_weight", 0.20))
        self.default_score_threshold: Optional[float] = retrieval_cfg.get("score_threshold", None)
        self.default_engine: str = retrieval_cfg.get("engine", "auto")

        # Cấu hình phân bổ slot trong chế độ Hybrid (LightRAG + Vector)
        self.hybrid_graph_ratio: float = retrieval_cfg.get("hybrid_graph_ratio", 0.2)
        self.hybrid_max_graph_k: int = retrieval_cfg.get("hybrid_max_graph_k", 2)

        # Cấu hình Nhánh Đồ thị Chính (LightRAG-First + LightRAGChunkBridge phụ trợ)
        self.enable_graph: bool = bool(retrieval_cfg.get("enable_graph", True))
        self.graph_top_k: int = int(retrieval_cfg.get("graph_top_k", 60))
        self.graph_weight: float = float(retrieval_cfg.get("graph_weight", 0.30))
        self.lightrag_hybrid_weight: float = float(
            retrieval_cfg.get("lightrag_hybrid_weight", max(self.graph_weight, 0.40))
        )
        self.graph_max_hops: int = int(retrieval_cfg.get("graph_max_hops", graph_cfg.get("max_hops", 2)))
        self.graph_seed_top_m: int = int(retrieval_cfg.get("graph_seed_top_m", graph_cfg.get("seed_top_m", 10)))
        self.graph_hop_decay: float = float(retrieval_cfg.get("graph_hop_decay", graph_cfg.get("hop_decay", 0.65)))
        graph_rel = paths_cfg.get("graph_index_file", "database/graph_index.pkl")
        self._graph_index_path = Path(graph_rel) if Path(graph_rel).is_absolute() else (BASE_DIR / graph_rel)

        # Cấu hình Maximal Marginal Relevance (MMR)
        self.enable_mmr: bool = bool(retrieval_cfg.get("enable_mmr", False))
        self.mmr_stage: str = str(retrieval_cfg.get("mmr_stage", "pre_rerank")).lower().strip()
        self.mmr_lambda: float = float(retrieval_cfg.get("mmr_lambda", 0.75))
        self.mmr_top_n: int = int(retrieval_cfg.get("mmr_top_n", 40))
        self.mmr_similarity_threshold: float = float(retrieval_cfg.get("mmr_similarity_threshold", 0.0))
        self.mmr_protect_tabular: bool = bool(retrieval_cfg.get("mmr_protect_tabular", True))

        # Cấu hình Reranker & Score Blending
        reranker_type = retrieval_cfg.get("reranker_type", "auto")
        reranker_model = retrieval_cfg.get("reranker_model", None)
        onnx_filename = retrieval_cfg.get("onnx_filename", None)
        self.rerank_max_length: int = int(retrieval_cfg.get("rerank_max_length", 512))
        self.rerank_batch_size: int = int(retrieval_cfg.get("rerank_batch_size", 8))
        self.rerank_blend_alpha: float = float(retrieval_cfg.get("rerank_blend_alpha", 0.65))
        self.enable_entity_anchor_boost: bool = bool(retrieval_cfg.get("enable_entity_anchor_boost", True))

        # Cấu hình Mở rộng ngữ cảnh Liền kề (Adjacent Chunk Expansion)
        self.enable_adjacent_expansion: bool = bool(retrieval_cfg.get("enable_adjacent_expansion", False))
        self.adjacent_expansion_top_k: int = int(retrieval_cfg.get("adjacent_expansion_top_k", 3))
        self.adjacent_window_size: int = int(retrieval_cfg.get("adjacent_window_size", 1))
        self.merge_contiguous_chunks: bool = bool(retrieval_cfg.get("merge_contiguous_chunks", True))

        # Chỉ nạp artifact từ ổ đĩa khi chạy với persistent VectorStore thật (không nạp trong unit test mock store).
        has_persistent_store = getattr(self.vector_store, "persist_dir", None) is not None

        self._effective_lightrag_cfg = (
            {"lightrag": {"enabled": False, "router": {"mode": "always_vector"}}}
            if (
                (not has_persistent_store or (config is not None and "lightrag" not in config))
                and lightrag_retriever is None
            )
            else raw_config
        )
        self._lightrag_retriever: Optional[LightRAGRetriever] = lightrag_retriever
        self.router = router or QueryRouter(config=self._effective_lightrag_cfg)
        lr_use_llm = bool(self._effective_lightrag_cfg.get("lightrag", {}).get("use_llm_query_extraction", True))
        self.query_entity_extractor: Optional[QueryEntityExtractor] = (
            getattr(lightrag_retriever, "query_extractor", None)
            or (QueryEntityExtractor(config=self._effective_lightrag_cfg) if (has_persistent_store and lr_use_llm) else None)
        )
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Chỉ sử dụng BM25Index từ src.indexing.sparse.
        # Cho phép dependency injection khi kiểm thử, không nạp file ổ cứng nếu vector_store là mock in-memory.
        self.bm25_index = bm25_index if bm25_index is not None else BM25Index()
        self.graph_index = graph_index
        self._adjacent_lookup: Optional[Dict[Tuple[str, int], DataChunk]] = None

        self.reranker = reranker or RerankerFactory.create(
            reranker_type=reranker_type,
            model_name=reranker_model,
            onnx_filename=onnx_filename,
            max_length=self.rerank_max_length,
            batch_size=self.rerank_batch_size,
        )

        if bm25_index is None and has_persistent_store:
            self._load_bm25_index()
        if graph_index is None and bm25_index is None and self.enable_graph and has_persistent_store:
            self._load_graph_index()

        # Gắn LightRAGChunkBridge (self.graph_index) vào LightRAGRetriever và khởi tạo sẵn làm Động cơ Đồ thị Chính
        if self._lightrag_retriever is not None and self.graph_index is not None:
            if hasattr(self._lightrag_retriever, "set_chunk_bridge"):
                self._lightrag_retriever.set_chunk_bridge(self.graph_index)
        elif self._lightrag_retriever is None:
            lr_enabled = bool(self._effective_lightrag_cfg.get("lightrag", {}).get("enabled", False))
            if (has_persistent_store and (lr_enabled or self.graph_index is not None)) or (
                self.enable_graph and self.graph_index is not None and hasattr(self.graph_index, "ground_lightrag_subgraph")
            ):
                self._lightrag_retriever = LightRAGRetriever(
                    config=self._effective_lightrag_cfg,
                    embedding_model=self.embedding_model,
                    query_extractor=self.query_entity_extractor,
                    chunk_bridge=self.graph_index,
                )

    @property
    def lightrag_retriever(self) -> LightRAGRetriever:
        """Trả về hoặc khởi tạo LightRAGRetriever (tự động gắn LightRAGChunkBridge nếu có)."""
        if self._lightrag_retriever is None:
            self._lightrag_retriever = LightRAGRetriever(
                config=self._effective_lightrag_cfg,
                embedding_model=self.embedding_model,
                query_extractor=self.query_entity_extractor,
                chunk_bridge=self.graph_index,
            )
        return self._lightrag_retriever

    @lightrag_retriever.setter
    def lightrag_retriever(self, value: Optional[LightRAGRetriever]) -> None:
        self._lightrag_retriever = value

    def index_chunks(self, chunks: List[DataChunk]) -> int:
        """Lập chỉ mục BM25 in-memory cho danh sách DataChunk (phục vụ kiểm thử hoặc dữ liệu nhỏ)."""
        self.bm25_index = BM25Index(chunks)
        self._adjacent_lookup = None
        return self.bm25_index.count()

    def _load_graph_index(self) -> int:
        """Đọc chỉ mục GraphRAG từ file nếu đã được build; bỏ qua an toàn nếu chưa có."""
        if not self._graph_index_path.exists():
            logger.debug("Chưa có GraphRAG artifact tại '%s'; bỏ qua nhánh Graph.", self._graph_index_path)
            return 0
        try:
            from src.graph.graph import GraphIndex

            self.graph_index = GraphIndex.load(self._graph_index_path, embedding_model=self.embedding_model)
            count = self.graph_index.count()
            logger.info(
                "Đã nạp GraphRAG artifact '%s' (%s chunks, %s entities, %s triples).",
                self._graph_index_path,
                count,
                self.graph_index.entity_count(),
                self.graph_index.triple_count(),
            )
            return count
        except (OSError, ValueError, EOFError, ImportError, AttributeError, pickle.UnpicklingError) as error:
            logger.warning("Không thể nạp GraphRAG artifact '%s': %s", self._graph_index_path, error)
            self.graph_index = None
            return 0

    def _load_bm25_index(self) -> int:
        """Đọc chỉ mục BM25 đã được UETIndexer tạo; không build lại trong Retrieval."""
        bm25_path = BASE_DIR / "database" / "bm25_index.pkl"
        if not bm25_path.exists():
            logger.warning(
                "Chưa có BM25 artifact tại '%s'; chạy UETIndexer trước khi tìm kiếm.",
                bm25_path,
            )
            return 0

        try:
            self.bm25_index = BM25Index.load(bm25_path)
            count = self.bm25_index.count()
            logger.info("Đã nạp %s chunks từ BM25 artifact '%s'.", count, bm25_path)
            return count
        except (OSError, ValueError, EOFError, ImportError, AttributeError, pickle.UnpicklingError) as error:
            logger.error("Không thể nạp BM25 artifact '%s': %s", bm25_path, error)
            return 0

    def _apply_mmr(
        self,
        query: UserQuery,
        query_vector: Any,
        candidates: List[RetrievedContext],
        top_n: int,
    ) -> List[RetrievedContext]:
        """Áp dụng MMR bảo toàn điểm Hybrid/Rerank và bảo vệ chunk bảng."""
        if len(candidates) <= top_n:
            return candidates

        try:
            q_vec: Optional[List[float]] = None
            if query_vector is not None and getattr(query_vector, "vector", None):
                q_vec = query_vector.vector
            else:
                q_vec = self.embedding_model.embed_query(query).vector

            chunk_ids = [c.chunk_id for c in candidates]
            embeddings_map: Dict[str, List[float]] = {}
            if hasattr(self.vector_store, "get_embeddings_by_ids"):
                embeddings_map = self.vector_store.get_embeddings_by_ids(chunk_ids)

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
                    missing_texts = [c.text for c in missing_chunks]
                    generated_embs = self.embedding_model.embed_texts(missing_texts)
                    for c, emb in zip(missing_chunks, generated_embs):
                        embeddings_map[c.chunk_id] = emb
                except Exception as emb_err:
                    logger.error(f"[MMR Fallback] Lỗi khi tạo vector bổ sung: {emb_err}")

            candidate_vectors: List[List[float]] = []
            valid_candidates: List[RetrievedContext] = []
            for c in candidates:
                if c.chunk_id in embeddings_map:
                    candidate_vectors.append(embeddings_map[c.chunk_id])
                    valid_candidates.append(c)

            if q_vec and candidate_vectors and len(valid_candidates) >= top_n:
                selected = MaximalMarginalRelevance.select(
                    query_vector=q_vec,
                    candidates=valid_candidates,
                    candidate_vectors=candidate_vectors,
                    lambda_mult=self.mmr_lambda,
                    top_n=top_n,
                    use_hybrid_relevance=True,
                    similarity_threshold=self.mmr_similarity_threshold,
                    protect_tabular=self.mmr_protect_tabular,
                )
                logger.info(
                    "Đã áp dụng MMR (%s): %d -> %d candidates (lambda=%.2f, thr=%.2f).",
                    self.mmr_stage,
                    len(candidates),
                    len(selected),
                    self.mmr_lambda,
                    self.mmr_similarity_threshold,
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

    def _blend_rerank_and_hybrid(
        self,
        query_text: str,
        reranked_chunks: List[RetrievedContext],
        hybrid_map: Dict[str, RetrievedContext],
        extracted_entities: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        """
        Hòa trộn điểm Reranker với điểm Hybrid RRF + boost nhẹ khi khớp chính xác năm/mã học phần/số QĐ/thực thể PA1.
        Tránh Cross-Encoder ghi đè 100% làm rơi câu khớp từ khóa/năm học cứng hoặc cầu nối khẩu ngữ.
        """
        if not reranked_chunks or isinstance(self.reranker, HeuristicReranker) or self.rerank_blend_alpha >= 1.0:
            return reranked_chunks

        alpha = max(0.0, min(1.0, self.rerank_blend_alpha))
        q_years = set(_YEAR_RE.findall(query_text)) if self.enable_entity_anchor_boost else set()
        q_codes = {m.group(0).lower() for m in _ENTITY_CODE_RE.finditer(query_text)} if self.enable_entity_anchor_boost else set()
        target_faculties = detect_query_faculties(query_text) if self.enable_entity_anchor_boost else set()
        pa1_terms = (
            [e.strip().lower() for e in (extracted_entities or []) if e and len(e.strip()) >= 3]
            if self.enable_entity_anchor_boost
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
        for rank, c in enumerate(blended_list, 1):
            c.rank = rank
        return blended_list

    def _get_adjacent_lookup(self) -> Dict[Tuple[str, int], DataChunk]:
        """Xây dựng bảng tra cứu lười (document_id, chunk_index) -> DataChunk từ BM25Index."""
        if self._adjacent_lookup is not None:
            return self._adjacent_lookup
        lookup: Dict[Tuple[str, int], DataChunk] = {}
        chunks = getattr(self.bm25_index, "chunks", None) or []
        for ch in chunks:
            meta = ch.metadata if isinstance(getattr(ch, "metadata", None), dict) else {}
            doc_id = getattr(ch, "document_id", None) or meta.get("document_id") or meta.get("doc_id")
            cidx = getattr(ch, "chunk_index", None)
            if cidx is None:
                cidx = meta.get("chunk_index")
            if doc_id and cidx is not None:
                try:
                    lookup[(str(doc_id), int(cidx))] = ch
                except (TypeError, ValueError):
                    continue
        self._adjacent_lookup = lookup
        return lookup

    @staticmethod
    def _strip_chunk_header(text: str) -> str:
        """Lược bỏ dòng breadcrumb [Title] và [META: ...] lặp lại khi gộp chunk liền kề."""
        lines = (text or "").splitlines()
        idx = 0
        while idx < len(lines):
            stripped = lines[idx].strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                idx += 1
                continue
            break
        body = "\n".join(lines[idx:]).strip()
        return body if body else (text or "").strip()

    def _expand_adjacent_chunks(self, final_contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """
        Mở rộng ngữ cảnh liền kề (±adjacent_window_size) cho Top-K chunks điểm cao nhất.
        Giữ nguyên chunk_id, rank và độ dài final_contexts để không làm sai lệch IR evaluation.
        """
        if not self.enable_adjacent_expansion or not self.merge_contiguous_chunks or not final_contexts:
            return final_contexts

        lookup = self._get_adjacent_lookup()
        if not lookup:
            return final_contexts

        window = max(1, int(self.adjacent_window_size))
        expand_limit = min(max(0, int(self.adjacent_expansion_top_k)), len(final_contexts))
        present_ids = {c.chunk_id for c in final_contexts}
        expanded_results = list(final_contexts)

        for i in range(expand_limit):
            ctx = expanded_results[i]
            meta = dict(ctx.metadata) if ctx.metadata else {}
            doc_id = meta.get("document_id") or meta.get("doc_id")
            cidx_raw = meta.get("chunk_index")
            if not doc_id or cidx_raw is None:
                continue
            try:
                center_idx = int(cidx_raw)
            except (TypeError, ValueError):
                continue

            merged_parts: List[str] = []
            merged_ids: List[str] = []

            for offset in range(-window, window + 1):
                nb_idx = center_idx + offset
                if offset == 0:
                    merged_parts.append(ctx.text)
                    merged_ids.append(ctx.chunk_id)
                    continue
                nb = lookup.get((str(doc_id), nb_idx))
                if nb is None or nb.chunk_id in present_ids:
                    continue
                nb_body = self._strip_chunk_header(nb.text)
                if nb_body and nb_body not in ctx.text:
                    merged_parts.append(nb_body)
                    merged_ids.append(nb.chunk_id)
                    present_ids.add(nb.chunk_id)

            if len(merged_ids) > 1:
                meta["expanded_chunk_ids"] = merged_ids
                meta["expanded_chunk_count"] = len(merged_ids)
                expanded_results[i] = RetrievedContext(
                    chunk_id=ctx.chunk_id,
                    text="\n".join(merged_parts),
                    similarity_score=ctx.similarity_score,
                    rank=ctx.rank,
                    metadata=meta,
                )

        return expanded_results

    def retrieve(
        self,
        query: UserQuery,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
    ) -> List[RetrievedContext]:
        """
        Quy trình truy xuất ngữ cảnh thống nhất (Unified Retrieval Router & Multi-Engine):
        Hỗ trợ các động cơ (engines):
          - "vector": BM25 + Vector + GraphRAG -> Fusion -> Rerank + Blend -> Top-K -> Adjacent Expansion.
          - "lightrag": Knowledge Graph Dual-Level Retrieval (truy xuất thực thể, quan hệ, tóm tắt toàn cục).
          - "hybrid": Dung hợp kết quả từ cả Đồ thị tri thức (LightRAG) và Đoạn cắt văn bản (Vector DB).
          - "auto": Tự động phân loại ý định qua QueryRouter để chọn engine tối ưu nhất.

        :param query: Câu hỏi người dùng (UserQuery)
        :param top_k: Số lượng văn bản liên quan cuối cùng cần lấy (mặc định 5)
        :param score_threshold: Ngưỡng điểm tin cậy tối thiểu (tùy chọn)
        :param filters: Bộ lọc metadata (ví dụ lọc theo khoa, năm học)
        :param engine: Động cơ truy xuất: 'vector', 'lightrag', 'hybrid', 'auto' (None = lấy từ config)
        :return: Danh sách RetrievedContext đã xếp hạng
        """
        active_engine = (engine or self.default_engine).lower().strip()

        # Nếu là chế độ auto, dùng QueryRouter để ra quyết định
        if active_engine == "auto":
            active_engine = self.router.route(query)
            logger.info(f"[UETRetriever] QueryRouter tự động định tuyến truy vấn sang: '{active_engine.upper()}'")

        effective_top_k = top_k or self.default_top_k

        # ---------------------------------------------------------------------
        # 1. ĐỘNG CƠ LIGHTRAG THUẦN TÚY (Chạy độc lập không qua ChunkBridge/Reranker)
        # ---------------------------------------------------------------------
        if active_engine == "lightrag":
            graph_results = self.lightrag_retriever.retrieve(query=query, top_k=effective_top_k)
            if graph_results:
                return graph_results
            # Nếu đồ thị chưa xây dựng hoặc không tìm thấy, fallback sang luồng 3-Way Grounded
            logger.info("[UETRetriever] LightRAG thuần không có dữ liệu phù hợp. Tự động fallback sang Grounded 3-Way.")
            return self._retrieve_vector(
                query=query,
                top_k=effective_top_k,
                score_threshold=score_threshold,
                filters=filters,
                route_intent="hybrid",
            )

        # ---------------------------------------------------------------------
        # 2. TƯƠNG THÍCH NGƯỢC / SLOT-MERGE DỰ PHÒNG:
        #    - Khi đã gắn LightRAGChunkBridge (graph_index.pkl), LightRAGRetriever.retrieve_grounded()
        #      chạy trực tiếp làm nhánh Đồ thị Chính trong 3-Way RRF + Dual-Output (Bước 3 bên dưới).
        #    - Chỉ dùng slot-merge khi truyền mock lightrag_retriever tùy chỉnh (vd: unit test) hoặc khi
        #      chưa có file cầu nối graph_index.pkl.
        # ---------------------------------------------------------------------
        supports_lightrag_grounded = (
            self.enable_graph
            and self._lightrag_retriever is not None
            and hasattr(self._lightrag_retriever, "retrieve_grounded")
            and hasattr(self._lightrag_retriever, "has_chunk_bridge")
            and self._lightrag_retriever.has_chunk_bridge()
        )

        if active_engine == "hybrid" and not supports_lightrag_grounded:
            query_text_clean = (query.query_text or "").strip()
            is_tabular_lookup = bool(self._TABULAR_NUMERIC_LOOKUP_RE.search(query_text_clean))

            if effective_top_k <= 1:
                graph_k = 0
            elif is_tabular_lookup:
                graph_k = 1
            else:
                graph_k = min(
                    self.hybrid_max_graph_k,
                    max(1, round(effective_top_k * self.hybrid_graph_ratio)),
                )
            vector_k = effective_top_k

            if graph_k > 0:
                future_graph = self._executor.submit(
                    self.lightrag_retriever.retrieve,
                    query=query,
                    top_k=graph_k,
                )
                future_vector = self._executor.submit(
                    self._retrieve_vector,
                    query=query,
                    top_k=vector_k,
                    score_threshold=score_threshold,
                    filters=filters,
                )
                try:
                    graph_results = future_graph.result()
                except Exception as g_err:
                    logger.warning(f"[UETRetriever] Nhánh LightRAG gặp sự cố ({g_err}), tiếp tục với Vector.")
                    graph_results = []
                vector_results = future_vector.result()
            else:
                graph_results = []
                vector_results = self._retrieve_vector(
                    query=query,
                    top_k=vector_k,
                    score_threshold=score_threshold,
                    filters=filters,
                )

            if not graph_results:
                return vector_results
            if not vector_results:
                return graph_results

            combined: List[RetrievedContext] = []
            seen_ids: Set[str] = set()
            seen_texts: Set[str] = set()

            for g in graph_results[:graph_k]:
                norm_txt = g.text.strip()
                if g.chunk_id not in seen_ids and norm_txt not in seen_texts:
                    combined.append(g)
                    seen_ids.add(g.chunk_id)
                    seen_texts.add(norm_txt)

            for v in vector_results:
                norm_txt = v.text.strip()
                if v.chunk_id not in seen_ids and norm_txt not in seen_texts:
                    combined.append(v)
                    seen_ids.add(v.chunk_id)
                    seen_texts.add(norm_txt)

            final_merged = combined[:effective_top_k]
            for i, ctx in enumerate(final_merged, 1):
                ctx.rank = i

            logger.info(
                f"[UETRetriever] Dung hợp Hybrid thành công (tabular_lookup={is_tabular_lookup}): "
                f"{len(graph_results[:graph_k])} graph contexts + {len(vector_results)} vector chunks "
                f"-> {len(final_merged)} contexts cuối cùng."
            )
            return final_merged

        # ---------------------------------------------------------------------
        # 3. ĐỘNG CƠ CHÍNH: LIGHTRAG-FIRST 3-WAY RRF (BM25 + VECTOR + LIGHTRAG GROUNDED + DUAL-OUTPUT)
        # ---------------------------------------------------------------------
        return self._retrieve_vector(
            query=query,
            top_k=effective_top_k,
            score_threshold=score_threshold,
            filters=filters,
            route_intent=active_engine,
        )

    def _run_bm25_and_vector_recall(
        self,
        query: UserQuery,
        query_text: str,
        active_filters: Optional[Dict[str, Any]],
    ) -> Tuple[List[RetrievedContext], List[RetrievedContext], Any]:
        """Thực thi BM25 Search và Dense Vector Search."""
        bm25_results: List[RetrievedContext] = []
        try:
            bm25_results = self.bm25_index.search(query=query_text, top_k=self.bm25_top_k)
            logger.debug(f"[BM25] Thu được {len(bm25_results)} kết quả.")
        except Exception as e:
            logger.warning(f"Lỗi nhánh BM25 Search: {e}. Vẫn tiếp tục với Vector Search.")

        vector_results: List[RetrievedContext] = []
        query_vector = None
        try:
            query_vector = self.embedding_model.embed_query(query)
            vector_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=self.vector_top_k,
                filters=active_filters,
            )
            logger.debug(f"[Vector Search] Thu được {len(vector_results)} kết quả.")
        except Exception as e:
            logger.warning(f"Lỗi nhánh Vector Search: {e}. Vẫn tiếp tục với BM25/Graph Search.")

        return bm25_results, vector_results, query_vector

    def _retrieve_vector(
        self,
        query: UserQuery,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        route_intent: str = "vector",
    ) -> List[RetrievedContext]:
        """
        Quy trình truy xuất ngữ cảnh LightRAG-First đa tầng:
        (PA1 Smart Router song song BM25 + Vector) -> LightRAGRetriever.retrieve_grounded (kèm ChunkBridge & Regex 0.65x)
        -> 3-Way RRF -> MMR -> Jina Reranker -> Adjacent Expansion -> Đính kèm lightrag_overview cho LLM.
        """
        query_text = (query.query_text or "").strip()
        if not query_text:
            logger.warning("Truy vấn người dùng rỗng. Trả về kết quả rỗng.")
            return []

        logger.info(f"Bắt đầu quy trình LightRAG-First 3-Way Retrieval (intent='{route_intent}') cho: '{query_text}'")

        effective_top_k = top_k or self.default_top_k
        active_filters = filters or query.filters
        has_graph = self.enable_graph and self.graph_index is not None and self.graph_index.count() > 0
        has_lightrag_grounded = (
            self.enable_graph
            and self._lightrag_retriever is not None
            and hasattr(self._lightrag_retriever, "retrieve_grounded")
            and (
                self._lightrag_retriever.is_available()
                or (
                    hasattr(self._lightrag_retriever, "has_chunk_bridge")
                    and self._lightrag_retriever.has_chunk_bridge()
                )
            )
        )

        # =====================================================================
        # BƯỚC 1: CHẠY SONG SONG PA1 (NẾU ROUTER BẬT) CÙNG VỚI BM25 + VECTOR
        # =====================================================================
        extracted_entities: List[str] = []
        high_level_themes: List[str] = []
        use_pa1 = (
            (has_lightrag_grounded or has_graph)
            and self.query_entity_extractor is not None
            and hasattr(self.router, "should_use_llm_extraction")
            and self.router.should_use_llm_extraction(query)
        )

        if use_pa1:
            fut_pa1 = self._executor.submit(self.query_entity_extractor.extract, query_text)
            fut_recall = self._executor.submit(
                self._run_bm25_and_vector_recall, query, query_text, active_filters
            )
            bm25_results, vector_results, query_vector = fut_recall.result()
            try:
                pa1_res = fut_pa1.result() or {}
                extracted_entities = list(pa1_res.get("low_level_entities") or [])
                high_level_themes = list(pa1_res.get("high_level_themes") or [])
            except Exception as pa1_err:
                logger.warning(f"Lỗi PA1 QueryEntityExtractor ({pa1_err}), fallback 0-LLM LightRAG.")
        else:
            bm25_results, vector_results, query_vector = self._run_bm25_and_vector_recall(
                query, query_text, active_filters
            )

        graph_results: List[RetrievedContext] = []
        lightrag_overview: Optional[str] = None
        is_tabular_lookup = bool(self._TABULAR_NUMERIC_LOOKUP_RE.search(query_text))
        is_pure_passage_lookup = bool(
            route_intent == "vector"
            and hasattr(self.router, "_pas_regex")
            and self.router._pas_regex.search(query_text)
            and not self.router._rel_regex.search(query_text)
        )
        want_overview = (route_intent == "hybrid") and not is_tabular_lookup

        if has_lightrag_grounded and self._lightrag_retriever is not None:
            try:
                if query_vector is None:
                    query_vector = self.embedding_model.embed_query(query)
                q_vec_list = getattr(query_vector, "vector", None) if query_vector is not None else None
                graph_results, lightrag_overview = self._lightrag_retriever.retrieve_grounded(
                    query=query,
                    query_vector=q_vec_list,
                    top_k=self.graph_top_k,
                    filters=active_filters,
                    extracted_entities=extracted_entities if use_pa1 else None,
                    high_level_themes=high_level_themes if use_pa1 else None,
                    allow_llm=False,
                    include_overview=want_overview,
                    seed_top_m=self.graph_seed_top_m,
                    max_hops=self.graph_max_hops,
                    hop_decay=self.graph_hop_decay,
                )
                logger.debug(
                    f"[LightRAG Primary] Thu được {len(graph_results)} grounded chunks "
                    f"(pa1={use_pa1}, seeds={extracted_entities}, overview={bool(lightrag_overview)})."
                )
            except Exception as e:
                logger.warning(f"Lỗi nhánh LightRAG Grounded Search: {e}. Vẫn tiếp tục với BM25 + Vector.")
        elif has_graph:
            try:
                if query_vector is None:
                    query_vector = self.embedding_model.embed_query(query)
                q_vec_list = getattr(query_vector, "vector", None) if query_vector is not None else None
                search_kwargs: Dict[str, Any] = {
                    "query": query_text,
                    "query_vector": q_vec_list,
                    "top_k": self.graph_top_k,
                    "seed_top_m": self.graph_seed_top_m,
                    "max_hops": self.graph_max_hops,
                    "hop_decay": self.graph_hop_decay,
                    "filters": active_filters,
                }
                if extracted_entities or high_level_themes:
                    search_kwargs["extracted_entities"] = extracted_entities
                    search_kwargs["high_level_themes"] = high_level_themes
                graph_results = self.graph_index.search(**search_kwargs)
            except TypeError:
                graph_results = self.graph_index.search(
                    query=query_text,
                    query_vector=q_vec_list,
                    top_k=self.graph_top_k,
                    seed_top_m=self.graph_seed_top_m,
                    max_hops=self.graph_max_hops,
                    filters=active_filters,
                )
            except Exception as e:
                logger.warning(f"Lỗi nhánh Graph Supplement Search: {e}. Vẫn tiếp tục với BM25 + Vector.")

        # =====================================================================
        # BƯỚC 2: HYBRID SEARCH FUSION (RRF hoặc LINEAR - Điều chỉnh trọng số theo ý định truy vấn)
        # =====================================================================
        if route_intent == "hybrid" and not is_tabular_lookup:
            effective_graph_weight = self.lightrag_hybrid_weight
        elif is_pure_passage_lookup:
            effective_graph_weight = min(self.graph_weight, 0.15)
        else:
            effective_graph_weight = self.graph_weight

        if self.fusion_method == "linear":
            candidate_chunks = HybridSearcher.linear_score_fusion(
                bm25_results=bm25_results,
                vector_results=vector_results,
                top_n=self.candidate_pool_size,
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
                graph_results=graph_results,
                graph_weight=effective_graph_weight,
            )
        else:
            candidate_chunks = HybridSearcher.reciprocal_rank_fusion(
                bm25_results=bm25_results,
                vector_results=vector_results,
                top_n=self.candidate_pool_size,
                rrf_k=self.rrf_k,
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
                graph_results=graph_results,
                graph_weight=effective_graph_weight,
            )

        if not candidate_chunks:
            logger.warning("Không tìm thấy đoạn văn bản nào từ BM25, Vector và GraphRAG Search.")
            return []

        hybrid_map = {c.chunk_id: c for c in candidate_chunks}

        # =====================================================================
        # BƯỚC 2.5: PRE-RERANK MMR (chỉ chạy khi bật và mmr_stage != 'post_rerank')
        # =====================================================================
        rerank_candidates = candidate_chunks
        if self.enable_mmr and self.mmr_stage != "post_rerank" and len(candidate_chunks) > self.mmr_top_n:
            rerank_candidates = self._apply_mmr(query, query_vector, candidate_chunks, top_n=self.mmr_top_n)

        # =====================================================================
        # BƯỚC 3: RERANKER + SCORE BLENDING
        # =====================================================================
        effective_rerank_n = max(self.rerank_top_n, effective_top_k)
        if self.enable_mmr and self.mmr_stage == "post_rerank":
            effective_rerank_n = max(effective_rerank_n, self.mmr_top_n)

        rerank_fetch_n = (
            max(effective_rerank_n, len(rerank_candidates))
            if (self.enable_entity_anchor_boost and not isinstance(self.reranker, HeuristicReranker) and self.rerank_blend_alpha < 1.0)
            else effective_rerank_n
        )
        try:
            ranked_chunks = self.reranker.rerank(
                query=query_text,
                candidates=rerank_candidates,
                top_n=rerank_fetch_n,
            )
        except Exception as e:
            logger.warning(f"Lỗi khi chạy Reranker ({e}). Sử dụng thứ tự từ Hybrid Search.")
            ranked_chunks = rerank_candidates[:effective_rerank_n]

        ranked_chunks = self._blend_rerank_and_hybrid(
            query_text,
            ranked_chunks,
            hybrid_map,
            extracted_entities=extracted_entities,
        )[:effective_rerank_n]

        # =====================================================================
        # BƯỚC 3.5: POST-RERANK MMR (nếu bật ở chế độ post_rerank)
        # =====================================================================
        if self.enable_mmr and self.mmr_stage == "post_rerank":
            post_top_n = max(effective_top_k, self.mmr_top_n)
            if len(ranked_chunks) > post_top_n:
                ranked_chunks = self._apply_mmr(query, query_vector, ranked_chunks, top_n=post_top_n)

        # =====================================================================
        # BƯỚC 4: TOP-K SELECTION & ADJACENT CHUNK EXPANSION
        # =====================================================================
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold

        filtered_chunks: List[RetrievedContext] = []
        if threshold is not None:
            filtered_chunks = [c for c in ranked_chunks if c.similarity_score >= threshold]
            if not filtered_chunks and ranked_chunks:
                logger.info(f"Không có chunk nào vượt qua ngưỡng {threshold}, giữ lại top 1 chunk tốt nhất.")
                filtered_chunks = [ranked_chunks[0]]
        else:
            filtered_chunks = ranked_chunks

        final_contexts = filtered_chunks[:effective_top_k]
        for i, ctx in enumerate(final_contexts, 1):
            ctx.rank = i

        final_contexts = self._expand_adjacent_chunks(final_contexts)

        # Đính kèm Bản đồ Liên kết Quy chế 2 tầng (LightRAG Dual-Output) lên chunk hạng 1 để bơm thẳng vào Prompt LLM
        if lightrag_overview and final_contexts:
            top_ctx = final_contexts[0]
            top_meta = dict(top_ctx.metadata) if top_ctx.metadata else {}
            top_meta["lightrag_overview"] = lightrag_overview
            final_contexts[0] = RetrievedContext(
                chunk_id=top_ctx.chunk_id,
                text=top_ctx.text,
                similarity_score=top_ctx.similarity_score,
                rank=top_ctx.rank,
                metadata=top_meta,
            )

        logger.info(
            f"Hoàn thành Retrieval: {len(candidate_chunks)} candidates -> "
            f"{len(rerank_candidates)} qua MMR/pool -> "
            f"{len(ranked_chunks)} reranked -> {len(final_contexts)} chunks cuối cùng "
            f"(lightrag_overview={bool(lightrag_overview)})."
        )
        return final_contexts


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n🔍 ĐANG KIỂM THỬ ĐƠN VỊ MODULE RETRIEVAL (SRC/RETRIEVAL)...")

    # Mô hình Embedding giả lập phục vụ unit test độc lập
    class DummyEmbedder(BaseEmbeddingModel):
        @property
        def dimension(self) -> int:
            return 384

        @property
        def model_name(self) -> str:
            return "dummy-test-embedder"

        def embed_texts(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * self.dimension for _ in texts]

    # Vector Store giả lập
    class DummyStore(BaseVectorStore):
        def __init__(self):
            self.mock_chunks = [
                DataChunk(
                    chunk_id="chunk_test_1",
                    text="Sinh viên UET có điểm GPA từ 3.60 trở lên và điểm rèn luyện xuất sắc được nhận học bổng xuất sắc.",
                    metadata={"title": "Quy chế học bổng UET", "source": "quy_che_hoc_bong.pdf"},
                ),
                DataChunk(
                    chunk_id="chunk_test_2",
                    text="Điều kiện xét tốt nghiệp: tích lũy đủ số tín chỉ, GPA từ 2.00 trở lên, chuẩn đầu ra ngoại ngữ B2.",
                    metadata={"title": "Quy định tốt nghiệp", "source": "quy_che_tot_nghiep.pdf"},
                ),
                DataChunk(
                    chunk_id="chunk_test_3",
                    text="Sinh viên bị cảnh báo học tập nếu điểm trung bình chung học kỳ dưới 0.80 đối với học kỳ 1.",
                    metadata={"title": "Quy chế cảnh báo học vụ", "source": "canh_bao_hoc_vu.pdf"},
                ),
            ]

        def store(self, chunks, vectors=None) -> int:
            self.mock_chunks.extend(chunks)
            return len(chunks)

        def search(self, query_vector, top_k: int = 5, filters=None) -> List[RetrievedContext]:
            return [
                RetrievedContext(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    similarity_score=0.88 - (i * 0.1),
                    rank=i + 1,
                    metadata=c.metadata,
                )
                for i, c in enumerate(self.mock_chunks[:top_k])
            ]

        def count(self) -> int:
            return len(self.mock_chunks)

        def clear(self) -> None:
            self.mock_chunks.clear()

    # Khởi tạo UETRetriever
    store = DummyStore()

    class DummyBM25:
        def __init__(self, chunks):
            self.chunks = chunks

        def search(self, query: str, top_k: int = 5):
            return [
                RetrievedContext(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    similarity_score=round(10.0 - (i * 0.5), 2),
                    rank=i + 1,
                    metadata=c.metadata,
                )
                for i, c in enumerate(self.chunks[:top_k])
            ]

    # Inject mock BM25 và cấu hình MMR để kiểm thử
    mock_bm25 = DummyBM25(chunks=store.mock_chunks)
    test_cfg = {
        "retrieval": {
            "bm25_top_k": 3,
            "vector_top_k": 3,
            "candidate_pool_size": 3,
            "enable_mmr": True,
            "mmr_lambda": 0.6,
            "mmr_top_n": 2,  # Rút từ 3 candidates xuống 2 qua MMR
            "rerank_top_n": 2,
            "final_top_k": 2,
            "reranker_type": "heuristic",
        }
    }
    retriever = UETRetriever(
        embedding_model=DummyEmbedder(),
        vector_store=store,
        bm25_index=mock_bm25,
        config=test_cfg,
    )

    # Thử nghiệm câu hỏi
    test_q = UserQuery(query_text="học bổng xuất sắc UET cần điều kiện GPA bao nhiêu?")
    results = retriever.retrieve(test_q, top_k=2)

    print(f"\nCâu hỏi: {test_q.query_text}")
    print(f"Số kết quả lấy về: {len(results)}")
    for r in results:
        print(f" - [Rank {r.rank} | Score: {r.similarity_score}] {r.metadata.get('title')}: {r.text[:80]}...")

    assert len(results) > 0, "Lỗi: Không tìm thấy kết quả nào!"
    print("\n✅ KIỂM THỬ MODULE RETRIEVAL THÀNH CÔNG RỰC RỠ!\n")

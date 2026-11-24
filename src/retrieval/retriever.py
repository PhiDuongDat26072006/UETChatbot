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
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.retrieval.reranker import BaseReranker, HeuristicReranker, RerankerFactory
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

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        config: Optional[Dict[str, Any]] = None,
        bm25_index: Optional[BM25Index] = None,
        graph_index: Optional[Any] = None,
    ):
        """
        Khởi tạo UETRetriever.
        :param embedding_model: Mô hình nhúng câu hỏi kế thừa BaseEmbeddingModel
        :param vector_store: Cơ sở dữ liệu vector kế thừa BaseVectorStore
        :param reranker: Mô hình Reranker (nếu None sẽ tự động khởi tạo qua RerankerFactory)
        :param config: Dict cấu hình tham số tùy chỉnh (hoặc tự đọc từ config.yaml)
        :param bm25_index: Chỉ mục đã tạo sẵn (dành cho test/in-memory); production tải từ file.
        :param graph_index: Chỉ mục GraphRAG (dành cho test/in-memory); production tải từ file.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        # Đọc cấu hình từ config.yaml nếu không truyền trực tiếp
        raw_config = config or load_yaml_config()
        retrieval_cfg = raw_config.get("retrieval", {})
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

        # Cấu hình GraphRAG Vector Index
        self.enable_graph: bool = bool(retrieval_cfg.get("enable_graph", True))
        self.graph_top_k: int = int(retrieval_cfg.get("graph_top_k", 60))
        self.graph_weight: float = float(retrieval_cfg.get("graph_weight", 0.30))
        self.graph_max_hops: int = int(retrieval_cfg.get("graph_max_hops", 2))
        self.graph_seed_top_m: int = int(retrieval_cfg.get("graph_seed_top_m", 10))
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

        # Chỉ nạp artifact từ ổ đĩa khi chạy với persistent VectorStore thật (không nạp trong unit test mock store).
        has_persistent_store = getattr(self.vector_store, "persist_dir", None) is not None
        if bm25_index is None and has_persistent_store:
            self._load_bm25_index()
        if graph_index is None and bm25_index is None and self.enable_graph and has_persistent_store:
            self._load_graph_index()

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
            from src.indexing.graph import GraphIndex

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
    ) -> List[RetrievedContext]:
        """
        Hòa trộn điểm Reranker với điểm Hybrid RRF + boost nhẹ khi khớp chính xác năm/mã học phần/số QĐ.
        Tránh Cross-Encoder ghi đè 100% làm rơi câu khớp từ khóa/năm học cứng.
        """
        if not reranked_chunks or isinstance(self.reranker, HeuristicReranker) or self.rerank_blend_alpha >= 1.0:
            return reranked_chunks

        alpha = max(0.0, min(1.0, self.rerank_blend_alpha))
        q_years = set(_YEAR_RE.findall(query_text)) if self.enable_entity_anchor_boost else set()
        q_codes = {m.group(0).lower() for m in _ENTITY_CODE_RE.finditer(query_text)} if self.enable_entity_anchor_boost else set()

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
    ) -> List[RetrievedContext]:
        """
        Quy trình truy xuất ngữ cảnh đa tầng (BM25 + Vector + GraphRAG -> Fusion -> Rerank + Blend -> Top-K -> Adjacent Expansion).
        """
        query_text = (query.query_text or "").strip()
        if not query_text:
            logger.warning("Truy vấn người dùng rỗng. Trả về kết quả rỗng.")
            return []

        logger.info(f"Bắt đầu quy trình Hybrid Retrieval cho: '{query_text}'")

        effective_top_k = top_k or self.default_top_k
        active_filters = filters or query.filters

        # =====================================================================
        # BƯỚC 1: SONG SONG BM25 SEARCH, VECTOR SEARCH VÀ GRAPHRAG
        # =====================================================================
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

        graph_results: List[RetrievedContext] = []
        if self.enable_graph and self.graph_index is not None and self.graph_index.count() > 0:
            try:
                if query_vector is None:
                    query_vector = self.embedding_model.embed_query(query)
                q_vec_list = getattr(query_vector, "vector", None) if query_vector is not None else None
                graph_results = self.graph_index.search(
                    query=query_text,
                    query_vector=q_vec_list,
                    top_k=self.graph_top_k,
                    seed_top_m=self.graph_seed_top_m,
                    max_hops=self.graph_max_hops,
                    filters=active_filters,
                )
                logger.debug(f"[GraphRAG] Thu được {len(graph_results)} kết quả.")
            except Exception as e:
                logger.warning(f"Lỗi nhánh GraphRAG Search: {e}. Vẫn tiếp tục với BM25 + Vector.")

        # =====================================================================
        # BƯỚC 2: HYBRID SEARCH FUSION (RRF hoặc LINEAR)
        # =====================================================================
        if self.fusion_method == "linear":
            candidate_chunks = HybridSearcher.linear_score_fusion(
                bm25_results=bm25_results,
                vector_results=vector_results,
                top_n=self.candidate_pool_size,
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
                graph_results=graph_results,
                graph_weight=self.graph_weight,
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
                graph_weight=self.graph_weight,
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

        try:
            ranked_chunks = self.reranker.rerank(
                query=query_text,
                candidates=rerank_candidates,
                top_n=effective_rerank_n,
            )
        except Exception as e:
            logger.warning(f"Lỗi khi chạy Reranker ({e}). Sử dụng thứ tự từ Hybrid Search.")
            ranked_chunks = rerank_candidates[:effective_rerank_n]

        ranked_chunks = self._blend_rerank_and_hybrid(query_text, ranked_chunks, hybrid_map)

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

        logger.info(
            f"Hoàn thành Retrieval: {len(candidate_chunks)} candidates -> "
            f"{len(rerank_candidates)} qua MMR/pool -> "
            f"{len(ranked_chunks)} reranked -> {len(final_contexts)} chunks cuối cùng."
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

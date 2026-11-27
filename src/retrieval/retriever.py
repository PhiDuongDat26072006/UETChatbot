"""
src/retrieval/retriever.py - Bộ điều phối truy xuất ngữ cảnh đa tầng (Retrieval Pipeline).

Scoring/MMR helpers live in ``src/retrieval/scoring.py`` and chunk expansion helpers live
in ``src/retrieval/expansion.py``.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.base import (
    BaseEmbeddingModel,
    BaseVectorStore,
    DataChunk,
    RetrievedContext,
    UserQuery,
)
from src.graph.lightrag_retriever import LightRAGRetriever
from src.graph.query_entity_extractor import QueryEntityExtractor
from src.indexing.sparse import BM25Index
from src.retrieval.expansion import (
    build_adjacent_lookup,
    build_doc_chunks_lookup,
    compact_table_row,
    expand_adjacent_chunks,
    expand_tabular_siblings,
    extract_header_lines,
    extract_meta_header,
    is_list_query,
    is_tabular_chunk,
    stitch_contiguous_chunks,
    strip_chunk_header,
)
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.reranker import BaseReranker, HeuristicReranker, RerankerFactory
from src.retrieval.router import QueryRouter
from src.retrieval.scoring import (
    apply_mmr_selection,
    blend_rerank_and_hybrid,
    blend_rerank_and_rrf,
)
from src.utils.helpers import BASE_DIR, get_logger, load_yaml_config, resolve_project_path

logger = get_logger("retrieval")


class UETRetriever:
    """Bộ điều phối truy xuất ngữ cảnh đa tầng (BM25 + Vector + GraphRAG -> RRF -> MMR -> Rerank)."""

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
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        raw_config = config or load_yaml_config()
        retrieval_cfg = raw_config.get("retrieval", {})
        graph_cfg = raw_config.get("graph", {})
        paths_cfg = raw_config.get("paths", {})

        self.bm25_top_k: int = int(retrieval_cfg.get("bm25_top_k", 50))
        self.vector_top_k: int = int(retrieval_cfg.get("vector_top_k", 50))
        self.candidate_pool_size: int = int(retrieval_cfg.get("candidate_pool_size", 25))
        self.rerank_top_n: int = int(retrieval_cfg.get("rerank_top_n", 15))
        self.default_top_k: int = int(retrieval_cfg.get("final_top_k", 5))
        self.fusion_method: str = str(retrieval_cfg.get("fusion_method", "rrf")).lower().strip()
        self.rrf_k: int = int(retrieval_cfg.get("rrf_k", 30))
        self.bm25_weight: float = float(retrieval_cfg.get("bm25_weight", 0.60))
        self.vector_weight: float = float(retrieval_cfg.get("vector_weight", 0.20))
        self.rerank_weight: float = float(retrieval_cfg.get("rerank_weight", retrieval_cfg.get("rerank_blend_alpha", 0.65)))
        self.rrf_blend_weight: float = float(retrieval_cfg.get("rrf_blend_weight", round(1.0 - self.rerank_weight, 4)))
        self.default_score_threshold: Optional[float] = retrieval_cfg.get("score_threshold", None)
        self.default_engine: str = retrieval_cfg.get("engine", "auto")

        self.hybrid_graph_ratio: float = retrieval_cfg.get("hybrid_graph_ratio", 0.2)
        self.hybrid_max_graph_k: int = retrieval_cfg.get("hybrid_max_graph_k", 2)

        env_enable_graph = os.environ.get("ENABLE_GRAPH") if config is None else None
        if env_enable_graph is not None:
            self.enable_graph: bool = env_enable_graph.strip().lower() in ("1", "true", "yes", "on")
        else:
            self.enable_graph = bool(retrieval_cfg.get("enable_graph", True))
        self.graph_top_k: int = int(retrieval_cfg.get("graph_top_k", 60))
        self.graph_weight: float = float(retrieval_cfg.get("graph_weight", 0.30))
        self.lightrag_hybrid_weight: float = float(
            retrieval_cfg.get("lightrag_hybrid_weight", max(self.graph_weight, 0.40))
        )
        self.graph_max_hops: int = int(retrieval_cfg.get("graph_max_hops", graph_cfg.get("max_hops", 2)))
        self.graph_seed_top_m: int = int(retrieval_cfg.get("graph_seed_top_m", graph_cfg.get("seed_top_m", 10)))
        self.graph_hop_decay: float = float(retrieval_cfg.get("graph_hop_decay", graph_cfg.get("hop_decay", 0.65)))
        self._graph_index_path = resolve_project_path(paths_cfg.get("graph_index_file", "database/graph_index.pkl"))

        self.enable_mmr: bool = bool(retrieval_cfg.get("enable_mmr", False))
        self.mmr_stage: str = str(retrieval_cfg.get("mmr_stage", "pre_rerank")).lower().strip()
        self.mmr_lambda: float = float(retrieval_cfg.get("mmr_lambda", 0.75))
        self.mmr_top_n: int = int(retrieval_cfg.get("mmr_top_n", 40))
        self.mmr_similarity_threshold: float = float(retrieval_cfg.get("mmr_similarity_threshold", 0.0))
        self.mmr_protect_tabular: bool = bool(retrieval_cfg.get("mmr_protect_tabular", True))

        agent_retrieval_cfg = raw_config.get("agent", {}).get("retrieval", {})
        self.shared_faculties: List[str] = agent_retrieval_cfg.get("shared_faculties", ["uet", "tuyensinh", "handbook"])
        self.min_results_after_filter: int = int(agent_retrieval_cfg.get("min_results_after_filter", 10))
        self.faculty_filter_mode: str = str(agent_retrieval_cfg.get("faculty_filter", "soft")).lower()

        reranker_type = retrieval_cfg.get("reranker_type", "auto")
        reranker_model = retrieval_cfg.get("reranker_model", None)
        onnx_filename = retrieval_cfg.get("onnx_filename", None)
        agent_reranker_cfg = raw_config.get("agent", {}).get("reranker", {})
        env_reranker_gpu = os.environ.get("RERANKER_USE_GPU") if config is None else None
        if env_reranker_gpu is not None and env_reranker_gpu.strip() != "":
            self.reranker_use_gpu: bool = env_reranker_gpu.strip().lower() in ("1", "true", "yes", "on", "gpu", "cuda")
        else:
            self.reranker_use_gpu = bool(retrieval_cfg.get("reranker_use_gpu", agent_reranker_cfg.get("use_gpu", True)))
        self.rerank_max_length: int = int(retrieval_cfg.get("rerank_max_length", 512))
        self.rerank_batch_size: int = int(retrieval_cfg.get("rerank_batch_size", 8))
        self.rerank_blend_alpha: float = float(retrieval_cfg.get("rerank_blend_alpha", 0.65))
        self.enable_entity_anchor_boost: bool = bool(retrieval_cfg.get("enable_entity_anchor_boost", True))

        self.enable_adjacent_expansion: bool = bool(retrieval_cfg.get("enable_adjacent_expansion", False))
        self.adjacent_expansion_top_k: int = int(retrieval_cfg.get("adjacent_expansion_top_k", 2))
        self.adjacent_window_size: int = int(retrieval_cfg.get("adjacent_window_size", 1))
        self.merge_contiguous_chunks: bool = bool(retrieval_cfg.get("merge_contiguous_chunks", False))

        self.enable_tabular_expansion: bool = bool(retrieval_cfg.get("enable_tabular_expansion", True))
        self.tabular_expansion_max_chunks: int = int(retrieval_cfg.get("tabular_expansion_max_chunks", 25))
        self.tabular_expansion_max_chars: int = int(retrieval_cfg.get("tabular_expansion_max_chars", 7000))

        has_persistent_store = getattr(self.vector_store, "persist_dir", None) is not None
        self._effective_lightrag_cfg = (
            {"lightrag": {"enabled": False, "router": {"mode": "always_vector"}}}
            if ((not has_persistent_store or (config is not None and "lightrag" not in config)) and lightrag_retriever is None)
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

        self.bm25_index = bm25_index if bm25_index is not None else BM25Index()
        self.graph_index = graph_index
        self._adjacent_lookup: Optional[Dict[Tuple[str, int], DataChunk]] = None
        self._doc_chunks_lookup: Optional[Dict[str, List[Tuple[int, DataChunk]]]] = None

        self.reranker = reranker or RerankerFactory.create(
            reranker_type=reranker_type,
            model_name=reranker_model,
            onnx_filename=onnx_filename,
            max_length=self.rerank_max_length,
            batch_size=self.rerank_batch_size,
            use_gpu=self.reranker_use_gpu,
        )

        if bm25_index is None and has_persistent_store:
            self._load_bm25_index()
        if graph_index is None and bm25_index is None and self.enable_graph and has_persistent_store:
            self._load_graph_index()

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
        self._doc_chunks_lookup = None
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
            logger.warning("Chưa có BM25 artifact tại '%s'; chạy UETIndexer trước khi tìm kiếm.", bm25_path)
            return 0

        try:
            self.bm25_index = BM25Index.load(bm25_path)
            count = self.bm25_index.count()
            logger.info("Đã nạp %s chunks từ BM25 artifact '%s'.", count, bm25_path)
            return count
        except (OSError, ValueError, EOFError, ImportError, AttributeError, pickle.UnpicklingError) as error:
            logger.error("Không thể nạp BM25 artifact '%s': %s", bm25_path, error)
            return 0

    # -------------------------------------------------------------------------
    # BACKWARD-COMPATIBLE SCORING & EXPANSION DELEGATES
    # -------------------------------------------------------------------------

    def _apply_mmr(
        self,
        query: UserQuery,
        query_vector: Any,
        candidates: List[RetrievedContext],
        top_n: int,
    ) -> List[RetrievedContext]:
        return apply_mmr_selection(
            query,
            query_vector,
            candidates,
            top_n=top_n,
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            mmr_lambda=self.mmr_lambda,
            mmr_stage=self.mmr_stage,
            mmr_similarity_threshold=self.mmr_similarity_threshold,
            mmr_protect_tabular=self.mmr_protect_tabular,
        )

    def _blend_rerank_and_hybrid(
        self,
        query_text: str,
        reranked_chunks: List[RetrievedContext],
        hybrid_map: Dict[str, RetrievedContext],
        extracted_entities: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        return blend_rerank_and_hybrid(
            query_text,
            reranked_chunks,
            hybrid_map,
            reranker=self.reranker,
            rerank_blend_alpha=self.rerank_blend_alpha,
            enable_entity_anchor_boost=self.enable_entity_anchor_boost,
            extracted_entities=extracted_entities,
        )

    def _blend_rerank_and_rrf(
        self,
        reranked: List[RetrievedContext],
        rrf_norm_map: Dict[str, float],
        top_n: int,
    ) -> List[RetrievedContext]:
        return blend_rerank_and_rrf(
            reranked,
            rrf_norm_map,
            top_n,
            rerank_weight=float(getattr(self, "rerank_weight", 0.6)),
            rrf_blend_weight=float(getattr(self, "rrf_blend_weight", 0.4)),
        )

    def _get_adjacent_lookup(self) -> Dict[Tuple[str, int], DataChunk]:
        if self._adjacent_lookup is None:
            chunks = getattr(self.bm25_index, "chunks", None) or []
            self._adjacent_lookup = build_adjacent_lookup(chunks)
        return self._adjacent_lookup

    def _get_doc_chunks_lookup(self) -> Dict[str, List[Tuple[int, DataChunk]]]:
        if self._doc_chunks_lookup is None:
            self._doc_chunks_lookup = build_doc_chunks_lookup(self._get_adjacent_lookup())
        return self._doc_chunks_lookup

    @staticmethod
    def _strip_chunk_header(text: str) -> str:
        return strip_chunk_header(text)

    @staticmethod
    def _is_list_query(query_text: str) -> bool:
        return is_list_query(query_text)

    @staticmethod
    def _extract_meta_header(text: str) -> str:
        return extract_meta_header(text)

    @staticmethod
    def _is_tabular_chunk(text: str) -> bool:
        return is_tabular_chunk(text)

    @staticmethod
    def _compact_table_row(text: str) -> str:
        return compact_table_row(text)

    @staticmethod
    def _extract_header_lines(text: str) -> str:
        return extract_header_lines(text)

    def _expand_tabular_siblings(
        self,
        query_text: str,
        final_contexts: List[RetrievedContext],
    ) -> List[RetrievedContext]:
        return expand_tabular_siblings(
            query_text,
            final_contexts,
            self._get_doc_chunks_lookup(),
            enabled=self.enable_tabular_expansion,
            max_chunks=self.tabular_expansion_max_chunks,
            max_chars=self.tabular_expansion_max_chars,
        )

    def _expand_adjacent_chunks(self, final_contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        return expand_adjacent_chunks(
            final_contexts,
            self._get_adjacent_lookup(),
            enabled=self.enable_adjacent_expansion,
            expansion_top_k=self.adjacent_expansion_top_k,
            window_size=self.adjacent_window_size,
        )

    @staticmethod
    def stitch_contiguous_chunks(contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        return stitch_contiguous_chunks(contexts)

    # -------------------------------------------------------------------------
    # ENGINE ROUTING & MULTI-BRANCH RECALL
    # -------------------------------------------------------------------------

    def _retrieve_hybrid_slot_merge(
        self,
        query: UserQuery,
        effective_top_k: int,
        score_threshold: Optional[float],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievedContext]:
        """Fallback slot-merge between LightRAG and Vector when no ChunkBridge is attached."""
        query_text_clean = (query.query_text or "").strip()
        is_tabular_lookup = bool(self._TABULAR_NUMERIC_LOOKUP_RE.search(query_text_clean))

        if effective_top_k <= 1:
            graph_k = 0
        elif is_tabular_lookup:
            graph_k = 1
        else:
            graph_k = min(self.hybrid_max_graph_k, max(1, round(effective_top_k * self.hybrid_graph_ratio)))

        if graph_k > 0:
            future_graph = self._executor.submit(self.lightrag_retriever.retrieve, query=query, top_k=graph_k)
            future_vector = self._executor.submit(
                self._retrieve_vector,
                query=query,
                top_k=effective_top_k,
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
                top_k=effective_top_k,
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

        for candidate in [*graph_results[:graph_k], *vector_results]:
            norm_txt = candidate.text.strip()
            if candidate.chunk_id not in seen_ids and norm_txt not in seen_texts:
                combined.append(candidate)
                seen_ids.add(candidate.chunk_id)
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

    def retrieve(
        self,
        query: UserQuery,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
    ) -> List[RetrievedContext]:
        """Quy trình truy xuất ngữ cảnh thống nhất (Unified Retrieval Router & Multi-Engine)."""
        active_engine = (engine or self.default_engine).lower().strip()
        if active_engine == "auto":
            active_engine = self.router.route(query)
            logger.info(f"[UETRetriever] QueryRouter tự động định tuyến truy vấn sang: '{active_engine.upper()}'")

        effective_top_k = top_k or self.default_top_k

        if active_engine == "lightrag":
            graph_results = self.lightrag_retriever.retrieve(query=query, top_k=effective_top_k)
            if graph_results:
                return graph_results
            logger.info("[UETRetriever] LightRAG thuần không có dữ liệu phù hợp. Tự động fallback sang Grounded 3-Way.")
            return self._retrieve_vector(
                query=query,
                top_k=effective_top_k,
                score_threshold=score_threshold,
                filters=filters,
                route_intent="hybrid",
            )

        supports_lightrag_grounded = (
            self.enable_graph
            and self._lightrag_retriever is not None
            and hasattr(self._lightrag_retriever, "retrieve_grounded")
            and hasattr(self._lightrag_retriever, "has_chunk_bridge")
            and self._lightrag_retriever.has_chunk_bridge()
        )
        if active_engine == "hybrid" and not supports_lightrag_grounded:
            return self._retrieve_hybrid_slot_merge(query, effective_top_k, score_threshold, filters)

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
        """Thực thi BM25 Search và Dense Vector Search (kèm Soft Faculty Filter & tự động nới lỏng)."""
        allowed_faculties = None
        vector_filters = active_filters

        if active_filters and "faculty" in active_filters and self.faculty_filter_mode != "off":
            raw_fac = active_filters["faculty"]
            fac_list = list(raw_fac) if isinstance(raw_fac, (list, tuple, set)) else [raw_fac]
            allowed_faculties = set(fac_list) | set(self.shared_faculties)
            if len(allowed_faculties) > 1:
                vector_filters = dict(active_filters)
                vector_filters["faculty"] = {"$in": list(allowed_faculties)}

        bm25_results: List[RetrievedContext] = []
        try:
            raw_bm25 = self.bm25_index.search(query=query_text, top_k=max(self.bm25_top_k, 50))
            if allowed_faculties is not None:
                bm25_results = [
                    c for c in raw_bm25
                    if not c.metadata.get("faculty") or c.metadata.get("faculty") in allowed_faculties
                ][: self.bm25_top_k]
            else:
                bm25_results = raw_bm25[: self.bm25_top_k]
            logger.debug(f"[BM25] Thu được {len(bm25_results)} kết quả (sau lọc soft={allowed_faculties is not None}).")
        except Exception as exc:
            logger.warning(f"Lỗi nhánh BM25 Search: {exc}. Vẫn tiếp tục với Vector Search.")

        vector_results: List[RetrievedContext] = []
        query_vector = None
        try:
            query_vector = self.embedding_model.embed_query(query)
            vector_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=self.vector_top_k,
                filters=vector_filters,
            )
            logger.debug(f"[Vector Search] Thu được {len(vector_results)} kết quả.")
        except Exception as exc:
            logger.warning(f"Lỗi nhánh Vector Search: {exc}. Vẫn tiếp tục với BM25/Graph Search.")

        if allowed_faculties is not None and (len(bm25_results) + len(vector_results) < self.min_results_after_filter):
            logger.info(
                f"[Soft Faculty Filter] Kết quả sau lọc ({len(bm25_results)} BM25 + {len(vector_results)} Vector) "
                f"< ngưỡng {self.min_results_after_filter}. Tự động nới lỏng bỏ lọc khoa."
            )
            try:
                bm25_results = self.bm25_index.search(query=query_text, top_k=self.bm25_top_k)
            except Exception:
                pass
            if query_vector is not None:
                try:
                    vector_results = self.vector_store.search(
                        query_vector=query_vector,
                        top_k=self.vector_top_k,
                        filters=None,
                    )
                except Exception:
                    pass

        return bm25_results, vector_results, query_vector

    def _run_graph_recall(
        self,
        *,
        query: UserQuery,
        query_text: str,
        query_vector: Any,
        active_filters: Optional[Dict[str, Any]],
        has_graph: bool,
        has_lightrag_grounded: bool,
        use_pa1: bool,
        extracted_entities: List[str],
        high_level_themes: List[str],
        want_overview: bool,
    ) -> Tuple[List[RetrievedContext], Optional[str]]:
        """Execute LightRAG grounded retrieval or fallback GraphIndex search."""
        graph_results: List[RetrievedContext] = []
        lightrag_overview: Optional[str] = None

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
            except Exception as exc:
                logger.warning(f"Lỗi nhánh LightRAG Grounded Search: {exc}. Vẫn tiếp tục với BM25 + Vector.")
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
            except Exception as exc:
                logger.warning(f"Lỗi nhánh Graph Supplement Search: {exc}. Vẫn tiếp tục với BM25 + Vector.")

        return graph_results, lightrag_overview

    def _retrieve_vector(
        self,
        query: UserQuery,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        route_intent: str = "vector",
    ) -> List[RetrievedContext]:
        """Quy trình truy xuất ngữ cảnh LightRAG-First 3-Way Retrieval."""
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

        # Bước 1: PA1 Smart Router song song BM25 + Vector + Graph recall
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
            fut_recall = self._executor.submit(self._run_bm25_and_vector_recall, query, query_text, active_filters)
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

        is_tabular_lookup = bool(self._TABULAR_NUMERIC_LOOKUP_RE.search(query_text))
        is_pure_passage_lookup = bool(
            route_intent == "vector"
            and hasattr(self.router, "_pas_regex")
            and self.router._pas_regex.search(query_text)
            and not self.router._rel_regex.search(query_text)
        )
        want_overview = (route_intent == "hybrid") and not is_tabular_lookup

        graph_results, lightrag_overview = self._run_graph_recall(
            query=query,
            query_text=query_text,
            query_vector=query_vector,
            active_filters=active_filters,
            has_graph=has_graph,
            has_lightrag_grounded=has_lightrag_grounded,
            use_pa1=use_pa1,
            extracted_entities=extracted_entities,
            high_level_themes=high_level_themes,
            want_overview=want_overview,
        )

        # Bước 2: Hybrid Search Fusion (RRF hoặc Linear)
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

        if getattr(self, "merge_contiguous_chunks", False) and candidate_chunks:
            candidate_chunks = self.stitch_contiguous_chunks(candidate_chunks)

        if not candidate_chunks:
            logger.warning("Không tìm thấy đoạn văn bản nào từ BM25, Vector và GraphRAG Search.")
            return []

        hybrid_map = {c.chunk_id: c for c in candidate_chunks}

        # Bước 2.5: Pre-rerank MMR
        rerank_candidates = candidate_chunks
        if self.enable_mmr and self.mmr_stage != "post_rerank" and len(candidate_chunks) > self.mmr_top_n:
            rerank_candidates = self._apply_mmr(query, query_vector, candidate_chunks, top_n=self.mmr_top_n)

        # Bước 3: Reranker + Score Blending + Entity Anchor Boost
        effective_rerank_n = max(self.rerank_top_n, effective_top_k)
        if self.enable_mmr and self.mmr_stage == "post_rerank":
            effective_rerank_n = max(effective_rerank_n, self.mmr_top_n)

        try:
            scored_all = self.reranker.rerank(
                query=query_text,
                candidates=rerank_candidates,
                top_n=len(rerank_candidates),
            )
            ranked_chunks = self._blend_rerank_and_hybrid(
                query_text,
                scored_all,
                hybrid_map,
                extracted_entities=extracted_entities,
            )[:effective_rerank_n]
            for rank, ctx in enumerate(ranked_chunks, 1):
                ctx.rank = rank
        except Exception as exc:
            logger.warning(f"Lỗi khi chạy Reranker ({exc}). Sử dụng thứ tự từ Hybrid Search.")
            ranked_chunks = rerank_candidates[:effective_rerank_n]

        # Bước 3.5: Post-rerank MMR
        if self.enable_mmr and self.mmr_stage == "post_rerank":
            post_top_n = max(effective_top_k, self.mmr_top_n)
            if len(ranked_chunks) > post_top_n:
                ranked_chunks = self._apply_mmr(query, query_vector, ranked_chunks, top_n=post_top_n)

        # Bước 4: Top-K Selection & Tabular/Adjacent Expansion
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold
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

        final_contexts = self._expand_tabular_siblings(query_text, final_contexts)
        final_contexts = self._expand_adjacent_chunks(final_contexts)

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
    base_dir = Path(__file__).resolve().parent.parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n🔍 ĐANG KIỂM THỬ ĐƠN VỊ MODULE RETRIEVAL (SRC/RETRIEVAL)...")

    class DummyEmbedder(BaseEmbeddingModel):
        @property
        def dimension(self) -> int:
            return 384

        @property
        def model_name(self) -> str:
            return "dummy-test-embedder"

        def embed_texts(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * self.dimension for _ in texts]

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

    store = DummyStore()
    retriever = UETRetriever(
        embedding_model=DummyEmbedder(),
        vector_store=store,
        bm25_index=DummyBM25(chunks=store.mock_chunks),
        config={
            "retrieval": {
                "bm25_top_k": 3,
                "vector_top_k": 3,
                "candidate_pool_size": 3,
                "enable_mmr": True,
                "mmr_lambda": 0.6,
                "mmr_top_n": 2,
                "rerank_top_n": 2,
                "final_top_k": 2,
                "reranker_type": "heuristic",
            }
        },
    )

    test_q = UserQuery(query_text="học bổng xuất sắc UET cần điều kiện GPA bao nhiêu?")
    results = retriever.retrieve(test_q, top_k=2)

    print(f"\nCâu hỏi: {test_q.query_text}")
    print(f"Số kết quả lấy về: {len(results)}")
    for r in results:
        print(f" - [Rank {r.rank} | Score: {r.similarity_score}] {r.metadata.get('title')}: {r.text[:80]}...")

    assert len(results) > 0, "Lỗi: Không tìm thấy kết quả nào!"
    print("\n✅ KIỂM THỬ MODULE RETRIEVAL THÀNH CÔNG RỰC RỠ!\n")

"""LightRAG Support Bridge & Regex Supplement Index (src/graph/graph.py).

Subordinate modules:
- ``src/graph/entity_utils.py``: domain regexes, canonicalization, relation normalization, tokenization
- ``src/graph/hybrid_extractor.py``: ``HybridGraphExtractor`` (rule-based + optional LLM extraction)
- ``src/graph/graph_builder.py``: offline graph construction, LightRAG KG merging, vector derivation
- ``src/graph/graph_search.py``: spreading activation search and LightRAG subgraph grounding

``GraphIndex`` and ``_CompatGraphUnpickler`` remain defined directly in this module to preserve
100% pickle compatibility with ``database/graph_index.pkl``.
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.base import BaseEmbeddingModel, DataChunk, RetrievedContext
from src.graph.entity_utils import (
    FACULTY_LABELS,
    _ARTICLE_KEEP_NS_RE,
    _ARTICLE_RE,
    _BREADCRUMB_ARTICLE_COLON_RE,
    _CANONICAL_ENTITY_ALIASES,
    _CERT_RE,
    _CONCEPT_PATTERNS,
    _COURSE_CODE_RE,
    _DOC_CODE_RE,
    _DOC_PREFIX_STRIP_RE,
    _ENTITY_TYPE_WEIGHTS,
    _EXTRACTION_PROMPT_TEMPLATE,
    _GENERIC_NS_TAGS,
    _GRAPH_WORD_RE,
    _LIGHTRAG_CATEGORY_MAP,
    _NAMED_REG_RE,
    _ORG_RE,
    _PERSON_RE,
    _RELATION_SEED_PHRASES,
    _RELATION_WEIGHTS,
    _REL_ASCII_PREFIX_RULES,
    _UNIT_RE,
    _VN_STOP_BOUNDARY,
    _WRAP_PAIRS,
    _canonicalize_triple,
    _classify_sentence_relation,
    _graph_tokenize,
    _normalize_relation,
    _strip_accents,
    _strip_chunk_header_for_match,
    _trim_vn_phrase,
    canonicalize_entity_name,
    canonicalize_lightrag_entity_name,
    clean_entity_name,
    normalize_entity_id,
)
from src.graph.graph_builder import (
    build_doc_neighbors,
    build_graph,
    derive_vectors_from_chunks,
    embed_graph_items,
    embed_in_batches,
    ingest_lightrag_knowledge_graph,
    l2_normalize,
    l2_normalize_array,
    load_lightrag_entity_chroma_vectors,
)
from src.graph.graph_search import ground_lightrag_subgraph, search_graph_index
from src.graph.hybrid_extractor import HybridGraphExtractor
from src.indexing.chunks import normalize_chunks
from src.utils.helpers import BASE_DIR, get_logger

logger = get_logger("graph.bridge")


class _CompatGraphUnpickler(pickle.Unpickler):
    """Tương thích ngược cho các file graph_index.pkl cũ lưu tham chiếu 'src.indexing.graph'."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "src.indexing.graph":
            module = "src.graph.graph"
        return super().find_class(module, name)


class GraphIndex:
    """
    Cầu nối Chunk & Bộ bổ khuyết Regex/Vector phụ trợ cho LightRAGRetriever (lưu tại database/graph_index.pkl).
    """

    def __init__(
        self,
        chunks: Iterable[DataChunk] = (),
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[Any] = None,
        chunk_vectors: Optional[Dict[str, List[float]]] = None,
        llm_client: Optional[Any] = None,
        enable_llm_extraction: bool = False,
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = "tencent/hy4-flash",
        llm_domains: Optional[List[str]] = None,
        max_llm_chunks: Optional[int] = None,
        embed_batch_size: int = 256,
        cache_path: Optional[Path] = None,
        extractor: Optional[HybridGraphExtractor] = None,
        extraction_mode: str = "hybrid",
        lightrag_graph_path: Optional[Path] = None,
        semantic_units_path: Optional[Path] = None,
        entity_chroma_dir: Optional[Path] = None,
    ):
        self.chunks: List[DataChunk] = normalize_chunks(list(chunks))
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.llm_calls_count: int = 0
        self.lightrag_nodes_merged: int = 0
        self.lightrag_edges_merged: int = 0

        self.graph: nx.DiGraph = nx.DiGraph()
        self.chunk_id_to_idx: Dict[str, int] = {c.chunk_id: idx for idx, c in enumerate(self.chunks)}
        self.entity_ids: List[str] = []
        self.entity_texts: List[str] = []
        self.entity_vectors: Optional[np.ndarray] = None
        self.entity_idf: Dict[str, float] = {}
        self.entity_idf_array: Optional[np.ndarray] = None
        self.triple_records: List[Dict[str, Any]] = []
        self.triple_vectors: Optional[np.ndarray] = None
        self.token_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.entity_tokens: Dict[str, Set[str]] = {}
        self.chunk_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.chunk_to_triples: Dict[str, List[int]] = defaultdict(list)
        self.doc_chunk_neighbors: Dict[str, List[str]] = {}
        self.document_summaries: Dict[str, Dict[str, Any]] = {}
        self.doc_id_to_summary: Dict[str, Dict[str, Any]] = {}

        if not self.chunks:
            self.token_to_entities = {}
            self.chunk_to_entities = {}
            self.chunk_to_triples = {}
            return

        active_extractor = extractor or HybridGraphExtractor(
            llm_client=llm_client,
            enable_llm_extraction=enable_llm_extraction,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_domains=llm_domains,
            max_llm_chunks=max_llm_chunks,
            cache_path=cache_path,
            extraction_mode=extraction_mode,
        )

        self._build_graph(
            active_extractor,
            lightrag_graph_path=lightrag_graph_path,
            semantic_units_path=semantic_units_path,
        )
        active_extractor.flush_cache()
        self.llm_calls_count = active_extractor.llm_calls_count
        self._embed_graph_items(
            chunk_vectors=chunk_vectors,
            entity_chroma_dir=entity_chroma_dir,
        )

    def _build_graph(
        self,
        extractor: HybridGraphExtractor,
        lightrag_graph_path: Optional[Path] = None,
        semantic_units_path: Optional[Path] = None,
    ) -> None:
        build_graph(
            self,
            extractor,
            lightrag_graph_path=lightrag_graph_path,
            semantic_units_path=semantic_units_path,
        )

    def _ingest_lightrag_knowledge_graph(
        self,
        graph_path: Path,
        units_path: Optional[Path],
        triple_key_to_idx: Dict[Tuple[str, str, str], int],
    ) -> None:
        ingest_lightrag_knowledge_graph(self, graph_path, units_path, triple_key_to_idx)

    def _build_doc_neighbors(self) -> None:
        build_doc_neighbors(self)

    @staticmethod
    def _l2_normalize_array(arr: np.ndarray) -> np.ndarray:
        return l2_normalize_array(arr)

    @staticmethod
    def _l2_normalize(vectors: List[List[float]]) -> Optional[np.ndarray]:
        return l2_normalize(vectors)

    def _embed_in_batches(self, texts: List[str]) -> Optional[np.ndarray]:
        return embed_in_batches(self, texts)

    @staticmethod
    def _load_lightrag_entity_chroma_vectors(entity_chroma_dir: Optional[Path]) -> Dict[str, np.ndarray]:
        return load_lightrag_entity_chroma_vectors(entity_chroma_dir)

    def _derive_vectors_from_chunks(
        self,
        chunk_vectors: Dict[str, List[float]],
        precomputed_entity_vecs: Optional[Dict[str, np.ndarray]] = None,
    ) -> bool:
        return derive_vectors_from_chunks(self, chunk_vectors, precomputed_entity_vecs=precomputed_entity_vecs)

    def _embed_graph_items(
        self,
        chunk_vectors: Optional[Dict[str, List[float]]] = None,
        entity_chroma_dir: Optional[Path] = None,
    ) -> None:
        embed_graph_items(self, chunk_vectors=chunk_vectors, entity_chroma_dir=entity_chroma_dir)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["embedding_model"] = None
        state["vector_store"] = None
        state["llm_client"] = None
        if isinstance(state.get("entity_vectors"), np.ndarray) and state["entity_vectors"].shape[0] >= 500:
            state["entity_vectors"] = state["entity_vectors"].astype(np.float16)
        if isinstance(state.get("triple_vectors"), np.ndarray) and state["triple_vectors"].shape[0] >= 500:
            state["triple_vectors"] = state["triple_vectors"].astype(np.float16)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.embedding_model = None
        self.vector_store = None
        self.llm_client = None
        if isinstance(getattr(self, "entity_vectors", None), np.ndarray) and self.entity_vectors.dtype != np.float32:
            self.entity_vectors = self.entity_vectors.astype(np.float32)
        if isinstance(getattr(self, "triple_vectors", None), np.ndarray) and self.triple_vectors.dtype != np.float32:
            self.triple_vectors = self.triple_vectors.astype(np.float32)
        if not hasattr(self, "document_summaries"):
            self.document_summaries = {}
        if not hasattr(self, "doc_id_to_summary"):
            self.doc_id_to_summary = {}

    def count(self) -> int:
        return len(self.chunks)

    def entity_count(self) -> int:
        return len(self.entity_ids)

    def triple_count(self) -> int:
        return len(self.triple_records)

    def save(self, path: str | Path) -> None:
        """Lưu GraphIndex ra ổ đĩa bằng ghi nguyên tử (atomic replacement)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                pickle.dump({"version": 1, "index": self}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model: Optional[BaseEmbeddingModel] = None,
    ) -> "GraphIndex":
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Không tìm thấy file Graph index tại: {target}")
        with target.open("rb") as stream:
            state = _CompatGraphUnpickler(stream).load()

        index: Optional[GraphIndex] = None
        if isinstance(state, cls):
            index = state
        elif isinstance(state, dict) and state.get("version") == 1:
            raw_index = state.get("index")
            if isinstance(raw_index, cls):
                index = raw_index

        if index is None:
            raise ValueError(f"Invalid or incompatible Graph artifact: {target}")

        index.embedding_model = embedding_model
        return index

    @staticmethod
    def _matches_filters(chunk: DataChunk, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        meta = chunk.metadata or {}
        for key, val in filters.items():
            if val is None:
                continue
            actual = meta.get(key)
            if actual is None:
                return False
            if isinstance(val, (list, tuple, set)):
                allowed = {str(v).casefold() for v in val}
                if str(actual).casefold() not in allowed:
                    return False
            elif str(actual).casefold() != str(val).casefold():
                return False
        return True

    def _max_edge_relation_weight(self, u: str, v: str) -> float:
        best = 1.0
        found = False
        for src, dst in ((u, v), (v, u)):
            if self.graph.has_edge(src, dst):
                for rel in self.graph[src][dst].get("relations", ()):
                    w = _RELATION_WEIGHTS.get(rel, 1.0)
                    if not found or w > best:
                        best = w
                        found = True
        return best if found else 1.0

    def search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 50,
        seed_top_m: int = 10,
        max_hops: int = 2,
        hop_decay: float = 0.65,
        adjacent_alpha: float = 0.18,
        filters: Optional[Dict[str, Any]] = None,
        extracted_entities: Optional[List[str]] = None,
        high_level_themes: Optional[List[str]] = None,
        skip_thesaurus_scan: bool = False,
    ) -> List[RetrievedContext]:
        return search_graph_index(
            self,
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            seed_top_m=seed_top_m,
            max_hops=max_hops,
            hop_decay=hop_decay,
            adjacent_alpha=adjacent_alpha,
            filters=filters,
            extracted_entities=extracted_entities,
            high_level_themes=high_level_themes,
            skip_thesaurus_scan=skip_thesaurus_scan,
        )

    def max_hub_df(self, ratio: float = 0.12) -> int:
        """Ngưỡng số lượng chunk tối đa trước khi coi một thực thể là Siêu nút (Hub)."""
        total_chunks = len(self.chunks)
        return max(250, int(ratio * total_chunks)) if total_chunks >= 50 else total_chunks + 1

    def resolve_lightrag_node_id(self, raw_name: str, etype: str = "") -> Optional[str]:
        """Ánh xạ tên thực thể từ LightRAG graph.json sang entity_id trong cầu nối."""
        if not raw_name:
            return None
        canon = canonicalize_lightrag_entity_name(raw_name, etype)
        if canon:
            eid = normalize_entity_id(canon)
            if eid and self.graph.has_node(eid):
                return eid
        eid_raw = normalize_entity_id(raw_name)
        if eid_raw and self.graph.has_node(eid_raw):
            return eid_raw
        stripped = re.sub(r"^\[[^\]]+\]\s*", "", str(raw_name)).strip()
        if stripped:
            eid_stripped = normalize_entity_id(stripped)
            if eid_stripped and self.graph.has_node(eid_stripped):
                return eid_stripped
        return None

    def get_entity_idf(self, raw_name: str, etype: str = "") -> float:
        """Trả về điểm độ hiếm Entity IDF từ kho chunk thực tế cho LightRAGRetriever."""
        eid = self.resolve_lightrag_node_id(raw_name, etype)
        if eid:
            return float(getattr(self, "entity_idf", {}).get(eid, 1.0))
        return 1.0

    def is_hub_entity(self, raw_name: str, ratio: float = 0.12) -> bool:
        """Kiểm tra thực thể có vượt ngưỡng Siêu nút (Hub Gating) trên tập chunk hay không."""
        eid = self.resolve_lightrag_node_id(raw_name)
        if not eid or not self.graph.has_node(eid):
            return False
        return len(self.graph.nodes[eid].get("chunks", ())) > self.max_hub_df(ratio)

    @staticmethod
    def has_hard_code_entities(query: str) -> bool:
        """Kiểm tra câu hỏi có chứa mã học phần, số quyết định, điều khoản hoặc chứng chỉ cứng."""
        clean = (query or "").strip()
        if not clean:
            return False
        return any(rx.search(clean) is not None for rx in (_DOC_CODE_RE, _COURSE_CODE_RE, _ARTICLE_RE, _CERT_RE))

    def _get_title_to_doc_ids(self) -> Dict[str, Set[str]]:
        """Bảng ánh xạ lười từ tiêu đề văn bản (lowercase) sang tập document_id trong chunks."""
        cached = getattr(self, "_cached_title_to_doc_ids", None)
        if cached is not None:
            return cached
        mapping: Dict[str, Set[str]] = defaultdict(set)
        for ch in self.chunks:
            meta = ch.metadata or {}
            did = str(ch.document_id or meta.get("document_id") or meta.get("doc_id") or "").strip()
            t_low = str(meta.get("title") or "").strip().casefold()
            if did and t_low:
                mapping[t_low].add(did)
            if did:
                mapping[did.casefold()].add(did)
        self._cached_title_to_doc_ids = dict(mapping)
        return self._cached_title_to_doc_ids

    def ground_lightrag_subgraph(
        self,
        query: str,
        seed_entities: List[Tuple[str, float]],
        activated_nodes: Dict[str, float],
        activated_edges: List[Dict[str, Any]],
        high_level_blocks: Optional[List[Dict[str, Any]]] = None,
        query_vector: Optional[List[float]] = None,
        top_k: int = 60,
        filters: Optional[Dict[str, Any]] = None,
        supplement_weight: float = 0.65,
        adjacent_alpha: float = 0.18,
        extracted_entities: Optional[List[str]] = None,
        high_level_themes: Optional[List[str]] = None,
        seed_top_m: int = 10,
        max_hops: int = 2,
        hop_decay: float = 0.65,
    ) -> List[RetrievedContext]:
        return ground_lightrag_subgraph(
            self,
            query=query,
            seed_entities=seed_entities,
            activated_nodes=activated_nodes,
            activated_edges=activated_edges,
            high_level_blocks=high_level_blocks,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            supplement_weight=supplement_weight,
            adjacent_alpha=adjacent_alpha,
            extracted_entities=extracted_entities,
            high_level_themes=high_level_themes,
            seed_top_m=seed_top_m,
            max_hops=max_hops,
            hop_decay=hop_decay,
        )


LightRAGChunkBridge = GraphIndex
RegexSupplementGraph = GraphIndex

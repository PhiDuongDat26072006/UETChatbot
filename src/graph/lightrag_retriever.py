"""
src/graph/lightrag_retriever.py - Động cơ Truy xuất Đồ thị Tri thức Chính (Primary Knowledge Graph Dual-Level Retriever).
Kiến trúc LightRAG-First kết hợp cầu nối phụ trợ (LightRAGChunkBridge):
  1. LLM Query Entity Extraction (PA1) + Academic Thesaurus Normalization + Forbidden Hub Filtering.
  2. Semantic Entity Resolution siêu tốc (<2ms) bằng ma trận NumPy RAM nạp từ entity_chroma (14.037 thực thể BGE-M3),
     có hỗ trợ fallback sang ChromaDB collection trực tiếp khi kiểm thử mock.
  3. Low-level Retrieval nâng cấp: Lan truyền kích hoạt có trọng số (Entity IDF + Hub Gating + Relation Weights
     + Co-Activation Bonus) trong bán kính 1-2 hops.
  4. High-level Retrieval: Khai thác tóm tắt văn bản toàn cục (Pass 1) kết hợp high_level_themes.
  5. Dual-Output (retrieve_grounded):
       - Grounded Chunks: Ánh xạ subgraph 2 tầng của LightRAG về DataChunk nguyên văn (chunked_data) qua
         LightRAGChunkBridge (kèm bổ khuyết Regex 0.65x cho mã cứng/chunk mới) để đưa vào 3-Way RRF & Reranker.
       - LightRAG Synthesis Overview: Tổng hợp bản đồ liên kết quy chế 2 tầng gọn nhẹ bơm thẳng vào Prompt LLM.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.base import BaseEmbeddingModel, RetrievedContext, UserQuery
from src.graph.thesaurus_utils import (
    AcademicThesaurus,
    detect_query_faculties,
    get_thesaurus,
)
from src.graph.query_entity_extractor import (
    QueryEntityExtractor,
    is_forbidden_entity,
)
from src.graph.lightrag_matching import (
    extract_and_match_entities,
    extract_high_level_summaries,
    extract_low_level_subgraph_detailed,
    resolve_term_in_graph,
    scan_graph_node_ngrams,
    scan_thesaurus_ngrams,
    search_entity_vectors,
)
from src.graph.lightrag_formatter import (
    build_synthesis_overview,
    format_dual_level_contexts,
    format_edge_line,
)
from src.utils.helpers import get_ascii_safe_chroma_path, get_logger, load_yaml_config

logger = get_logger("graph.lightrag_retriever")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
_WORD_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)

# Trọng số ưu tiên theo ý nghĩa pháp quy của các loại cạnh trong LightRAG graph.json
LIGHTRAG_RELATION_WEIGHTS: Dict[str, float] = {
    "điều_kiện_tiên_quyết": 1.35,
    "ngoại_lệ_cho": 1.35,
    "chế_tài_xử_lý": 1.30,
    "bổ_sung_cho": 1.30,
    "thay_thế_bởi": 1.30,
    "dẫn_chiếu_đến": 1.25,
    "quy_định_bởi": 1.20,
    "áp_dụng_cho": 1.15,
    "bao_gồm": 1.10,
    "quản_lý_bởi": 1.05,
    "thuộc_về": 0.90,
    "liên_quan_đến": 0.85,
}

LIGHTRAG_CATEGORY_WEIGHTS: Dict[str, float] = {
    "VAN_BAN": 1.25,
    "BANG_TRA_CUU": 1.20,
    "DIEU_KHOAN": 1.20,
    "DIEU_KIEN": 1.15,
    "QUY_TRINH": 1.15,
    "KHOA_NGANH": 1.10,
    "NHAN_SU": 1.10,
    "DON_VI": 0.90,
    "OTHER": 0.85,
}


class LightRAGRetriever:
    """
    Động cơ truy xuất Đồ thị Tri thức 2 tầng chính (Primary Dual-Level Knowledge Graph Retriever).
    Vừa hỗ trợ xuất khối ngữ cảnh đồ thị độc lập (retrieve), vừa hỗ trợ xuất Kép (retrieve_grounded:
    DataChunk gốc từ chunked_data cho RRF/Reranker + Bản đồ liên kết 2 tầng cho LLM).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        graph_path: Optional[str | Path] = None,
        thesaurus: Optional[AcademicThesaurus] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        query_extractor: Optional[QueryEntityExtractor] = None,
        entity_collection: Optional[Any] = None,
        chunk_bridge: Optional[Any] = None,
    ):
        raw_cfg = config or load_yaml_config()
        lightrag_cfg = raw_cfg.get("lightrag", {})

        self.enabled: bool = lightrag_cfg.get("enabled", True)
        self.default_mode: str = lightrag_cfg.get("retrieval_mode", "hybrid")
        self.max_entities_per_query: int = int(lightrag_cfg.get("max_entities_per_query", 15))
        self.subgraph_depth: int = int(lightrag_cfg.get("subgraph_depth", 2))

        # Cấu hình PA1 (LLM Query Entity Extraction + Semantic Entity Vector Search)
        self.use_llm_query_extraction: bool = bool(lightrag_cfg.get("use_llm_query_extraction", True))
        self.entity_similarity_threshold: float = float(
            lightrag_cfg.get("entity_similarity_threshold", 0.72)
        )
        self.entity_vector_top_k: int = int(lightrag_cfg.get("entity_vector_top_k", 3))

        # Cấu hình Dual-Output & Cầu nối bổ khuyết (feat/graph helper)
        self.supplement_weight: float = float(lightrag_cfg.get("supplement_weight", 0.65))
        self.enable_synthesis_overview: bool = bool(lightrag_cfg.get("enable_synthesis_overview", True))
        self.synthesis_max_chars: int = int(lightrag_cfg.get("synthesis_max_chars", 900))
        self.chunk_bridge: Optional[Any] = chunk_bridge

        # Đường dẫn file đồ thị tri thức
        cfg_graph_path = graph_path or lightrag_cfg.get("graph_path", "database/lightrag_db/graph.json")
        self.graph_path = Path(cfg_graph_path) if Path(cfg_graph_path).is_absolute() else BASE_DIR / cfg_graph_path

        # Đường dẫn thư mục ChromaDB chứa vector của các thực thể LightRAG
        cfg_vec_dir = lightrag_cfg.get("entity_vector_dir", "database/lightrag_db/entity_chroma")
        resolved_vec_dir = Path(cfg_vec_dir) if Path(cfg_vec_dir).is_absolute() else BASE_DIR / cfg_vec_dir
        fallback_chroma_dir = BASE_DIR / "database" / "lightrag_db" / "entity_chroma"
        if not (resolved_vec_dir / "chroma.sqlite3").exists() and (fallback_chroma_dir / "chroma.sqlite3").exists():
            resolved_vec_dir = fallback_chroma_dir
        self.entity_vector_dir: Path = resolved_vec_dir

        # Tài nguyên dùng chung (tránh nạp trùng mô hình tốn RAM)
        self.thesaurus = thesaurus or get_thesaurus()
        self.embedding_model = embedding_model
        self.query_extractor = query_extractor or (
            QueryEntityExtractor(config=raw_cfg) if self.enabled else None
        )
        self._entity_collection = entity_collection

        # Cấu trúc lưu trữ in-memory đồ thị, thống kê IDF & ma trận NumPy RAM
        self.entities: Dict[str, Dict[str, Any]] = {}
        self._lower_to_canonical: Dict[str, str] = {}
        self._alias_to_graph_entity: Dict[str, str] = {}
        self.adjacency: Dict[str, List[Dict[str, Any]]] = {}
        self.reverse_adjacency: Dict[str, List[Dict[str, Any]]] = {}
        self.document_summaries: Dict[str, Dict[str, Any]] = {}
        self._entity_idf: Dict[str, float] = {}
        self._max_hub_degree: int = 120
        self._entity_vec_names: List[str] = []
        self._entity_vec_matrix: Optional[np.ndarray] = None

        if self.enabled:
            self._load_graph()
            if self._entity_collection is None:
                self._init_entity_collection()

    def set_chunk_bridge(self, bridge: Optional[Any]) -> None:
        """Gắn hoặc cập nhật cầu nối LightRAGChunkBridge (GraphIndex phụ trợ)."""
        self.chunk_bridge = bridge

    def has_chunk_bridge(self) -> bool:
        """Kiểm tra cầu nối sang DataChunk (chunked_data) đã sẵn sàng hay chưa."""
        if self.chunk_bridge is None:
            return False
        cnt_fn = getattr(self.chunk_bridge, "count", None)
        return callable(cnt_fn) and int(cnt_fn()) > 0

    def _init_entity_collection(self) -> None:
        """
        Kết nối tới ChromaDB collection chứa vector thực thể (lightrag_entity_collection)
        và nạp toàn bộ vector chuẩn hóa L2 lên ma trận NumPy RAM để tìm kiếm <2ms không tốn I/O đĩa.
        """
        sqlite_file = self.entity_vector_dir / "chroma.sqlite3"
        if not sqlite_file.exists():
            logger.debug(f"[LightRAGRetriever] Không tìm thấy entity_chroma tại '{self.entity_vector_dir}'.")
            return

        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=get_ascii_safe_chroma_path(self.entity_vector_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            col = client.get_collection(name="lightrag_entity_collection")
            self._entity_collection = col
            total_vecs = col.count()
            logger.info(
                f"[LightRAGRetriever] Đã kết nối kho vector thực thể 'lightrag_entity_collection' "
                f"({total_vecs} vectors)."
            )

            if total_vecs > 0 and self.entities:
                loaded_names: List[str] = []
                loaded_vecs: List[np.ndarray] = []
                seen_resolved: Set[str] = set()
                dim: Optional[int] = None
                page_size = 2500

                for offset in range(0, total_vecs, page_size):
                    fetched = col.get(
                        limit=page_size,
                        offset=offset,
                        include=["embeddings", "metadatas"],
                    )
                    embs = fetched.get("embeddings")
                    metas = fetched.get("metadatas") or []
                    if embs is None or len(embs) == 0:
                        continue
                    for emb, meta in zip(embs, metas):
                        if emb is None or not isinstance(meta, dict):
                            continue
                        cand_name = str(meta.get("canonical_name") or "").strip()
                        if not cand_name or is_forbidden_entity(cand_name):
                            continue
                        resolved = self._resolve_term_in_graph(cand_name)
                        if not resolved or resolved in seen_resolved:
                            continue
                        arr = np.asarray(emb, dtype=np.float32)
                        if arr.ndim != 1 or arr.shape[0] == 0:
                            continue
                        if dim is None:
                            dim = int(arr.shape[0])
                        elif arr.shape[0] != dim:
                            continue
                        norm = float(np.linalg.norm(arr))
                        if norm <= 0.0:
                            continue
                        seen_resolved.add(resolved)
                        loaded_names.append(resolved)
                        loaded_vecs.append(arr / norm)

                if loaded_vecs:
                    self._entity_vec_names = loaded_names
                    self._entity_vec_matrix = np.vstack(loaded_vecs).astype(np.float32)
                    logger.info(
                        f"[LightRAGRetriever] Đã nạp {len(self._entity_vec_names)} vector thực thể BGE-M3 "
                        f"({self._entity_vec_matrix.shape[1]}-d) lên RAM NumPy."
                    )
        except Exception as e:
            logger.warning(f"[LightRAGRetriever] Không thể mở lightrag_entity_collection: {e}")
            self._entity_collection = None
            self._entity_vec_names = []
            self._entity_vec_matrix = None

    def _load_graph(self) -> bool:
        """Nạp dữ liệu đồ thị tri thức từ file JSON, xây dựng chỉ mục tra cứu O(1) và thống kê Entity IDF / Hub Gating."""
        if not self.graph_path.exists():
            logger.info(
                f"[LightRAGRetriever] Chưa tìm thấy file đồ thị tại '{self.graph_path}'. "
                f"Sẵn sàng hoạt động khi đồ thị được xây dựng (chế độ Graceful Fallback)."
            )
            return False

        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_entities = data.get("entities", {})
            if not raw_entities and "nodes" in data:
                raw_entities = {n["canonical_name"]: n for n in data.get("nodes", []) if "canonical_name" in n}
            elif isinstance(raw_entities, list):
                raw_entities = {n["canonical_name"]: n for n in raw_entities if "canonical_name" in n}

            raw_relations = data.get("relations", []) or data.get("edges", [])
            raw_doc_summaries = data.get("document_summaries", {})

            # Lọc sạch các node chung chung thuộc FORBIDDEN_ENTITIES nếu còn sót trong file đồ thị cũ
            self.entities = {
                ent_k: ent_v
                for ent_k, ent_v in raw_entities.items()
                if ent_k and not is_forbidden_entity(ent_k, allow_school_names=True)
            }
            self.document_summaries = raw_doc_summaries

            # Xây dựng bảng tra cứu không phân biệt hoa/thường & bí danh sạch
            self._lower_to_canonical = {}
            self._alias_to_graph_entity = {}
            all_unit_ids: Set[str] = set()

            for ent_name, ent_data in self.entities.items():
                ent_low = ent_name.strip().lower()
                if ent_low:
                    self._lower_to_canonical[ent_low] = ent_name
                stripped_prefix_name = re.sub(r"^\[[^\]]+\]\s*", "", ent_name).strip().lower()
                if (
                    stripped_prefix_name
                    and stripped_prefix_name != ent_low
                    and len(stripped_prefix_name) >= 3
                    and not is_forbidden_entity(stripped_prefix_name)
                ):
                    self._alias_to_graph_entity.setdefault(stripped_prefix_name, ent_name)
                for alias in ent_data.get("aliases", []):
                    alias_low = str(alias).strip().lower()
                    if len(alias_low) >= 3 and not is_forbidden_entity(alias_low):
                        self._alias_to_graph_entity.setdefault(alias_low, ent_name)
                for uid in ent_data.get("source_unit_ids", ()) or ():
                    if uid:
                        all_unit_ids.add(str(uid))

            # Xây dựng danh sách kề cho đồ thị (bỏ qua cạnh nối vào thực thể cấm)
            self.adjacency = {}
            self.reverse_adjacency = {}

            for rel in raw_relations:
                src = rel.get("source")
                tgt = rel.get("target")
                if (
                    not src
                    or not tgt
                    or is_forbidden_entity(src, allow_school_names=True)
                    or is_forbidden_entity(tgt, allow_school_names=True)
                ):
                    continue

                self.adjacency.setdefault(src, []).append(rel)
                self.reverse_adjacency.setdefault(tgt, []).append(rel)

            # Tính toán Entity IDF & ngưỡng Hub Gating trực tiếp trên đồ thị LightRAG
            total_units = max(len(all_unit_ids), len(self.entities), 10)
            idf_denom = max(math.log(total_units + 1.0), 1e-6)
            self._entity_idf = {}
            degrees: List[int] = []

            for ent_name, ent_data in self.entities.items():
                deg = len(self.adjacency.get(ent_name, ())) + len(self.reverse_adjacency.get(ent_name, ()))
                degrees.append(deg)
                unit_df = max(1, len(ent_data.get("source_unit_ids") or ()), deg // 3)
                raw_idf = max(0.0, math.log((total_units + 1.0) / (unit_df + 0.5)) / idf_denom)
                cat = str(ent_data.get("entity_type") or ent_data.get("category") or "OTHER").upper()
                cat_w = LIGHTRAG_CATEGORY_WEIGHTS.get(cat, 1.0)
                self._entity_idf[ent_name] = round(min(1.35, max(0.20, raw_idf * cat_w)), 4)

            if len(degrees) >= 50:
                sorted_deg = sorted(degrees)
                p99 = sorted_deg[min(len(sorted_deg) - 1, int(0.99 * len(sorted_deg)))]
                self._max_hub_degree = max(80, p99)
            else:
                self._max_hub_degree = 120

            logger.info(
                f"[LightRAGRetriever] Đã nạp thành công đồ thị tri thức: "
                f"{len(self.entities)} thực thể, {len(raw_relations)} quan hệ, "
                f"{len(self.document_summaries)} tóm tắt tài liệu từ '{self.graph_path}' "
                f"(hub_degree_cap={self._max_hub_degree})."
            )
            return True

        except Exception as e:
            logger.error(f"[LightRAGRetriever] Lỗi khi đọc file đồ thị '{self.graph_path}': {e}")
            return False

    def is_available(self) -> bool:
        """Kiểm tra xem đồ thị tri thức đã sẵn sàng phục vụ truy vấn hay chưa."""
        return self.enabled and len(self.entities) > 0

    def _effective_node_idf(self, node_name: str) -> float:
        """Kết hợp IDF nội tại của graph.json với IDF mức chunk từ LightRAGChunkBridge (nếu có)."""
        graph_idf = self._entity_idf.get(node_name, 1.0)
        if self.chunk_bridge is not None and hasattr(self.chunk_bridge, "get_entity_idf"):
            bridge_idf = float(self.chunk_bridge.get_entity_idf(node_name))
            return round(math.sqrt(max(0.05, graph_idf * bridge_idf)), 4)
        return graph_idf

    def _is_hub_node(self, node_name: str) -> bool:
        """Kiểm tra siêu nút để chặn bùng nổ lan truyền đa bước."""
        deg = len(self.adjacency.get(node_name, ())) + len(self.reverse_adjacency.get(node_name, ()))
        if deg > self._max_hub_degree:
            return True
        if self.chunk_bridge is not None and hasattr(self.chunk_bridge, "is_hub_entity"):
            return bool(self.chunk_bridge.is_hub_entity(node_name))
        return False

    def _scan_thesaurus_ngrams(self, query_text: str) -> List[Tuple[str, float]]:
        return scan_thesaurus_ngrams(self, query_text)

    def _scan_graph_node_ngrams(self, query_text: str) -> List[Tuple[str, float]]:
        return scan_graph_node_ngrams(self, query_text)

    def _resolve_term_in_graph(self, term: str) -> Optional[str]:
        return resolve_term_in_graph(self, term)

    def _search_entity_vectors(
        self,
        terms: List[str],
        threshold: Optional[float] = None,
        precomputed_query_text: Optional[str] = None,
        precomputed_query_vector: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        return search_entity_vectors(
            self,
            terms=terms,
            threshold=threshold,
            precomputed_query_text=precomputed_query_text,
            precomputed_query_vector=precomputed_query_vector,
        )

    def extract_and_match_entities(
        self,
        query_text: str,
        pre_extracted_entities: Optional[List[str]] = None,
        pre_extracted_themes: Optional[List[str]] = None,
        allow_llm: Optional[bool] = None,
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        return extract_and_match_entities(
            self,
            query_text=query_text,
            pre_extracted_entities=pre_extracted_entities,
            pre_extracted_themes=pre_extracted_themes,
            allow_llm=allow_llm,
            query_vector=query_vector,
        )

    def find_seed_entities(self, query_text: str) -> List[Tuple[str, float]]:
        """API tương thích ngược: Trả về danh sách (canonical_name, confidence_score)."""
        seeds, _ = self.extract_and_match_entities(query_text)
        return seeds

    def _extract_low_level_subgraph_detailed(
        self,
        seed_entities: List[Tuple[str, float]],
        depth: int = 1,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]]]:
        return extract_low_level_subgraph_detailed(
            self,
            seed_entities=seed_entities,
            depth=depth,
            relation_weights=LIGHTRAG_RELATION_WEIGHTS,
        )

    def _extract_low_level_subgraph(
        self, seed_entities: List[Tuple[str, float]], depth: int = 1
    ) -> List[Dict[str, Any]]:
        """API tương thích ngược: Trả về danh sách Low-level entity blocks đã xếp hạng."""
        blocks, _, _ = self._extract_low_level_subgraph_detailed(seed_entities, depth=depth)
        return blocks

    def _extract_high_level_summaries(
        self,
        query_text: str,
        seed_entities: List[Tuple[str, float]],
        high_level_themes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return extract_high_level_summaries(
            self,
            query_text=query_text,
            seed_entities=seed_entities,
            high_level_themes=high_level_themes,
        )

    @staticmethod
    def _format_edge_line(rel: Dict[str, Any], max_ev_chars: int = 110) -> str:
        return format_edge_line(rel, max_ev_chars=max_ev_chars)

    def build_synthesis_overview(
        self,
        high_level_blocks: List[Dict[str, Any]],
        low_level_blocks: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
    ) -> Optional[str]:
        return build_synthesis_overview(
            self,
            high_level_blocks=high_level_blocks,
            low_level_blocks=low_level_blocks,
            max_chars=max_chars,
        )

    def _format_dual_level_contexts(
        self,
        high_level_blocks: List[Dict[str, Any]],
        low_level_blocks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[RetrievedContext]:
        return format_dual_level_contexts(
            self,
            high_level_blocks=high_level_blocks,
            low_level_blocks=low_level_blocks,
            top_k=top_k,
        )

    def retrieve(
        self,
        query: UserQuery,
        top_k: int = 5,
        mode: Optional[str] = None,
    ) -> List[RetrievedContext]:
        """
        Thực thi truy xuất LightRAG 2 tầng độc lập (dùng cho --engine lightrag hoặc khi không gắn chunk_bridge).
        """
        query_text = (query.query_text or "").strip()
        if not query_text or not self.enabled:
            return []

        if not self.is_available():
            if not self._load_graph():
                return []

        active_mode = mode or self.default_mode
        seed_entities, high_level_themes = self.extract_and_match_entities(query_text)
        logger.info(
            f"[LightRAGRetriever] Seed entities={seed_entities[:5]} | Themes={high_level_themes}"
        )

        high_level_blocks = (
            self._extract_high_level_summaries(query_text, seed_entities, high_level_themes=high_level_themes)
            if active_mode in ["high_level", "hybrid"]
            else []
        )
        low_level_blocks = (
            self._extract_low_level_subgraph(seed_entities, depth=self.subgraph_depth)
            if (active_mode in ["low_level", "hybrid"] and seed_entities)
            else []
        )

        final_contexts = self._format_dual_level_contexts(high_level_blocks, low_level_blocks, top_k=top_k)
        logger.info(
            f"[LightRAGRetriever] Đã truy xuất {len(final_contexts)} ngữ cảnh đồ thị cho query '{query_text[:60]}...'"
        )
        return final_contexts

    def retrieve_grounded(
        self,
        query: UserQuery,
        query_vector: Optional[List[float]] = None,
        top_k: int = 60,
        filters: Optional[Dict[str, Any]] = None,
        extracted_entities: Optional[List[str]] = None,
        high_level_themes: Optional[List[str]] = None,
        allow_llm: Optional[bool] = None,
        include_overview: bool = True,
        seed_top_m: int = 10,
        max_hops: int = 2,
        hop_decay: float = 0.65,
    ) -> Tuple[List[RetrievedContext], Optional[str]]:
        """
        [ĐỘNG CƠ ĐỒ THỊ CHÍNH - DUAL OUTPUT]:
          1. Thực thi suy luận 2 tầng trên đồ thị LightRAG (High-level Pass 1 + Low-level Pass 2 có IDF/Hub-Gating).
          2. Tổng hợp Bản đồ Liên kết Quy chế (lightrag_overview) cho câu hỏi quan hệ/tổng hợp.
          3. Dùng LightRAGChunkBridge ánh xạ subgraph về DataChunk nguyên văn (chunked_data) kèm bổ khuyết
             Regex/Vector (0.65x) của feat/graph cho mã học phần/số QĐ/chunk mới, sẵn sàng cho 3-Way RRF & Reranker.
        """
        query_text = (query.query_text or "").strip()
        if not query_text or top_k <= 0:
            return [], None

        graph_ready = self.is_available() or (self.enabled and self._load_graph())

        # Nếu graph.json chưa dựng (vd: môi trường test chỉ nạp chunk_bridge/GraphIndex), fallback sang cầu nối Regex
        if not graph_ready:
            if self.has_chunk_bridge() and hasattr(self.chunk_bridge, "search"):
                fallback_hits = self.chunk_bridge.search(
                    query=query_text,
                    query_vector=query_vector,
                    top_k=top_k,
                    seed_top_m=seed_top_m,
                    max_hops=max_hops,
                    hop_decay=hop_decay,
                    filters=filters,
                    extracted_entities=extracted_entities,
                    high_level_themes=high_level_themes,
                )
                return fallback_hits, None
            return [], None

        seed_entities, resolved_themes = self.extract_and_match_entities(
            query_text,
            pre_extracted_entities=extracted_entities,
            pre_extracted_themes=high_level_themes,
            allow_llm=allow_llm,
            query_vector=query_vector,
        )

        high_level_blocks = self._extract_high_level_summaries(
            query_text,
            seed_entities,
            high_level_themes=resolved_themes,
        )
        low_level_blocks, activated_nodes, scored_edges = self._extract_low_level_subgraph_detailed(
            seed_entities,
            depth=self.subgraph_depth,
        )

        overview = (
            self.build_synthesis_overview(high_level_blocks, low_level_blocks)
            if (include_overview and self.enable_synthesis_overview)
            else None
        )

        if self.has_chunk_bridge() and hasattr(self.chunk_bridge, "ground_lightrag_subgraph"):
            grounded_chunks = self.chunk_bridge.ground_lightrag_subgraph(
                query=query_text,
                seed_entities=seed_entities,
                activated_nodes=activated_nodes,
                activated_edges=scored_edges,
                high_level_blocks=high_level_blocks,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                supplement_weight=self.supplement_weight,
                extracted_entities=extracted_entities,
                high_level_themes=resolved_themes,
                seed_top_m=seed_top_m,
                max_hops=max_hops,
                hop_decay=hop_decay,
            )
            logger.debug(
                "[LightRAGRetriever] Grounded output: %d chunks (seeds=%d, edges=%d, overview=%s).",
                len(grounded_chunks),
                len(seed_entities),
                len(scored_edges),
                bool(overview),
            )
            return grounded_chunks, overview

        # Nếu chưa có chunk_bridge, trả về trực tiếp các khối Dual-Level
        direct_contexts = self._format_dual_level_contexts(high_level_blocks, low_level_blocks, top_k=top_k)
        return direct_contexts, overview

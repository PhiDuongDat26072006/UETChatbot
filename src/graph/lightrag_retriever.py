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
from src.utils.helpers import get_logger, load_yaml_config

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
                path=str(self.entity_vector_dir),
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
        """
        Quét n-grams nguyên từ (từ dài xuống ngắn, không chồng lấp) qua Academic Thesaurus.
        Loại bỏ hoàn toàn lỗi bắt nhầm chuỗi con (như bắt 'Thông tin' trong 'Công nghệ thông tin').
        """
        tokens = _WORD_TOKEN_RE.findall(query_text.lower())
        if not tokens:
            return []

        covered_positions: Set[int] = set()
        found: Dict[str, float] = {}

        for n in range(min(6, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                span = set(range(i, i + n))
                if span & covered_positions:
                    continue
                phrase = " ".join(tokens[i : i + n])
                if is_forbidden_entity(phrase):
                    continue
                canonical = self.thesaurus.get_canonical(phrase)
                if canonical and not is_forbidden_entity(canonical):
                    covered_positions.update(span)
                    score = 1.0 if phrase in self.thesaurus.get_aliases(canonical) else 0.95
                    found[canonical] = max(found.get(canonical, 0.0), score)

        return sorted(found.items(), key=lambda x: x[1], reverse=True)

    def _scan_graph_node_ngrams(self, query_text: str) -> List[Tuple[str, float]]:
        """
        Quét n-grams đa từ (3..10 từ, từ dài xuống ngắn) khớp chính xác tên node nguyên gốc trong graph.json
        (ví dụ: 'Ban chủ nhiệm khoa', 'Bộ môn Điện tử và Kỹ thuật máy tính') vốn không nằm trong 169 mục Thesaurus.
        Chỉ tra self._lower_to_canonical (không dùng alias đã cắt tiền tố [QĐ-...]) để tránh bắt nhầm cụm 2 từ chung chung
        như 'đại học', 'đơn vị', 'bộ môn', và cho phép chồng lấn tối đa 1 từ biên (vd: 'Ban chủ nhiệm [Khoa] Công nghệ Nông nghiệp').
        """
        if not self._lower_to_canonical:
            return []
        tokens = _WORD_TOKEN_RE.findall(query_text.lower())
        if len(tokens) < 3:
            return []

        covered_positions: Set[int] = set()
        found: Dict[str, float] = {}
        max_n = min(10, len(tokens))

        for n in range(max_n, 2, -1):
            for i in range(len(tokens) - n + 1):
                span = set(range(i, i + n))
                overlap_cnt = len(span & covered_positions)
                if overlap_cnt > 1 or (n - overlap_cnt) < 2:
                    continue
                phrase = " ".join(tokens[i : i + n])
                if len(phrase) < 10 or is_forbidden_entity(phrase):
                    continue
                node_name = self._lower_to_canonical.get(phrase)
                if not node_name or is_forbidden_entity(node_name) or self._is_hub_node(node_name):
                    continue
                covered_positions.update(span)
                score = 0.96 if n >= 4 else 0.92
                found[node_name] = max(found.get(node_name, 0.0), score)

        return sorted(found.items(), key=lambda x: x[1], reverse=True)

    def _resolve_term_in_graph(self, term: str) -> Optional[str]:
        """Khớp chính xác tên chuẩn hóa hoặc alias sạch vào tập node thực tế của graph.json."""
        low = term.strip().lower()
        if not low or is_forbidden_entity(low):
            return None
        if low in self._lower_to_canonical:
            return self._lower_to_canonical[low]
        if low in self._alias_to_graph_entity:
            return self._alias_to_graph_entity[low]
        return None

    def _search_entity_vectors(
        self,
        terms: List[str],
        threshold: Optional[float] = None,
        precomputed_query_text: Optional[str] = None,
        precomputed_query_vector: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Tìm kiếm ngữ nghĩa các thực thể:
          - Tái sử dụng query_vector đã nhúng sẵn từ UETRetriever nếu term trùng câu hỏi gốc (tiết kiệm 1 lượt BGE-M3).
          - Ưu tiên nhánh ma trận NumPy trong RAM (<2ms, nạp từ entity_chroma).
          - Fallback sang truy vấn trực tiếp _entity_collection (hỗ trợ mock collection trong unit test).
        """
        if not terms:
            return []
        if self._entity_vec_matrix is None and self._entity_collection is None:
            return []

        min_sim = self.entity_similarity_threshold if threshold is None else float(threshold)
        try:
            vectors: List[List[float]] = []
            if (
                precomputed_query_vector is not None
                and precomputed_query_text
                and len(terms) == 1
                and terms[0].strip() == precomputed_query_text.strip()
            ):
                vectors = [list(precomputed_query_vector)]
            elif self.embedding_model is not None:
                if precomputed_query_vector is not None and precomputed_query_text:
                    to_embed = [t for t in terms if t.strip() != precomputed_query_text.strip()]
                    embedded_map: Dict[str, List[float]] = {}
                    if to_embed:
                        for t_str, emb_v in zip(to_embed, self.embedding_model.embed_texts(to_embed)):
                            embedded_map[t_str] = emb_v
                    for t in terms:
                        if t.strip() == precomputed_query_text.strip():
                            vectors.append(list(precomputed_query_vector))
                        elif t in embedded_map:
                            vectors.append(embedded_map[t])
                else:
                    vectors = self.embedding_model.embed_texts(terms)
            if not vectors:
                return []

            # Nhánh 1: Ma trận NumPy RAM siêu tốc
            if (
                self._entity_vec_matrix is not None
                and len(self._entity_vec_names) == self._entity_vec_matrix.shape[0]
            ):
                q_arr = np.asarray(vectors, dtype=np.float32)
                if q_arr.ndim == 2 and q_arr.shape[1] == self._entity_vec_matrix.shape[1]:
                    norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
                    norms = np.where(norms == 0.0, 1.0, norms)
                    q_norm = q_arr / norms
                    sims_mat = q_norm @ self._entity_vec_matrix.T
                    k_pick = min(max(1, self.entity_vector_top_k), sims_mat.shape[1])

                    matched_ram: Dict[str, float] = {}
                    for row in sims_mat:
                        top_idx = np.argpartition(row, -k_pick)[-k_pick:]
                        for idx in top_idx:
                            sim = float(row[int(idx)])
                            if sim >= min_sim:
                                node_name = self._entity_vec_names[int(idx)]
                                score = round(min(0.94, sim), 4)
                                if score > matched_ram.get(node_name, 0.0):
                                    matched_ram[node_name] = score
                    return sorted(matched_ram.items(), key=lambda x: x[1], reverse=True)

            # Nhánh 2: Truy vấn trực tiếp ChromaDB collection (hoặc mock DummyEntityCollection)
            if self._entity_collection is None:
                return []

            res = self._entity_collection.query(
                query_embeddings=vectors,
                n_results=max(1, self.entity_vector_top_k),
                include=["metadatas", "distances"],
            )
            metadatas_list = res.get("metadatas") or []
            distances_list = res.get("distances") or []

            matched: Dict[str, float] = {}
            for metas, dists in zip(metadatas_list, distances_list):
                for meta, dist in zip(metas or [], dists or []):
                    if not isinstance(meta, dict):
                        continue
                    cand_name = str(meta.get("canonical_name") or "").strip()
                    if not cand_name or is_forbidden_entity(cand_name):
                        continue
                    resolved = self._resolve_term_in_graph(cand_name)
                    if not resolved:
                        continue
                    sim = 1.0 - float(dist)
                    if sim >= min_sim:
                        score = round(min(0.94, sim), 4)
                        if score > matched.get(resolved, 0.0):
                            matched[resolved] = score

            return sorted(matched.items(), key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"[LightRAGRetriever] Lỗi truy vấn vector thực thể: {e}")
            return []

    def extract_and_match_entities(
        self,
        query_text: str,
        pre_extracted_entities: Optional[List[str]] = None,
        pre_extracted_themes: Optional[List[str]] = None,
        allow_llm: Optional[bool] = None,
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        Phễu 3 bước chuẩn LightRAG (PA1 + Graceful Fallback PA2):
          1. Trích xuất thực thể & chủ đề bằng LLM (hoặc nhận kết quả PA1 đã chạy song song) + quét Academic Thesaurus.
          2. Lọc thực thể rác (FORBIDDEN_ENTITIES) & chuẩn hóa tên về Canonical Name.
          3. Ánh xạ vào graph.json: ưu tiên Exact Match (O(1)), phần còn lại Vector Search trên RAM NumPy / entity_chroma.
        Trả về: (seed_entities, high_level_themes).
        """
        clean_query = (query_text or "").strip()
        if not clean_query:
            return [], []

        matched: Dict[str, float] = {}
        unmatched_for_vector: List[str] = []
        high_level_themes: List[str] = []
        used_llm = False

        # Bước 1A: Dùng kết quả PA1 truyền sẵn (nếu UETRetriever đã chạy song song) hoặc gọi QueryEntityExtractor
        raw_low_level: List[str] = []
        if pre_extracted_entities is not None or pre_extracted_themes is not None:
            raw_low_level = list(pre_extracted_entities or [])
            high_level_themes = list(pre_extracted_themes or [])
            used_llm = bool(raw_low_level or high_level_themes)
        else:
            should_call_llm = self.use_llm_query_extraction if allow_llm is None else bool(allow_llm)
            if should_call_llm and self.query_extractor is not None:
                llm_res = self.query_extractor.extract(clean_query)
                used_llm = bool(llm_res.get("used_llm", False))
                raw_low_level = list(llm_res.get("low_level_entities") or [])
                high_level_themes = list(llm_res.get("high_level_themes") or [])

        for idx, raw_ent in enumerate(raw_low_level):
            if is_forbidden_entity(raw_ent):
                continue
            canonical = self.thesaurus.normalize(raw_ent)
            if is_forbidden_entity(canonical):
                continue

            base_conf = round(max(0.88, 1.0 - idx * 0.02), 4)
            graph_node = self._resolve_term_in_graph(canonical) or self._resolve_term_in_graph(raw_ent)
            if graph_node:
                matched[graph_node] = max(matched.get(graph_node, 0.0), base_conf)
            elif canonical not in unmatched_for_vector:
                unmatched_for_vector.append(canonical)

        theme_vector_fallbacks: List[str] = []
        for raw_theme in high_level_themes:
            norm_theme = self.thesaurus.normalize(raw_theme)
            if not is_forbidden_entity(norm_theme):
                graph_node = self._resolve_term_in_graph(norm_theme)
                if graph_node:
                    matched[graph_node] = max(matched.get(graph_node, 0.0), 0.85)
                elif norm_theme not in theme_vector_fallbacks:
                    theme_vector_fallbacks.append(norm_theme)

        # Bước 1B: Quét n-gram nguyên cụm qua Academic Thesaurus (bắt chắc CN1, KKHT, ĐRL...)
        for canonical_th, th_score in self._scan_thesaurus_ngrams(clean_query):
            graph_node = self._resolve_term_in_graph(canonical_th)
            if graph_node:
                matched[graph_node] = max(matched.get(graph_node, 0.0), th_score)
            elif canonical_th not in unmatched_for_vector:
                unmatched_for_vector.append(canonical_th)

        # Bước 1C: Quét n-gram đa từ khớp trực tiếp tên node trong graph.json
        # (bắt các mục/bộ môn/chức năng cụ thể như 'Ban chủ nhiệm khoa', 'Bộ môn Điện tử và Kỹ thuật máy tính')
        for node_name, gn_score in self._scan_graph_node_ngrams(clean_query):
            matched[node_name] = max(matched.get(node_name, 0.0), gn_score)

        # Bước 2: Semantic Entity Search trên ma trận RAM NumPy / entity_chroma
        # Không đẩy high_level_themes sang nhúng BGE-M3 nếu đã có seed thực thể để tránh tốn thêm 1 lượt ONNX
        vector_lookup_terms = list(unmatched_for_vector)
        if not matched and not vector_lookup_terms:
            vector_lookup_terms.extend(theme_vector_fallbacks[:2])
        if (not used_llm or len(matched) == 0) and clean_query not in vector_lookup_terms:
            vector_lookup_terms.append(clean_query)

        if vector_lookup_terms:
            vec_matches = self._search_entity_vectors(
                vector_lookup_terms,
                precomputed_query_text=clean_query,
                precomputed_query_vector=query_vector,
            )
            for node_name, vec_score in vec_matches:
                matched[node_name] = max(matched.get(node_name, 0.0), vec_score)

        sorted_seeds = sorted(matched.items(), key=lambda x: x[1], reverse=True)
        return sorted_seeds[: self.max_entities_per_query], high_level_themes

    def find_seed_entities(self, query_text: str) -> List[Tuple[str, float]]:
        """API tương thích ngược: Trả về danh sách (canonical_name, confidence_score)."""
        seeds, _ = self.extract_and_match_entities(query_text)
        return seeds

    def _extract_low_level_subgraph_detailed(
        self,
        seed_entities: List[Tuple[str, float]],
        depth: int = 1,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]]]:
        """
        Trích xuất đồ thị con cục bộ bằng Lan truyền Kích hoạt có Trọng số (Weighted Spreading Activation):
          - Tích hợp Entity IDF, Hub Gating, trọng số quan hệ pháp quy (LIGHTRAG_RELATION_WEIGHTS)
            và điểm thưởng đồng kích hoạt (Co-Activation Bonus).
        Trả về: (blocks, activated_nodes, scored_edges).
        """
        valid_seeds = [(s, float(sc)) for s, sc in seed_entities if s in self.entities]
        if not valid_seeds:
            return [], {}, []

        seed_set: Set[str] = {s for s, _ in valid_seeds}
        activated_nodes: Dict[str, float] = {}
        frontier: Dict[str, float] = {}

        for s_name, s_score in valid_seeds:
            weighted_seed = round(s_score * max(0.55, self._effective_node_idf(s_name)), 4)
            activated_nodes[s_name] = weighted_seed
            frontier[s_name] = weighted_seed

        max_hops = max(0, int(depth))
        hop_decay_base = 0.68
        raw_edges_seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for hop in range(1, max_hops + 1):
            if not frontier:
                break
            decay = hop_decay_base ** hop
            next_frontier: Dict[str, float] = {}

            for u, u_score in frontier.items():
                # Chặn lan truyền từ các siêu nút ở bước hop >= 2 (vẫn cho phép hop 1 từ seed nếu không quá lớn)
                if hop > 1 and self._is_hub_node(u):
                    continue
                out_rels = self.adjacency.get(u, [])
                in_rels = self.reverse_adjacency.get(u, [])
                total_deg = len(out_rels) + len(in_rels)
                if total_deg == 0 or (u not in seed_set and self._is_hub_node(u)):
                    continue

                damping = 1.0 / math.sqrt(max(1, min(total_deg, 64)))

                for rel in out_rels:
                    tgt = rel.get("target")
                    if not tgt or tgt not in self.entities:
                        continue
                    rtype = str(rel.get("relation_type") or rel.get("predicate") or "liên_quan_đến").lower()
                    rel_w = LIGHTRAG_RELATION_WEIGHTS.get(rtype, 0.95)
                    raw_edges_seen.setdefault((u, rtype, tgt), rel)
                    if self._is_hub_node(tgt) and tgt not in seed_set:
                        continue
                    prop = u_score * decay * damping * rel_w * self._effective_node_idf(tgt)
                    if prop > activated_nodes.get(tgt, 0.0):
                        activated_nodes[tgt] = round(prop, 4)
                        next_frontier[tgt] = max(next_frontier.get(tgt, 0.0), prop)

                for rel in in_rels:
                    src = rel.get("source")
                    if not src or src not in self.entities:
                        continue
                    rtype = str(rel.get("relation_type") or rel.get("predicate") or "liên_quan_đến").lower()
                    rel_w = LIGHTRAG_RELATION_WEIGHTS.get(rtype, 0.95)
                    raw_edges_seen.setdefault((src, rtype, u), rel)
                    if self._is_hub_node(src) and src not in seed_set:
                        continue
                    prop = u_score * decay * damping * rel_w * self._effective_node_idf(src)
                    if prop > activated_nodes.get(src, 0.0):
                        activated_nodes[src] = round(prop, 4)
                        next_frontier[src] = max(next_frontier.get(src, 0.0), prop)

            if len(next_frontier) > self.max_entities_per_query * 3:
                top_f = heapq.nlargest(self.max_entities_per_query * 3, next_frontier.items(), key=lambda x: x[1])
                frontier = dict(top_f)
            else:
                frontier = next_frontier

        # Chấm điểm từng cạnh dựa trên Co-Activation và trọng số quan hệ pháp quy
        scored_edges: List[Dict[str, Any]] = []
        node_co_bonus: Dict[str, float] = {}
        for (src, rtype, tgt), rel in raw_edges_seen.items():
            s_act = activated_nodes.get(src, 0.0)
            t_act = activated_nodes.get(tgt, 0.0)
            if s_act <= 0.0 and t_act <= 0.0:
                continue
            rel_w = LIGHTRAG_RELATION_WEIGHTS.get(rtype, 0.95)
            both_seed_mult = 1.35 if (src in seed_set and tgt in seed_set) else (1.15 if (s_act > 0 and t_act > 0) else 1.0)
            e_score = round(0.5 * (s_act + t_act) * rel_w * both_seed_mult, 4)
            enriched_edge = dict(rel)
            enriched_edge["score"] = e_score
            scored_edges.append(enriched_edge)

            if src in seed_set and tgt in seed_set:
                node_co_bonus[src] = node_co_bonus.get(src, 0.0) + 0.12 * rel_w
                node_co_bonus[tgt] = node_co_bonus.get(tgt, 0.0) + 0.12 * rel_w

        scored_edges.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        # Chọn danh sách thực thể tiêu biểu để dựng Low-level blocks (ưu tiên seeds + top bridge nodes)
        ordered_candidates: List[Tuple[str, float]] = []
        seen_block_nodes: Set[str] = set()
        for seed, orig_score in valid_seeds:
            final_s = min(0.99, round(orig_score + node_co_bonus.get(seed, 0.0), 4))
            ordered_candidates.append((seed, final_s))
            seen_block_nodes.add(seed)

        if len(ordered_candidates) < self.max_entities_per_query:
            extra_slots = self.max_entities_per_query - len(ordered_candidates)
            bridge_nodes = [
                (n, sc) for n, sc in activated_nodes.items() if n not in seen_block_nodes and sc >= 0.25
            ]
            bridge_nodes.sort(key=lambda x: x[1], reverse=True)
            for b_node, b_sc in bridge_nodes[: min(4, extra_slots)]:
                ordered_candidates.append((b_node, round(min(0.86, b_sc), 4)))
                seen_block_nodes.add(b_node)

        blocks: List[Dict[str, Any]] = []
        for node_name, rel_score in ordered_candidates:
            node_info = self.entities.get(node_name)
            if not node_info:
                continue

            direct_outgoing = [r for r in scored_edges if r.get("source") == node_name]
            direct_incoming = [r for r in scored_edges if r.get("target") == node_name]

            ent_cat = node_info.get("entity_type") or node_info.get("category", "GENERAL")
            descriptions = node_info.get("descriptions", [])
            if isinstance(descriptions, list):
                ent_desc = " | ".join(descriptions) if descriptions else node_info.get("description", "")
            else:
                ent_desc = str(descriptions)

            blocks.append({
                "entity": node_name,
                "category": ent_cat,
                "description": ent_desc,
                "outgoing_relations": direct_outgoing,
                "incoming_relations": direct_incoming,
                "relevance_score": rel_score,
            })

        blocks.sort(key=lambda b: float(b.get("relevance_score", 0.0)), reverse=True)
        return blocks, activated_nodes, scored_edges

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
        """
        Trích xuất tóm tắt chủ đề tài liệu toàn cục (High-level Retrieval).
        Kết hợp cả high_level_themes do LLM bóc tách và thực thể hạt giống.
        """
        if not self.document_summaries:
            return []

        query_lower = query_text.lower()
        target_faculties = detect_query_faculties(query_text)
        seed_names = {s.lower() for s, _ in seed_entities}
        themes_lower = [
            self.thesaurus.normalize(t).lower()
            for t in (high_level_themes or [])
            if not is_forbidden_entity(t)
        ]

        matched_summaries: List[Tuple[str, Dict[str, Any], float]] = []

        for doc_id, summary_info in self.document_summaries.items():
            doc_title = summary_info.get("title", "")
            doc_summary = summary_info.get("summary", "")
            title_low = doc_title.lower()
            summary_low = doc_summary.lower()
            key_entities = {e.lower() for e in summary_info.get("key_entities", [])}

            # Lọc/giảm mạnh tóm tắt thuộc Khoa/Viện khác khi câu hỏi nêu đích danh Khoa/Viện cụ thể
            faculty_mult = 1.0
            if target_faculties:
                summary_blob = f"{doc_title} {doc_summary} {' '.join(summary_info.get('key_entities', []))}"
                summary_faculties = detect_query_faculties(summary_blob)
                if summary_faculties:
                    if summary_faculties & target_faculties:
                        faculty_mult = 1.25
                    else:
                        faculty_mult = 0.15

            score = 0.0
            if title_low and title_low in query_lower:
                score += 0.5

            for theme in themes_lower:
                if theme in title_low:
                    score += 0.45
                elif theme in summary_low:
                    score += 0.25
                if theme in key_entities:
                    score += 0.30

            overlap = key_entities.intersection(seed_names)
            score += len(overlap) * 0.30

            for word in _WORD_TOKEN_RE.findall(query_lower):
                if len(word) > 3 and not is_forbidden_entity(word) and word in summary_low:
                    score += 0.04

            score *= faculty_mult
            if score > 0.15:
                matched_summaries.append((doc_id, summary_info, min(score, 0.96)))

        matched_summaries.sort(key=lambda x: x[2], reverse=True)
        return [
            {
                "doc_id": doc_id,
                "title": s.get("title", ""),
                "summary": s.get("summary", ""),
                "document_type": s.get("document_type", ""),
                "key_entities": s.get("key_entities", []),
                "score": score,
            }
            for doc_id, s, score in matched_summaries[:3]
        ]

    @staticmethod
    def _format_edge_line(rel: Dict[str, Any], max_ev_chars: int = 110) -> str:
        rel_type = rel.get("relation_type") or rel.get("predicate", "liên_quan")
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        desc = rel.get("description") or (
            rel.get("evidence_texts", [""])[0]
            if rel.get("evidence_texts")
            else rel.get("evidence", "")
        )
        desc_clean = re.sub(r"\s+", " ", str(desc or "").strip())[:max_ev_chars]
        desc_part = f" (Căn cứ: {desc_clean})" if desc_clean else ""
        return f"  └─ [{src}] ──({rel_type})──> [{tgt}]{desc_part}"

    def build_synthesis_overview(
        self,
        high_level_blocks: List[Dict[str, Any]],
        low_level_blocks: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
    ) -> Optional[str]:
        """
        Tổng hợp Bản đồ Liên kết Quy chế 2 tầng (High-level + Low-level) gọn nhẹ (~150-220 từ)
        để đính kèm trực tiếp vào vùng ngữ cảnh của LLM cho các câu hỏi quan hệ/tổng hợp,
        mà không bị Cross-Encoder Reranker chấm lệch điểm hay chiếm mất slot của DataChunk gốc.
        """
        limit = max(250, int(max_chars if max_chars is not None else self.synthesis_max_chars))
        lines: List[str] = []

        for hl in (high_level_blocks or [])[:2]:
            title = str(hl.get("title") or "").strip()
            summary = re.sub(r"\s+", " ", str(hl.get("summary") or "").strip())[:160]
            if title and summary:
                lines.append(f"• Phạm vi văn bản [{title}]: {summary}")

        seen_edges: Set[str] = set()
        added_entities = 0
        for ll in (low_level_blocks or []):
            if added_entities >= 3:
                break
            ent_name = str(ll.get("entity") or "").strip()
            ent_desc = re.sub(r"\s+", " ", str(ll.get("description") or "").strip())[:160]
            rels = list(ll.get("outgoing_relations") or [])[:2] + list(ll.get("incoming_relations") or [])[:2]
            if not ent_desc and not rels:
                continue

            if ent_desc:
                lines.append(f"• Thực thể [{ent_name}]: {ent_desc}")
            else:
                lines.append(f"• Thực thể [{ent_name}]:")

            for r in rels[:3]:
                edge_line = self._format_edge_line(r, max_ev_chars=95)
                if edge_line not in seen_edges:
                    seen_edges.add(edge_line)
                    lines.append(edge_line)
            added_entities += 1

        if not lines:
            return None

        assembled: List[str] = []
        curr_len = 0
        for ln in lines:
            if curr_len + len(ln) + 1 > limit and assembled:
                break
            assembled.append(ln)
            curr_len += len(ln) + 1

        return "\n".join(assembled).strip() or None

    def _format_dual_level_contexts(
        self,
        high_level_blocks: List[Dict[str, Any]],
        low_level_blocks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[RetrievedContext]:
        """Đóng gói các khối High-level và Low-level thành List[RetrievedContext] độc lập."""
        contexts: List[RetrievedContext] = []
        rank_counter = 1

        for hl in high_level_blocks:
            entities_str = ", ".join(hl["key_entities"][:6]) if hl["key_entities"] else "N/A"
            text_block = (
                f"【TỔNG QUAN QUY CHẾ / VĂN BẢN ĐÀO TẠO UET】\n"
                f"• Văn bản: {hl['title']}\n"
                f"• Phân loại: {hl['document_type']}\n"
                f"• Các thực thể trọng tâm: {entities_str}\n"
                f"• Nội dung bao quát: {hl['summary']}"
            )
            contexts.append(
                RetrievedContext(
                    chunk_id=f"lightrag_high_{hl['doc_id']}",
                    text=text_block,
                    similarity_score=round(hl["score"], 4),
                    rank=rank_counter,
                    metadata={
                        "source": "knowledge_graph_high_level",
                        "doc_id": hl["doc_id"],
                        "document_type": hl["document_type"],
                        "retrieval_tier": "high_level",
                    },
                )
            )
            rank_counter += 1

        for ll in low_level_blocks:
            ent_name = ll["entity"]
            ent_cat = ll["category"]
            ent_desc = ll["description"]

            lines = [
                f"【TRI THỨC LIÊN KẾT HỌC VỤ UET: {ent_name.upper()}】",
                f"• Danh mục: {ent_cat}",
                f"• Định nghĩa / Quy định: {ent_desc}",
            ]

            relations_text = []
            for out_rel in ll["outgoing_relations"][:5]:
                relations_text.append(self._format_edge_line(out_rel, max_ev_chars=140))
            for in_rel in ll["incoming_relations"][:5]:
                relations_text.append(self._format_edge_line(in_rel, max_ev_chars=140))

            if relations_text:
                lines.append("• Các mối liên kết quy chế liên quan:")
                lines.extend(relations_text)

            formatted_text = "\n".join(lines)
            contexts.append(
                RetrievedContext(
                    chunk_id=f"lightrag_low_{ent_name}",
                    text=formatted_text,
                    similarity_score=round(ll["relevance_score"], 4),
                    rank=rank_counter,
                    metadata={
                        "source": "knowledge_graph_low_level",
                        "entity": ent_name,
                        "category": ent_cat,
                        "retrieval_tier": "low_level",
                    },
                )
            )
            rank_counter += 1

        contexts.sort(key=lambda x: x.similarity_score, reverse=True)
        final_contexts = contexts[:top_k]
        for i, ctx in enumerate(final_contexts, 1):
            ctx.rank = i
        return final_contexts

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

"""
src/graph/lightrag_indexer.py - Cỗ máy Offline nạp đồ thị tri thức LightRAG.

⚠️  LƯU Ý QUAN TRỌNG:
    Module này CHỈ định nghĩa các class và hàm. KHÔNG tự động chạy khi import.
    Việc xây dựng toàn bộ Knowledge Graph cho dữ liệu thực phải được kích hoạt
    thủ công qua lệnh CLI: python main.py --ingest --to lightrag [--clear]
    và phải có sự phê duyệt của quản trị viên dự án.

Kiến trúc Hierarchical 2-Pass Extraction:
    - CHẶNG 1 (Document-level): Trích xuất Root Entities từ phần mở đầu/mục lục văn bản.
    - CHẶNG 2 (Unit-level): Trích xuất thực thể chi tiết từng Semantic Unit,
      nối cạnh về Root Entities đã có từ Chặng 1.

Chống trùng lặp:
    - AcademicThesaurus: Chuẩn hóa alias/viết tắt -> Canonical Name (O(1)).
    - EntityRegistry: In-memory registry ngăn tạo node trùng lặp.
    - Graph Persistence: Lưu trữ bằng NetworkX + JSON (không cần Neo4j).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.helpers import get_logger, load_yaml_config
from src.graph.thesaurus_utils import (
    AcademicThesaurus,
    FORBIDDEN_ENTITIES,
    get_thesaurus,
    is_forbidden_entity,
)
from src.graph.adaptive_extractor import SemanticUnit, AdaptiveSemanticExtractor

logger = get_logger("graph.lightrag_indexer")

# ---------------------------------------------------------------------------
# CẤU TRÚC DỮ LIỆU ĐỒ THỊ
# ---------------------------------------------------------------------------

DEFAULT_GRAPH_PATH = BASE_DIR / "database" / "lightrag_db" / "graph.json"
DEFAULT_ENTITY_VECTOR_DIR = BASE_DIR / "database" / "lightrag_db" / "entity_chroma"

# Các loại quan hệ được phép (Closed Relation Schema)
ALLOWED_RELATION_TYPES = {
    "quy_định_bởi", "điều_kiện_tiên_quyết", "ngoại_lệ_cho",
    "dẫn_chiếu_đến", "chế_tài_xử_lý", "áp_dụng_cho",
    "bổ_sung_cho", "thay_thế_bởi", "liên_quan_đến",
    "thuộc_về", "quản_lý_bởi", "bao_gồm",
}

# Danh sách category loại thực thể hợp lệ
VALID_ENTITY_CATEGORIES = {
    "VAN_BAN", "DIEU_KHOAN", "QUY_TRINH",
    "DIEU_KIEN", "KHOA_NGANH", "NHAN_SU",
    "DON_VI", "BANG_TRA_CUU", "OTHER",
}


# ---------------------------------------------------------------------------
# ENTITY REGISTRY: Chống trùng lặp trong phiên Ingestion
# ---------------------------------------------------------------------------

class EntityRegistry:
    """
    In-Memory Registry quản lý thực thể đã trích xuất trong một phiên nạp đồ thị.
    Mỗi Canonical Name chỉ tồn tại duy nhất 1 node trong đồ thị.
    Khi LLM trích xuất cùng thực thể nhiều lần, Registry chỉ bổ sung bằng chứng
    (source_unit_ids) chứ KHÔNG tạo node mới.
    """

    def __init__(self):
        # canonical_name (lowercase key) -> node metadata dict
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edge_key_set: Set[str] = set()
        self._edges_by_key: Dict[str, Dict[str, Any]] = {}

    def node_exists(self, canonical: str) -> bool:
        return canonical.lower() in self._nodes

    def register_node(
        self,
        canonical: str,
        entity_type: str,
        description: str,
        source_unit_id: str,
        aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Đăng ký thực thể mới hoặc cập nhật node đã tồn tại.
        KHÔNG BAO GIỜ tạo 2 node cho cùng 1 Canonical Name.
        """
        key = canonical.lower()
        if key in self._nodes:
            # Node đã tồn tại -> chỉ bổ sung bằng chứng
            node = self._nodes[key]
            if source_unit_id not in node["source_unit_ids"]:
                node["source_unit_ids"].append(source_unit_id)
            if description and len(node["descriptions"]) < 5:
                node["descriptions"].append(description)
            return node

        # Node mới -> tạo lần đầu
        node = {
            "id": self._make_node_id(canonical),
            "canonical_name": canonical,
            "entity_type": entity_type or "OTHER",
            "descriptions": [description] if description else [],
            "aliases": aliases or [],
            "source_unit_ids": [source_unit_id],
        }
        self._nodes[key] = node
        return node

    def get_node(self, canonical: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(canonical.lower())

    def register_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        evidence_text: str,
        source_unit_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Đăng ký cạnh (quan hệ) giữa 2 node.
        Mỗi cặp (source, target, relation_type) chỉ có 1 cạnh duy nhất;
        khi gặp lại ở SemanticUnit khác, gộp bổ sung bằng chứng (evidence_texts) và source_unit_ids.
        """
        edge_key = f"{source.lower()}::{relation_type}::{target.lower()}"
        if edge_key in self._edges_by_key:
            existing = self._edges_by_key[edge_key]
            unit_ids = existing.setdefault("source_unit_ids", [])
            if source_unit_id and source_unit_id not in unit_ids:
                unit_ids.append(source_unit_id)
            ev_list = existing.setdefault("evidence_texts", [])
            if evidence_text and evidence_text not in ev_list and len(ev_list) < 5:
                ev_list.append(evidence_text)
            return None  # Đã cập nhật in-place trên cạnh hiện có

        edge = {
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "evidence_texts": [evidence_text] if evidence_text else [],
            "source_unit_ids": [source_unit_id] if source_unit_id else [],
        }
        self._edge_key_set.add(edge_key)
        self._edges_by_key[edge_key] = edge
        return edge

    def all_nodes(self) -> List[Dict[str, Any]]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    @staticmethod
    def _make_node_id(canonical: str) -> str:
        return hashlib.md5(canonical.lower().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# PROMPT BUILDER: Tạo Prompt trích xuất thực thể tiếng Việt
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Tạo Prompt chuẩn hóa cho LLM trích xuất thực thể và quan hệ.
    Hỗ trợ 2 loại prompt:
      - Chặng 1 (Document-level): Trích xuất Root Entities từ mở đầu văn bản (chỉ gửi danh sách đơn vị/cơ quan).
      - Chặng 2 (Unit-level): Trích xuất chi tiết Điều/Phần cụ thể với Dynamic Contextual Filtering
        (chỉ gửi các thực thể có mặt trong đoạn văn bản, tiết kiệm ~95% token).
    """

    def __init__(self, thesaurus: AcademicThesaurus):
        self.thesaurus = thesaurus

    def build_pass1_prompt(self, doc_header: str, doc_title: str) -> str:
        """
        Chặng 1: Trích xuất Root Entities từ phần mở đầu/tiêu đề văn bản.
        Output: Danh sách JSON các thực thể gốc (tên văn bản, cơ quan ban hành,
                phạm vi áp dụng, đối tượng áp dụng).
        """
        pass1_context = self.thesaurus.build_pass1_entity_context()
        return f"""Bạn là chuyên gia trích xuất tri thức học vụ Trường Đại học Công nghệ – ĐHQGHN.

{pass1_context}

=== NHIỆM VỤ CHẶNG 1: TRÍCH XUẤT THỰC THỂ GỐC (ROOT ENTITIES) ===
Phân tích phần MỞ ĐẦU của văn bản dưới đây và trích xuất các thực thể cấp cao nhất:
1. Tên đầy đủ và số hiệu của văn bản (nếu là văn bản pháp quy).
2. Cơ quan ban hành.
3. Phạm vi và đối tượng áp dụng (ví dụ: Sinh viên chính quy K64 trở đi).
4. Các chương/phần chính trong văn bản.

=== VĂN BẢN: {doc_title} ===
{doc_header}

=== YÊU CẦU ĐẦU RA (JSON NGHIÊM NGẶT) ===
Chỉ trả về JSON theo đúng cấu trúc sau, KHÔNG thêm lời giải thích:
{{
  "document_title": "Tên đầy đủ của văn bản",
  "document_code": "Số hiệu quyết định (nếu có, ví dụ: QĐ-5115/ĐHQGHN, để null nếu không có)",
  "issuing_body": "Cơ quan ban hành (ví dụ: Giám đốc ĐHQGHN, Hiệu trưởng Trường ĐH Công nghệ)",
  "scope": "Mô tả ngắn phạm vi áp dụng",
  "root_entities": [
    {{
      "canonical_name": "Tên thực thể (dùng tên chuẩn hóa trong danh sách ưu tiên nếu có)",
      "entity_type": "VAN_BAN|DON_VI|KHOA_NGANH|OTHER",
      "namespace_prefix": "Tiền tố định danh namespace (ví dụ: [QĐ-5115])"
    }}
  ]
}}"""

    def build_pass2_prompt(
        self,
        unit: SemanticUnit,
        root_entities: List[Dict[str, Any]],
        namespace_prefix: str,
    ) -> str:
        """
        Chặng 2: Trích xuất thực thể và quan hệ chi tiết từng Semantic Unit.
        Áp dụng Dynamic Contextual Filtering: Chỉ gửi các thực thể có xuất hiện trong unit.content.
        Output: Danh sách JSON entities + relationships.
        """
        root_names = [e.get("canonical_name", "") for e in root_entities if e.get("canonical_name")]
        root_context = ", ".join(root_names[:10]) if root_names else "Không có thực thể gốc"

        forbidden_list = ", ".join(sorted(list(FORBIDDEN_ENTITIES))[:20])
        allowed_relations = ", ".join(sorted(ALLOWED_RELATION_TYPES))

        # Dynamic Contextual Filtering: Chỉ lọc thực thể có liên quan đến nội dung unit này
        dynamic_entity_context = self.thesaurus.build_dynamic_entity_context(unit.content)

        return f"""Bạn là chuyên gia trích xuất Đồ thị Tri thức học vụ Trường ĐH Công nghệ – ĐHQGHN.

{dynamic_entity_context}

=== CÁC THỰC THỂ GỐC CỦA VĂN BẢN NÀY (ĐÃ BIẾT) ===
{root_context}
(Hãy ưu tiên kết nối thực thể mới với các thực thể gốc này bằng quan hệ phù hợp.)

=== TIỀN TỐ ĐỊNH DANH ===
Tất cả thực thể mới PHẢI gắn tiền tố: {namespace_prefix}
Ví dụ đúng: "[QĐ-5115] Điều 15: Cảnh báo học vụ"

=== NỘI DUNG CẦN PHÂN TÍCH ===
{unit.to_llm_prompt_context()}

=== QUY TẮC TRÍCH XUẤT BẮT BUỘC ===
1. LOẠI THỰC THỂ ĐƯỢC PHÉP: VAN_BAN, DIEU_KHOAN, QUY_TRINH, DIEU_KIEN, KHOA_NGANH, NHAN_SU, DON_VI, BANG_TRA_CUU
2. THỰC THỂ BỊ CẤM TUYỆT ĐỐI (không tạo node): {forbidden_list}
3. QUAN HỆ ĐƯỢC PHÉP: {allowed_relations}
4. MỌI thực thể kiểu DIEU_KHOAN phải có tiền tố namespace: {namespace_prefix}
5. Nếu thực thể xuất hiện trong danh sách ưu tiên ở trên, PHẢI dùng ĐÚNG tên chuẩn hóa đó.
6. MÔ TẢ (description): Chỉ trích dẫn CHÍNH XÁC câu chữ từ văn bản, KHÔNG diễn giải lại.

=== YÊU CẦU ĐẦU RA (JSON NGHIÊM NGẶT) ===
Chỉ trả về JSON, KHÔNG thêm lời giải thích:
{{
  "entities": [
    {{
      "canonical_name": "Tên thực thể (kèm namespace nếu là Điều/Khoản)",
      "entity_type": "VAN_BAN|DIEU_KHOAN|QUY_TRINH|DIEU_KIEN|KHOA_NGANH|NHAN_SU|DON_VI|BANG_TRA_CUU",
      "description": "Mô tả ngắn gọn từ văn bản gốc (tối đa 200 ký tự)"
    }}
  ],
  "relationships": [
    {{
      "source": "Tên thực thể nguồn (canonical_name)",
      "target": "Tên thực thể đích (canonical_name)",
      "relation_type": "quy_định_bởi|điều_kiện_tiên_quyết|ngoại_lệ_cho|dẫn_chiếu_đến|chế_tài_xử_lý|áp_dụng_cho|bổ_sung_cho|thay_thế_bởi|liên_quan_đến|thuộc_về|quản_lý_bởi|bao_gồm",
      "evidence": "Câu trích dẫn trực tiếp từ văn bản làm bằng chứng (tối đa 100 ký tự)"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# RESPONSE PARSER: Phân tích JSON đầu ra của LLM
# ---------------------------------------------------------------------------

class LLMResponseParser:
    """Phân tích và làm sạch JSON đầu ra từ LLM trích xuất thực thể."""

    @staticmethod
    def extract_json(raw_response: str) -> Optional[Dict[str, Any]]:
        """
        Trích xuất JSON từ response LLM có thể chứa markdown code block.
        Xử lý các trường hợp: ```json...```, {... }, hoặc JSON thuần.
        """
        if not raw_response:
            return None

        # Thử tìm trong code block markdown
        code_block = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_response)
        if code_block:
            json_str = code_block.group(1).strip()
        else:
            # Tìm JSON object đầu tiên trong response
            obj_match = re.search(r'\{[\s\S]*\}', raw_response)
            if obj_match:
                json_str = obj_match.group(0)
            else:
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"Lỗi parse JSON: {e} | snippet: {json_str[:200]}")
            return None

    @staticmethod
    def parse_pass1(raw: str) -> Optional[Dict[str, Any]]:
        """Phân tích output Chặng 1 (Root Entities)."""
        data = LLMResponseParser.extract_json(raw)
        if not data:
            return None
        return data

    @staticmethod
    def parse_pass2(
        raw: str,
        thesaurus: AcademicThesaurus,
        namespace_prefix: str,
        unit_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Phân tích output Chặng 2 (Entities + Relationships).
        Áp dụng chuẩn hóa thực thể qua thesaurus và lọc bỏ thực thể cấm.

        Returns:
            Tuple (cleaned_entities, cleaned_relationships)
        """
        data = LLMResponseParser.extract_json(raw)
        if not data:
            return [], []

        raw_entities = data.get("entities", [])
        raw_rels = data.get("relationships", [])

        # Làm sạch entities
        entities: List[Dict[str, Any]] = []
        entity_names_seen: Set[str] = set()

        for ent in raw_entities:
            name = str(ent.get("canonical_name", "")).strip()
            if not name or len(name) < 2:
                continue
            if is_forbidden_entity(name, allow_school_names=True):
                continue
            # Chuẩn hóa qua thesaurus
            normalized, was_normalized = thesaurus.normalize_entity_name(name)
            if is_forbidden_entity(normalized, allow_school_names=True):
                continue
            if not was_normalized and len(normalized) < 4:
                continue  # Bỏ qua thực thể quá ngắn không có trong từ điển

            # Chống trùng lặp trong 1 response
            norm_key = normalized.lower()
            if norm_key in entity_names_seen:
                continue
            entity_names_seen.add(norm_key)

            entity_type = str(ent.get("entity_type", "OTHER")).upper()
            if entity_type not in VALID_ENTITY_CATEGORIES:
                entity_type = "OTHER"

            entities.append({
                "canonical_name": normalized,
                "entity_type": entity_type,
                "description": str(ent.get("description", ""))[:300],
                "source_unit_id": unit_id,
            })

        # Làm sạch relationships
        relationships: List[Dict[str, Any]] = []
        for rel in raw_rels:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            rtype = str(rel.get("relation_type", "liên_quan_đến")).strip().lower()

            if not source or not target:
                continue
            # Chuẩn hóa tên & lọc thực thể cấm ở cả 2 đầu mút
            source = thesaurus.normalize(source)
            target = thesaurus.normalize(target)
            if (
                not source
                or not target
                or source.strip().lower() == target.strip().lower()
                or is_forbidden_entity(source, allow_school_names=True)
                or is_forbidden_entity(target, allow_school_names=True)
            ):
                continue
            # Kiểm tra loại quan hệ
            if rtype not in ALLOWED_RELATION_TYPES:
                rtype = "liên_quan_đến"

            relationships.append({
                "source": source,
                "target": target,
                "relation_type": rtype,
                "evidence": str(rel.get("evidence", ""))[:200],
                "source_unit_id": unit_id,
            })

        return entities, relationships


# ---------------------------------------------------------------------------
# LIGHTRAG INDEXER: Cỗ máy chính
# ---------------------------------------------------------------------------

class LightRAGIndexer:
    """
    Cỗ máy Offline xây dựng Knowledge Graph cho UET Chatbot LightRAG.

    Pipeline Hierarchical 2-Pass:
      CHẶNG 1: Scan doc header -> Root Entities
      CHẶNG 2: Từng Semantic Unit -> chi tiết Entities + Edges -> gộp vào Registry

    ⚠️  KHÔNG gọi run() hoặc run_full_pipeline() mà không có sự phê duyệt.
    Chỉ nên dùng các phương thức riêng lẻ để kiểm thử (PoC, test nhỏ).

    Cách khởi chạy ĐÚNG (khi đã được phê duyệt):
        python main.py --ingest --to lightrag [--clear]
    """

    def __init__(
        self,
        graph_path: Optional[Path] = None,
        entity_vector_dir: Optional[Path] = None,
        llm_provider: Optional[str] = None,
        delay_between_calls: Optional[float] = None,
        max_pass2_retries: Optional[int] = None,
        max_workers: Optional[int] = None,
        header_max_chars: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config if config is not None else load_yaml_config()
        lr_cfg = (cfg or {}).get("lightrag", {})
        idx_cfg = lr_cfg.get("indexer", {})

        cfg_graph_rel = lr_cfg.get("graph_path")
        resolved_graph_path = (
            Path(graph_path)
            if graph_path is not None
            else (
                Path(cfg_graph_rel) if Path(cfg_graph_rel).is_absolute() else (BASE_DIR / cfg_graph_rel)
                if cfg_graph_rel
                else DEFAULT_GRAPH_PATH
            )
        )
        cfg_vec_rel = lr_cfg.get("entity_vector_dir")
        resolved_vec_dir = (
            Path(entity_vector_dir)
            if entity_vector_dir is not None
            else (
                Path(cfg_vec_rel) if Path(cfg_vec_rel).is_absolute() else (BASE_DIR / cfg_vec_rel)
                if cfg_vec_rel
                else DEFAULT_ENTITY_VECTOR_DIR
            )
        )

        self.graph_path = resolved_graph_path
        self.entity_vector_dir = resolved_vec_dir
        self.delay = float(
            delay_between_calls
            if delay_between_calls is not None
            else idx_cfg.get("delay_between_calls", 0.5)
        )
        self.max_retries = int(
            max_pass2_retries
            if max_pass2_retries is not None
            else idx_cfg.get("max_pass2_retries", 2)
        )
        self.max_workers = max(
            1,
            int(max_workers if max_workers is not None else idx_cfg.get("max_workers", 5)),
        )
        self.header_max_chars = max(
            200,
            int(header_max_chars if header_max_chars is not None else idx_cfg.get("header_max_chars", 2000)),
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Khởi tạo hạ tầng
        self.thesaurus = get_thesaurus()
        self.extractor = AdaptiveSemanticExtractor(config=cfg)
        self.registry = EntityRegistry()
        self.prompt_builder = PromptBuilder(self.thesaurus)
        self.parser = LLMResponseParser()

        # Lưu trữ đồ thị (nodes + edges riêng để dễ serialize)
        self._nodes: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []
        self._doc_summaries: Dict[str, Dict[str, Any]] = {}

        # Thống kê phiên nạp
        self._stats = {
            "docs_processed": 0,
            "docs_failed": 0,
            "units_processed": 0,
            "units_failed": 0,
            "llm_calls": 0,
            "llm_errors": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "rate_limit_waits": 0,
        }
        self._consecutive_llm_failures = 0

        # Đường dẫn cache Semantic Units
        cache_rel = lr_cfg.get("semantic_units_cache", "database/lightrag_db/semantic_units_cache.jsonl")
        self.semantic_units_cache_path = (
            Path(cache_rel) if Path(cache_rel).is_absolute() else (BASE_DIR / cache_rel)
        )
        self.semantic_units_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_units: List[SemanticUnit] = []
        self._cached_unit_ids: Set[str] = set()

        # Khởi tạo LLM Client
        self._llm = self._init_llm(llm_provider, config=cfg)

        # Tạo thư mục lưu trữ
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.entity_vector_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _init_llm(provider: Optional[str], config: Optional[Dict[str, Any]] = None):
        """Khởi tạo LLM Client tương thích với UETLLMClient có sẵn."""
        try:
            from src.llm.llm_client import UETLLMClient
            cfg = config if config is not None else load_yaml_config()
            effective_provider = (
                provider
                or cfg.get("lightrag", {}).get("indexer", {}).get("llm_provider")
                or cfg.get("llm", {}).get("provider", "openrouter")
            )
            llm = UETLLMClient(provider=effective_provider, enable_cache=False)
            logger.info(f"Khởi tạo LLM Client: provider={effective_provider}")
            return llm
        except Exception as e:
            logger.error(f"Không thể khởi tạo LLM Client: {e}")
            return None

    def _provider_has_keys(self, provider_name: str) -> bool:
        """Kiểm tra provider có cấu hình API key hợp lệ trước khi gọi để tránh spam cảnh báo."""
        key_attr_map = {
            "openrouter": "openrouter_keys",
            "groq": "groq_keys",
            "gemini": "gemini_keys",
            "sambanova": "sambanova_keys",
        }
        attr = key_attr_map.get(provider_name)
        if not attr or not hasattr(self._llm, attr):
            return True
        keys = getattr(self._llm, attr, None)
        return bool(keys)

    def _call_llm_raw(self, prompt_text: str) -> Optional[str]:
        """
        Gọi LLM API và trả về text thô (thread-safe).
        Tự động bỏ qua provider chưa cấu hình key, fallback giữa các provider khả dụng
        (OpenRouter -> Groq -> Gemini -> SambaNova), và retry có backoff theo max_retries.
        """
        if self._stop_event.is_set():
            return None

        if not self._llm:
            logger.error("LLM Client chưa được khởi tạo.")
            with self._lock:
                self._consecutive_llm_failures += 1
            return None

        # Danh sách callers ưu tiên (lọc sẵn provider có API key)
        providers_order = [self._llm.provider]
        for p in ["openrouter", "groq", "gemini", "sambanova"]:
            if p not in providers_order:
                providers_order.append(p)

        caller_map = {
            "openrouter": self._llm._call_openrouter,
            "groq": self._llm._call_groq,
            "gemini": self._llm._call_gemini,
            "sambanova": self._llm._call_sambanova,
        }

        active_providers = [
            p for p in providers_order
            if p in caller_map and self._provider_has_keys(p)
        ]
        if not active_providers:
            logger.error("Không có LLM provider nào được cấu hình API key hợp lệ.")
            with self._lock:
                self._stats["llm_errors"] += 1
                self._consecutive_llm_failures += 1
            return None

        max_attempts = max(1, int(self.max_retries) + 1)
        for attempt in range(max_attempts):
            if self._stop_event.is_set():
                return None

            for p_name in active_providers:
                if self._stop_event.is_set():
                    return None
                caller = caller_map[p_name]
                try:
                    text, model_name, _ = caller(prompt_text)
                    if text and text.strip():
                        with self._lock:
                            self._stats["llm_calls"] += 1
                            self._consecutive_llm_failures = 0
                        if self.delay > 0 and not self._stop_event.is_set():
                            time.sleep(self.delay)  # Rate limit delay trên từng luồng
                        return text
                except Exception as e:
                    logger.warning(f"[{p_name}] Gặp sự cố: {e}. Đổi ngay sang provider tiếp theo...")
                    continue

            if attempt + 1 < max_attempts and not self._stop_event.is_set():
                backoff = min(2.0 * (attempt + 1), 6.0)
                with self._lock:
                    self._stats["rate_limit_waits"] += 1
                logger.warning(
                    f"Tất cả provider khả dụng đều lỗi (lần {attempt + 1}/{max_attempts}). "
                    f"Nghỉ {backoff:.1f}s rồi thử lại..."
                )
                time.sleep(backoff)

        with self._lock:
            self._stats["llm_errors"] += 1
            self._consecutive_llm_failures += 1
        return None

    # -------------------------------------------------------------------------
    # CHẶNG 1: Trích xuất Root Entities từ phần mở đầu văn bản
    # -------------------------------------------------------------------------

    def _get_consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_llm_failures

    def _extract_pass1_data(
        self,
        doc_title: str,
        doc_content: str,
        header_max_chars: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """Trích xuất Chặng 1 thuần (thread-safe, không thay đổi Registry dùng chung)."""
        eff_header_chars = int(
            header_max_chars if header_max_chars is not None else self.header_max_chars
        )
        header = doc_content[:eff_header_chars]
        prompt = self.prompt_builder.build_pass1_prompt(header, doc_title)
        raw = self._call_llm_raw(prompt)
        llm_ok = bool(raw)

        fallback_ctx = {
            "document_title": doc_title,
            "document_code": None,
            "namespace_prefix": f"[{doc_title[:40]}]",
            "root_entities": [],
        }
        if not raw:
            logger.warning(f"Chặng 1 không có phản hồi cho: {doc_title}")
            return fallback_ctx, None, [], False

        parsed = self.parser.parse_pass1(raw)
        if not parsed:
            logger.warning(f"Chặng 1 parse thất bại cho: {doc_title}")
            return fallback_ctx, None, [], llm_ok

        doc_code = parsed.get("document_code") or doc_title[:40]
        namespace_prefix = f"[{doc_code}]"

        summary_key = str(doc_id).strip() if (doc_id and str(doc_id).strip()) else doc_title
        pass1_unit_id = f"pass1_{hashlib.md5(summary_key.encode('utf-8')).hexdigest()[:8]}"

        root_entity_names: List[str] = []
        root_nodes: List[Dict[str, Any]] = []
        seen_roots: Set[str] = set()
        for ent in parsed.get("root_entities", []):
            raw_name = str(ent.get("canonical_name", "")).strip()
            if not raw_name or is_forbidden_entity(raw_name, allow_school_names=True):
                continue
            name = self.thesaurus.normalize(raw_name)
            if not name or len(name) < 2 or is_forbidden_entity(name, allow_school_names=True):
                continue
            low_name = name.lower()
            if low_name in seen_roots:
                continue
            seen_roots.add(low_name)
            root_entity_names.append(name)
            root_nodes.append({
                "canonical": name,
                "entity_type": ent.get("entity_type", "VAN_BAN"),
                "description": parsed.get("scope", ""),
                "source_unit_id": pass1_unit_id,
                "aliases": self.thesaurus.get_aliases(name),
            })

        doc_summary = {
            "doc_id": summary_key,
            "title": parsed.get("document_title") or doc_title,
            "document_code": doc_code,
            "document_type": parsed.get("document_type", "LEGAL"),
            "issuing_body": parsed.get("issuing_body", ""),
            "summary": parsed.get("scope", "") or parsed.get("document_title", ""),
            "key_entities": root_entity_names,
        }

        logger.info(
            f"Chặng 1 '{doc_title}': namespace={namespace_prefix}, "
            f"{len(root_nodes)} root entities"
        )
        return {**parsed, "namespace_prefix": namespace_prefix}, doc_summary, root_nodes, True

    def _register_pass1_result(
        self,
        summary_key: str,
        doc_summary: Optional[Dict[str, Any]],
        root_nodes: List[Dict[str, Any]],
    ) -> None:
        """Gộp kết quả Chặng 1 vào trạng thái đồ thị dùng chung (phải gọi dưới self._lock hoặc main thread)."""
        if doc_summary is not None:
            self._doc_summaries[summary_key] = doc_summary
        for node in root_nodes:
            before_count = self.registry.node_count()
            self.registry.register_node(
                canonical=node["canonical"],
                entity_type=node["entity_type"],
                description=node["description"],
                source_unit_id=node["source_unit_id"],
                aliases=node["aliases"],
            )
            if self.registry.node_count() > before_count:
                self._stats["nodes_created"] += 1

    def run_pass1(
        self,
        doc_title: str,
        doc_content: str,
        header_max_chars: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Chặng 1: Gửi phần mở đầu văn bản cho LLM để lấy Root Entities.
        Chỉ chạy 1 lần cho mỗi tài liệu.
        """
        summary_key = str(doc_id).strip() if (doc_id and str(doc_id).strip()) else doc_title
        root_ctx, doc_summary, root_nodes, _ = self._extract_pass1_data(
            doc_title, doc_content, header_max_chars=header_max_chars, doc_id=doc_id
        )
        with self._lock:
            self._register_pass1_result(summary_key, doc_summary, root_nodes)
        return root_ctx

    # -------------------------------------------------------------------------
    # CHẶNG 2: Trích xuất chi tiết từng Semantic Unit
    # -------------------------------------------------------------------------

    def _extract_pass2_unit_data(
        self,
        unit: SemanticUnit,
        root_context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """Trích xuất Chặng 2 thuần cho 1 Semantic Unit (thread-safe, không sửa Registry)."""
        namespace_prefix = root_context.get("namespace_prefix", f"[{unit.document_title[:40]}]")
        root_entities = root_context.get("root_entities", [])

        prompt = self.prompt_builder.build_pass2_prompt(unit, root_entities, namespace_prefix)
        raw = self._call_llm_raw(prompt)
        if not raw:
            logger.warning(f"Chặng 2 không có phản hồi: {unit.unit_id}")
            return [], [], False

        entities, relationships = self.parser.parse_pass2(
            raw=raw,
            thesaurus=self.thesaurus,
            namespace_prefix=namespace_prefix,
            unit_id=unit.unit_id,
        )
        return entities, relationships, True

    def _register_pass2_result(
        self,
        unit_id: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        llm_ok: bool,
    ) -> Tuple[int, int]:
        """Gộp thực thể/cạnh của 1 unit vào Registry (phải gọi dưới self._lock hoặc main thread)."""
        if not llm_ok:
            self._stats["units_failed"] += 1
            return 0, 0

        nodes_added = 0
        edges_added = 0
        for ent in entities:
            before_count = self.registry.node_count()
            self.registry.register_node(
                canonical=ent["canonical_name"],
                entity_type=ent["entity_type"],
                description=ent["description"],
                source_unit_id=unit_id,
                aliases=self.thesaurus.get_aliases(ent["canonical_name"]),
            )
            if self.registry.node_count() > before_count:
                nodes_added += 1

        for rel in relationships:
            edge = self.registry.register_edge(
                source=rel["source"],
                target=rel["target"],
                relation_type=rel["relation_type"],
                evidence_text=rel["evidence"],
                source_unit_id=unit_id,
            )
            if edge is not None:
                self._edges.append(edge)
                edges_added += 1

        self._stats["units_processed"] += 1
        self._stats["nodes_created"] += nodes_added
        self._stats["edges_created"] += edges_added
        return nodes_added, edges_added

    def run_pass2_unit(
        self,
        unit: SemanticUnit,
        root_context: Dict[str, Any],
    ) -> Tuple[int, int]:
        """
        Chặng 2: Trích xuất thực thể và quan hệ từ 1 Semantic Unit.
        Gộp kết quả vào EntityRegistry (chống trùng lặp tự động).
        """
        entities, relationships, llm_ok = self._extract_pass2_unit_data(unit, root_context)
        with self._lock:
            return self._register_pass2_result(unit.unit_id, entities, relationships, llm_ok)

    # -------------------------------------------------------------------------
    # PIPELINE TỔNG: Chạy cho 1 tài liệu (PoC / Test lẻ & Worker song song)
    # -------------------------------------------------------------------------

    def _add_cached_units(self, units: List[SemanticUnit]) -> None:
        """Thêm Semantic Units vào cache với cơ chế chống trùng lặp theo unit_id."""
        for u in units:
            uid = getattr(u, "unit_id", None)
            if uid and uid in self._cached_unit_ids:
                continue
            self._cached_units.append(u)
            if uid:
                self._cached_unit_ids.add(uid)

    def _extract_document_payload(
        self,
        doc,  # ProcessedData
        max_consecutive_failures: Optional[int] = None,
        unit_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Chạy trích xuất 2 chặng cho 1 tài liệu ở chế độ cô lập (thread-safe).
        Trả về payload thuần để gộp tuần tự vào Registry trên main thread / dưới self._lock.
        """
        logger.info(f"Bắt đầu xử lý: '{doc.title}' ({len(doc.content)} ký tự)")

        doc_id = str(getattr(doc, "id", "") or "").strip()
        summary_key = doc_id if doc_id else doc.title

        if self._stop_event.is_set():
            return {
                "doc_id": summary_key,
                "doc_title": doc.title,
                "doc_summary": None,
                "root_nodes": [],
                "pass1_ok": False,
                "units": [],
                "unit_results": [],
                "units_succeeded": 0,
                "units_failed": 0,
            }

        root_context, doc_summary, root_nodes, pass1_ok = self._extract_pass1_data(
            doc.title,
            doc.content,
            header_max_chars=self.header_max_chars,
            doc_id=doc_id or None,
        )
        if (
            max_consecutive_failures is not None
            and self._get_consecutive_failures() >= max_consecutive_failures
        ):
            self._stop_event.set()

        units = self.extractor.extract(doc)
        logger.info(f"  -> {len(units)} Semantic Units")

        unit_results: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], bool]] = []
        units_succeeded = 0
        units_failed = 0

        eff_unit_workers = max(1, int(unit_workers))
        if eff_unit_workers > 1 and len(units) > 1 and not self._stop_event.is_set():
            executor = ThreadPoolExecutor(max_workers=min(eff_unit_workers, len(units)))
            futures = {}
            try:
                for unit in units:
                    if self._stop_event.is_set():
                        units_failed += 1
                        continue
                    futures[executor.submit(self._extract_pass2_unit_data, unit, root_context)] = unit

                for fut in as_completed(futures):
                    unit = futures[fut]
                    entities, relationships, llm_ok = fut.result()
                    unit_results.append((unit.unit_id, entities, relationships, llm_ok))
                    if llm_ok:
                        units_succeeded += 1
                    else:
                        units_failed += 1
                        if (
                            max_consecutive_failures is not None
                            and self._get_consecutive_failures() >= max_consecutive_failures
                        ):
                            self._stop_event.set()
            except KeyboardInterrupt:
                self._stop_event.set()
                for fut in futures:
                    fut.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=not self._stop_event.is_set(), cancel_futures=self._stop_event.is_set())
        else:
            for i, unit in enumerate(units, 1):
                if self._stop_event.is_set() or (
                    max_consecutive_failures is not None
                    and self._get_consecutive_failures() >= max_consecutive_failures
                ):
                    self._stop_event.set()
                    remaining = len(units) - i + 1
                    units_failed += remaining
                    logger.error(
                        f"🛑 Circuit breaker kích hoạt giữa tài liệu '{doc.title}' "
                        f"({self._get_consecutive_failures()} lỗi LLM liên tiếp). Bỏ qua {remaining} units còn lại."
                    )
                    break
                logger.debug(f"  [{i}/{len(units)}] {unit.breadcrumb[:60]}")
                entities, relationships, llm_ok = self._extract_pass2_unit_data(unit, root_context)
                unit_results.append((unit.unit_id, entities, relationships, llm_ok))
                if llm_ok:
                    units_succeeded += 1
                else:
                    units_failed += 1
                    if (
                        max_consecutive_failures is not None
                        and self._get_consecutive_failures() >= max_consecutive_failures
                    ):
                        self._stop_event.set()

        return {
            "doc_id": summary_key,
            "doc_title": doc.title,
            "doc_summary": doc_summary,
            "root_nodes": root_nodes,
            "pass1_ok": pass1_ok,
            "units": units,
            "unit_results": unit_results,
            "units_succeeded": units_succeeded,
            "units_failed": units_failed,
        }

    def _merge_document_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gộp payload trích xuất của 1 tài liệu vào đồ thị chung (thread-safe qua self._lock)."""
        doc_title = payload["doc_title"]
        summary_key = payload.get("doc_id") or doc_title
        pass1_ok = payload["pass1_ok"]
        units = payload["units"]
        units_succeeded = payload["units_succeeded"]
        units_failed = payload["units_failed"]

        with self._lock:
            self._register_pass1_result(
                summary_key, payload["doc_summary"], payload["root_nodes"]
            )

            total_nodes = 0
            total_edges = 0
            for unit_id, entities, relationships, llm_ok in payload["unit_results"]:
                n, e = self._register_pass2_result(unit_id, entities, relationships, llm_ok)
                total_nodes += n
                total_edges += e

            doc_fully_succeeded = pass1_ok and units_failed == 0 and len(units) > 0
            if pass1_ok and units_failed == 0 and len(units) == 0:
                doc_fully_succeeded = True

            if doc_fully_succeeded:
                self._add_cached_units(units)
                self._stats["docs_processed"] += 1
            else:
                self._stats["docs_failed"] += 1
                logger.warning(
                    f"⚠️  Tài liệu '{doc_title}' chưa trích xuất trọn vẹn "
                    f"(pass1_ok={pass1_ok}, units_ok={units_succeeded}/{len(units)}, units_failed={units_failed}). "
                    f"KHÔNG lưu vào cache hoàn thành để tự động chạy bù ở lần resume tiếp theo."
                )

            self._nodes = self.registry.all_nodes()
            node_total = self.registry.node_count()
            edge_total = len(self._edges)

        logger.info(f"Hoàn thành '{doc_title}': +{total_nodes} nodes, +{total_edges} edges")
        return {
            "document": doc_title,
            "units_processed": units_succeeded,
            "units_failed": units_failed,
            "pass1_ok": pass1_ok,
            "fully_succeeded": doc_fully_succeeded,
            "nodes_added": total_nodes,
            "edges_added": total_edges,
            "total_nodes_in_graph": node_total,
            "total_edges_in_graph": edge_total,
        }

    def process_single_document(
        self,
        doc,  # ProcessedData
        max_consecutive_failures: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý 1 tài liệu hoàn chỉnh qua 2 chặng:
          1. Chặng 1: Lấy Root Entities.
          2. Chặng 2: Lấy chi tiết từng Semantic Unit.

        Chỉ đánh dấu tài liệu vào semantic_units_cache khi KHÔNG có unit nào lỗi gọi LLM,
        đảm bảo các tài liệu gặp sự cố API sẽ tự động được chạy bù ở lần resume sau.
        """
        payload = self._extract_document_payload(
            doc, max_consecutive_failures=max_consecutive_failures
        )
        return self._merge_document_payload(payload)

    # -------------------------------------------------------------------------
    # PIPELINE TỔNG: Chạy cho toàn bộ dữ liệu (PHẢI CÓ PHÊ DUYỆT)
    # -------------------------------------------------------------------------

    def run_full_pipeline(
        self,
        docs,            # List[ProcessedData]
        checkpoint_every: int = 10,
        clear_existing: bool = False,
        max_consecutive_failures: int = 10,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        ⚠️  PHƯƠNG THỨC NÀY PHẢI CÓ SỰ PHÊ DUYỆT CỦA QUẢN TRỊ VIÊN TRƯỚC KHI CHẠY.
        ⚠️  Sẽ tiêu tốn nhiều lượt gọi LLM API.

        Chạy pipeline hoàn chỉnh cho toàn bộ danh sách tài liệu:
          - Nạp đồ thị checkpoint hiện tại (nếu có, có tự động fallback từ .bak).
          - Hỗ trợ chạy đa luồng an toàn (workers gọi LLM song song, main thread gộp vào Registry).
          - Lưu checkpoint atomic sau mỗi `checkpoint_every` tài liệu mới xử lý.
          - Circuit breaker tự dừng an toàn & lưu checkpoint nếu LLM lỗi liên tiếp.
          - Bắt KeyboardInterrupt (Ctrl+C) để lưu checkpoint trước khi thoát, không hỏng file.
          - Lưu đồ thị cuối cùng và vector hóa thực thể.

        Kích hoạt qua: python main.py --ingest --to lightrag [--clear]
        """
        self._stop_event.clear()
        with self._lock:
            self._consecutive_llm_failures = 0

        if clear_existing:
            logger.info("⚠️  --clear: Xóa đồ thị cũ và bắt đầu lại từ đầu.")
            self._nodes = []
            self._edges = []
            self._doc_summaries = {}
            self.registry = EntityRegistry()
            self._cached_units = []
            self._cached_unit_ids = set()
            # Xóa file cũ
            for p in (
                self.graph_path,
                self.graph_path.with_name(self.graph_path.name + ".bak"),
                self.semantic_units_cache_path,
            ):
                if p.exists():
                    p.unlink()
            if (self.entity_vector_dir / "chroma.sqlite3").exists():
                try:
                    import chromadb
                    from chromadb.config import Settings

                    client = chromadb.PersistentClient(
                        path=str(self.entity_vector_dir),
                        settings=Settings(anonymized_telemetry=False),
                    )
                    client.delete_collection(name="lightrag_entity_collection")
                    logger.info("✅ Đã làm sạch collection 'lightrag_entity_collection' khi chạy --clear.")
                except Exception:
                    pass
        else:
            # Nạp checkpoint đồ thị hiện có (nếu có)
            self._load_graph()

        processed_doc_ids = set()
        if not clear_existing and self._cached_units:
            for u in self._cached_units:
                d_id = getattr(u, "document_id", None) or getattr(u, "doc_id", None)
                if d_id:
                    processed_doc_ids.add(d_id)
        if processed_doc_ids:
            logger.info(f"🔄 Checkpoint: Đã tìm thấy {len(processed_doc_ids)} tài liệu đã hoàn thành. Sẽ tự động bỏ qua và tiếp tục từ tài liệu chưa xử lý!")

        pending_docs = [
            doc for doc in docs
            if clear_existing or doc.id not in processed_doc_ids
        ]
        eff_workers = max(1, int(max_workers if max_workers is not None else self.max_workers))
        failure_limit = max_consecutive_failures if max_consecutive_failures > 0 else None

        total = len(docs)
        pending_total = len(pending_docs)
        logger.info(
            f"🚀 Bắt đầu nạp {pending_total}/{total} tài liệu cần xử lý vào Knowledge Graph LightRAG "
            f"(workers={eff_workers})..."
        )

        newly_processed = 0
        aborted_due_to_llm_errors = False
        interrupted = False

        try:
            if eff_workers > 1 and pending_total > 1:
                # Xếp tài liệu dài nhất chạy trước (LPT) để các luồng kết thúc đều nhau
                ordered_docs = sorted(
                    pending_docs,
                    key=lambda d: len(getattr(d, "content", "") or ""),
                    reverse=True,
                )
                pool_size = min(eff_workers, pending_total)
                executor = ThreadPoolExecutor(max_workers=pool_size)
                future_to_doc = {}
                try:
                    for doc in ordered_docs:
                        fut = executor.submit(
                            self._extract_document_payload,
                            doc,
                            failure_limit,
                            1,
                        )
                        future_to_doc[fut] = doc

                    for fut in as_completed(future_to_doc):
                        doc = future_to_doc[fut]
                        payload = fut.result()
                        newly_processed += 1
                        logger.info(f"[{newly_processed}/{pending_total}] Gộp kết quả: {doc.title}")
                        self._merge_document_payload(payload)

                        if failure_limit is not None and self._get_consecutive_failures() >= failure_limit:
                            aborted_due_to_llm_errors = True
                            self._stop_event.set()
                            for pending_fut in future_to_doc:
                                pending_fut.cancel()
                            logger.error(
                                f"🛑 Circuit breaker: Đã xảy ra {self._get_consecutive_failures()} lỗi LLM liên tiếp. "
                                f"Tạm dừng an toàn và lưu checkpoint!"
                            )
                            break

                        if checkpoint_every > 0 and newly_processed % checkpoint_every == 0:
                            self.save_graph()
                            logger.info(
                                f"  Checkpoint saved ({newly_processed}/{pending_total}): "
                                f"{self.registry.node_count()} nodes, {len(self._edges)} edges"
                            )
                except KeyboardInterrupt:
                    interrupted = True
                    self._stop_event.set()
                    for pending_fut in future_to_doc:
                        pending_fut.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    executor.shutdown(wait=not self._stop_event.is_set(), cancel_futures=self._stop_event.is_set())
            else:
                unit_workers = eff_workers if (eff_workers > 1 and pending_total == 1) else 1
                for idx, doc in enumerate(pending_docs, 1):
                    if failure_limit is not None and self._get_consecutive_failures() >= failure_limit:
                        aborted_due_to_llm_errors = True
                        self._stop_event.set()
                        logger.error(
                            f"🛑 Circuit breaker: Đã xảy ra {self._get_consecutive_failures()} lỗi LLM liên tiếp. "
                            f"Tạm dừng an toàn và lưu checkpoint để tránh tạo dữ liệu rỗng!"
                        )
                        break

                    logger.info(f"[{idx}/{pending_total}] Xử lý: {doc.title}")
                    payload = self._extract_document_payload(
                        doc,
                        max_consecutive_failures=failure_limit,
                        unit_workers=unit_workers,
                    )
                    self._merge_document_payload(payload)
                    newly_processed += 1

                    if failure_limit is not None and self._get_consecutive_failures() >= failure_limit:
                        aborted_due_to_llm_errors = True
                        self._stop_event.set()
                        logger.error(
                            f"🛑 Circuit breaker: Đã xảy ra {self._get_consecutive_failures()} lỗi LLM liên tiếp. "
                            f"Tạm dừng an toàn và lưu checkpoint!"
                        )
                        break

                    if checkpoint_every > 0 and newly_processed % checkpoint_every == 0:
                        self.save_graph()
                        logger.info(f"  Checkpoint saved: {self.registry.node_count()} nodes, {len(self._edges)} edges")
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("⚠️  Nhận tín hiệu dừng (KeyboardInterrupt). Đang lưu checkpoint an toàn trước khi thoát...")
            self.save_graph()
            raise

        # Lưu đồ thị cuối cùng (atomic)
        self.save_graph()

        # Chỉ vector hóa toàn bộ khi chạy xong bình thường (không bị dừng sớm do outage)
        if not aborted_due_to_llm_errors and not interrupted:
            self.vectorize_entities(force_reembed=clear_existing)
        else:
            logger.warning(
                "⚠️  Bỏ qua bước vector hóa do phiên nạp dừng sớm. "
                "Hãy kiểm tra lại LLM API rồi chạy lại lệnh resume để hoàn tất phần còn lại."
            )

        summary = {
            **self._stats,
            "final_nodes": self.registry.node_count(),
            "final_edges": len(self._edges),
            "graph_path": str(self.graph_path),
            "aborted_due_to_llm_errors": aborted_due_to_llm_errors,
            "workers": eff_workers,
        }
        logger.info(f"✅ Hoàn thành! {summary}")
        return summary

    # -------------------------------------------------------------------------
    # LƯU TRỮ & NẠP ĐỒ THỊ
    # -------------------------------------------------------------------------

    def save_graph(self) -> None:
        """Lưu đồ thị (nodes + edges) ra file JSON theo cơ chế atomic write + sao lưu .bak."""
        self._nodes = self.registry.all_nodes()
        graph_data = {
            "meta": {
                "created_by": "LightRAGIndexer",
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "summary_count": len(self._doc_summaries),
                "stats": self._stats,
            },
            "nodes": self._nodes,
            "edges": self._edges,
            "entities": {node["canonical_name"]: node for node in self._nodes},
            "relations": self._edges,
            "document_summaries": self._doc_summaries,
        }
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.graph_path.with_name(self.graph_path.name + ".tmp")
        bak_path = self.graph_path.with_name(self.graph_path.name + ".bak")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Sao lưu bản cũ hợp lệ sang .bak trước khi thay thế nguyên tử
            if self.graph_path.exists() and self.graph_path.stat().st_size > 0:
                try:
                    shutil.copy2(self.graph_path, bak_path)
                except OSError as bak_err:
                    logger.warning(f"Không thể tạo file sao lưu {bak_path}: {bak_err}")

            os.replace(tmp_path, self.graph_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

        # Lưu cache Semantic Units ra file JSONL (atomic) phục vụ tra cứu và resume
        if self._cached_units:
            self.extractor.save_units_jsonl(self._cached_units, self.semantic_units_cache_path)

        logger.info(
            f"Đã lưu đồ thị: {len(self._nodes)} nodes, {len(self._edges)} edges, "
            f"{len(self._doc_summaries)} summaries -> {self.graph_path}"
        )

    def _populate_from_graph_data(self, data: Dict[str, Any]) -> None:
        """Khôi phục trạng thái in-memory (Registry, edges, summaries, stats) từ dict JSON."""
        self.registry = EntityRegistry()
        self._edges = data.get("edges", []) or data.get("relations", [])
        self._doc_summaries = data.get("document_summaries", {})
        if "meta" in data and "stats" in data["meta"]:
            self._stats = {**self._stats, **data["meta"]["stats"]}
        nodes = data.get("nodes", [])
        if not nodes and "entities" in data:
            if isinstance(data["entities"], dict):
                nodes = list(data["entities"].values())
            elif isinstance(data["entities"], list):
                nodes = data["entities"]
        for node in nodes:
            canonical = node.get("canonical_name", "")
            if canonical:
                self.registry._nodes[canonical.lower()] = node
        dedup_edges: List[Dict[str, Any]] = []
        for edge in self._edges:
            rel_t = edge.get("relation_type") or edge.get("predicate", "liên_quan")
            edge_key = f"{edge['source'].lower()}::{rel_t}::{edge['target'].lower()}"
            if edge_key in self.registry._edges_by_key:
                existing = self.registry._edges_by_key[edge_key]
                for uid in edge.get("source_unit_ids", ()) or ():
                    if uid and uid not in existing.setdefault("source_unit_ids", []):
                        existing["source_unit_ids"].append(uid)
                for ev in edge.get("evidence_texts", ()) or ():
                    if ev and ev not in existing.setdefault("evidence_texts", []) and len(existing["evidence_texts"]) < 5:
                        existing["evidence_texts"].append(ev)
                continue
            self.registry._edge_key_set.add(edge_key)
            self.registry._edges_by_key[edge_key] = edge
            dedup_edges.append(edge)
        self._edges = dedup_edges
        self._nodes = list(self.registry._nodes.values())

    def _load_graph(self) -> bool:
        """
        Nạp đồ thị từ file JSON (checkpoint).
        Tự động thử bản dự phòng .bak nếu graph.json bị hỏng.
        Nếu file tồn tại nhưng cả bản chính lẫn .bak đều hỏng -> ném RuntimeError
        thay vì âm thầm chạy lại từ đầu và ghi đè dữ liệu.
        """
        bak_path = self.graph_path.with_name(self.graph_path.name + ".bak")
        if not self.graph_path.exists() and not bak_path.exists():
            logger.info("Không tìm thấy đồ thị cũ, bắt đầu từ đầu.")
            return False

        candidates = []
        if self.graph_path.exists():
            candidates.append(self.graph_path)
        if bak_path.exists():
            candidates.append(bak_path)

        last_err: Optional[Exception] = None
        loaded_from: Optional[Path] = None
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._populate_from_graph_data(data)
                loaded_from = path
                break
            except Exception as e:
                last_err = e
                logger.error(f"Lỗi nạp đồ thị từ {path}: {e}")

        if loaded_from is None:
            raise RuntimeError(
                f"File checkpoint đồ thị '{self.graph_path}' bị hỏng và không có bản .bak hợp lệ "
                f"(lỗi cuối: {last_err}). Hãy khôi phục file hoặc dùng cờ --clear nếu muốn tạo lại từ đầu."
            )

        if loaded_from == bak_path:
            logger.warning(f"⚠️  Đã tự động khôi phục đồ thị từ bản sao lưu {bak_path}!")

        # Khôi phục cache Semantic Units nếu có
        if self.semantic_units_cache_path.exists():
            self._cached_units = self.extractor.load_units_jsonl(self.semantic_units_cache_path)
            self._cached_unit_ids = {
                u.unit_id for u in self._cached_units if getattr(u, "unit_id", None)
            }

        logger.info(
            f"Đã nạp đồ thị checkpoint ({loaded_from.name}): "
            f"{self.registry.node_count()} nodes, {len(self._edges)} edges"
        )
        return True

    def get_graph_stats(self) -> Dict[str, Any]:
        """Trả về thống kê đồ thị hiện tại (không cần nạp file)."""
        return {
            "nodes": self.registry.node_count(),
            "edges": len(self._edges),
            "graph_path": str(self.graph_path),
            "graph_exists": self.graph_path.exists(),
            "session_stats": self._stats,
        }

    # -------------------------------------------------------------------------
    # VECTOR HÓA THỰC THỂ (phục vụ Retrieval)
    # -------------------------------------------------------------------------

    def vectorize_entities(
        self,
        force_reembed: bool = False,
        chroma_dir: Optional[Path] = None,
    ) -> int:
        """
        Vector hóa tên + mô tả của các Node trong đồ thị bằng BGE-M3 ONNX.
        Lưu vào ChromaDB collection riêng: lightrag_entity_collection.

        Tối ưu gia tăng (Incremental Embedding):
          - Mặc định (force_reembed=False): Kiểm tra collection hiện có, BỎ QUA các entity
            đã có vector với cùng nội dung văn bản; nếu chỉ đổi metadata (vd: thêm source_unit_ids)
            thì chỉ cập nhật metadata mà KHÔNG gọi lại model embedding.
          - Chỉ nhúng lại toàn bộ khi force_reembed=True (vd: cờ --clear).

        Returns:
            Số lượng entity mới/thay đổi đã được vector hóa trong phiên này.
        """
        nodes = self.registry.all_nodes()
        if not nodes:
            logger.warning("Không có node nào để vector hóa.")
            return 0

        try:
            import chromadb
            from chromadb.config import Settings

            # Mở ChromaDB collection dành riêng cho LightRAG Entity trước khi tải model nặng
            target_chroma_dir = Path(chroma_dir) if chroma_dir else self.entity_vector_dir
            target_chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(target_chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection(
                name="lightrag_entity_collection",
                metadata={"hnsw:space": "cosine"},
            )

            existing_map: Dict[str, Tuple[str, Dict[str, Any]]] = {}
            if not force_reembed:
                try:
                    existing = collection.get(include=["documents", "metadatas"])
                    ex_ids = existing.get("ids") or []
                    ex_docs = existing.get("documents") or []
                    ex_metas = existing.get("metadatas") or []
                    for eid, edoc, emeta in zip(ex_ids, ex_docs, ex_metas):
                        existing_map[eid] = (edoc or "", emeta or {})
                except Exception as read_err:
                    logger.warning(f"Không thể đọc entity hiện có từ Chroma ({read_err}); sẽ nhúng toàn bộ.")

            items_to_embed: List[Tuple[str, str, Dict[str, Any]]] = []
            meta_only_updates: List[Tuple[str, Dict[str, Any]]] = []
            skipped_unchanged = 0

            for node in nodes:
                name = node["canonical_name"]
                desc = " ".join(node.get("descriptions", []))[:300]
                text_to_embed = f"{name}. {desc}".strip()
                nid = node["id"]
                meta = {
                    "canonical_name": name,
                    "entity_type": node.get("entity_type", "OTHER"),
                    "source_unit_ids": json.dumps(node.get("source_unit_ids", [])),
                }

                if not force_reembed and nid in existing_map:
                    old_doc, old_meta = existing_map[nid]
                    if old_doc == text_to_embed:
                        if old_meta == meta:
                            skipped_unchanged += 1
                        else:
                            meta_only_updates.append((nid, meta))
                        continue

                items_to_embed.append((nid, text_to_embed, meta))

            # Cập nhật các node chỉ thay đổi metadata (không cần nhúng lại vector)
            if meta_only_updates:
                batch_size = 500
                for i in range(0, len(meta_only_updates), batch_size):
                    sub = meta_only_updates[i : i + batch_size]
                    collection.update(
                        ids=[x[0] for x in sub],
                        metadatas=[x[1] for x in sub],
                    )

            logger.info(
                f"📊 Incremental Entity Embedding: tổng={len(nodes)}, "
                f"bỏ qua không đổi={skipped_unchanged}, "
                f"chỉ cập nhật metadata={len(meta_only_updates)}, "
                f"cần nhúng mới/thay đổi={len(items_to_embed)}"
            )

            if not items_to_embed:
                logger.info("✅ Không có entity mới/thay đổi nào cần nhúng thêm.")
                return 0

            from src.embeddings.embedder import UETEmbedder
            embedder = UETEmbedder()

            batch_size = 100
            count = 0

            for i in range(0, len(items_to_embed), batch_size):
                batch = items_to_embed[i : i + batch_size]
                ids = [x[0] for x in batch]
                texts = [x[1] for x in batch]
                metadatas = [x[2] for x in batch]

                # Embedding batch bằng BGE-M3 ONNX (tối ưu hóa batch inference)
                if hasattr(embedder, "embed_texts"):
                    vectors = embedder.embed_texts(texts)
                elif hasattr(embedder, "embed_batch"):
                    vectors = embedder.embed_batch(texts)
                else:
                    vectors = []
                    for text in texts:
                        vec = embedder.embed_query(text)
                        vectors.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))

                collection.upsert(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=metadatas,
                )
                count += len(batch)
                logger.info(f"  Vector hóa {count}/{len(items_to_embed)} entities mới...")

            logger.info(
                f"✅ Đã vector hóa {count} entities mới/thay đổi vào lightrag_entity_collection "
                f"(tổng trong collection: {collection.count()})"
            )
            return count

        except Exception as e:
            logger.error(f"Lỗi vector hóa entities: {e}")
            return 0


if __name__ == "__main__":
    # Test nhanh với 1 tài liệu giả lập (không gọi LLM thật)
    print("LightRAGIndexer module loaded. Các class sẵn sàng sử dụng.")
    print("Để chạy PoC với 1 văn bản: LightRAGIndexer().process_single_document(doc)")
    print("Để chạy toàn pipeline (CẦN PHÊ DUYỆT): python main.py --ingest --to lightrag")

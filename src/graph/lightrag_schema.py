"""
src/graph/lightrag_schema.py - Schema constants, EntityRegistry, PromptBuilder,
and LLMResponseParser for LightRAG ingestion.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.graph.adaptive_extractor import SemanticUnit
from src.graph.thesaurus_utils import (
    AcademicThesaurus,
    FORBIDDEN_ENTITIES,
    is_forbidden_entity,
)
from src.utils.helpers import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
logger = get_logger("graph.lightrag_indexer")

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

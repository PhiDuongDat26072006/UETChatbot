"""
src/retrieval/router.py - Phân loại Ý định Truy vấn & Điều phối Động (Query Intent Classifier & Router).
Điều phối thông minh giữa:
  - "vector": Truy xuất lai Hybrid BM25 + Vector Search (truy xuất đoạn văn bản, văn bản đơn lẻ, liên hệ/giờ giấc).
  - "lightrag": Truy xuất Đồ thị Tri thức 2 tầng (quan hệ đa thực thể, điều kiện ràng buộc, so sánh, tiên quyết).
  - "hybrid": Dung hợp cả kết quả từ Đồ thị Tri thức (LightRAG) và Đoạn cắt văn bản (Vector/BM25).
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set

from src.base import UserQuery
from src.graph.thesaurus_utils import AcademicThesaurus, get_thesaurus, is_forbidden_entity
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("retrieval.router")


class QueryRouter:
    """
    Bộ điều phối truy vấn thông minh dựa trên mật độ thực thể và từ khóa quan hệ học vụ.
    Hoạt động với độ trễ cực thấp (<1ms) không phụ thuộc vào LLM phân loại.
    """

    # Nhóm từ khóa chỉ định mối quan hệ / điều kiện / quy trình đa thực thể -> Ưu tiên LightRAG / Hybrid
    RELATIONAL_PATTERNS = [
        r"\b(điều kiện|yêu cầu|tiêu chí|tiêu chuẩn)\b",
        r"\b(mối quan hệ|liên quan|khác nhau|so sánh|phân biệt)\b",
        r"\b(tiên quyết|học trước|ràng buộc|bắt buộc trước)\b",
        r"\b(quy trình|thủ tục|các bước|trình tự|hướng dẫn)\b",
        r"\b(ảnh hưởng|dẫn đến|hậu quả|bị gì|bị sao|có sao không|có bị)\b",
        r"\b(được không|có được|khi nào|bao giờ|mức nào)\b",
        r"\b(xét tốt nghiệp|ra trường|chuẩn đầu ra)\b",
        r"\b(học bổng|cảnh báo học vụ|buộc thôi học|thôi học|đuổi học|bảo lưu|học lại|học cải thiện|tạch|trượt|rớt môn|nợ môn|kỷ luật)\b",
        r"\b(chuyển ngành|song ngành|bằng kép|học cùng lúc)\b",
    ]

    # Nhóm từ khóa tra cứu thông tin tra cứu sự kiện/đơn vị cụ thể -> Ưu tiên Vector
    PASSAGE_PATTERNS = [
        r"\b(ở đâu|địa chỉ|tại đâu|phòng nào|tòa nhà|cơ sở)\b",
        r"\b(số điện thoại|sđt|hotline|liên hệ|email|mail|website|web)\b",
        r"\b(thời gian|lịch|mấy giờ|ngày nào|hạn chót|deadline)\b",
        r"\b(link|đường link|trang web|mẫu đơn|biểu mẫu)\b",
    ]

    # Nhóm khẩu ngữ / tiếng lóng sinh viên bắt buộc kích hoạt LLM PA1 để dịch sang thuật ngữ chuẩn
    COLLOQUIAL_SLANG_PATTERNS = [
        r"\b(tạch|rớt môn|trượt môn|trượt|nợ môn|nợ tín|học lại|cải thiện)\b",
        r"\b(đuổi học|bị đuổi|cảnh báo|thôi học|bảo lưu|đình chỉ|cấm thi)\b",
        r"\b(học 2 ngành|hai ngành|song ngành|song bằng|bằng kép|văn bằng 2)\b",
        r"\b(ra trường|bị gì|bị sao|có sao không|mất học bổng|chậm tiến độ|quá hạn)\b",
    ]

    # Nhóm tra cứu số liệu / bảng điểm / mã cứng đã rõ từ khóa -> Chạy thẳng 0-LLM (<15ms)
    TABULAR_OR_CODE_PATTERNS = [
        r"\b(điểm chuẩn|điểm trúng tuyển|điểm sàn|ngưỡng đầu vào|chỉ tiêu|học phí|mã ngành|mã xét tuyển)\b",
        r"\b(\d{1,5}/(?:QĐ|NQ|TB|KH|HD|CV)-[A-ZĐƠƯĂÂÊÔ0-9\-]+|[A-Z]{2,4}\d{4}[A-Z0-9]*)\b",
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        thesaurus: Optional[AcademicThesaurus] = None,
    ):
        raw_cfg = config or load_yaml_config()
        lightrag_cfg = raw_cfg.get("lightrag", {})
        router_cfg = lightrag_cfg.get("router", {})

        # Chế độ điều phối: "auto", "always_hybrid", "always_lightrag", "always_vector"
        self.mode: str = router_cfg.get("mode", "auto")
        self.confidence_threshold: float = float(router_cfg.get("confidence_threshold", 0.65))
        self.use_llm_query_extraction: bool = bool(lightrag_cfg.get("use_llm_query_extraction", True))
        self.llm_extraction_mode: str = str(
            router_cfg.get("llm_extraction_mode", lightrag_cfg.get("llm_extraction_mode", "auto"))
        ).lower().strip()
        self.thesaurus = thesaurus or get_thesaurus()

        self._word_regex = re.compile(r"[\w\+\-\./]+", re.UNICODE)
        self._rel_regex = re.compile("|".join(self.RELATIONAL_PATTERNS), re.IGNORECASE)
        self._pas_regex = re.compile("|".join(self.PASSAGE_PATTERNS), re.IGNORECASE)
        self._slang_regex = re.compile("|".join(self.COLLOQUIAL_SLANG_PATTERNS), re.IGNORECASE)
        self._tab_code_regex = re.compile("|".join(self.TABULAR_OR_CODE_PATTERNS), re.IGNORECASE)

    def should_use_llm_extraction(self, query: str | UserQuery) -> bool:
        """
        Quyết định thông minh (Lựa chọn 1 - <0.5ms) có cần gọi LLM QueryEntityExtractor (PA1) hay không:
          - Bật PA1 LLM (chạy song song với BM25 + Vector) khi câu hỏi chứa khẩu ngữ/tiếng lóng sinh viên
            hoặc hỏi quy chế quan hệ/điều kiện/hậu quả đa thực thể phức tạp.
          - Chạy thẳng 0-LLM (Thesaurus + Regex + Vector NumPy <15ms) cho tra cứu bảng điểm, địa chỉ/liên hệ, mã môn/số QĐ.
        """
        if not self.use_llm_query_extraction or self.llm_extraction_mode in ("off", "false", "none", "0"):
            return False
        if self.llm_extraction_mode == "always":
            return True

        query_text = (query.query_text if isinstance(query, UserQuery) else str(query or "")).strip()
        if not query_text:
            return False

        clean_text = query_text.lower()

        # 1. Nếu có tiếng lóng/khẩu ngữ sinh viên -> BẮT BUỘC dùng PA1 LLM để dịch sang thuật ngữ chuẩn
        if self._slang_regex.search(clean_text):
            return True

        has_relational = bool(self._rel_regex.search(clean_text))
        has_passage = bool(self._pas_regex.search(clean_text))
        has_tab_or_code = bool(self._tab_code_regex.search(query_text))

        # 2. Câu hỏi tra cứu bảng điểm/mã ngành/học phí hoặc địa chỉ/liên hệ đơn thuần -> 0-LLM siêu tốc
        if has_tab_or_code and not has_relational:
            return False
        if has_passage and not has_relational:
            return False
        if has_tab_or_code and any(k in clean_text for k in ("điểm chuẩn", "điểm trúng tuyển", "điểm sàn", "chỉ tiêu")):
            return False

        # 3. Câu hỏi quan hệ/điều kiện/quy trình học vụ đủ dài (>= 5 từ) -> Bật PA1 LLM
        word_count = len(clean_text.split())
        if has_relational and word_count >= 5:
            return True

        return False

    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Phân tích các đặc trưng học vụ của truy vấn:
        - Mật độ thực thể UET đặc thù (loại trừ từ cấm chung chung như 'UET', 'môn học', 'sinh viên')
        - Dấu hiệu câu hỏi quan hệ/điều kiện (relational signals)
        - Dấu hiệu câu hỏi sự kiện/tra cứu thông tin liên hệ (passage signals)
        """
        clean_text = (query_text or "").lower().strip()

        # 1. Quét thực thể không chồng lấp qua Academic Thesaurus (tách sạch dấu câu)
        detected_entities: List[str] = []
        covered_positions: Set[int] = set()
        words = self._word_regex.findall(clean_text)
        for n in range(min(5, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                span = set(range(i, i + n))
                if span & covered_positions:
                    continue
                phrase = " ".join(words[i : i + n])
                if is_forbidden_entity(phrase):
                    continue
                canonical = self.thesaurus.get_canonical(phrase)
                if canonical and not is_forbidden_entity(canonical):
                    covered_positions.update(span)
                    if canonical not in detected_entities:
                        detected_entities.append(canonical)

        # 2. Quét mẫu từ khóa
        has_relational = bool(self._rel_regex.search(clean_text))
        has_passage = bool(self._pas_regex.search(clean_text))

        ent_cnt = len(detected_entities)
        if ent_cnt >= 2 and has_relational:
            confidence = 0.90
        elif ent_cnt >= 1 and has_relational and not has_passage:
            confidence = 0.82
        elif has_passage and not has_relational:
            confidence = 0.85
        elif ent_cnt <= 1 and not has_relational:
            confidence = 0.75
        else:
            confidence = 0.60

        return {
            "query_text": query_text,
            "detected_entities": detected_entities,
            "entity_count": ent_cnt,
            "has_relational_intent": has_relational,
            "has_passage_intent": has_passage,
            "confidence": confidence,
        }

    def route(self, query: str | UserQuery) -> str:
        """
        Ra quyết định chọn chế độ điều phối truy xuất:
          - Chế độ "auto": chọn "hybrid" (tăng cường nhánh LightRAG + Overview) hoặc "vector" (ưu tiên BM25/Vector);
            nếu độ tự tin phân loại < confidence_threshold, fallback an toàn sang "hybrid".
          - Chế độ ghi đè cứng: "always_vector" | "always_lightrag" | "always_hybrid".
        """
        if self.mode == "always_vector":
            return "vector"
        if self.mode == "always_lightrag":
            return "lightrag"
        if self.mode == "always_hybrid":
            return "hybrid"

        query_text = query.query_text if isinstance(query, UserQuery) else str(query)
        analysis = self.analyze_query(query_text)

        entity_count = analysis["entity_count"]
        has_relational = analysis["has_relational_intent"]
        has_passage = analysis["has_passage_intent"]
        confidence = float(analysis.get("confidence", 0.75))

        if entity_count >= 2 and has_relational:
            decision = "hybrid"
        elif entity_count >= 1 and has_relational and not has_passage:
            decision = "hybrid"
        elif has_passage and not has_relational:
            decision = "vector"
        elif entity_count <= 1 and not has_relational:
            decision = "vector" if confidence >= self.confidence_threshold else "hybrid"
        else:
            decision = "hybrid"

        logger.debug(
            f"[QueryRouter] Query: '{query_text[:50]}...' -> Route: {decision.upper()} "
            f"(Entities: {analysis['detected_entities']}, Rel: {has_relational}, Pass: {has_passage}, Conf: {confidence:.2f})"
        )
        return decision

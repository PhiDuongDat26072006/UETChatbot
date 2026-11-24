"""
src/agent/rewriter.py - Mở rộng từ viết tắt, trích xuất thực thể khoa và chuẩn hóa truy vấn học vụ UET.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple


# Từ điển từ viết tắt học vụ UET - ĐHQGHN
ACRONYM_MAP: Dict[str, str] = {
    r"\bctđt\b": "chương trình đào tạo",
    r"\bclc\b": "chất lượng cao",
    r"\bknlnn\b": "khung năng lực ngoại ngữ",
    r"\bđrl\b": "điểm rèn luyện",
    r"\bđkhp\b": "đăng ký học phần",
    r"\bcđr\b": "chuẩn đầu ra",
    r"\bgpa\b": "điểm trung bình chung",
    r"\bcpa\b": "điểm trung bình chung tích lũy",
    r"\bkltn\b": "khóa luận tốt nghiệp",
    r"\btttn\b": "thực tập tốt nghiệp",
    r"\bctsv\b": "phòng công tác sinh viên",
    r"\bpđt\b": "phòng đào tạo",
    r"\bgdqp\b": "giáo dục quốc phòng an ninh",
    r"\bgdqp-an\b": "giáo dục quốc phòng an ninh",
    r"\bgdtc\b": "giáo dục thể chất",
    r"\bkkht\b": "khuyến khích học tập",
    r"\bhb\b": "học bổng",
    r"\bhbkh\b": "học bổng khuyến khích học tập",
    r"\bhbkk\b": "học bổng khuyến khích học tập",
    r"\bhbkkht\b": "học bổng khuyến khích học tập",
    r"\bbth\b": "buộc thôi học",
    r"\bcb\b": "cảnh báo",
    r"\bcb-hv\b": "cảnh báo học vụ",
    r"\bcbhv\b": "cảnh báo học vụ",
    r"\btc\b": "tín chỉ",
}

# Ánh xạ thuật ngữ đời thường sang thuật ngữ văn bản chính thức của UET khi truy xuất
# Mỗi phần tử: (regex_khớp_câu_hỏi, từ_khóa_kiểm_tra_đã_có, cụm_thuật_ngữ_bổ_sung)
SYNONYM_EXPANSION_RULES: List[Tuple[str, str, str]] = [
    (r"\bđiểm chuẩn\b", "trúng tuyển", "điểm trúng tuyển"),
    (r"\bđiểm tuyển sinh\b", "trúng tuyển", "điểm trúng tuyển"),
    (r"\bđiểm đầu vào\b", "trúng tuyển", "điểm trúng tuyển"),
    (r"\bđiểm sàn\b", "ngưỡng", "ngưỡng đầu vào ngưỡng đảm bảo chất lượng đầu vào"),
    (r"\b(?:nợ môn|rớt môn|trượt môn)\b", "học lại", "học lại xử lý học tập"),
    (r"\bđiểm kém\b", "cảnh báo", "cảnh báo học vụ"),
    (r"\b(?:chuẩn tiếng anh|bằng tiếng anh)\b", "chuẩn đầu ra", "chuẩn đầu ra ngoại ngữ"),
    (r"\b(?:bảo lưu|nghỉ học tạm)\b", "nghỉ học tạm thời", "nghỉ học tạm thời"),
]

# Ánh xạ từ khóa sang mã khoa / viện
FACULTY_PATTERNS: List[Tuple[str, List[str]]] = [
    ("fit", ["cntt", "công nghệ thông tin", "fit", "khoa học máy tính", "kỹ thuật phần mềm", "hệ thống thông tin", "mạng máy tính"]),
    ("fet", ["đtvt", "điện tử viễn thông", "fet", "viễn thông", "kỹ thuật điện tử", "hệ thống nhúng", "iot"]),
    ("fema", ["fema", "cơ học kỹ thuật", "tự động hóa", "cơ điện tử", "kỹ thuật điều khiển", "cơ kỹ thuật"]),
    ("fepn", ["fepn", "vật lý kỹ thuật", "công nghệ nano", "quang điện tử"]),
    ("fat", ["fat", "công nghệ nông nghiệp", "nông nghiệp"]),
    ("fce", ["fce", "xây dựng", "giao thông", "công nghệ xây dựng"]),
    ("sae", ["sae", "hàng không vũ trụ", "hàng không"]),
    ("iai", ["iai", "trí tuệ nhân tạo", "khoa học dữ liệu"]),
]


class UETQueryRewriter:
    """
    Bộ phân tích và viết lại truy vấn (Query Rewriter & Expander):
    1. Chuẩn hóa & mở rộng từ viết tắt học vụ UET (CTĐT -> chương trình đào tạo, ĐRL -> điểm rèn luyện...).
    2. Nhận diện khoa / viện để định tuyến metadata filter.
    3. Sinh query tối ưu hóa cho hệ thống Hybrid Retrieval (BM25 + BGE-M3).
    """

    def __init__(
        self,
        llm_client=None,
        enable_llm: bool = True,
        model: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.enable_llm = enable_llm
        self.model = model

    @staticmethod
    def expand_acronyms(text: str) -> str:
        """Thay thế các từ viết tắt học vụ phổ biến bằng cụm từ đầy đủ."""
        expanded = text
        for pattern, replacement in ACRONYM_MAP.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        return expanded

    @staticmethod
    def expand_synonyms(text: str) -> str:
        """
        Bổ sung thuật ngữ tương đương trong văn bản chính thức UET ngay cạnh từ khóa đời thường
        (ví dụ: 'điểm chuẩn' -> 'điểm chuẩn điểm trúng tuyển') để tăng độ khớp cho BM25, Vector và Reranker.
        """
        expanded = text
        for pattern, guard, addition in SYNONYM_EXPANSION_RULES:
            if guard.lower() in expanded.lower():
                continue
            match = re.search(pattern, expanded, flags=re.IGNORECASE)
            if match:
                expanded = (
                    expanded[: match.end()] + " " + addition + expanded[match.end() :]
                )
        return expanded

    @staticmethod
    def expand_for_retrieval(text: str) -> str:
        """Chuẩn hóa truy vấn trước khi tìm kiếm: mở rộng từ viết tắt và bổ sung thuật ngữ đồng nghĩa UET."""
        if not text:
            return text
        expanded = UETQueryRewriter.expand_acronyms(text)
        expanded = UETQueryRewriter.expand_synonyms(expanded)
        return re.sub(r"\s+", " ", expanded).strip()

    @staticmethod
    def detect_faculties(text: str) -> List[str]:
        """
        Trích xuất TẤT CẢ các mã khoa khớp từ câu hỏi (fit, fet, fema, sae, fat, fepn, fce, iai).
        Tuân thủ Phase 2.1 (D5) - Bộ lọc khoa mềm hỗ trợ nhiều khoa.
        """
        lower = text.lower()
        matched = []
        for faculty_code, keywords in FACULTY_PATTERNS:
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                    if faculty_code not in matched:
                        matched.append(faculty_code)
                    break
        return matched

    @staticmethod
    def detect_faculty(text: str) -> Optional[str]:
        """Trích xuất mã khoa đầu tiên từ câu hỏi (giữ tương thích ngược)."""
        faculties = UETQueryRewriter.detect_faculties(text)
        return faculties[0] if faculties else None

    def rewrite_for_retrieval(self, query: str, retry_feedback: Optional[str] = None) -> str:
        """
        Viết lại câu hỏi thành chuỗi tìm kiếm học vụ chuẩn mực.
        Nếu có retry_feedback (khi lần trước không tìm thấy), bổ sung thêm các từ khóa quy chế tương đương.
        """
        base_expanded = self.expand_acronyms(query.strip())

        # Nếu có LLM và enable_llm bật, dùng LLM để viết lại
        if self.enable_llm and self.llm_client and retry_feedback:
            try:
                prompt_text = (
                    f"Câu hỏi của sinh viên: '{query}'\n"
                    f"Lý do lần tìm kiếm trước thất bại: {retry_feedback}\n"
                    f"Hãy viết lại một truy vấn tìm kiếm ngắn gọn (dưới 15 từ), chứa các thuật ngữ pháp lý / quy chế chính thức "
                    f"của Trường Đại học Công nghệ (VNU-UET) như Điều, Khoản, Quy định, Tiêu chuẩn, Điều kiện để tìm kiếm chính xác."
                )
                from src.base import AugmentedPrompt, UserQuery
                uq = UserQuery(query_text=query)
                ap = AugmentedPrompt(
                    user_query=uq,
                    contexts=[],
                    system_instruction="Bạn là chuyên viên tối ưu hóa truy vấn học vụ UET.",
                    formatted_prompt=prompt_text,
                )
                llm_resp = self.llm_client.generate(ap, model=self.model) if hasattr(self.llm_client, "model") else self.llm_client.generate(ap)
                clean_rewrite = llm_resp.answer.strip().replace('"', '').replace("'", "")
                if clean_rewrite and len(clean_rewrite) > 5:
                    return clean_rewrite
            except Exception:
                pass

        # Chiến lược Heuristic mặc định: Thêm từ khóa ngữ cảnh quy chế
        tokens = [base_expanded]
        lower_q = base_expanded.lower()
        if any(w in lower_q for w in ["học bổng", "kkht"]) and "điều kiện" not in lower_q:
            tokens.append("tiêu chuẩn điều kiện xét cấp học bổng")
        elif any(w in lower_q for w in ["tốt nghiệp", "ra trường"]) and "điều kiện" not in lower_q:
            tokens.append("quy định điều kiện công nhận tốt nghiệp")
        elif any(w in lower_q for w in ["ngoại ngữ", "tiếng anh", "ielts", "toefl", "vstep"]) and "chuẩn" not in lower_q:
            tokens.append("quy đổi chuẩn đầu ra ngoại ngữ bậc 3 bậc 4")
        elif any(w in lower_q for w in ["thôi học", "cảnh báo"]) and "mức" not in lower_q:
            tokens.append("quy chế xử lý cảnh báo học vụ buộc thôi học")

        return " ".join(tokens)

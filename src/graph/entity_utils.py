"""Domain patterns, canonicalization, relation normalization, and tokenization for Knowledge Graph."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from src.graph.thesaurus_utils import get_thesaurus

FACULTY_LABELS: Dict[str, str] = {
    "fit": "Khoa Công nghệ Thông tin (FIT)",
    "fet": "Khoa Điện tử Viễn thông (FET)",
    "fema": "Khoa Cơ học kỹ thuật và Tự động hóa (FEMA)",
    "fepn": "Khoa Vật lý kỹ thuật và Công nghệ Nano (FEPN)",
    "fat": "Khoa Công nghệ Nông nghiệp (FAT)",
    "fce": "Khoa Công nghệ Xây dựng - Giao thông (FCE)",
    "iai": "Viện Trí tuệ nhân tạo (IAI)",
    "sae": "Khoa Hàng không Vũ trụ (SAE)",
    "handbook": "Sổ tay sinh viên UET (HANDBOOK)",
    "tuyensinh": "Thông tin Tuyển sinh UET",
    "studenthub": "Cổng thông tin Sinh viên UET",
    "tuoitrecongnghe": "Tuổi trẻ Công nghệ UET",
    "uet": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
}

_VN_STOP_BOUNDARY = {
    "của", "tại", "về", "được", "là", "có", "theo", "cho", "với", "trong",
    "từ", "đến", "khi", "nếu", "nhằm", "như", "gồm", "bao", "đã", "đang",
    "sẽ", "phải", "này", "đó", "trên", "dưới", "và", "hoặc", "do", "bởi",
    "trở", "làm", "giữ", "đóng", "tổ", "chức", "thực", "hiện", "quy", "định",
}

_DOC_CODE_RE = re.compile(r"\b(\d{1,5}/(?:QĐ|NQ|TB|KH|HD|CV)-[A-ZĐƠƯĂÂÊÔ0-9\-]+)\b")
_NAMED_REG_RE = re.compile(r"\b((?:Quy\s+chế|Quy\s+định)\s+[A-ZĐÀ-Ỹ][^\n.;,()\"”]{4,75})")
_ARTICLE_RE = re.compile(
    r"\b((?:Điều|Khoản)\s+\d+(?:,\s*(?:Chương|chương)\s+[IVXLCDM\d]+)?|Chương\s+[IVXLCDM\d]+)\b"
)
_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4}\d{4}[A-Z0-9]*)\b")
_UNIT_RE = re.compile(r"\b((?:Khoa|Bộ\s+môn|Phòng|Viện|Trung\s+tâm)\s+[A-ZĐÀ-Ỹ][^\n.;,:()\"”]{2,70})")
_PERSON_RE = re.compile(r"\b((?:GS|PGS|TS|ThS)\.\s*(?:[A-ZĐ][a-zà-ỹ]+\s+){1,3}[A-ZĐ][a-zà-ỹ]+)\b")
_CERT_RE = re.compile(
    r"\b(IELTS|TOEFL(?:\s+(?:ITP|iBT))?|TOEIC|VSTEP(?:\.3-5)?|Aptis|JLPT|VNU-TEST)\b",
    re.IGNORECASE,
)
_ORG_RE = re.compile(r"\b(Trường\s+Đại\s+học\s+Công\s+nghệ|Đại\s+học\s+Quốc\s+gia\s+Hà\s+Nội|ĐHQGHN|UET)\b")

_CONCEPT_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bhọc\s+bổng(?:\s+(?:khuyến\s+khích\s+học\s+tập|xuất\s+sắc|giỏi|khá))?\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bthủ\s+khoa(?:\s+ngành\s+học)?\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bkhóa\s+luận\s+tốt\s+nghiệp\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bxét\s+tốt\s+nghiệp\b|\btốt\s+nghiệp\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bchuẩn\s+đầu\s+ra(?:\s+ngoại\s+ngữ)?\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bcảnh\s+báo\s+học\s+(?:tập|vụ)\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bbuộc\s+thôi\s+học\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bđiểm\s+rèn\s+luyện\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bđiểm\s+trung\s+bình\s+chung(?:\s+(?:học\s+kỳ|toàn\s+khóa|tích\s+lũy))?\b|\bGPA\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bhọc\s+cải\s+thiện(?:\s+điểm)?\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\brút\s+bớt\s+học\s+phần\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bđăng\s+ký\s+(?:học\s+phần|tín\s+chỉ)\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\bhọc\s+phí\b", re.IGNORECASE), "CONCEPT"),
    (re.compile(r"\btín\s+chỉ(?:\s+(?:tối\s+thiểu|tối\s+đa))?\b", re.IGNORECASE), "CONCEPT"),
]

_EXTRACTION_PROMPT_TEMPLATE = """Bạn là hệ thống trích xuất Đồ thị Tri thức (Knowledge Graph) chuyên về quy chế đào tạo, học vụ và tuyển sinh Trường Đại học Công nghệ - ĐHQGHN (UET).
Hãy đọc đoạn văn bản dưới đây và trích xuất tối đa 8 bộ ba quan hệ (triples) quan trọng nhất dạng JSON array.

Mỗi phần tử trong mảng JSON phải đúng cấu trúc:
{{"head": "tên thực thể nguồn", "head_type": "REGULATION|ARTICLE|COURSE|FACULTY|DEPARTMENT|CONCEPT|CERTIFICATE|PERSON", "relation": "TEN_QUAN_HE_IN_HOA", "tail": "tên thực thể đích", "tail_type": "REGULATION|ARTICLE|COURSE|FACULTY|DEPARTMENT|CONCEPT|CERTIFICATE|PERSON"}}

Quy tắc:
- Giữ nguyên tên riêng, mã học phần (VD: INT3401), số quyết định (VD: 798/QĐ-ĐHQGHN), mức điểm GPA/chứng chỉ nếu có.
- Quan hệ viết ngắn gọn dạng SNAKE_CASE IN HOA tiếng Việt (VD: YEU_CAU_DIEU_KIEN, SUA_DOI, CAN_CU, QUY_DINH_VE, THUOC_DON_VI, BAO_GOM, AP_DUNG_CHO).
- Chỉ trả về JSON array hợp lệ, không giải thích thêm.

Văn bản:
{text}
"""

_CANONICAL_ENTITY_ALIASES: Dict[str, str] = {
    "uet": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "đhcn": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "đại học công nghệ": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "trường đại học công nghệ": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "trường đh công nghệ": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "trường đại học công nghệ - đhqghn": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "trường đại học công nghệ đhqghn": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "trường đại học công nghệ - đại học quốc gia hà nội": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "trường đại học công nghệ - đhqghn (uet)": "Trường Đại học Công nghệ - ĐHQGHN (UET)",
    "đhqghn": "Đại học Quốc gia Hà Nội (ĐHQGHN)",
    "vnu": "Đại học Quốc gia Hà Nội (ĐHQGHN)",
    "đại học quốc gia hà nội": "Đại học Quốc gia Hà Nội (ĐHQGHN)",
    "đại học quốc gia hà nội (đhqghn)": "Đại học Quốc gia Hà Nội (ĐHQGHN)",
    "fit": "Khoa Công nghệ Thông tin (FIT)",
    "khoa cntt": "Khoa Công nghệ Thông tin (FIT)",
    "khoa công nghệ thông tin": "Khoa Công nghệ Thông tin (FIT)",
    "khoa công nghệ thông tin (fit)": "Khoa Công nghệ Thông tin (FIT)",
    "fet": "Khoa Điện tử Viễn thông (FET)",
    "khoa đtvt": "Khoa Điện tử Viễn thông (FET)",
    "khoa điện tử viễn thông": "Khoa Điện tử Viễn thông (FET)",
    "khoa điện tử - viễn thông": "Khoa Điện tử Viễn thông (FET)",
    "khoa điện tử viễn thông (fet)": "Khoa Điện tử Viễn thông (FET)",
    "fema": "Khoa Cơ học kỹ thuật và Tự động hóa (FEMA)",
    "khoa chkttđh": "Khoa Cơ học kỹ thuật và Tự động hóa (FEMA)",
    "khoa cơ học kỹ thuật và tự động hóa": "Khoa Cơ học kỹ thuật và Tự động hóa (FEMA)",
    "khoa cơ học kỹ thuật & tự động hóa": "Khoa Cơ học kỹ thuật và Tự động hóa (FEMA)",
    "khoa cơ học kỹ thuật và tự động hóa (fema)": "Khoa Cơ học kỹ thuật và Tự động hóa (FEMA)",
    "fepn": "Khoa Vật lý kỹ thuật và Công nghệ Nano (FEPN)",
    "khoa vlkt&cnnn": "Khoa Vật lý kỹ thuật và Công nghệ Nano (FEPN)",
    "khoa vật lý kỹ thuật và công nghệ nano": "Khoa Vật lý kỹ thuật và Công nghệ Nano (FEPN)",
    "khoa vật lý kỹ thuật và công nghệ nano (fepn)": "Khoa Vật lý kỹ thuật và Công nghệ Nano (FEPN)",
    "fat": "Khoa Công nghệ Nông nghiệp (FAT)",
    "khoa cnnn": "Khoa Công nghệ Nông nghiệp (FAT)",
    "khoa công nghệ nông nghiệp": "Khoa Công nghệ Nông nghiệp (FAT)",
    "khoa công nghệ nông nghiệp (fat)": "Khoa Công nghệ Nông nghiệp (FAT)",
    "fce": "Khoa Công nghệ Xây dựng - Giao thông (FCE)",
    "khoa công nghệ xây dựng - giao thông": "Khoa Công nghệ Xây dựng - Giao thông (FCE)",
    "khoa công nghệ xây dựng giao thông": "Khoa Công nghệ Xây dựng - Giao thông (FCE)",
    "khoa công nghệ xây dựng - giao thông (fce)": "Khoa Công nghệ Xây dựng - Giao thông (FCE)",
    "khoa công nghệ xây dựng – giao thông": "Khoa Công nghệ Xây dựng - Giao thông (FCE)",
    "iai": "Viện Trí tuệ nhân tạo (IAI)",
    "viện trí tuệ nhân tạo": "Viện Trí tuệ nhân tạo (IAI)",
    "viện công nghệ trí tuệ nhân tạo": "Viện Trí tuệ nhân tạo (IAI)",
    "viện trí tuệ nhân tạo (iai)": "Viện Trí tuệ nhân tạo (IAI)",
    "sae": "Khoa Hàng không Vũ trụ (SAE)",
    "khoa hàng không vũ trụ": "Khoa Hàng không Vũ trụ (SAE)",
    "viện công nghệ hàng không vũ trụ": "Khoa Hàng không Vũ trụ (SAE)",
    "khoa hàng không vũ trụ (sae)": "Khoa Hàng không Vũ trụ (SAE)",
    "handbook": "Sổ tay sinh viên UET (HANDBOOK)",
    "sổ tay sinh viên uet": "Sổ tay sinh viên UET (HANDBOOK)",
    "gpa": "Điểm trung bình chung (GPA)",
    "điểm tbc": "Điểm trung bình chung (GPA)",
    "điểm trung bình chung": "Điểm trung bình chung (GPA)",
    "điểm trung bình chung tích lũy": "Điểm trung bình chung (GPA)",
    "điểm trung bình chung học kỳ": "Điểm trung bình chung (GPA)",
    "điểm trung bình chung toàn khóa": "Điểm trung bình chung (GPA)",
    "điểm trung bình chung (gpa)": "Điểm trung bình chung (GPA)",
    "đrl": "Điểm rèn luyện",
    "điểm rèn luyện": "Điểm rèn luyện",
    "kết quả rèn luyện": "Điểm rèn luyện",
    "kltn": "Khóa luận tốt nghiệp",
    "khóa luận tốt nghiệp": "Khóa luận tốt nghiệp",
    "cảnh báo học tập": "Cảnh báo học vụ",
    "cảnh báo học vụ": "Cảnh báo học vụ",
    "chuẩn đầu ra ngoại ngữ": "Chuẩn đầu ra ngoại ngữ",
    "chuẩn đầu ra về ngoại ngữ": "Chuẩn đầu ra ngoại ngữ",
    "vstep": "VSTEP",
    "vstep.3-5": "VSTEP",
    "vstep 3-5": "VSTEP",
    "chứng chỉ vstep": "VSTEP",
    "ielts": "IELTS",
    "chứng chỉ ielts": "IELTS",
    "toeic": "TOEIC",
    "chứng chỉ toeic": "TOEIC",
    "toefl": "TOEFL",
    "chứng chỉ toefl": "TOEFL",
    "toefl itp": "TOEFL ITP",
    "toefl ibt": "TOEFL iBT",
    "jlpt": "JLPT",
    "chứng chỉ tiếng nhật jlpt": "JLPT",
    "aptis": "Aptis",
    "chứng chỉ aptis": "Aptis",
    "vnu-test": "VNU-TEST",
}

_DOC_PREFIX_STRIP_RE = re.compile(
    r"^(?:quyết\s+định|nghị\s+quyết|thông\s+báo|kế\s+hoạch|hướng\s+dẫn|công\s+văn|quy\s+định|qđ|vb)(?:\s+số)?\s+(\d{1,5}/(?:QĐ|NQ|TB|KH|HD|CV)-[A-ZĐƠƯĂÂÊÔ0-9\-]+)\b.*$",
    re.IGNORECASE,
)

_REL_ASCII_PREFIX_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("SUA_DOI", "BO_SUNG", "THAY_THE"), "SỬA_ĐỔI_BỔ_SUNG"),
    (("CAN_CU", "DUA_TREN", "DAN_CHIEU"), "CĂN_CỨ"),
    (("NGOAI_LE", "MIEN_"), "NGOẠI_LỆ_CHO"),
    (("CHE_TAI", "XU_LY", "KY_LUAT"), "CHẾ_TÀI_XỬ_LÝ"),
    (("YEU_CAU", "DIEU_KIEN", "CAN_CO", "BAT_BUOC", "TIEN_QUYET"), "YÊU_CẦU_ĐIỀU_KIỆN"),
    (("TRUC_THUOC",), "TRỰC_THUỘC"),
    (("THUOC",), "THUỘC_ĐƠN_VỊ"),
    (("BAO_GOM", "CHUA_NOI_DUNG"), "BAO_GỒM"),
    (("PHU_TRACH", "CHU_TRI", "GIANG_DAY", "QUAN_LY"), "PHỤ_TRÁCH"),
    (("QUY_DINH", "AP_DUNG", "BAN_HANH", "XET_"), "QUY_ĐỊNH_VỀ"),
    (("LIEN_QUAN",), "LIÊN_QUAN_ĐẾN"),
)

_RELATION_WEIGHTS: Dict[str, float] = {
    "SỬA_ĐỔI_BỔ_SUNG": 1.35,
    "CĂN_CỨ": 1.30,
    "NGOẠI_LỆ_CHO": 1.30,
    "CHẾ_TÀI_XỬ_LÝ": 1.30,
    "YÊU_CẦU_ĐIỀU_KIỆN": 1.25,
    "QUY_ĐỊNH_VỀ": 1.15,
    "BAO_GỒM": 1.10,
    "PHỤ_TRÁCH": 1.10,
    "TRỰC_THUỘC": 0.95,
    "THUỘC_ĐƠN_VỊ": 0.85,
    "LIÊN_QUAN_ĐẾN": 0.85,
}

_RELATION_SEED_PHRASES: Dict[str, str] = {
    "SỬA_ĐỔI_BỔ_SUNG": "sửa đổi bổ sung thay thế quy định điều khoản",
    "CĂN_CỨ": "căn cứ quyết định dẫn chiếu văn bản pháp quy",
    "NGOẠI_LỆ_CHO": "ngoại lệ miễn trừ trường hợp đặc biệt ưu tiên",
    "CHẾ_TÀI_XỬ_LÝ": "chế tài xử lý kỷ luật cảnh báo học vụ buộc thôi học",
    "YÊU_CẦU_ĐIỀU_KIỆN": "điều kiện tiên quyết yêu cầu tiêu chuẩn bắt buộc đạt",
    "QUY_ĐỊNH_VỀ": "quy định áp dụng đối với ban hành xét duyệt",
    "BAO_GỒM": "bao gồm cấu thành nội dung danh mục",
    "PHỤ_TRÁCH": "phụ trách quản lý chủ trì giảng dạy",
    "TRỰC_THUỘC": "trực thuộc đơn vị cấp trên",
    "THUỘC_ĐƠN_VỊ": "thuộc khoa viện bộ môn đơn vị đào tạo",
    "LIÊN_QUAN_ĐẾN": "liên quan kết nối tham chiếu",
}

_LIGHTRAG_CATEGORY_MAP: Dict[str, str] = {
    "VAN_BAN": "REGULATION",
    "BANG_TRA_CUU": "REGULATION",
    "DIEU_KHOAN": "ARTICLE",
    "DIEU_KIEN": "CONCEPT",
    "QUY_TRINH": "CONCEPT",
    "KHOA_NGANH": "PROGRAM",
    "NHAN_SU": "PERSON",
    "DON_VI": "DEPARTMENT",
    "ORGANIZATION": "FACULTY",
    "ACADEMIC_METRIC": "CONCEPT",
    "ACADEMIC_STATUS": "CONCEPT",
    "PROCEDURE": "CONCEPT",
    "CERTIFICATE_AND_STANDARD": "CERTIFICATE",
    "SCHOLARSHIP_AND_POLICY": "CONCEPT",
    "PROGRAM_AND_DEGREE": "PROGRAM",
}

_ENTITY_TYPE_WEIGHTS: Dict[str, float] = {
    "REGULATION": 1.25,
    "COURSE": 1.25,
    "ARTICLE": 1.15,
    "CERTIFICATE": 1.15,
    "PERSON": 1.15,
    "PROGRAM": 1.10,
    "CONCEPT": 1.00,
    "DEPARTMENT": 0.95,
    "FACULTY": 0.70,
}

_WRAP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
)
_GRAPH_WORD_RE = re.compile(
    r"\d{1,5}/(?:qđ|nq|tb|kh|hd|cv)-[a-zđơưăâêô0-9\-]+|[a-z0-9à-ỹđ]+",
    re.IGNORECASE,
)
_BREADCRUMB_ARTICLE_COLON_RE = re.compile(
    r"^((?:Điều|Khoản|Chương|Mục)\s+[IVXLCDM\d]+(?:\s*,\s*(?:Điều|Khoản|Chương|Mục)\s+[IVXLCDM\d]+)?)\s*:.*$",
    re.IGNORECASE,
)

_ARTICLE_KEEP_NS_RE = re.compile(
    r"^(?:Điều|Khoản|Chương|Mục| Phụ\s+lục|Bảng)\b",
    re.IGNORECASE,
)
_GENERIC_NS_TAGS: Set[str] = set(_LIGHTRAG_CATEGORY_MAP.keys()) | {"OTHER", "GENERAL", "THỰC THỂ"}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return no_marks.replace("đ", "d").replace("Đ", "D")


def _graph_tokenize(text: str) -> Set[str]:
    """Lightweight sub-millisecond tokenizer (unigrams + bigrams) for Graph entity matching."""
    norm = unicodedata.normalize("NFC", str(text or "")).casefold()
    words = _GRAPH_WORD_RE.findall(norm)
    if not words:
        return set()
    tokens: Set[str] = set()
    for i, w in enumerate(words):
        is_stop = w in _VN_STOP_BOUNDARY and not any(c.isdigit() for c in w)
        if not is_stop and (len(w) >= 2 or w.isdigit()):
            tokens.add(w)
        if i + 1 < len(words):
            nxt = words[i + 1]
            nxt_stop = nxt in _VN_STOP_BOUNDARY and not any(c.isdigit() for c in nxt)
            if not (is_stop and nxt_stop):
                tokens.add(f"{w}_{nxt}")
    return tokens


def clean_entity_name(name: str) -> str:
    """Chuẩn hóa tên hiển thị của thực thể mà vẫn bảo toàn dấu ngoặc cân bằng như (UET)."""
    text = unicodedata.normalize("NFC", str(name or ""))
    changed = True
    while changed and text:
        changed = False
        stripped = text.strip(" \t\r\n.,;:-–—")
        if stripped != text:
            text = stripped
            changed = True
        for left, right in _WRAP_PAIRS:
            if len(text) >= 2 and text.startswith(left) and text.endswith(right):
                if left != right or text.count(left) >= 2:
                    text = text[len(left):-len(right)]
                    changed = True
            elif left == right:
                if text.count(left) % 2 == 1:
                    if text.startswith(left):
                        text = text[len(left):]
                        changed = True
                    elif text.endswith(right):
                        text = text[:-len(right)]
                        changed = True
            else:
                if text.startswith(left) and text.count(left) > text.count(right):
                    text = text[len(left):]
                    changed = True
                elif text.endswith(right) and text.count(right) > text.count(left):
                    text = text[:-len(right)]
                    changed = True
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 120:
        text = text[:120].rsplit(" ", 1)[0].strip() or text[:120]
    return text


def canonicalize_entity_name(name: str, etype: str = "") -> str:
    """Gộp các biến thể viết tắt/từ đồng nghĩa UET về tên thực thể chuẩn."""
    cleaned = clean_entity_name(name)
    if not cleaned:
        return ""
    m_doc = _DOC_PREFIX_STRIP_RE.match(cleaned)
    if m_doc:
        return m_doc.group(1).upper()
    low = cleaned.casefold()
    alias = _CANONICAL_ENTITY_ALIASES.get(low)
    if alias:
        return alias
    try:
        thesaurus_canon = get_thesaurus().get_canonical(cleaned)
        if thesaurus_canon:
            mapped = _CANONICAL_ENTITY_ALIASES.get(thesaurus_canon.casefold())
            return mapped if mapped else thesaurus_canon
    except Exception:
        pass
    return cleaned


def normalize_entity_id(name: str) -> str:
    """Tạo định danh bất biến cho thực thể từ tên đã chuẩn hóa đồng nghĩa."""
    cleaned = canonicalize_entity_name(name).casefold()
    cleaned = re.sub(r"[^\w]+", "_", cleaned, flags=re.UNICODE)
    return cleaned.strip("_")


def _trim_vn_phrase(phrase: str, max_words: int = 8) -> str:
    """Cắt cụm danh từ tiếng Việt trước động từ/giới từ thường gặp."""
    words = clean_entity_name(phrase).split()
    kept: List[str] = []
    for idx, w in enumerate(words):
        low = w.strip(".,;:").casefold()
        if idx >= 2 and low in _VN_STOP_BOUNDARY:
            break
        kept.append(w)
        if len(kept) >= max_words:
            break
    return clean_entity_name(" ".join(kept))


def _normalize_relation(rel: str) -> str:
    text = unicodedata.normalize("NFC", str(rel or "LIÊN_QUAN_ĐẾN")).strip().upper()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    if not text:
        return "LIÊN_QUAN_ĐẾN"
    ascii_rel = _strip_accents(text).upper()
    for prefixes, target in _REL_ASCII_PREFIX_RULES:
        if any(ascii_rel.startswith(p) for p in prefixes):
            return target
    return text


def _canonicalize_triple(tr: Dict[str, str]) -> Optional[Dict[str, str]]:
    h_type = str(tr.get("head_type") or "CONCEPT").upper()
    t_type = str(tr.get("tail_type") or "CONCEPT").upper()
    head = canonicalize_entity_name(str(tr.get("head") or ""), h_type)
    tail = canonicalize_entity_name(str(tr.get("tail") or ""), t_type)
    rel = _normalize_relation(str(tr.get("relation") or "LIÊN_QUAN_ĐẾN"))
    if not head or not tail or normalize_entity_id(head) == normalize_entity_id(tail):
        return None
    return {
        "head": head,
        "head_type": h_type,
        "relation": rel,
        "tail": tail,
        "tail_type": t_type,
    }


def _classify_sentence_relation(sentence_lower: str) -> str:
    if any(k in sentence_lower for k in ("sửa đổi", "bổ sung", "thay thế")):
        return "SỬA_ĐỔI_BỔ_SUNG"
    if "căn cứ" in sentence_lower:
        return "CĂN_CỨ"
    if any(k in sentence_lower for k in ("điều kiện", "yêu cầu", "tối thiểu", "từ ", "trở lên", "đạt")):
        return "YÊU_CẦU_ĐIỀU_KIỆN"
    if any(k in sentence_lower for k in ("trực thuộc", "thuộc")):
        return "TRỰC_THUỘC"
    if any(k in sentence_lower for k in ("chủ trì", "giảng dạy", "phụ trách")):
        return "PHỤ_TRÁCH"
    if any(k in sentence_lower for k in ("quy định", "ban hành", "áp dụng", "xét")):
        return "QUY_ĐỊNH_VỀ"
    return "LIÊN_QUAN_ĐẾN"


def canonicalize_lightrag_entity_name(raw_name: str, etype: str = "") -> str:
    """Chuẩn hóa tên node/cạnh từ LightRAG graph.json sang không gian tên hợp nhất của GraphIndex."""
    text = str(raw_name or "").strip()
    if not text:
        return ""
    m_ns = re.match(r"^\[([^\]]+)\]\s*(.+)$", text)
    if m_ns:
        ns_tag = m_ns.group(1).strip()
        remainder = m_ns.group(2).strip()
        if ns_tag.upper() in _GENERIC_NS_TAGS:
            text = remainder
        elif _ARTICLE_KEEP_NS_RE.match(remainder):
            return clean_entity_name(f"[{ns_tag}] {remainder}")
        else:
            text = remainder
    return canonicalize_entity_name(text, etype)


def _strip_chunk_header_for_match(text: str) -> str:
    lines = (text or "").splitlines()
    idx = 0
    while idx < len(lines):
        s = lines[idx].strip()
        if s.startswith("[") and s.endswith("]"):
            idx += 1
            continue
        break
    body = " ".join(lines[idx:]).strip()
    return re.sub(r"\s+", " ", (body or text or "")).casefold()

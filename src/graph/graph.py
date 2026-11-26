"""LightRAG Support Bridge & Regex Supplement Index (src/graph/graph.py).

Serves as a subordinate helper module for LightRAGRetriever (src/graph/lightrag_retriever.py):
1. Chunk Bridge: Maps LightRAG 2-Pass SemanticUnits, entities, relations, and Pass 1 summaries
   back to authentic DataChunks from chunked_data (Primary 1.0x / Secondary 0.25x).
2. Scoring & Structural Helper: Provides Entity IDF, Hub Gating, Co-Activation, and Adjacent Smoothing.
3. Regex & Vector Supplement: Fills coverage gaps for hard UET codes (course codes, decision numbers)
   and newly ingested chunks not yet processed by LightRAGIndexer.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import pickle
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import BaseEmbeddingModel, DataChunk, RetrievedContext
from src.indexing.chunks import normalize_chunks
from src.graph.thesaurus_utils import (
    KNOWN_FACULTY_KEYS,
    detect_query_faculties,
    get_thesaurus,
    is_forbidden_entity,
)
from src.utils.helpers import get_logger

logger = get_logger("graph.bridge")


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

_DOC_CODE_RE = re.compile(
    r"\b(\d{1,5}/(?:QĐ|NQ|TB|KH|HD|CV)-[A-ZĐƠƯĂÂÊÔ0-9\-]+)\b"
)
_NAMED_REG_RE = re.compile(
    r"\b((?:Quy\s+chế|Quy\s+định)\s+[A-ZĐÀ-Ỹ][^\n.;,()\"”]{4,75})"
)
_ARTICLE_RE = re.compile(
    r"\b((?:Điều|Khoản)\s+\d+(?:,\s*(?:Chương|chương)\s+[IVXLCDM\d]+)?|Chương\s+[IVXLCDM\d]+)\b"
)
_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4}\d{4}[A-Z0-9]*)\b")
_UNIT_RE = re.compile(
    r"\b((?:Khoa|Bộ\s+môn|Phòng|Viện|Trung\s+tâm)\s+[A-ZĐÀ-Ỹ][^\n.;,:()\"”]{2,70})"
)
_PERSON_RE = re.compile(
    r"\b((?:GS|PGS|TS|ThS)\.\s*(?:[A-ZĐ][a-zà-ỹ]+\s+){1,3}[A-ZĐ][a-zà-ỹ]+)\b"
)
_CERT_RE = re.compile(
    r"\b(IELTS|TOEFL(?:\s+(?:ITP|iBT))?|TOEIC|VSTEP(?:\.3-5)?|Aptis|JLPT|VNU-TEST)\b",
    re.IGNORECASE,
)
_ORG_RE = re.compile(
    r"\b(Trường\s+Đại\s+học\s+Công\s+nghệ|Đại\s+học\s+Quốc\s+gia\s+Hà\s+Nội|ĐHQGHN|UET)\b"
)
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
    "khoa công nghệ xây dựng – giao thông": "Khoa Công nghệ Xây dựng - Giao thông (FCE)",
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


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return no_marks.replace("đ", "d").replace("Đ", "D")


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
    """Gộp các biến thể viết tắt/từ đồng nghĩa UET (bao gồm 2.348 aliases của AcademicThesaurus) về tên thực thể chuẩn."""
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


_VALID_EXTRACTION_MODES = ("hybrid", "regex_only", "llm_only")


class HybridGraphExtractor:
    """
    Bộ trích xuất Đồ thị Tri thức lai (Rule-based Academic Regex + OpenRouter Tencent Hy4 LLM).
    Hỗ trợ 3 chế độ độc lập: extraction_mode = "hybrid" | "regex_only" | "llm_only".
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        enable_llm_extraction: bool = False,
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = "tencent/hy4-flash",
        llm_domains: Optional[List[str]] = None,
        max_llm_chunks: Optional[int] = None,
        cache_path: Optional[Path] = None,
        extraction_mode: str = "hybrid",
    ):
        mode = str(extraction_mode or "hybrid").strip().lower()
        if mode not in _VALID_EXTRACTION_MODES:
            raise ValueError(f"Unsupported extraction_mode '{extraction_mode}'. Choose from {_VALID_EXTRACTION_MODES}.")
        self.extraction_mode = mode
        self.llm_client = llm_client
        self.enable_llm_extraction = enable_llm_extraction
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_domains = [d.lower() for d in llm_domains] if llm_domains else []
        self.max_llm_chunks = int(max_llm_chunks) if max_llm_chunks is not None else None
        self.llm_calls_count: int = 0
        self.cache_path = Path(cache_path) if cache_path else (BASE_DIR / "database" / "graph_llm_cache.json")
        self._llm_cache: Dict[str, Any] = self._load_cache()
        self._cache_dirty = False

    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"Không thể đọc graph LLM cache '{self.cache_path}': {exc}")
            return {}

    def flush_cache(self) -> None:
        if not self._cache_dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as fp:
                json.dump(self._llm_cache, fp, ensure_ascii=False, indent=2)
            self._cache_dirty = False
        except Exception as exc:
            logger.warning(f"Không thể lưu graph LLM cache '{self.cache_path}': {exc}")

    @staticmethod
    def _extract_entities_from_text(text: str) -> List[Dict[str, str]]:
        """Trích xuất thực thể học vụ UET bằng regex từ một đoạn hoặc câu."""
        found: Dict[str, Dict[str, str]] = {}

        def _add(name: str, etype: str) -> None:
            cleaned = canonicalize_entity_name(name, etype)
            if len(cleaned) < 2 or is_forbidden_entity(cleaned, allow_school_names=True):
                return
            eid = normalize_entity_id(cleaned)
            if not eid:
                return
            if eid not in found:
                found[eid] = {"id": eid, "name": cleaned, "type": etype}

        for m in _DOC_CODE_RE.finditer(text):
            _add(m.group(1), "REGULATION")
        for m in _NAMED_REG_RE.finditer(text):
            _add(_trim_vn_phrase(m.group(1), max_words=10), "REGULATION")
        for m in _ARTICLE_RE.finditer(text):
            _add(m.group(1), "ARTICLE")
        for m in _COURSE_CODE_RE.finditer(text):
            _add(m.group(1).upper(), "COURSE")
        for m in _UNIT_RE.finditer(text):
            unit = _trim_vn_phrase(m.group(1), max_words=8)
            if len(unit.split()) >= 2:
                etype = "DEPARTMENT" if unit.casefold().startswith("bộ môn") else "FACULTY"
                _add(unit, etype)
        for m in _PERSON_RE.finditer(text):
            _add(m.group(1), "PERSON")
        for m in _CERT_RE.finditer(text):
            _add(m.group(1).upper() if "vnu" in m.group(1).lower() or m.group(1).isupper() else m.group(1), "CERTIFICATE")
        for m in _ORG_RE.finditer(text):
            _add(m.group(1), "FACULTY")
        for pattern, etype in _CONCEPT_PATTERNS:
            for m in pattern.finditer(text):
                _add(m.group(0), etype)

        return list(found.values())

    @staticmethod
    def _extract_breadcrumb_segments(chunk: DataChunk) -> List[str]:
        meta = chunk.metadata or {}
        raw_bc = meta.get("breadcrumb")
        if not raw_bc and chunk.text.lstrip().startswith("["):
            first_line = chunk.text.lstrip().splitlines()[0]
            if first_line.startswith("[") and "]" in first_line and ">" in first_line:
                raw_bc = first_line[1:first_line.index("]")]
        if not raw_bc:
            return []
        segments: List[str] = []
        for seg in str(raw_bc).split(">"):
            cleaned = clean_entity_name(seg)
            if not cleaned or re.fullmatch(r"Trang\s+\d+", cleaned, flags=re.IGNORECASE):
                continue
            m_art = _BREADCRUMB_ARTICLE_COLON_RE.match(cleaned)
            if m_art:
                cleaned = clean_entity_name(m_art.group(1))
            if cleaned:
                segments.append(cleaned)
        return segments

    def extract_rules(self, chunk: DataChunk) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Trích xuất thực thể và bộ ba cấu trúc từ metadata + nội dung chunk."""
        meta = chunk.metadata or {}
        text = unicodedata.normalize("NFC", chunk.text or "")
        entities_map: Dict[str, Dict[str, str]] = {}
        triples: List[Dict[str, str]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        def _register_entity(name: str, etype: str) -> Optional[Dict[str, str]]:
            cleaned = canonicalize_entity_name(name, etype)
            if len(cleaned) < 2 or is_forbidden_entity(cleaned, allow_school_names=True):
                return None
            eid = normalize_entity_id(cleaned)
            if not eid:
                return None
            if eid not in entities_map:
                entities_map[eid] = {"id": eid, "name": cleaned, "type": etype}
            return entities_map[eid]

        def _add_triple(h_name: str, h_type: str, rel: str, t_name: str, t_type: str) -> None:
            h_ent = _register_entity(h_name, h_type)
            t_ent = _register_entity(t_name, t_type)
            if not h_ent or not t_ent or h_ent["id"] == t_ent["id"]:
                return
            norm_rel = _normalize_relation(rel)
            sig = (h_ent["id"], norm_rel, t_ent["id"])
            if sig in seen_triples:
                return
            seen_triples.add(sig)
            triples.append({
                "head": h_ent["name"],
                "head_type": h_ent["type"],
                "relation": norm_rel,
                "tail": t_ent["name"],
                "tail_type": t_ent["type"],
            })

        # 1. Anchor tài liệu & Breadcrumb
        doc_title = clean_entity_name(str(meta.get("title") or ""))
        bc_segments = self._extract_breadcrumb_segments(chunk)
        if not doc_title and bc_segments:
            doc_title = bc_segments[0]

        doc_code_match = _DOC_CODE_RE.search(text)
        doc_anchor_name = doc_code_match.group(1) if (doc_code_match and chunk.chunk_index == 0) else (doc_title or "")
        doc_anchor_type = "REGULATION"

        if doc_anchor_name:
            _register_entity(doc_anchor_name, doc_anchor_type)

        for idx in range(len(bc_segments) - 1):
            _add_triple(bc_segments[idx], "REGULATION", "BAO_GỒM", bc_segments[idx + 1], "ARTICLE")

        faculty_key = str(meta.get("faculty") or meta.get("site") or "").strip().lower()
        if faculty_key and faculty_key in FACULTY_LABELS:
            fac_label = FACULTY_LABELS[faculty_key]
            _register_entity(fac_label, "FACULTY")
            if doc_anchor_name:
                _add_triple(doc_anchor_name, doc_anchor_type, "THUỘC_ĐƠN_VỊ", fac_label, "FACULTY")

        # 2. Thực thể trong toàn chunk
        chunk_entities = self._extract_entities_from_text(text)
        for ent in chunk_entities:
            _register_entity(ent["name"], ent["type"])

        anchor = bc_segments[-1] if bc_segments else doc_anchor_name
        anchor_type = "ARTICLE" if len(bc_segments) > 1 else doc_anchor_type

        # 3. Quan hệ theo từng câu
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]
        for sent in sentences:
            sent_entities = self._extract_entities_from_text(sent)
            if not sent_entities:
                continue
            sent_lower = sent.casefold()
            rel = _classify_sentence_relation(sent_lower)

            if "căn cứ" in sent_lower and doc_anchor_name:
                for ent in sent_entities:
                    if ent["type"] == "REGULATION":
                        _add_triple(doc_anchor_name, doc_anchor_type, "CĂN_CỨ", ent["name"], "REGULATION")
            elif any(k in sent_lower for k in ("sửa đổi", "bổ sung")) and doc_anchor_name:
                for ent in sent_entities:
                    if ent["type"] in ("REGULATION", "ARTICLE"):
                        _add_triple(doc_anchor_name, doc_anchor_type, "SỬA_ĐỔI_BỔ_SUNG", ent["name"], ent["type"])

            limited = sent_entities[:5]
            for i in range(len(limited)):
                for j in range(i + 1, len(limited)):
                    _add_triple(
                        limited[i]["name"],
                        limited[i]["type"],
                        rel,
                        limited[j]["name"],
                        limited[j]["type"],
                    )

        # 4. Neo thực thể trong chunk vào anchor tài liệu/mục
        if anchor:
            for ent in chunk_entities[:10]:
                if normalize_entity_id(ent["name"]) != normalize_entity_id(anchor):
                    _add_triple(anchor, anchor_type, "QUY_ĐỊNH_VỀ", ent["name"], ent["type"])

        return list(entities_map.values()), triples

    def _is_llm_eligible(self, chunk: DataChunk) -> bool:
        if not self.enable_llm_extraction or self.llm_client is None:
            return False
        if self.max_llm_chunks is not None and self.llm_calls_count >= self.max_llm_chunks:
            return False
        if not self.llm_domains:
            return True
        meta = chunk.metadata or {}
        haystack = " ".join(
            str(meta.get(k) or "") for k in ("faculty", "site", "domain", "file_name")
        ).casefold()
        return any(dom in haystack for dom in self.llm_domains)

    @staticmethod
    def _parse_llm_triples(raw_response: str) -> List[Dict[str, str]]:
        if not raw_response:
            return []
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_response).strip()
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []

        valid: List[Dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            canon = _canonicalize_triple(item)
            if canon is not None:
                valid.append(canon)
        return valid

    def extract_llm(self, chunk: DataChunk) -> List[Dict[str, str]]:
        """Trích xuất triples bổ sung qua OpenRouter (Tencent Hy4) có Disk Cache theo chunk."""
        content_hash = hashlib.sha256((chunk.text or "").encode("utf-8", errors="replace")).hexdigest()
        cached_entry = self._llm_cache.get(chunk.chunk_id)
        if isinstance(cached_entry, dict) and cached_entry.get("hash") == content_hash:
            res: List[Dict[str, str]] = []
            for raw_tr in (cached_entry.get("triples") or []):
                canon = _canonicalize_triple(raw_tr)
                if canon is not None:
                    res.append(canon)
            return res

        if not self._is_llm_eligible(chunk):
            return []

        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(text=(chunk.text or "")[:2500])
        self.llm_calls_count += 1
        try:
            raw_resp = self.llm_client.complete_text(
                prompt,
                preferred_provider=self.llm_provider,
                model=self.llm_model,
            )
            triples = self._parse_llm_triples(raw_resp)
            self._llm_cache[chunk.chunk_id] = {
                "hash": content_hash,
                "triples": triples,
            }
            self._cache_dirty = True
            return triples
        except Exception as exc:
            logger.warning(f"Lỗi gọi LLM trích xuất Graph cho chunk {chunk.chunk_id}: {exc}")
            return []

    @staticmethod
    def _merge_triples(
        base_entities: List[Dict[str, str]],
        base_triples: List[Dict[str, str]],
        extra_triples: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        entities_map = {e["id"]: e for e in base_entities}
        triples = list(base_triples)
        seen = {
            (normalize_entity_id(t["head"]), t["relation"], normalize_entity_id(t["tail"]))
            for t in triples
        }
        for lt in extra_triples:
            hid, tid = normalize_entity_id(lt["head"]), normalize_entity_id(lt["tail"])
            if not hid or not tid or hid == tid:
                continue
            entities_map.setdefault(hid, {"id": hid, "name": lt["head"], "type": lt["head_type"]})
            entities_map.setdefault(tid, {"id": tid, "name": lt["tail"], "type": lt["tail_type"]})
            sig = (hid, lt["relation"], tid)
            if sig not in seen:
                seen.add(sig)
                triples.append(lt)
        return list(entities_map.values()), triples

    def extract(self, chunk: DataChunk) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Trích xuất thực thể & triples theo extraction_mode (hybrid / regex_only / llm_only)."""
        if self.extraction_mode == "regex_only":
            return self.extract_rules(chunk)
        llm_triples = self.extract_llm(chunk)
        if self.extraction_mode == "llm_only":
            return self._merge_triples([], [], llm_triples)
        entities, triples = self.extract_rules(chunk)
        return self._merge_triples(entities, triples, llm_triples) if llm_triples else (entities, triples)


_ARTICLE_KEEP_NS_RE = re.compile(
    r"^(?:Điều|Khoản|Chương|Mục| Phụ\s+lục|Bảng)\b",
    re.IGNORECASE,
)
_GENERIC_NS_TAGS: Set[str] = set(_LIGHTRAG_CATEGORY_MAP.keys()) | {"OTHER", "GENERAL", "THỰC THỂ"}


def canonicalize_lightrag_entity_name(raw_name: str, etype: str = "") -> str:
    """
    Chuẩn hóa tên node/cạnh từ LightRAG graph.json sang không gian tên hợp nhất của GraphIndex:
      - Gỡ bỏ tiền tố danh mục như '[ACADEMIC_STATUS] Buộc thôi học' -> 'Buộc thôi học'.
      - Giữ lại tiền tố văn bản '[QĐ-5115] Điều 15...' cho các Điều/Khoản/Chương/Bảng để không trùng chéo giữa các QĐ.
      - Gỡ bỏ tiền tố tiêu đề bài viết lặp lại cho tên người/đơn vị/khái niệm độc lập.
    """
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


class _CompatGraphUnpickler(pickle.Unpickler):
    """Tương thích ngược cho các file graph_index.pkl cũ lưu tham chiếu 'src.indexing.graph'."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "src.indexing.graph":
            module = "src.graph.graph"
        return super().find_class(module, name)


class GraphIndex:
    """
    Cầu nối Chunk & Bộ bổ khuyết Regex/Vector phụ trợ cho LightRAGRetriever (lưu tại database/graph_index.pkl).
    Giữ tên lớp GraphIndex để tương thích 100% với pickle artifact hiện có và các bài kiểm thử.
    - Offline: Ánh xạ tri thức 2-Pass của LightRAG (graph.json + semantic_units_cache + entity_chroma)
      về DataChunk gốc (chunked_data) kết hợp tầng Regex học vụ bổ khuyết mã cứng -> ma trận NumPy RAM.
    - Online (Phụ trợ cho LightRAGRetriever):
        + ground_lightrag_subgraph(): Neo subgraph 2 tầng của LightRAG về DataChunk thật (Primary 1.0x / Secondary 0.25x),
          đồng thời bổ khuyết Regex/Vector ở trọng số phụ (0.65x) khi câu hỏi chứa mã cứng hoặc chunk mới.
        + Cung cấp Entity IDF, Hub Gating và Adjacent Smoothing cho LightRAGRetriever.
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
        self.chunk_id_to_idx: Dict[str, int] = {
            c.chunk_id: idx for idx, c in enumerate(self.chunks)
        }
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
        triple_key_to_idx: Dict[Tuple[str, str, str], int] = {}

        for chunk in self.chunks:
            cid = chunk.chunk_id
            entities, triples = extractor.extract(chunk)
            chunk_ent_ids: List[str] = []

            for ent in entities:
                eid = ent["id"]
                if not self.graph.has_node(eid):
                    self.graph.add_node(
                        eid,
                        name=ent["name"],
                        type=ent["type"],
                        chunks=set(),
                        secondary_chunks=set(),
                        descriptions=[],
                    )
                self.graph.nodes[eid]["chunks"].add(cid)
                if eid not in chunk_ent_ids:
                    chunk_ent_ids.append(eid)

            self.chunk_to_entities[cid] = chunk_ent_ids

            for tr in triples:
                hid = normalize_entity_id(tr["head"])
                tid = normalize_entity_id(tr["tail"])
                rel = tr["relation"]
                if not hid or not tid or hid == tid:
                    continue

                if not self.graph.has_node(hid):
                    self.graph.add_node(
                        hid, name=tr["head"], type=tr.get("head_type", "CONCEPT"),
                        chunks=set(), secondary_chunks=set(), descriptions=[],
                    )
                if not self.graph.has_node(tid):
                    self.graph.add_node(
                        tid, name=tr["tail"], type=tr.get("tail_type", "CONCEPT"),
                        chunks=set(), secondary_chunks=set(), descriptions=[],
                    )

                self.graph.nodes[hid]["chunks"].add(cid)
                self.graph.nodes[tid]["chunks"].add(cid)

                if self.graph.has_edge(hid, tid):
                    edge = self.graph[hid][tid]
                    edge["relations"].add(rel)
                    edge["chunks"].add(cid)
                    edge["weight"] += 1.0
                else:
                    self.graph.add_edge(
                        hid,
                        tid,
                        relations={rel},
                        chunks={cid},
                        weight=1.0,
                    )

                t_sig = (hid, rel, tid)
                if t_sig not in triple_key_to_idx:
                    t_idx = len(self.triple_records)
                    triple_key_to_idx[t_sig] = t_idx
                    self.triple_records.append({
                        "head_id": hid,
                        "head_name": self.graph.nodes[hid]["name"],
                        "relation": rel,
                        "tail_id": tid,
                        "tail_name": self.graph.nodes[tid]["name"],
                        "text": f"{self.graph.nodes[hid]['name']} -[{rel}]-> {self.graph.nodes[tid]['name']}",
                        "evidence": "",
                        "chunk_ids": {cid},
                        "secondary_chunk_ids": set(),
                    })
                else:
                    t_idx = triple_key_to_idx[t_sig]
                    self.triple_records[t_idx]["chunk_ids"].add(cid)

                if t_idx not in self.chunk_to_triples[cid]:
                    self.chunk_to_triples[cid].append(t_idx)

        if lightrag_graph_path is not None and Path(lightrag_graph_path).exists():
            self._ingest_lightrag_knowledge_graph(
                graph_path=Path(lightrag_graph_path),
                units_path=Path(semantic_units_path) if semantic_units_path else None,
                triple_key_to_idx=triple_key_to_idx,
            )

        self.entity_ids = list(self.graph.nodes())
        self.entity_texts = []
        for eid in self.entity_ids:
            nd = self.graph.nodes[eid]
            descs = nd.get("descriptions") or []
            desc_snippet = f" - {descs[0][:180]}" if descs else ""
            self.entity_texts.append(f"{nd['name']} ({nd['type']}){desc_snippet}")

        total_docs = max(1, len(self.chunks))
        idf_denom = max(math.log(total_docs + 1.0), 1e-6)
        self.entity_idf = {}
        for eid in self.entity_ids:
            node = self.graph.nodes[eid]
            df = max(1, len(node.get("chunks", ())))
            raw_idf = max(0.0, math.log((total_docs + 1.0) / (df + 0.5)) / idf_denom)
            type_w = _ENTITY_TYPE_WEIGHTS.get(str(node.get("type") or "CONCEPT"), 1.0)
            self.entity_idf[eid] = round(min(1.35, max(0.15, raw_idf * type_w)), 4)

        self.entity_idf_array = (
            np.asarray([self.entity_idf[eid] for eid in self.entity_ids], dtype=np.float32)
            if self.entity_ids
            else None
        )

        self._build_doc_neighbors()

        for eid in self.entity_ids:
            name = self.graph.nodes[eid]["name"]
            stripped_name = re.sub(r"^\[[^\]]+\]\s*", "", name).strip()
            toks = _graph_tokenize(name) | _graph_tokenize(stripped_name)
            self.entity_tokens[eid] = toks
            for tok in toks:
                self.token_to_entities[tok].append(eid)

        self.token_to_entities = dict(self.token_to_entities)
        self.chunk_to_entities = dict(self.chunk_to_entities)
        self.chunk_to_triples = dict(self.chunk_to_triples)

        logger.info(
            f"Đã dựng Unified Knowledge Graph: {len(self.chunks)} chunks -> "
            f"{len(self.entity_ids)} entities, {len(self.triple_records)} triples "
            f"(LightRAG merged: +{self.lightrag_nodes_merged} nodes, +{self.lightrag_edges_merged} edges, "
            f"{len(self.document_summaries)} summaries)."
        )

    def _ingest_lightrag_knowledge_graph(
        self,
        graph_path: Path,
        units_path: Optional[Path],
        triple_key_to_idx: Dict[Tuple[str, str, str], int],
    ) -> None:
        """
        Hợp nhất tri thức 2-Pass từ LightRAG (graph.json + semantic_units_cache.jsonl) vào GraphIndex:
          - Ánh xạ 2 mức từ SemanticUnit -> DataChunk:
              + Primary chunks (100% trọng số): chứa câu bằng chứng evidence / mô tả / tên thực thể.
              + Secondary chunks (25% trọng số): các chunk khác trong cùng SemanticUnit (chống pha loãng điểm).
          - Nạp toàn bộ document_summaries phục vụ High-Level Summary Boost.
        """
        try:
            with graph_path.open("r", encoding="utf-8") as fp:
                lr_data = json.load(fp)
        except Exception as exc:
            logger.warning(f"Không thể đọc LightRAG graph '{graph_path}': {exc}")
            return

        raw_entities = lr_data.get("entities") or {}
        if isinstance(raw_entities, list):
            raw_entities = {n["canonical_name"]: n for n in raw_entities if "canonical_name" in n}
        elif not raw_entities and "nodes" in lr_data:
            raw_entities = {n["canonical_name"]: n for n in lr_data.get("nodes", []) if "canonical_name" in n}

        raw_relations = lr_data.get("relations") or lr_data.get("edges") or []
        raw_summaries = lr_data.get("document_summaries") or {}
        self.document_summaries = dict(raw_summaries) if isinstance(raw_summaries, dict) else {}

        # Lập bảng tra cứu nhanh chunk theo document_id và title
        doc_chunks_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        title_chunks_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        title_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        orig_titles_to_chunks: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for chunk in self.chunks:
            meta = chunk.metadata or {}
            doc_id = str(chunk.document_id or meta.get("document_id") or meta.get("doc_id") or "").strip()
            raw_title = str(meta.get("title") or "").strip()
            title_low = raw_title.casefold()
            body_low = _strip_chunk_header_for_match(chunk.text)
            if doc_id:
                doc_chunks_map[doc_id].append((chunk.chunk_id, body_low))
            if raw_title:
                orig_titles_to_chunks[raw_title].append((chunk.chunk_id, body_low))
            if title_low:
                title_chunks_map[title_low].append((chunk.chunk_id, body_low))
                if doc_id:
                    title_to_doc_ids[title_low].add(doc_id)

        # Liên kết document_summaries sang document_id (hỗ trợ cả key theo doc_id mới lẫn key theo doc_title cũ)
        for sum_key, sum_info in self.document_summaries.items():
            if not isinstance(sum_info, dict):
                continue
            explicit_did = str(sum_info.get("doc_id") or "").strip()
            if explicit_did and explicit_did in doc_chunks_map:
                self.doc_id_to_summary[explicit_did] = sum_info
            if str(sum_key).strip() in doc_chunks_map:
                self.doc_id_to_summary[str(sum_key).strip()] = sum_info
            t_low = str(sum_info.get("title") or sum_key).strip().casefold()
            for did in title_to_doc_ids.get(t_low, ()):
                self.doc_id_to_summary.setdefault(did, sum_info)
            sum_key_low = str(sum_key).strip().casefold()
            for did in title_to_doc_ids.get(sum_key_low, ()):
                self.doc_id_to_summary.setdefault(did, sum_info)

        # Ánh xạ unit_id -> danh sách (chunk_id, body_low) thuộc SemanticUnit đó
        unit_to_chunks: Dict[str, List[Tuple[str, str]]] = {}
        pass1_hash_to_chunks: Dict[str, List[Tuple[str, str]]] = {}
        # Fallback trực tiếp từ chunks cho cả hash theo doc_id (mới) và doc_title (tương thích ngược)
        for did, c_list in doc_chunks_map.items():
            p1_did = f"pass1_{hashlib.md5(did.encode('utf-8')).hexdigest()[:8]}"
            pass1_hash_to_chunks[p1_did] = c_list[:2]
        for raw_title, c_list in orig_titles_to_chunks.items():
            p1_title = f"pass1_{hashlib.md5(raw_title.encode('utf-8')).hexdigest()[:8]}"
            pass1_hash_to_chunks.setdefault(p1_title, c_list[:2])

        if units_path is None:
            default_units = graph_path.parent / "semantic_units_cache.jsonl"
            if default_units.exists():
                units_path = default_units

        seen_doc_titles: Dict[str, str] = {}
        if units_path is not None and units_path.exists():
            try:
                with units_path.open("r", encoding="utf-8") as ufp:
                    for line in ufp:
                        line = line.strip()
                        if not line:
                            continue
                        u = json.loads(line)
                        uid = str(u.get("unit_id") or "")
                        did = str(u.get("document_id") or "")
                        d_title = str(u.get("document_title") or "")
                        if d_title and did:
                            seen_doc_titles[d_title] = did
                        cand_chunks = doc_chunks_map.get(did) or title_chunks_map.get(d_title.strip().casefold()) or []
                        if not uid or not cand_chunks:
                            continue
                        if len(cand_chunks) <= 2:
                            unit_to_chunks[uid] = cand_chunks
                            continue
                        u_norm = re.sub(r"\s+", " ", str(u.get("content") or "")).casefold()
                        matched_c: List[Tuple[str, str]] = []
                        for cid, c_body in cand_chunks:
                            if not c_body:
                                continue
                            # Lấy 2 lát cắt đại diện ở đầu và giữa chunk để đối chiếu nhanh O(1) với unit.content
                            probe_1 = c_body[: min(65, len(c_body))]
                            mid = len(c_body) // 2
                            probe_2 = c_body[mid : min(mid + 65, len(c_body))]
                            if (len(probe_1) >= 20 and probe_1 in u_norm) or (len(probe_2) >= 20 and probe_2 in u_norm):
                                matched_c.append((cid, c_body))
                        unit_to_chunks[uid] = matched_c if matched_c else cand_chunks[:3]
            except Exception as exc:
                logger.warning(f"Lỗi khi đọc semantic_units_cache '{units_path}': {exc}")

        for orig_title, did in seen_doc_titles.items():
            p1_id = f"pass1_{hashlib.md5(orig_title.encode('utf-8')).hexdigest()[:8]}"
            c_list = doc_chunks_map.get(did) or []
            if c_list:
                pass1_hash_to_chunks[p1_id] = c_list[:2]

        def _split_primary_secondary(
            unit_ids: Iterable[str],
            evidence_probes: List[str],
            name_probes: List[str],
        ) -> Tuple[Set[str], Set[str]]:
            pool: List[Tuple[str, str]] = []
            seen_c: Set[str] = set()
            for uid in unit_ids:
                cands = unit_to_chunks.get(uid) or pass1_hash_to_chunks.get(uid) or ()
                for cid, body_low in cands:
                    if cid not in seen_c:
                        seen_c.add(cid)
                        pool.append((cid, body_low))
            if not pool:
                return set(), set()

            ev_clean = [re.sub(r"\s+", " ", p).strip().casefold() for p in evidence_probes if p and len(p.strip()) >= 12]
            nm_clean = [re.sub(r"\s+", " ", p).strip().casefold() for p in name_probes if p and len(p.strip()) >= 3]

            primary: Set[str] = set()
            secondary: Set[str] = set()
            for cid, body_low in pool:
                hit = False
                for ev in ev_clean:
                    snippet = ev[: min(48, len(ev))]
                    if snippet in body_low:
                        hit = True
                        break
                if not hit:
                    for nm in nm_clean:
                        if nm in body_low:
                            hit = True
                            break
                if hit:
                    primary.add(cid)
                else:
                    secondary.add(cid)

            if not primary:
                # Nếu LLM diễn giải nhẹ câu bằng chứng, ưu tiên tối đa 2 chunk đầu của SemanticUnit
                for cid, _ in pool[:2]:
                    primary.add(cid)
                    secondary.discard(cid)

            if len(secondary) > 6:
                secondary = set(sorted(secondary)[:6])
            return primary, secondary

        # 1. Nạp toàn bộ Nodes từ LightRAG graph.json
        raw_name_to_eid: Dict[str, str] = {}
        for raw_key, node_info in raw_entities.items():
            if not isinstance(node_info, dict):
                continue
            raw_canon = str(node_info.get("canonical_name") or raw_key).strip()
            raw_cat = str(node_info.get("entity_type") or node_info.get("category") or "CONCEPT").upper()
            mapped_type = _LIGHTRAG_CATEGORY_MAP.get(raw_cat, "CONCEPT")
            canon_name = canonicalize_lightrag_entity_name(raw_canon, mapped_type)
            if len(canon_name) < 2 or is_forbidden_entity(canon_name, allow_school_names=True):
                continue
            eid = normalize_entity_id(canon_name)
            if not eid:
                continue

            raw_name_to_eid[raw_canon.casefold()] = eid
            raw_name_to_eid[str(raw_key).strip().casefold()] = eid

            descs_raw = node_info.get("descriptions") or []
            if isinstance(descs_raw, str):
                descs = [descs_raw.strip()] if descs_raw.strip() else []
            else:
                descs = [str(d).strip() for d in descs_raw if str(d).strip()]

            stripped_probe = re.sub(r"^\[[^\]]+\]\s*", "", canon_name).strip()
            prim_cids, sec_cids = _split_primary_secondary(
                node_info.get("source_unit_ids") or (),
                evidence_probes=descs[:2],
                name_probes=[stripped_probe, raw_canon],
            )
            if not prim_cids and not self.graph.has_node(eid):
                continue

            if not self.graph.has_node(eid):
                self.graph.add_node(
                    eid,
                    name=canon_name,
                    type=mapped_type,
                    chunks=set(),
                    secondary_chunks=set(),
                    descriptions=[],
                )
                self.lightrag_nodes_merged += 1

            g_node = self.graph.nodes[eid]
            g_node["chunks"].update(prim_cids)
            g_node.setdefault("secondary_chunks", set()).update(sec_cids - g_node["chunks"])
            existing_descs: List[str] = g_node.setdefault("descriptions", [])
            for d in descs:
                if d not in existing_descs and len(existing_descs) < 3:
                    existing_descs.append(d)

            for cid in prim_cids:
                ent_list = self.chunk_to_entities[cid]
                if eid not in ent_list:
                    ent_list.append(eid)

        # 2. Nạp toàn bộ Edges (quan hệ có bằng chứng) từ LightRAG graph.json
        for rel_item in raw_relations:
            if not isinstance(rel_item, dict):
                continue
            src_raw = str(rel_item.get("source") or "").strip()
            tgt_raw = str(rel_item.get("target") or "").strip()
            if not src_raw or not tgt_raw:
                continue

            hid = raw_name_to_eid.get(src_raw.casefold())
            if not hid:
                h_canon = canonicalize_lightrag_entity_name(src_raw)
                if not h_canon or is_forbidden_entity(h_canon, allow_school_names=True):
                    continue
                hid = normalize_entity_id(h_canon)
            else:
                h_canon = self.graph.nodes[hid]["name"] if self.graph.has_node(hid) else canonicalize_lightrag_entity_name(src_raw)

            tid = raw_name_to_eid.get(tgt_raw.casefold())
            if not tid:
                t_canon = canonicalize_lightrag_entity_name(tgt_raw)
                if not t_canon or is_forbidden_entity(t_canon, allow_school_names=True):
                    continue
                tid = normalize_entity_id(t_canon)
            else:
                t_canon = self.graph.nodes[tid]["name"] if self.graph.has_node(tid) else canonicalize_lightrag_entity_name(tgt_raw)

            if not hid or not tid or hid == tid:
                continue

            raw_rel_type = str(rel_item.get("relation_type") or rel_item.get("predicate") or "LIÊN_QUAN_ĐẾN")
            norm_rel = _normalize_relation(raw_rel_type)

            ev_list = rel_item.get("evidence_texts") or []
            if isinstance(ev_list, str):
                ev_list = [ev_list]
            elif not ev_list and rel_item.get("evidence"):
                ev_list = [str(rel_item["evidence"])]
            best_evidence = re.sub(r"\s+", " ", str(ev_list[0]).strip())[:140] if ev_list else ""

            h_probe = re.sub(r"^\[[^\]]+\]\s*", "", h_canon).strip()
            t_probe = re.sub(r"^\[[^\]]+\]\s*", "", t_canon).strip()
            prim_cids, sec_cids = _split_primary_secondary(
                rel_item.get("source_unit_ids") or (),
                evidence_probes=[best_evidence] if best_evidence else [],
                name_probes=[h_probe, t_probe],
            )

            # Fallback sang giao của chunks hai đầu mút nếu cạnh thiếu unit_id
            if not prim_cids:
                h_chunks = self.graph.nodes[hid].get("chunks", set()) if self.graph.has_node(hid) else set()
                t_chunks = self.graph.nodes[tid].get("chunks", set()) if self.graph.has_node(tid) else set()
                shared = h_chunks & t_chunks
                prim_cids = shared if shared else set(sorted(h_chunks)[:2] + sorted(t_chunks)[:2])
            if not prim_cids:
                continue

            if not self.graph.has_node(hid):
                self.graph.add_node(
                    hid, name=h_canon, type="CONCEPT",
                    chunks=set(prim_cids), secondary_chunks=set(sec_cids), descriptions=[],
                )
            else:
                self.graph.nodes[hid]["chunks"].update(prim_cids)

            if not self.graph.has_node(tid):
                self.graph.add_node(
                    tid, name=t_canon, type="CONCEPT",
                    chunks=set(prim_cids), secondary_chunks=set(sec_cids), descriptions=[],
                )
            else:
                self.graph.nodes[tid]["chunks"].update(prim_cids)

            if self.graph.has_edge(hid, tid):
                edge = self.graph[hid][tid]
                edge["relations"].add(norm_rel)
                edge["chunks"].update(prim_cids)
                edge["weight"] += 1.25
            else:
                self.graph.add_edge(
                    hid,
                    tid,
                    relations={norm_rel},
                    chunks=set(prim_cids),
                    weight=1.25,
                )
                self.lightrag_edges_merged += 1

            h_display = self.graph.nodes[hid]["name"]
            t_display = self.graph.nodes[tid]["name"]
            base_text = f"{h_display} -[{norm_rel}]-> {t_display}"
            rich_text = f"{base_text} (Căn cứ: {best_evidence})" if best_evidence else base_text

            t_sig = (hid, norm_rel, tid)
            if t_sig not in triple_key_to_idx:
                t_idx = len(self.triple_records)
                triple_key_to_idx[t_sig] = t_idx
                self.triple_records.append({
                    "head_id": hid,
                    "head_name": h_display,
                    "relation": norm_rel,
                    "tail_id": tid,
                    "tail_name": t_display,
                    "text": rich_text,
                    "evidence": best_evidence,
                    "chunk_ids": set(prim_cids),
                    "secondary_chunk_ids": set(sec_cids - prim_cids),
                })
            else:
                t_idx = triple_key_to_idx[t_sig]
                rec = self.triple_records[t_idx]
                rec["chunk_ids"].update(prim_cids)
                rec.setdefault("secondary_chunk_ids", set()).update(sec_cids - rec["chunk_ids"])
                if best_evidence and not rec.get("evidence"):
                    rec["evidence"] = best_evidence
                    rec["text"] = rich_text

            for cid in prim_cids:
                if t_idx not in self.chunk_to_triples[cid]:
                    self.chunk_to_triples[cid].append(t_idx)

    def _build_doc_neighbors(self) -> None:
        doc_groups: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        for idx, chunk in enumerate(self.chunks):
            doc_id = chunk.document_id or (chunk.metadata or {}).get("document_id")
            if not doc_id:
                continue
            c_idx = chunk.chunk_index if chunk.chunk_index is not None else (chunk.metadata or {}).get("chunk_index", idx)
            doc_groups[str(doc_id)].append((int(c_idx), idx, chunk.chunk_id))

        neighbors: Dict[str, List[str]] = defaultdict(list)
        for entries in doc_groups.values():
            if len(entries) < 2:
                continue
            entries.sort(key=lambda item: (item[0], item[1]))
            for pos, (_, _, cid) in enumerate(entries):
                if pos > 0:
                    neighbors[cid].append(entries[pos - 1][2])
                if pos + 1 < len(entries):
                    neighbors[cid].append(entries[pos + 1][2])
        self.doc_chunk_neighbors = dict(neighbors)

    @staticmethod
    def _l2_normalize_array(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (arr / norms).astype(np.float32)

    @staticmethod
    def _l2_normalize(vectors: List[List[float]]) -> Optional[np.ndarray]:
        if not vectors:
            return None
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        return GraphIndex._l2_normalize_array(arr)

    def _embed_in_batches(self, texts: List[str]) -> Optional[np.ndarray]:
        if not texts or self.embedding_model is None:
            return None
        all_vecs: List[List[float]] = []
        for start in range(0, len(texts), self.embed_batch_size):
            batch = texts[start:start + self.embed_batch_size]
            all_vecs.extend(self.embedding_model.embed_texts(batch))
        return self._l2_normalize(all_vecs)

    @staticmethod
    def _load_lightrag_entity_chroma_vectors(entity_chroma_dir: Optional[Path]) -> Dict[str, np.ndarray]:
        """Đọc trực tiếp kho vector thực thể BGE-M3 chuẩn (Tên + Định nghĩa) từ entity_chroma sang RAM."""
        if entity_chroma_dir is None:
            return {}
        sqlite_path = Path(entity_chroma_dir) / "chroma.sqlite3"
        if not sqlite_path.exists():
            return {}
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(entity_chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            col = client.get_collection(name="lightrag_entity_collection")
            total_ents = col.count()
            if total_ents == 0:
                return {}

            vec_map: Dict[str, np.ndarray] = {}
            page_size = 2500
            for offset in range(0, total_ents, page_size):
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
                    raw_name = str(meta.get("canonical_name") or "").strip()
                    if not raw_name:
                        continue
                    arr = np.asarray(emb, dtype=np.float32)
                    if arr.ndim != 1 or arr.shape[0] == 0:
                        continue
                    norm = float(np.linalg.norm(arr))
                    if norm > 0.0:
                        arr = arr / norm
                    canon = canonicalize_lightrag_entity_name(raw_name)
                    eid = normalize_entity_id(canon) if canon else ""
                    if eid:
                        vec_map[eid] = arr
            logger.info(f"Đã nạp {len(vec_map)} vector thực thể BGE-M3 chuẩn từ '{entity_chroma_dir}'.")
            return vec_map
        except Exception as exc:
            logger.warning(f"Không thể đọc vector từ entity_chroma '{entity_chroma_dir}': {exc}")
            return {}

    def _derive_vectors_from_chunks(
        self,
        chunk_vectors: Dict[str, List[float]],
        precomputed_entity_vecs: Optional[Dict[str, np.ndarray]] = None,
    ) -> bool:
        """
        Tổng hợp ma trận NumPy cho cả Entities và Triples:
          - Ưu tiên vector BGE-M3 chuẩn (Tên + Định nghĩa) từ entity_chroma cho các thực thể LightRAG.
          - Các thực thể Regex còn lại tổng hợp từ vector chunk trong ChromaDB.
          - Vector cạnh (Triple) kết hợp có trọng số: Head + Tail + Loại quan hệ + Chunk bằng chứng.
        """
        valid_vecs: Dict[str, np.ndarray] = {}
        dim: Optional[int] = None
        if precomputed_entity_vecs:
            first_ent_vec = next(iter(precomputed_entity_vecs.values()), None)
            if first_ent_vec is not None:
                dim = int(first_ent_vec.shape[0])

        for cid, vec in (chunk_vectors or {}).items():
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] == 0:
                continue
            if dim is None:
                dim = int(arr.shape[0])
            if arr.shape[0] == dim:
                valid_vecs[cid] = arr

        if (not valid_vecs and not precomputed_entity_vecs) or dim is None:
            return False

        rel_vec_map: Dict[str, np.ndarray] = {}
        if self.embedding_model is not None and self.triple_records:
            try:
                rel_keys = list(_RELATION_SEED_PHRASES.keys())
                rel_phrases = [_RELATION_SEED_PHRASES[k] for k in rel_keys]
                rel_embs = self._embed_in_batches(rel_phrases)
                if rel_embs is not None and rel_embs.shape[1] == dim:
                    for k, r_vec in zip(rel_keys, rel_embs):
                        rel_vec_map[k] = r_vec
            except Exception:
                rel_vec_map = {}

        entity_vec_map: Dict[str, np.ndarray] = {}
        direct_ent_hits = 0
        if self.entity_ids:
            ent_mat = np.zeros((len(self.entity_ids), dim), dtype=np.float32)
            for idx, eid in enumerate(self.entity_ids):
                direct_v = precomputed_entity_vecs.get(eid) if precomputed_entity_vecs else None
                if direct_v is not None and direct_v.shape[0] == dim:
                    ent_mat[idx] = direct_v
                    entity_vec_map[eid] = direct_v
                    direct_ent_hits += 1
                    continue

                cids = self.graph.nodes[eid].get("chunks", ())
                acc = np.zeros(dim, dtype=np.float32)
                cnt = 0
                for cid in cids:
                    v = valid_vecs.get(cid)
                    if v is not None:
                        acc += v
                        cnt += 1
                if cnt > 0:
                    acc /= float(cnt)
                    ent_mat[idx] = acc
                    entity_vec_map[eid] = acc
            self.entity_vectors = self._l2_normalize_array(ent_mat)

        if self.triple_records:
            tr_mat = np.zeros((len(self.triple_records), dim), dtype=np.float32)
            for idx, tr in enumerate(self.triple_records):
                acc = np.zeros(dim, dtype=np.float32)
                w_sum = 0.0
                h_vec = entity_vec_map.get(tr["head_id"])
                if h_vec is not None:
                    acc += 0.38 * h_vec
                    w_sum += 0.38
                t_vec = entity_vec_map.get(tr["tail_id"])
                if t_vec is not None:
                    acc += 0.38 * t_vec
                    w_sum += 0.38
                r_vec = rel_vec_map.get(tr["relation"])
                if r_vec is not None:
                    acc += 0.14 * r_vec
                    w_sum += 0.14

                c_acc = np.zeros(dim, dtype=np.float32)
                c_cnt = 0
                for cid in tr["chunk_ids"]:
                    v = valid_vecs.get(cid)
                    if v is not None:
                        c_acc += v
                        c_cnt += 1
                if c_cnt > 0:
                    acc += 0.10 * (c_acc / float(c_cnt))
                    w_sum += 0.10

                if w_sum > 0.0:
                    tr_mat[idx] = acc / w_sum
            self.triple_vectors = self._l2_normalize_array(tr_mat)

        logger.info(
            f"Đã dựng ma trận NumPy {dim} chiều cho {len(self.entity_ids)} entities "
            f"(entity_chroma chuẩn={direct_ent_hits}) và {len(self.triple_records)} triples."
        )
        return True

    def _embed_graph_items(
        self,
        chunk_vectors: Optional[Dict[str, List[float]]] = None,
        entity_chroma_dir: Optional[Path] = None,
    ) -> None:
        precomputed_ent_vecs = self._load_lightrag_entity_chroma_vectors(entity_chroma_dir)
        resolved_chunk_vecs = chunk_vectors
        if (
            not resolved_chunk_vecs
            and self.vector_store is not None
            and hasattr(self.vector_store, "get_embeddings_by_ids")
        ):
            try:
                resolved_chunk_vecs = self.vector_store.get_embeddings_by_ids(
                    [c.chunk_id for c in self.chunks]
                )
            except Exception as exc:
                logger.warning(f"Không thể lấy chunk vectors từ VectorStore ({exc}). Fallback sang embedding_model.")
                resolved_chunk_vecs = None

        if (resolved_chunk_vecs or precomputed_ent_vecs) and self._derive_vectors_from_chunks(
            resolved_chunk_vecs or {},
            precomputed_entity_vecs=precomputed_ent_vecs,
        ):
            return

        if self.embedding_model is None:
            return
        if self.entity_texts:
            self.entity_vectors = self._embed_in_batches(self.entity_texts)
        if self.triple_records:
            triple_texts = [tr["text"] for tr in self.triple_records]
            self.triple_vectors = self._embed_in_batches(triple_texts)

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
        """
        Truy xuất trên Unified Graph Vector Index (kết hợp PA1/Thesaurus/Regex + NumPy Vector + Spreading Activation):
        1. Kích hoạt Seed từ Regex mã cứng + AcademicThesaurus N-gram + PA1 LLM entities + Lexical + Vector Entity/Triple.
        2. Lan truyền kích hoạt 1-2 hops có Hub Gating, Entity IDF và trọng số quan hệ pháp lý.
        3. Tổng hợp điểm chunk 2 mức (Primary 100% / Secondary 25%) + Edge Co-Activation Bonus + Bridging + Doc Summary Boost.
        """
        clean_query = (query or "").strip()
        if top_k <= 0 or not self.chunks or not self.entity_ids or not clean_query:
            return []

        q_vec_arr: Optional[np.ndarray] = None
        if query_vector is None and self.embedding_model is not None:
            try:
                vecs = self.embedding_model.embed_texts([clean_query])
                if vecs:
                    query_vector = vecs[0]
            except Exception as exc:
                logger.warning(f"Không thể embed query trong GraphIndex.search: {exc}")

        if query_vector is not None:
            arr = np.asarray(query_vector, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0.0:
                q_vec_arr = arr / norm

        entity_scores: Dict[str, float] = defaultdict(float)
        triple_chunk_bonus: Dict[str, float] = defaultdict(float)
        activated_triple_indices: Dict[int, float] = {}
        entity_idf = getattr(self, "entity_idf", {})
        idf_arr = getattr(self, "entity_idf_array", None)
        ent_hits: Dict[str, int] = defaultdict(int)

        # 1.0a. Regex bắt nhanh Mã môn học (INT3401), Số hiệu QĐ (5115/QĐ-ĐHQGHN), Điều/Khoản, Chứng chỉ (<0.1ms)
        for rx in (_DOC_CODE_RE, _COURSE_CODE_RE, _ARTICLE_RE, _CERT_RE):
            for m in rx.finditer(clean_query):
                code_eid = normalize_entity_id(canonicalize_entity_name(m.group(1)))
                if code_eid and self.graph.has_node(code_eid):
                    idf = entity_idf.get(code_eid, 1.0)
                    entity_scores[code_eid] += 1.35 * idf
                    ent_hits[code_eid] += 2

        # 1.0b. Quét cụm dài-xuống-ngắn qua AcademicThesaurus (bỏ qua nếu LightRAGRetriever đã quét ở tầng trên)
        if not skip_thesaurus_scan:
            try:
                thesaurus = get_thesaurus()
                q_words = re.findall(r"[\w\+\-\./]+", clean_query.lower())
                covered: Set[int] = set()
                for n_gram in range(min(6, len(q_words)), 0, -1):
                    for pos in range(len(q_words) - n_gram + 1):
                        span = set(range(pos, pos + n_gram))
                        if span & covered:
                            continue
                        phrase = " ".join(q_words[pos : pos + n_gram])
                        if is_forbidden_entity(phrase, allow_school_names=False):
                            continue
                        th_canon = thesaurus.get_canonical(phrase)
                        if th_canon and not is_forbidden_entity(th_canon, allow_school_names=False):
                            covered.update(span)
                            th_eid = normalize_entity_id(canonicalize_entity_name(th_canon))
                            if th_eid and self.graph.has_node(th_eid):
                                idf = entity_idf.get(th_eid, 1.0)
                                entity_scores[th_eid] += 1.15 * idf
                                ent_hits[th_eid] += 2
            except Exception:
                pass

        # 1.0c. Tiêm thực thể chuẩn hóa từ PA1 LLM QueryEntityExtractor (hiểu khẩu ngữ/tiếng lóng sinh viên)
        if extracted_entities:
            for rank_idx, raw_ent in enumerate(extracted_entities[:8]):
                if is_forbidden_entity(raw_ent, allow_school_names=False):
                    continue
                canon_ent = canonicalize_lightrag_entity_name(raw_ent)
                if not canon_ent or is_forbidden_entity(canon_ent, allow_school_names=False):
                    continue
                pa1_eid = normalize_entity_id(canon_ent)
                decay_w = max(0.80, 1.0 - rank_idx * 0.04)
                if pa1_eid and self.graph.has_node(pa1_eid):
                    idf = entity_idf.get(pa1_eid, 1.0)
                    entity_scores[pa1_eid] += 1.30 * decay_w * idf
                    ent_hits[pa1_eid] += 2
                else:
                    # Fallback khớp token con khi LLM trả về cụm hơi khác nhẹ tên node
                    sub_toks = _graph_tokenize(canon_ent)
                    if sub_toks:
                        cand_counts: Dict[str, int] = defaultdict(int)
                        for st in sub_toks:
                            for ceid in self.token_to_entities.get(st, ()):
                                cand_counts[ceid] += 1
                        for ceid, c_cnt in cand_counts.items():
                            denom = max(1, len(self.entity_tokens.get(ceid, ())))
                            if c_cnt / denom >= 0.75 and c_cnt >= min(2, len(sub_toks)):
                                idf = entity_idf.get(ceid, 1.0)
                                entity_scores[ceid] += 0.90 * decay_w * idf
                                ent_hits[ceid] += 1

        # 1a. Lexical token matching (nhanh < 0.1ms, nhân Entity IDF + boost thực thể hiếm khớp hoàn toàn)
        q_tokens = _graph_tokenize(clean_query)
        if q_tokens:
            lex_hits: Dict[str, int] = defaultdict(int)
            for q_tok in q_tokens:
                for eid in self.token_to_entities.get(q_tok, ()):
                    lex_hits[eid] += 1
                    ent_hits[eid] += 1
            for eid, hit_count in lex_hits.items():
                denom = max(1, len(self.entity_tokens.get(eid, ())))
                overlap_ratio = hit_count / denom
                idf = entity_idf.get(eid, 1.0)
                exact_rare_boost = 1.25 if (overlap_ratio >= 0.99 and idf >= 0.75) else 1.0
                lex_score = (0.45 * overlap_ratio + 0.1 * min(hit_count, 3)) * idf * exact_rare_boost
                entity_scores[eid] += lex_score

        # 1b. Vector search trên Entities (gating COURSE/CERTIFICATE không khớp lexical & chống lạm phát df=1 cùng chunk)
        if (
            q_vec_arr is not None
            and self.entity_vectors is not None
            and self.entity_vectors.shape[1] == q_vec_arr.shape[0]
        ):
            sims = self.entity_vectors @ q_vec_arr
            weighted_sims = np.maximum(sims, 0.0) * (
                idf_arr if (idf_arr is not None and idf_arr.shape[0] == sims.shape[0]) else 1.0
            )
            k_cand = min(max(seed_top_m * 4, 20), len(weighted_sims))
            top_indices = np.argpartition(weighted_sims, -k_cand)[-k_cand:]
            ranked_vec_indices = sorted(
                (int(i) for i in top_indices if float(weighted_sims[int(i)]) > 0.0),
                key=lambda i: float(weighted_sims[i]),
                reverse=True,
            )
            df1_chunk_counts: Dict[str, int] = defaultdict(int)
            for idx in ranked_vec_indices:
                eid = self.entity_ids[idx]
                w_sim = float(weighted_sims[idx])
                node = self.graph.nodes[eid]
                etype = str(node.get("type") or "")
                has_lex = ent_hits.get(eid, 0) > 0
                if etype in {"COURSE", "CERTIFICATE"} and not has_lex:
                    w_sim *= 0.25
                chunks_set = node.get("chunks", ())
                if not has_lex and len(chunks_set) == 1:
                    only_cid = next(iter(chunks_set))
                    if df1_chunk_counts[only_cid] >= 2:
                        continue
                    df1_chunk_counts[only_cid] += 1
                entity_scores[eid] += w_sim

        # 1c. Vector search trên Triples (nhân trọng số quan hệ & endpoint IDF, cộng thưởng 2 mức chunk)
        if (
            q_vec_arr is not None
            and self.triple_vectors is not None
            and self.triple_vectors.shape[1] == q_vec_arr.shape[0]
        ):
            t_sims = self.triple_vectors @ q_vec_arr
            k_tr = min(max(seed_top_m * 2, 12), len(t_sims))
            top_t_indices = np.argpartition(t_sims, -k_tr)[-k_tr:]
            for idx in top_t_indices:
                sim = float(t_sims[idx])
                if sim > 0.0:
                    t_idx = int(idx)
                    tr = self.triple_records[t_idx]
                    rel_w = _RELATION_WEIGHTS.get(tr["relation"], 1.0)
                    h_idf = entity_idf.get(tr["head_id"], 1.0)
                    t_idf = entity_idf.get(tr["tail_id"], 1.0)
                    tr_score = sim * rel_w
                    activated_triple_indices[t_idx] = tr_score
                    entity_scores[tr["head_id"]] += 0.5 * tr_score * h_idf
                    entity_scores[tr["tail_id"]] += 0.5 * tr_score * t_idf
                    base_tr_bonus = 0.35 * tr_score * (0.5 + 0.25 * (h_idf + t_idf))
                    for cid in tr["chunk_ids"]:
                        triple_chunk_bonus[cid] += base_tr_bonus
                    for scid in tr.get("secondary_chunk_ids", ()):
                        triple_chunk_bonus[scid] += 0.25 * base_tr_bonus

        if not entity_scores and not triple_chunk_bonus:
            return []

        total_chunks = len(self.chunks)
        max_hub_df = max(250, int(0.12 * total_chunks)) if total_chunks >= 50 else total_chunks + 1

        top_seeds = heapq.nlargest(seed_top_m, entity_scores.items(), key=lambda item: item[1])
        seed_set: Set[str] = {eid for eid, _ in top_seeds}
        activated_entities: Dict[str, float] = dict(top_seeds)
        frontier: Dict[str, float] = dict(top_seeds)

        # 2. Multi-hop Spreading Activation (bỏ qua super-node > max_hub_df để chống bùng nổ độ trễ & nhiễu)
        for hop in range(1, max(0, max_hops) + 1):
            if not frontier:
                break
            decay = hop_decay ** hop
            next_frontier: Dict[str, float] = {}
            for u, u_score in frontier.items():
                if not self.graph.has_node(u) or len(self.graph.nodes[u].get("chunks", ())) > max_hub_df:
                    continue
                neighbors = set(self.graph.successors(u)) | set(self.graph.predecessors(u))
                if not neighbors:
                    continue
                damping = 1.0 / math.sqrt(len(neighbors))
                for v in neighbors:
                    if len(self.graph.nodes[v].get("chunks", ())) > max_hub_df:
                        continue
                    rel_w = self._max_edge_relation_weight(u, v)
                    v_idf = entity_idf.get(v, 1.0)
                    prop = u_score * decay * damping * rel_w * v_idf
                    prev = activated_entities.get(v, 0.0)
                    if prop > prev:
                        activated_entities[v] = prop
                        next_frontier[v] = max(next_frontier.get(v, 0.0), prop)
            if len(next_frontier) > seed_top_m * 3:
                top_next = heapq.nlargest(seed_top_m * 3, next_frontier.items(), key=lambda item: item[1])
                frontier = dict(top_next)
            else:
                frontier = next_frontier

        # 3. Tổng hợp điểm 2 mức cho từng chunk (Primary 1.0x / Secondary 0.25x + Co-Activation + Doc Summary Boost)
        chunk_scores: Dict[str, float] = defaultdict(float)
        chunk_matched_entities: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        def _add_entity_chunks(eid: str, contrib: float, limit: Optional[int] = None) -> None:
            nd = self.graph.nodes[eid]
            cids = nd.get("chunks", ())
            if limit is not None and len(cids) > limit:
                cids = sorted(cids)[:limit]
            for cid in cids:
                chunk_scores[cid] += contrib
                chunk_matched_entities[cid].append((eid, contrib))
            sec_cids = nd.get("secondary_chunks", ())
            if sec_cids:
                sec_contrib = 0.25 * contrib
                for scid in sec_cids:
                    chunk_scores[scid] += sec_contrib

        # Lưu ý: e_score trong activated_entities đã được nhân entity_idf ở bước Seed & Multi-hop (tránh phạt bình phương idf^2)
        for eid, e_score in activated_entities.items():
            if not self.graph.has_node(eid):
                continue
            node_chunks = self.graph.nodes[eid].get("chunks", ())
            if len(node_chunks) > max_hub_df:
                continue
            _add_entity_chunks(eid, e_score)

        for cid, bonus in triple_chunk_bonus.items():
            chunk_scores[cid] += bonus

        if not chunk_scores and activated_entities:
            best_eid, best_score = max(activated_entities.items(), key=lambda item: item[1])
            if self.graph.has_node(best_eid):
                _add_entity_chunks(best_eid, best_score, limit=max(1, top_k * 4))

        if not chunk_scores:
            return []

        # High-Level Document Summary Boost (Mitigation #3 cho câu hỏi bao quát nhiều văn bản)
        doc_summary_scores: Dict[str, float] = {}
        doc_id_summaries = getattr(self, "doc_id_to_summary", None) or {}
        if doc_id_summaries:
            theme_terms = [
                canonicalize_entity_name(t).casefold()
                for t in (high_level_themes or [])
                if t and not is_forbidden_entity(t, allow_school_names=False)
            ]
            seed_names_low = {
                self.graph.nodes[eid]["name"].casefold()
                for eid in seed_set
                if self.graph.has_node(eid)
            }
            q_low = clean_query.casefold()
            for did, s_info in doc_id_summaries.items():
                s_title = str(s_info.get("title") or "").casefold()
                s_body = str(s_info.get("summary") or "").casefold()
                s_keys = {canonicalize_entity_name(str(k)).casefold() for k in (s_info.get("key_entities") or [])}
                s_score = 0.0
                for th in theme_terms:
                    if th in s_title:
                        s_score += 0.45
                    elif th in s_body:
                        s_score += 0.25
                    if th in s_keys:
                        s_score += 0.30
                overlap_seeds = len(s_keys & seed_names_low)
                if overlap_seeds:
                    s_score += min(0.60, 0.25 * overlap_seeds)
                if s_title and len(s_title) >= 10 and s_title in q_low:
                    s_score += 0.40
                if s_score >= 0.25:
                    doc_summary_scores[did] = min(1.0, s_score)

        if filters:
            filtered_items = [
                (cid, score)
                for cid, score in chunk_scores.items()
                if cid in self.chunk_id_to_idx and self._matches_filters(self.chunks[self.chunk_id_to_idx[cid]], filters)
            ]
        else:
            filtered_items = [(cid, score) for cid, score in chunk_scores.items() if cid in self.chunk_id_to_idx]

        if not filtered_items:
            return []

        top_base_chunks = heapq.nlargest(max(top_k * 4, 200), filtered_items, key=lambda item: item[1])
        pre_smooth_scores: Dict[str, float] = {}
        for cid, base_score in top_base_chunks:
            co_scores: List[float] = []
            for t_idx in self.chunk_to_triples.get(cid, ()):
                tr = self.triple_records[t_idx]
                h_score = activated_entities.get(tr["head_id"], 0.0)
                t_score = activated_entities.get(tr["tail_id"], 0.0)
                if h_score > 0.0 and t_score > 0.0:
                    rel_w = _RELATION_WEIGHTS.get(tr["relation"], 1.0)
                    both_seed_mult = 1.35 if (tr["head_id"] in seed_set and tr["tail_id"] in seed_set) else 1.0
                    co_scores.append(0.45 * math.sqrt(h_score * t_score) * rel_w * both_seed_mult)
            co_bonus = sum(sorted(co_scores, reverse=True)[:3])
            matched_cnt = len({eid for eid, _ in chunk_matched_entities.get(cid, ())})
            bridge_bonus = 0.25 * min(3, max(0, matched_cnt - 1))

            summary_bonus = 0.0
            if doc_summary_scores:
                ch_obj = self.chunks[self.chunk_id_to_idx[cid]]
                ch_did = str(ch_obj.document_id or (ch_obj.metadata or {}).get("document_id") or "")
                if ch_did in doc_summary_scores:
                    summary_bonus = 0.18 * doc_summary_scores[ch_did]

            pre_smooth_scores[cid] = base_score + co_bonus + bridge_bonus + summary_bonus

        if not pre_smooth_scores:
            return []

        # 4. Adjacent Structural Smoothing (lan tỏa điểm nhẹ sang chunk +/- 1 cùng văn bản)
        doc_neighbors = getattr(self, "doc_chunk_neighbors", {})
        smoothed_scores: Dict[str, float] = dict(pre_smooth_scores)
        if adjacent_alpha > 0.0 and doc_neighbors:
            smooth_pool = heapq.nlargest(max(top_k, seed_top_m * 3), pre_smooth_scores.items(), key=lambda item: item[1])
            for cid, base_val in smooth_pool:
                delta = adjacent_alpha * base_val
                for nbr_cid in doc_neighbors.get(cid, ()):
                    smoothed_scores[nbr_cid] = smoothed_scores.get(nbr_cid, 0.0) + delta

        target_faculties = detect_query_faculties(query)

        final_scores: List[Tuple[str, float]] = []
        for cid, score in smoothed_scores.items():
            idx = self.chunk_id_to_idx.get(cid)
            if idx is None:
                continue
            chunk = self.chunks[idx]
            if not self._matches_filters(chunk, filters):
                continue
            adj_score = score
            if target_faculties:
                ch_meta = chunk.metadata or {}
                ch_fac = str(ch_meta.get("faculty") or ch_meta.get("site") or "").strip().lower()
                if ch_fac in target_faculties:
                    adj_score *= 1.15
                elif ch_fac in KNOWN_FACULTY_KEYS:
                    adj_score *= 0.35
            final_scores.append((cid, adj_score))

        if not final_scores:
            return []

        top_chunks = heapq.nlargest(top_k, final_scores, key=lambda item: item[1])
        max_score = top_chunks[0][1] if top_chunks[0][1] > 0 else 1.0

        results: List[RetrievedContext] = []
        summary_injected_docs: Set[str] = set()
        for rank, (cid, raw_score) in enumerate(top_chunks, start=1):
            chunk = self.chunks[self.chunk_id_to_idx[cid]]
            norm_score = round(min(1.0, max(0.01, raw_score / max_score)), 4)

            matched_ents_sorted = sorted(
                chunk_matched_entities.get(cid, []),
                key=lambda x: x[1],
                reverse=True,
            )
            seen_enames: Set[str] = set()
            ent_names: List[str] = []
            for eid, _ in matched_ents_sorted:
                if self.graph.has_node(eid):
                    ename = self.graph.nodes[eid]["name"]
                    if ename not in seen_enames:
                        seen_enames.add(ename)
                        ent_names.append(ename)
                        if len(ent_names) >= 6:
                            break

            # Ưu tiên xếp hạng các cạnh đồng kích hoạt (bắc cầu trực tiếp giữa 2 thực thể đang kích hoạt) lên đầu
            triple_candidates: List[Tuple[float, str]] = []
            seen_t_texts: Set[str] = set()
            for t_idx in self.chunk_to_triples.get(cid, ()):
                tr = self.triple_records[t_idx]
                h_act = activated_entities.get(tr["head_id"], 0.0)
                t_act = activated_entities.get(tr["tail_id"], 0.0)
                co_mult = 2.5 if (h_act > 0.0 and t_act > 0.0) else 1.0
                t_score = co_mult * (h_act + t_act + activated_triple_indices.get(t_idx, 0.0))
                if t_score > 0.0 and tr["text"] not in seen_t_texts:
                    seen_t_texts.add(tr["text"])
                    triple_candidates.append((t_score, tr["text"]))
            triple_candidates.sort(key=lambda x: x[0], reverse=True)
            top_triples = [t_txt for _, t_txt in triple_candidates[:5]]

            # Đính kèm 1 dòng phạm vi tổng quan văn bản (Pass 1 Summary) cho chunk đại diện đầu tiên của tài liệu đó
            ch_did = str(chunk.document_id or (chunk.metadata or {}).get("document_id") or "")
            if ch_did and ch_did in doc_summary_scores and ch_did not in summary_injected_docs:
                s_info = doc_id_summaries.get(ch_did) or {}
                s_scope = str(s_info.get("summary") or "").strip()
                if s_scope and len(top_triples) < 5:
                    top_triples.insert(0, f"Phạm vi văn bản: {s_scope[:140]}")
                    summary_injected_docs.add(ch_did)

            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta["search_source"] = "GraphRAG"
            meta["search_type"] = "graph"
            meta["graph_score"] = round(raw_score, 4)
            meta["graph_entities"] = ent_names
            meta["graph_triples"] = top_triples

            results.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=norm_score,
                    rank=rank,
                    metadata=meta,
                )
            )

        return results

    # -------------------------------------------------------------------------
    # LIGHTRAG SUPPORT BRIDGE & REGEX SUPPLEMENT HELPERS
    # -------------------------------------------------------------------------

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
        """
        Neo kết quả suy luận 2 tầng của LightRAGRetriever về các DataChunk nguyên văn (chunked_data),
        kết hợp bổ khuyết từ tầng Regex/Vector của feat/graph ở trọng số phụ (supplement_weight < 1.0):
          1. Nguồn Chính (1.0x): Chunks ánh xạ từ activated_nodes, activated_edges (kèm bằng chứng) và
             high_level_blocks của LightRAGRetriever (Primary 1.0x / Secondary 0.25x + Co-Activation).
          2. Nguồn Phụ Bổ Khuyết (supplement_weight, mặc định 0.65x): Bổ sung các chunk khớp mã học phần,
             số quyết định, hoặc dữ liệu mới nạp chưa có trong graph.json của LightRAG.
        """
        if top_k <= 0 or not self.chunks:
            return []

        hub_cap = self.max_hub_df()

        seed_eids: Set[str] = set()
        for s_name, _ in (seed_entities or ()):
            seid = self.resolve_lightrag_node_id(s_name)
            if seid:
                seed_eids.add(seid)

        lr_chunk_scores: Dict[str, float] = defaultdict(float)
        chunk_matched_entities: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        chunk_matched_seeds: Dict[str, Set[str]] = defaultdict(set)
        chunk_edge_scores: Dict[str, List[float]] = defaultdict(list)
        chunk_lr_triples: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

        # 1A. Ánh xạ các thực thể kích hoạt từ LightRAGRetriever -> Primary (1.0x) & Secondary (0.25x) chunks
        # Lưu ý: act_score từ LightRAGRetriever đã tích hợp _effective_node_idf (không nhân lặp entity_idf lần 2)
        for raw_name, act_score in (activated_nodes or {}).items():
            if act_score <= 0.0:
                continue
            eid = self.resolve_lightrag_node_id(raw_name)
            if not eid or not self.graph.has_node(eid):
                continue
            nd = self.graph.nodes[eid]
            prim_cids = nd.get("chunks", ())
            if len(prim_cids) > hub_cap:
                continue
            contrib = float(act_score)
            disp_name = str(nd.get("name") or raw_name)
            is_seed = eid in seed_eids
            for cid in prim_cids:
                lr_chunk_scores[cid] += contrib
                chunk_matched_entities[cid].append((disp_name, contrib))
                if is_seed:
                    chunk_matched_seeds[cid].add(eid)
            for scid in nd.get("secondary_chunks", ()):
                lr_chunk_scores[scid] += 0.25 * contrib

        # 1B. Ánh xạ các cạnh quan hệ 2-hop của LightRAGRetriever (kèm bằng chứng Căn cứ)
        for edge in (activated_edges or ()):
            src_raw = str(edge.get("source") or "").strip()
            tgt_raw = str(edge.get("target") or "").strip()
            if not src_raw or not tgt_raw:
                continue
            rel_raw = str(edge.get("relation_type") or edge.get("predicate") or "LIÊN_QUAN_ĐẾN")
            norm_rel = _normalize_relation(rel_raw)
            rel_w = _RELATION_WEIGHTS.get(norm_rel, 1.0)

            hid = self.resolve_lightrag_node_id(src_raw)
            tid = self.resolve_lightrag_node_id(tgt_raw)
            if not hid and not tid:
                continue

            ev_list = edge.get("evidence_texts") or []
            if isinstance(ev_list, str):
                ev_list = [ev_list]
            ev = str(
                edge.get("evidence")
                or (ev_list[0] if ev_list else "")
                or edge.get("description")
                or ""
            ).strip()[:140]

            h_disp = self.graph.nodes[hid]["name"] if (hid and self.graph.has_node(hid)) else src_raw
            t_disp = self.graph.nodes[tid]["name"] if (tid and self.graph.has_node(tid)) else tgt_raw
            t_text = f"{h_disp} -[{norm_rel}]-> {t_disp}" + (f" (Căn cứ: {ev})" if ev else "")

            edge_cids: Set[str] = set()
            if hid and tid:
                if self.graph.has_edge(hid, tid):
                    edge_cids.update(self.graph[hid][tid].get("chunks", ()))
                if self.graph.has_edge(tid, hid):
                    edge_cids.update(self.graph[tid][hid].get("chunks", ()))
                if not edge_cids:
                    h_c = self.graph.nodes[hid].get("chunks", set()) if self.graph.has_node(hid) else set()
                    t_c = self.graph.nodes[tid].get("chunks", set()) if self.graph.has_node(tid) else set()
                    if len(h_c) <= hub_cap and len(t_c) <= hub_cap:
                        edge_cids = set(h_c) & set(t_c)

            if not edge_cids:
                continue

            both_seeds = 1.35 if (hid in seed_eids and tid in seed_eids) else 1.0
            base_edge_score = float(edge.get("score", 0.80)) * rel_w * both_seeds
            for cid in edge_cids:
                chunk_edge_scores[cid].append(0.45 * base_edge_score)
                chunk_lr_triples[cid].append((base_edge_score * 2.0, t_text))

        # Giới hạn tối đa Top-3 cạnh mạnh nhất mỗi chunk (chống lạm phát điểm ở các chunk CTĐT nhiều cạnh lặp)
        for cid, e_list in chunk_edge_scores.items():
            lr_chunk_scores[cid] += sum(sorted(e_list, reverse=True)[:3])

        # 1C. Thưởng đồng kích hoạt (Bridging Bonus) & High-Level Summary Boost từ LightRAG Pass 1
        doc_summary_map: Dict[str, Tuple[float, str]] = {}
        if high_level_blocks:
            title_map = self._get_title_to_doc_ids()
            for hl in high_level_blocks:
                hl_title_low = str(hl.get("title") or "").strip().casefold()
                hl_doc_low = str(hl.get("doc_id") or "").strip().casefold()
                hl_score = float(hl.get("score") or 0.0)
                hl_sum = str(hl.get("summary") or "").strip()
                matched_dids = set(title_map.get(hl_title_low, ())) | set(title_map.get(hl_doc_low, ()))
                for did in matched_dids:
                    if hl_score > doc_summary_map.get(did, (0.0, ""))[0]:
                        doc_summary_map[did] = (hl_score, hl_sum)

        for cid in list(lr_chunk_scores.keys()):
            distinct_ents = len({ename for ename, _ in chunk_matched_entities.get(cid, ())})
            if distinct_ents >= 2:
                lr_chunk_scores[cid] += 0.22 * min(3, distinct_ents - 1)
            # Thưởng mạnh khi chunk phủ >= 2 thực thể HẠT GIỐNG khác nhau của câu hỏi (vd: 'Ban chủ nhiệm khoa' + 'Khoa FAT')
            matched_seed_cnt = len(chunk_matched_seeds.get(cid, ()))
            if matched_seed_cnt >= 2:
                lr_chunk_scores[cid] *= 1.0 + 0.55 * min(3, matched_seed_cnt - 1)
            idx = self.chunk_id_to_idx.get(cid)
            if idx is not None and doc_summary_map:
                ch = self.chunks[idx]
                did = str(ch.document_id or (ch.metadata or {}).get("document_id") or "")
                if did in doc_summary_map:
                    lr_chunk_scores[cid] += 0.20 * doc_summary_map[did][0]

        # 1D. Adjacent Structural Smoothing trên điểm gốc của LightRAG
        doc_neighbors = getattr(self, "doc_chunk_neighbors", {})
        if lr_chunk_scores and adjacent_alpha > 0.0 and doc_neighbors:
            top_lr_pool = heapq.nlargest(max(top_k, 30), lr_chunk_scores.items(), key=lambda x: x[1])
            for cid, val in top_lr_pool:
                delta = adjacent_alpha * val
                for nbr_cid in doc_neighbors.get(cid, ()):
                    lr_chunk_scores[nbr_cid] = lr_chunk_scores.get(nbr_cid, 0.0) + delta

        # 2. Chạy tầng Phụ trợ Bổ khuyết (Regex mã cứng + Vector Graph trên các chunk chưa phủ)
        supp_hits: List[RetrievedContext] = []
        if not lr_chunk_scores or float(supplement_weight) > 0.0:
            supp_top_k = top_k if not lr_chunk_scores else min(top_k, 40)
            supp_hits = self.search(
                query=query,
                query_vector=query_vector,
                top_k=supp_top_k,
                seed_top_m=seed_top_m,
                max_hops=max_hops,
                hop_decay=hop_decay,
                adjacent_alpha=adjacent_alpha,
                filters=filters,
                extracted_entities=extracted_entities,
                high_level_themes=high_level_themes,
                skip_thesaurus_scan=bool(seed_entities),
            )

        # Nếu LightRAG graph.json không khớp được chunk nào (vd: chunk tin tức mới nạp), dùng trực tiếp Supplement Fallback
        if not lr_chunk_scores:
            for h in supp_hits:
                h.metadata["graph_engine"] = "regex_supplement_fallback"
            return supp_hits

        # Chuẩn hóa điểm LightRAG về [0, 1.0] để giữ vai trò Nguồn Chính (1.0x)
        valid_lr_items = [
            (cid, sc)
            for cid, sc in lr_chunk_scores.items()
            if cid in self.chunk_id_to_idx and self._matches_filters(self.chunks[self.chunk_id_to_idx[cid]], filters)
        ]
        if not valid_lr_items:
            for h in supp_hits:
                h.metadata["graph_engine"] = "regex_supplement_fallback"
            return supp_hits

        max_lr = max(sc for _, sc in valid_lr_items) or 1.0
        combined_scores: Dict[str, float] = {cid: (sc / max_lr) for cid, sc in valid_lr_items}
        supp_meta_map: Dict[str, Dict[str, Any]] = {}

        has_hard_code = self.has_hard_code_entities(query)
        eff_supp_w = min(0.85, max(0.0, float(supplement_weight)) + (0.15 if has_hard_code else 0.0))

        for sh in supp_hits:
            scid = sh.chunk_id
            supp_meta_map[scid] = sh.metadata or {}
            s_val = eff_supp_w * float(sh.similarity_score)
            if scid in combined_scores:
                # Hợp nhất đơn điệu: đồng thuận luôn >= điểm từng nhánh riêng lẻ (tránh phạt ngược khi lr_val thấp)
                lr_val = combined_scores[scid]
                combined_scores[scid] = max(lr_val, s_val) + 0.20 * min(lr_val, s_val)
            else:
                # Chunk chỉ có từ tầng bổ khuyết (mã cứng / chunk mới) -> giữ dưới trần eff_supp_w
                combined_scores[scid] = s_val

        # Lọc giảm nhiễu liên khoa khi câu hỏi nêu đích danh Khoa/Viện (vd: hỏi FAT không để FIT/FEMA/FET lấn át)
        target_faculties = detect_query_faculties(query)
        if target_faculties:
            for cid in list(combined_scores.keys()):
                idx = self.chunk_id_to_idx.get(cid)
                if idx is None:
                    continue
                ch_meta = self.chunks[idx].metadata or {}
                ch_fac = str(ch_meta.get("faculty") or ch_meta.get("site") or "").strip().lower()
                if ch_fac in target_faculties:
                    combined_scores[cid] *= 1.15
                elif ch_fac in KNOWN_FACULTY_KEYS:
                    combined_scores[cid] *= 0.35

        ranked_cids = heapq.nlargest(top_k, combined_scores.items(), key=lambda x: x[1])
        top_max = ranked_cids[0][1] if ranked_cids and ranked_cids[0][1] > 0 else 1.0

        results: List[RetrievedContext] = []
        summary_injected_docs: Set[str] = set()
        for rank, (cid, raw_comb) in enumerate(ranked_cids, start=1):
            chunk = self.chunks[self.chunk_id_to_idx[cid]]
            norm_score = round(min(1.0, max(0.01, raw_comb / top_max)), 4)
            supp_meta = supp_meta_map.get(cid, {})

            # Gộp danh sách thực thể (ưu tiên LightRAG trước, bổ sung từ supplement sau)
            ent_names: List[str] = []
            seen_ents: Set[str] = set()
            for ename, _ in sorted(chunk_matched_entities.get(cid, []), key=lambda x: x[1], reverse=True):
                if ename not in seen_ents:
                    seen_ents.add(ename)
                    ent_names.append(ename)
                    if len(ent_names) >= 6:
                        break
            for s_ename in (supp_meta.get("graph_entities") or []):
                if len(ent_names) >= 6:
                    break
                if s_ename not in seen_ents:
                    seen_ents.add(s_ename)
                    ent_names.append(s_ename)

            # Gộp danh sách cạnh đồ thị (ưu tiên cạnh có Căn cứ của LightRAG trước)
            triples_out: List[str] = []
            seen_triples: Set[str] = set()
            for _, t_txt in sorted(chunk_lr_triples.get(cid, []), key=lambda x: x[0], reverse=True):
                if t_txt not in seen_triples:
                    seen_triples.add(t_txt)
                    triples_out.append(t_txt)
                    if len(triples_out) >= 5:
                        break
            for s_tr in (supp_meta.get("graph_triples") or []):
                if len(triples_out) >= 5:
                    break
                if s_tr not in seen_triples and not str(s_tr).startswith("Phạm vi văn bản:"):
                    seen_triples.add(s_tr)
                    triples_out.append(s_tr)

            # Nếu vẫn chưa có triple trực tiếp nhưng chunk thuộc node LightRAG, lấy triple tiêu biểu từ bridge
            if not triples_out:
                for t_idx in self.chunk_to_triples.get(cid, ())[:3]:
                    tr_rec = self.triple_records[t_idx]
                    if tr_rec["text"] not in seen_triples:
                        seen_triples.add(tr_rec["text"])
                        triples_out.append(tr_rec["text"])

            ch_did = str(chunk.document_id or (chunk.metadata or {}).get("document_id") or "")
            if ch_did and ch_did in doc_summary_map and ch_did not in summary_injected_docs:
                s_scope = doc_summary_map[ch_did][1]
                if s_scope and len(triples_out) < 5:
                    triples_out.insert(0, f"Phạm vi văn bản: {s_scope[:140]}")
                    summary_injected_docs.add(ch_did)

            is_lr_primary = cid in lr_chunk_scores
            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta["search_source"] = "GraphRAG"
            meta["search_type"] = "graph"
            meta["graph_engine"] = "lightrag_primary" if is_lr_primary else "regex_supplement"
            meta["lightrag_grounded"] = is_lr_primary
            meta["graph_score"] = round(raw_comb, 4)
            meta["graph_entities"] = ent_names
            meta["graph_triples"] = triples_out

            results.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=norm_score,
                    rank=rank,
                    metadata=meta,
                )
            )

        return results


# Aliases làm rõ kiến trúc LightRAG-First (GraphIndex đóng vai trò Cầu nối Chunk & Bổ khuyết Regex)
LightRAGChunkBridge = GraphIndex
RegexSupplementGraph = GraphIndex

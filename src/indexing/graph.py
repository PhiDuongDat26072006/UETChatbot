"""Knowledge Graph + Vector Index (GraphRAG) for UETChatbot.

Builds an offline hybrid Knowledge Graph (deterministic UET academic rules + optional
OpenRouter Tencent Hy4 LLM extraction with persistent cache) and embeds entities/triples
with the shared embedding model (bge-m3) for 0-LLM multi-hop graph retrieval at query time.
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
from src.utils.helpers import get_logger

logger = get_logger("indexing.graph")


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
    "ielts": "IELTS",
    "toeic": "TOEIC",
    "toefl": "TOEFL",
    "toefl itp": "TOEFL ITP",
    "toefl ibt": "TOEFL iBT",
    "jlpt": "JLPT",
    "aptis": "Aptis",
    "vnu-test": "VNU-TEST",
}

_DOC_PREFIX_STRIP_RE = re.compile(
    r"^(?:quyết\s+định|nghị\s+quyết|thông\s+báo|kế\s+hoạch|hướng\s+dẫn|công\s+văn|quy\s+định|qđ|vb)(?:\s+số)?\s+(\d{1,5}/(?:QĐ|NQ|TB|KH|HD|CV)-[A-ZĐƠƯĂÂÊÔ0-9\-]+)\b.*$",
    re.IGNORECASE,
)

_REL_ASCII_PREFIX_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("SUA_DOI", "BO_SUNG", "THAY_THE"), "SỬA_ĐỔI_BỔ_SUNG"),
    (("CAN_CU", "DUA_TREN"), "CĂN_CỨ"),
    (("YEU_CAU", "DIEU_KIEN", "CAN_CO", "BAT_BUOC"), "YÊU_CẦU_ĐIỀU_KIỆN"),
    (("TRUC_THUOC",), "TRỰC_THUỘC"),
    (("THUOC",), "THUỘC_ĐƠN_VỊ"),
    (("BAO_GOM", "CHUA_NOI_DUNG"), "BAO_GỒM"),
    (("PHU_TRACH", "CHU_TRI", "GIANG_DAY"), "PHỤ_TRÁCH"),
    (("QUY_DINH", "AP_DUNG", "BAN_HANH", "XET_"), "QUY_ĐỊNH_VỀ"),
    (("LIEN_QUAN",), "LIÊN_QUAN_ĐẾN"),
)

_RELATION_WEIGHTS: Dict[str, float] = {
    "SỬA_ĐỔI_BỔ_SUNG": 1.35,
    "CĂN_CỨ": 1.30,
    "YÊU_CẦU_ĐIỀU_KIỆN": 1.25,
    "QUY_ĐỊNH_VỀ": 1.15,
    "BAO_GỒM": 1.10,
    "PHỤ_TRÁCH": 1.10,
    "TRỰC_THUỘC": 0.95,
    "THUỘC_ĐƠN_VỊ": 0.85,
    "LIÊN_QUAN_ĐẾN": 0.85,
}

_ENTITY_TYPE_WEIGHTS: Dict[str, float] = {
    "REGULATION": 1.25,
    "COURSE": 1.25,
    "ARTICLE": 1.15,
    "CERTIFICATE": 1.15,
    "PERSON": 1.15,
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
    """Gộp các biến thể viết tắt/từ đồng nghĩa UET về tên thực thể chuẩn duy nhất."""
    cleaned = clean_entity_name(name)
    if not cleaned:
        return ""
    m_doc = _DOC_PREFIX_STRIP_RE.match(cleaned)
    if m_doc:
        return m_doc.group(1).upper()
    alias = _CANONICAL_ENTITY_ALIASES.get(cleaned.casefold())
    if alias:
        return alias
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
            if len(cleaned) < 2:
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
            eid = normalize_entity_id(cleaned)
            if not eid or len(cleaned) < 2:
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


class GraphIndex:
    """
    Chỉ mục Đồ thị Tri thức Vector (Graph Vector Index) lưu cục bộ tại database/graph_index.pkl.
    - Offline: trích xuất Entity & Triples -> dựng đồ thị NetworkX -> nhúng vector bằng Embedding Model.
    - Online: 0-LLM kích hoạt Seed Entities (Vector + Lexical) -> lan truyền 1-2 hops -> trả về RetrievedContext.
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
    ):
        self.chunks: List[DataChunk] = normalize_chunks(list(chunks))
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.llm_calls_count: int = 0

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

        self._build_graph(active_extractor)
        active_extractor.flush_cache()
        self.llm_calls_count = active_extractor.llm_calls_count
        self._embed_graph_items(chunk_vectors=chunk_vectors)

    def _build_graph(self, extractor: HybridGraphExtractor) -> None:
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
                    self.graph.add_node(hid, name=tr["head"], type=tr.get("head_type", "CONCEPT"), chunks=set())
                if not self.graph.has_node(tid):
                    self.graph.add_node(tid, name=tr["tail"], type=tr.get("tail_type", "CONCEPT"), chunks=set())

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
                        "chunk_ids": {cid},
                    })
                else:
                    t_idx = triple_key_to_idx[t_sig]
                    self.triple_records[t_idx]["chunk_ids"].add(cid)

                if t_idx not in self.chunk_to_triples[cid]:
                    self.chunk_to_triples[cid].append(t_idx)

        self.entity_ids = list(self.graph.nodes())
        self.entity_texts = [
            f"{self.graph.nodes[eid]['name']} ({self.graph.nodes[eid]['type']})"
            for eid in self.entity_ids
        ]

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
            toks = _graph_tokenize(name)
            self.entity_tokens[eid] = toks
            for tok in toks:
                self.token_to_entities[tok].append(eid)

        self.token_to_entities = dict(self.token_to_entities)
        self.chunk_to_entities = dict(self.chunk_to_entities)
        self.chunk_to_triples = dict(self.chunk_to_triples)

        logger.info(
            f"Đã dựng Knowledge Graph: {len(self.chunks)} chunks -> "
            f"{len(self.entity_ids)} entities, {len(self.triple_records)} triples."
        )

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

    def _derive_vectors_from_chunks(self, chunk_vectors: Dict[str, List[float]]) -> bool:
        """Tổng hợp vector Entity/Triple trực tiếp từ vector chunk có sẵn trong ChromaDB."""
        valid_vecs: Dict[str, np.ndarray] = {}
        dim: Optional[int] = None
        for cid, vec in chunk_vectors.items():
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] == 0:
                continue
            if dim is None:
                dim = int(arr.shape[0])
            if arr.shape[0] == dim:
                valid_vecs[cid] = arr

        if not valid_vecs or dim is None:
            return False

        entity_vec_map: Dict[str, np.ndarray] = {}
        if self.entity_ids:
            ent_mat = np.zeros((len(self.entity_ids), dim), dtype=np.float32)
            for idx, eid in enumerate(self.entity_ids):
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
                cnt = 0
                for cid in tr["chunk_ids"]:
                    v = valid_vecs.get(cid)
                    if v is not None:
                        acc += v
                        cnt += 1
                h_vec = entity_vec_map.get(tr["head_id"])
                if h_vec is not None:
                    acc += h_vec
                    cnt += 1
                t_vec = entity_vec_map.get(tr["tail_id"])
                if t_vec is not None:
                    acc += t_vec
                    cnt += 1
                if cnt > 0:
                    acc /= float(cnt)
                    tr_mat[idx] = acc
            self.triple_vectors = self._l2_normalize_array(tr_mat)

        logger.info(
            f"Đã tổng hợp vector {dim} chiều cho {len(self.entity_ids)} entities và "
            f"{len(self.triple_records)} triples trực tiếp từ ChromaDB (0 lần chạy bge-m3 local)."
        )
        return True

    def _embed_graph_items(self, chunk_vectors: Optional[Dict[str, List[float]]] = None) -> None:
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

        if resolved_chunk_vecs and self._derive_vectors_from_chunks(resolved_chunk_vecs):
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
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.embedding_model = None
        self.vector_store = None
        self.llm_client = None

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
            state = pickle.load(stream)

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
    ) -> List[RetrievedContext]:
        """
        Truy xuất 0-LLM trên Graph Vector Index (tối ưu hóa IDF + Co-Activation + Adjacent Smoothing):
        1. Vector similarity (nhân Entity IDF & Relation Weight) + Lexical token matching (tăng trọng số thực thể hiếm).
        2. Lan truyền kích hoạt 1-2 hops theo trọng số quan hệ & IDF nút đích.
        3. Tổng hợp điểm chunk (Entity IDF + Edge Co-Activation Bonus + Bridging) và làm mượt cấu trúc liền kề.
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

        # 1a. Lexical token matching (nhanh < 0.1ms, nhân Entity IDF + boost thực thể hiếm khớp hoàn toàn)
        q_tokens = _graph_tokenize(clean_query)
        ent_hits: Dict[str, int] = defaultdict(int)
        if q_tokens:
            for q_tok in q_tokens:
                for eid in self.token_to_entities.get(q_tok, ()):
                    ent_hits[eid] += 1
            for eid, hit_count in ent_hits.items():
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

        # 1c. Vector search trên Triples (nhân trọng số quan hệ & endpoint IDF)
        if (
            q_vec_arr is not None
            and self.triple_vectors is not None
            and self.triple_vectors.shape[1] == q_vec_arr.shape[0]
        ):
            t_sims = self.triple_vectors @ q_vec_arr
            k_tr = min(max(seed_top_m, 1), len(t_sims))
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
                    for cid in tr["chunk_ids"]:
                        triple_chunk_bonus[cid] += 0.35 * tr_score * (0.5 + 0.25 * (h_idf + t_idf))

        if not entity_scores and not triple_chunk_bonus:
            return []

        total_chunks = len(self.chunks)
        max_hub_df = max(250, int(0.12 * total_chunks)) if total_chunks >= 50 else total_chunks + 1

        top_seeds = heapq.nlargest(seed_top_m, entity_scores.items(), key=lambda item: item[1])
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

        # 3. Tổng hợp điểm cho từng chunk (lọc bỏ fanout từ super-node & tính Co-Activation trên top pool)
        chunk_scores: Dict[str, float] = defaultdict(float)
        chunk_matched_entities: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        def _add_entity_chunks(eid: str, contrib: float, limit: Optional[int] = None) -> None:
            cids = self.graph.nodes[eid].get("chunks", ())
            if limit is not None and len(cids) > limit:
                cids = list(cids)[:limit]
            for cid in cids:
                chunk_scores[cid] += contrib
                chunk_matched_entities[cid].append((eid, contrib))

        for eid, e_score in activated_entities.items():
            if not self.graph.has_node(eid):
                continue
            node_chunks = self.graph.nodes[eid].get("chunks", ())
            if len(node_chunks) > max_hub_df:
                continue
            _add_entity_chunks(eid, e_score * entity_idf.get(eid, 1.0))

        for cid, bonus in triple_chunk_bonus.items():
            chunk_scores[cid] += bonus

        if not chunk_scores and activated_entities:
            best_eid, best_score = max(activated_entities.items(), key=lambda item: item[1])
            if self.graph.has_node(best_eid):
                _add_entity_chunks(best_eid, best_score * entity_idf.get(best_eid, 1.0), limit=max(1, top_k * 4))

        if not chunk_scores:
            return []

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
                    co_scores.append(0.45 * math.sqrt(h_score * t_score) * rel_w)
            co_bonus = sum(sorted(co_scores, reverse=True)[:3])
            matched_cnt = len(chunk_matched_entities.get(cid, ()))
            bridge_bonus = 0.25 * min(3, max(0, matched_cnt - 1))
            pre_smooth_scores[cid] = base_score + co_bonus + bridge_bonus

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

        final_scores: List[Tuple[str, float]] = []
        for cid, score in smoothed_scores.items():
            idx = self.chunk_id_to_idx.get(cid)
            if idx is None:
                continue
            chunk = self.chunks[idx]
            if not self._matches_filters(chunk, filters):
                continue
            final_scores.append((cid, score))

        if not final_scores:
            return []

        top_chunks = heapq.nlargest(top_k, final_scores, key=lambda item: item[1])
        max_score = top_chunks[0][1] if top_chunks[0][1] > 0 else 1.0

        results: List[RetrievedContext] = []
        for rank, (cid, raw_score) in enumerate(top_chunks, start=1):
            chunk = self.chunks[self.chunk_id_to_idx[cid]]
            norm_score = round(min(1.0, max(0.01, raw_score / max_score)), 4)

            matched_ents_sorted = sorted(
                chunk_matched_entities.get(cid, []),
                key=lambda x: x[1],
                reverse=True,
            )
            ent_names = [
                self.graph.nodes[eid]["name"]
                for eid, _ in matched_ents_sorted[:6]
                if self.graph.has_node(eid)
            ]

            triple_candidates: List[Tuple[float, str]] = []
            for t_idx in self.chunk_to_triples.get(cid, ()):
                tr = self.triple_records[t_idx]
                t_score = (
                    activated_entities.get(tr["head_id"], 0.0)
                    + activated_entities.get(tr["tail_id"], 0.0)
                    + activated_triple_indices.get(t_idx, 0.0)
                )
                if t_score > 0.0:
                    triple_candidates.append((t_score, tr["text"]))
            triple_candidates.sort(key=lambda x: x[0], reverse=True)
            top_triples = [t_txt for _, t_txt in triple_candidates[:6]]

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

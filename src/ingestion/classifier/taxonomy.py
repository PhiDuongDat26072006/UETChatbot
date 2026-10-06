"""Định nghĩa Taxonomy 7 danh mục phân loại URLs và các mẫu từ khóa regex."""
from __future__ import annotations

import enum
import re
import unicodedata

from .patterns_edu import EDU_PATTERNS
from .patterns_org import ORG_PATTERNS


class Category(str, enum.Enum):
    """Danh mục phân loại endpoints phục vụ RAG Chatbot."""
    DAO_TAO = "DAO_TAO"
    TUYEN_SINH = "TUYEN_SINH"
    QUY_CHE_BIEU_MAU = "QUY_CHE_BIEU_MAU"
    NGHIEN_CUU_HOP_TAC = "NGHIEN_CUU_HOP_TAC"
    CAN_BO_GIANG_VIEN = "CAN_BO_GIANG_VIEN"
    TIN_TUC_SU_KIEN = "TIN_TUC_SU_KIEN"
    KHAC = "KHAC"


CATEGORY_LABELS = {
    Category.DAO_TAO: "Đào tạo, CTĐT, môn học",
    Category.TUYEN_SINH: "Tuyển sinh ĐH, ThS, TS",
    Category.QUY_CHE_BIEU_MAU: "Quy chế, biểu mẫu, học phí",
    Category.NGHIEN_CUU_HOP_TAC: "NCKH, hội thảo, học bổng",
    Category.CAN_BO_GIANG_VIEN: "Cán bộ, giảng viên, bộ môn",
    Category.TIN_TUC_SU_KIEN: "Tin tức, sự kiện, thông báo",
    Category.KHAC: "Khác / Chưa phân loại",
}

PRIORITY_ORDER = [
    Category.TUYEN_SINH,
    Category.QUY_CHE_BIEU_MAU,
    Category.CAN_BO_GIANG_VIEN,
    Category.NGHIEN_CUU_HOP_TAC,
    Category.DAO_TAO,
    Category.TIN_TUC_SU_KIEN,
]

# Kết hợp mẫu từ các danh mục giáo dục và hoạt động
_RAW_PATTERNS = {**EDU_PATTERNS, **ORG_PATTERNS}
CATEGORY_PATTERNS = {
    Category[name]: pats for name, pats in _RAW_PATTERNS.items()
}

COMPILED_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in CATEGORY_PATTERNS.items()
}


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: chữ thường, giữ cả dạng có dấu và không dấu để khớp regex."""
    if not text:
        return ""
    text_lower = text.lower()
    text_no_accents = ''.join(
        c for c in unicodedata.normalize('NFD', text_lower)
        if unicodedata.category(c) != 'Mn'
    ).replace('đ', 'd').replace('Đ', 'd')
    return f"{text_lower} {text_no_accents}"

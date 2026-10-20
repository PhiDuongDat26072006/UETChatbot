"""Định nghĩa Taxonomy 7 danh mục phân loại URLs và các mẫu từ khóa regex."""
from __future__ import annotations

import enum
import re
import unicodedata


class Category(str, enum.Enum):
    """Danh mục phân loại endpoints phục vụ RAG Chatbot."""
    DAO_TAO = "DAO_TAO"
    TUYEN_SINH = "TUYEN_SINH"
    QUY_CHE_BIEU_MAU = "QUY_CHE_BIEU_MAU"
    NGHIEN_CUU_HOP_TAC = "NGHIEN_CUU_HOP_TAC"
    CAN_BO_GIANG_VIEN = "CAN_BO_GIANG_VIEN"
    TIN_TUC_SU_KIEN = "TIN_TUC_SU_KIEN"
    KHAC = "KHAC"


URL_MATCH_WEIGHT = 4
TITLE_MATCH_WEIGHT = 3

PRIORITY_ORDER: list[Category] = [
    Category.TUYEN_SINH,
    Category.QUY_CHE_BIEU_MAU,
    Category.CAN_BO_GIANG_VIEN,
    Category.NGHIEN_CUU_HOP_TAC,
    Category.DAO_TAO,
    Category.TIN_TUC_SU_KIEN,
]

CATEGORY_PATTERNS: dict[Category, list[str]] = {
    Category.TUYEN_SINH: [
        r'\btuyen[-_ ]?sinh\b',
        r'\bxet[-_ ]?tuyen\b',
        r'\bthi[-_ ]?tuyen\b',
        r'\bnhap[-_ ]?hoc\b',
        r'\bdiem[-_ ]?chuan\b',
        r'\bchi[-_ ]?tieu\b',
        r'\bde[-_ ]?an[-_ ]?tuyen[-_ ]?sinh\b',
        r'\badmission\b',
        r'\badmissions\b',
        r'\benrollment\b',
        r'\btan[-_ ]?sinh[-_ ]?vien\b',
        r'\btuyen[-_ ]?thanh\b',
    ],
    Category.QUY_CHE_BIEU_MAU: [
        r'\bquy[-_ ]?che\b',
        r'\bquy[-_ ]?dinh\b',
        r'\bbieu[-_ ]?mau\b',
        r'\bmau[-_ ]?don\b',
        r'\bdon[-_ ]?tu\b',
        r'\bhoc[-_ ]?phi\b',
        r'\bthu[-_ ]?tuc\b',
        r'\bquy[-_ ]?trinh\b',
        r'\bso[-_ ]?tay[-_ ]?sinh[-_ ]?vien\b',
        r'\bnoi[-_ ]?quy\b',
        r'\bdieu[-_ ]?le\b',
        r'\bchuan[-_ ]?dau[-_ ]?ra[-_ ]?ngoai[-_ ]?ngu\b',
        r'\bchuan[-_ ]?tieng[-_ ]?anh\b',
        r'\bvstep\b',
        r'\btoeic\b',
        r'\bielts\b',
        r'\bvan[-_ ]?ban[-_ ]?quy[-_ ]?pham\b',
        r'\bhoc[-_ ]?bong[-_ ]?khuyen[-_ ]?khich\b',
        r'\bche[-_ ]?do[-_ ]?chinh[-_ ]?sach\b',
    ],
    Category.DAO_TAO: [
        r'\bdao[-_ ]?tao\b',
        r'\bchuong[-_ ]?trinh[-_ ]?dao[-_ ]?tao\b',
        r'\bctdt\b',
        r'\bkhung[-_ ]?chuong[-_ ]?trinh\b',
        r'\bmon[-_ ]?hoc\b',
        r'\bhoc[-_ ]?phan\b',
        r'\btin[-_ ]?chi\b',
        r'\bthoi[-_ ]?khoa[-_ ]?bieu\b',
        r'\btkb\b',
        r'\blich[-_ ]?thi\b',
        r'\blich[-_ ]?hoc\b',
        r'\bchuan[-_ ]?dau[-_ ]?ra\b',
        r'\bdai[-_ ]?hoc\b',
        r'\bsau[-_ ]?dai[-_ ]?hoc\b',
        r'\bthac[-_ ]?si\b',
        r'\btien[-_ ]?si\b',
        r'\bcao[-_ ]?hoc\b',
        r'\bcu[-_ ]?nhan\b',
        r'\bky[-_ ]?su\b',
        r'\bkhoa[-_ ]?luan\b',
        r'\bkltn\b',
        r'\bdo[-_ ]?an\b',
        r'\bdatn\b',
        r'\btot[-_ ]?nghiep\b',
        r'\bhoc[-_ ]?vu\b',
        r'\bgiao[-_ ]?trinh\b',
        r'\bde[-_ ]?cuong\b',
        r'\bsyllabus\b',
        r'\bchuyen[-_ ]?nganh\b',
        r'\bnganh[-_ ]?hoc\b',
        r'\blop[-_ ]?hoc[-_ ]?phan\b',
        r'\bket[-_ ]?qua[-_ ]?hoc[-_ ]?tap\b',
    ],
    Category.CAN_BO_GIANG_VIEN: [
        r'\bcan[-_ ]?bo\b',
        r'\bgiang[-_ ]?vien\b',
        r'\bnhan[-_ ]?su\b',
        r'\bbo[-_ ]?mon\b',
        r'\bban[-_ ]?chu[-_ ]?nhiem\b',
        r'\bban[-_ ]?lanh[-_ ]?dao\b',
        r'\bly[-_ ]?lich[-_ ]?khoa[-_ ]?hoc\b',
        r'\bly[-_ ]?lich\b',
        r'\bcurriculum[-_ ]?vitae\b',
        r'\bfaculty\b',
        r'\bstaff\b',
        r'\blecturer\b',
        r'\bprofessor\b',
        r'\bgiao[-_ ]?su\b',
        r'\bpho[-_ ]?giao[-_ ]?su\b',
        r'\bchuyen[-_ ]?gia\b',
        r'\bdanh[-_ ]?sach[-_ ]?can[-_ ]?bo\b',
        r'\bdoi[-_ ]?ngu\b',
        r'\bphong[-_ ]?thi[-_ ]?nghiem\b',
    ],
    Category.NGHIEN_CUU_HOP_TAC: [
        r'\bnghien[-_ ]?cuu\b',
        r'\bnckh\b',
        r'\bsvnckh\b',
        r'\bkhoa[-_ ]?hoc[-_ ]?cong[-_ ]?nghe\b',
        r'\bkhcn\b',
        r'\bhoi[-_ ]?thao\b',
        r'\bhoi[-_ ]?nghi\b',
        r'\bconference\b',
        r'\bworkshop\b',
        r'\bseminar\b',
        r'\bsymposium\b',
        r'\bde[-_ ]?tai\b',
        r'\bdu[-_ ]?an\b',
        r'\bproject\b',
        r'\bcong[-_ ]?bo\b',
        r'\bbai[-_ ]?bao\b',
        r'\btap[-_ ]?chi\b',
        r'\bpublication\b',
        r'\bpatent\b',
        r'\bsang[-_ ]?che\b',
        r'\bhop[-_ ]?tac\b',
        r'\bdoi[-_ ]?ngoai\b',
        r'\binternational\b',
        r'\bquoc[-_ ]?te\b',
        r'\bhoc[-_ ]?bong\b',
        r'\bscholarship\b',
        r'\btrao[-_ ]?doi\b',
        r'\bexchange\b',
        r'\bdoanh[-_ ]?nghiep\b',
        r'\btai[-_ ]?tro\b',
        r'\bmou\b',
        r'\blab\b',
        r'\bpartner\b',
    ],
    Category.TIN_TUC_SU_KIEN: [
        r'\btin[-_ ]?tuc\b',
        r'\bnews\b',
        r'\bsu[-_ ]?kien\b',
        r'\bevent\b',
        r'\bevents\b',
        r'\bthong[-_ ]?bao\b',
        r'\bannouncement\b',
        r'\bnotice\b',
        r'\bhoat[-_ ]?dong\b',
        r'\bphong[-_ ]?trao\b',
        r'\bdoan[-_ ]?thanh[-_ ]?nien\b',
        r'\bhoi[-_ ]?sinh[-_ ]?vien\b',
        r'\bcuu[-_ ]?sinh[-_ ]?vien\b',
        r'\balumni\b',
        r'\bkhai[-_ ]?giang\b',
        r'\bbe[-_ ]?giang\b',
        r'\ble[-_ ]?tot[-_ ]?nghiep\b',
        r'\btrao[-_ ]?bang\b',
        r'\bgiai[-_ ]?thuong\b',
        r'\bvinh[-_ ]?danh\b',
        r'\btuyen[-_ ]?duong\b',
        r'\bcuoc[-_ ]?thi\b',
        r'\bhackathon\b',
        r'\bgiao[-_ ]?luu\b',
        r'\bgap[-_ ]?mat\b',
        r'\btoa[-_ ]?dam\b',
        r'\btuyen[-_ ]?dung\b',
        r'\bviec[-_ ]?lam\b',
    ],
}

COMPILED_PATTERNS: dict[Category, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in CATEGORY_PATTERNS.items()
}


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản phục vụ so khớp từ khóa.

    Chuyển về chữ thường và ghép đồng thời cả bản giữ dấu lẫn bản bỏ dấu tiếng Việt,
    giúp biểu thức regex khớp chính xác cả hai dạng.
    """
    if not text:
        return ""
    text_lower = text.lower()
    text_no_accents = ''.join(
        c for c in unicodedata.normalize('NFD', text_lower)
        if unicodedata.category(c) != 'Mn'
    ).replace('đ', 'd').replace('Đ', 'd')
    return f"{text_lower} {text_no_accents}"

"""Cấu hình các Khoa, Viện, domain và hàm resolve_unit chuẩn hóa."""
from __future__ import annotations

# Danh sách tên miền chính thức của các Khoa và Viện trực thuộc UET
FACULTY_TARGETS = [
    {
        "code": "FIT",
        "name": "Khoa Công nghệ thông tin",
        "url": "https://fit.uet.vnu.edu.vn",
        "domain": "fit.uet.vnu.edu.vn",
        "output_filename": "fit.uet.vnu.edu.vn.txt",
    },
    {
        "code": "FET",
        "name": "Khoa Điện tử Viễn thông",
        "url": "https://fet.uet.vnu.edu.vn",
        "domain": "fet.uet.vnu.edu.vn",
        "output_filename": "fet.uet.vnu.edu.vn.txt",
    },
    {
        "code": "FEPN",
        "name": "Khoa Vật lý kỹ thuật & Công nghệ Nano",
        "url": "https://fepn.uet.vnu.edu.vn",
        "domain": "fepn.uet.vnu.edu.vn",
        "output_filename": "fepn.uet.vnu.edu.vn.txt",
    },
    {
        "code": "FEMA",
        "name": "Khoa Cơ học kỹ thuật & Tự động hoá",
        "url": "https://fema.uet.vnu.edu.vn",
        "domain": "fema.uet.vnu.edu.vn",
        "output_filename": "fema.uet.vnu.edu.vn.txt",
    },
    {
        "code": "FAT",
        "name": "Khoa Công nghệ Nông nghiệp",
        "url": "https://fat.uet.vnu.edu.vn",
        "domain": "fat.uet.vnu.edu.vn",
        "output_filename": "fat.uet.vnu.edu.vn.txt",
    },
    {
        "code": "FCE",
        "name": "Khoa Công nghệ Xây dựng – Giao thông",
        "url": "https://fce.uet.vnu.edu.vn",
        "domain": "fce.uet.vnu.edu.vn",
        "output_filename": "fce.uet.vnu.edu.vn.txt",
    },
    {
        "code": "SAE",
        "name": "Viện Công nghệ Hàng không Vũ trụ",
        "url": "https://sae.uet.vnu.edu.vn",
        "domain": "sae.uet.vnu.edu.vn",
        "output_filename": "sae.uet.vnu.edu.vn.txt",
    },
    {
        "code": "IAI",
        "name": "Viện Trí tuệ nhân tạo",
        "url": "https://uet.vnu.edu.vn/vien-tri-tue-nhan-tao/",
        "domain": "iai.uet.vnu.edu.vn",
        "target_domain": "uet.vnu.edu.vn",
        "subpath": "vien-tri-tue-nhan-tao",
        "category_id": 146,
        "output_filename": "iai.uet.vnu.edu.vn.txt",
    },
]

TARGET_DOMAINS = [t["url"] for t in FACULTY_TARGETS]

# Danh sách tên thư mục dữ liệu cho từng đơn vị (dẫn xuất từ FACULTY_TARGETS)
DOMAIN_FOLDERS: list[dict[str, str]] = [
    {"code": t["code"], "name": t["name"], "folder": t["domain"]}
    for t in FACULTY_TARGETS
]
# Thêm UET chính (không nằm trong FACULTY_TARGETS)
DOMAIN_FOLDERS.append({"code": "UET", "name": "Trang chủ ĐH Công nghệ", "folder": "uet.edu.vn"})

# Bảng ánh xạ chuẩn hóa tên miền sang mã định danh đơn vị (Unit Code) viết hoa
DOMAIN_TO_UNIT_MAP: dict[str, str] = {
    "fit.uet.vnu.edu.vn": "FIT",
    "fet.uet.vnu.edu.vn": "FET",
    "fepn.uet.vnu.edu.vn": "FEPN",
    "fema.uet.vnu.edu.vn": "FEMA",
    "fat.uet.vnu.edu.vn": "FAT",
    "fce.uet.vnu.edu.vn": "FCE",
    "sae.uet.vnu.edu.vn": "SAE",
    "iai.uet.vnu.edu.vn": "IAI",
    "uet.vnu.edu.vn": "UET",
    "uet.edu.vn": "UET",
}


def resolve_unit(domain: str) -> str:
    """Trả về mã unit chuẩn hóa (FIT, FET, IAI, UET...). Mặc định trả về 'UET' nếu không khớp."""
    if not domain or not isinstance(domain, str):
        return "UET"
    clean_domain = domain.lower().strip()
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]
    return DOMAIN_TO_UNIT_MAP.get(clean_domain, "UET")

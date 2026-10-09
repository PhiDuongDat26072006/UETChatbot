"""Cấu hình các Khoa, Viện và hàm chuẩn hóa đơn vị (resolve_unit)."""
from __future__ import annotations

import posixpath
from typing import TypedDict
from urllib.parse import unquote, urlparse


class FacultyTarget(TypedDict):
    """A source's identity, real host, crawl seeds, and optional path scope."""

    faculty_id: str
    name: str
    domain: str
    start_urls: list[str]
    allowed_path_prefixes: list[str]


# Danh sách tên miền chính thức của các Khoa/Viện trực thuộc UET và Cổng chính
FACULTY_TARGETS: list[FacultyTarget] = [
    {
        "faculty_id": "FIT",
        "name": "Khoa Công nghệ thông tin",
        "start_urls": ["https://fit.uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "fit.uet.vnu.edu.vn",
    },
    {
        "faculty_id": "FET",
        "name": "Khoa Điện tử Viễn thông",
        "start_urls": ["https://fet.uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "fet.uet.vnu.edu.vn",
    },
    {
        "faculty_id": "FEPN",
        "name": "Khoa Vật lý kỹ thuật & Công nghệ Nano",
        "start_urls": ["https://fepn.uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "fepn.uet.vnu.edu.vn",
    },
    {
        "faculty_id": "FEMA",
        "name": "Khoa Cơ học kỹ thuật & Tự động hoá",
        "start_urls": ["https://fema.uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "fema.uet.vnu.edu.vn",
    },
    {
        "faculty_id": "FAT",
        "name": "Khoa Công nghệ Nông nghiệp",
        "start_urls": ["https://fat.uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "fat.uet.vnu.edu.vn",
    },
    {
        "faculty_id": "FCE",
        "name": "Khoa Công nghệ Xây dựng – Giao thông",
        "start_urls": ["https://fce.uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "fce.uet.vnu.edu.vn",
    },
    {
        "faculty_id": "SAE",
        "name": "Viện Công nghệ Hàng không Vũ trụ",
        "start_urls": ["https://sae.uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "sae.uet.vnu.edu.vn",
    },
    {
        "faculty_id": "IAI",
        "name": "Viện Trí tuệ nhân tạo",
        "start_urls": ["https://uet.vnu.edu.vn/vien-tri-tue-nhan-tao/"],
        "allowed_path_prefixes": ["/vien-tri-tue-nhan-tao/"],
        "domain": "uet.vnu.edu.vn",
    },
    {
        "faculty_id": "UET",
        "name": "Trường Đại học Công nghệ",
        "start_urls": ["https://uet.vnu.edu.vn"],
        "allowed_path_prefixes": [],
        "domain": "uet.vnu.edu.vn",
    },
]


def faculty_storage_name(faculty_id: str) -> str:
    """Derive established storage names independently of the request host."""
    return "uet.edu.vn" if faculty_id == "UET" else f"{faculty_id.lower()}.uet.vnu.edu.vn"


def find_faculty(source: str, source_url: str | None = None) -> FacultyTarget | None:
    """Resolve an ID, storage name, or host; URL paths disambiguate shared hosts.

    Explicit IDs and storage names take precedence. Unknown sources return None
    so callers can retain their own domain and storage conventions.
    """
    if not isinstance(source, str) or not source.strip():
        return None
    source = source.strip()
    for target in FACULTY_TARGETS:
        if source.upper() == target["faculty_id"] or source.lower().removeprefix("www.") == faculty_storage_name(target["faculty_id"]):
            return target
    parsed = urlparse(source_url or (source if "://" in source else f"https://{source}"))
    candidates = [target for target in FACULTY_TARGETS if url_in_scope(parsed.geturl(), target["domain"])]
    for target in candidates:
        if target["allowed_path_prefixes"] and url_in_scope(parsed.geturl(), target["domain"], target["allowed_path_prefixes"]):
            return target
    return next((target for target in candidates if not target["allowed_path_prefixes"]), None)


def resolve_unit(domain: str, source_url: str | None = None) -> str:
    """Normalize known source identities to faculty IDs, defaulting to UET."""
    target = find_faculty(domain, source_url)
    return target["faculty_id"] if target else "UET"


def url_in_scope(url: str, domain: str, prefixes: list[str] | tuple[str, ...] = ()) -> bool:
    """Match source hosts and path segments after decoding and resolving dot paths."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    domain = domain.lower().split(":")[0].removeprefix("www.")
    aliases = ("uet.vnu.edu.vn", "uet.edu.vn")
    if parsed.scheme not in ("http", "https"):
        return False
    if host != domain and not (host in aliases and domain in aliases):
        return False
    path = posixpath.normpath(unquote(parsed.path or "/"))
    return not prefixes or any(
        path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")
        for prefix in prefixes
    )

"""
src/graph/thesaurus_utils.py - Bộ chuẩn hóa thực thể học vụ UET.
Nạp academic_thesaurus.json và cung cấp:
  - Tra cứu O(1) alias -> canonical_name
  - Danh sách aliases cho Entity Matching
  - Thông tin category, description cho prompt trích xuất
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("graph.thesaurus")

THESAURUS_PATH = Path(__file__).resolve().parent / "academic_thesaurus.json"


def _resolve_thesaurus_path(
    path: Optional[Path | str] = None,
    config: Optional[dict] = None,
) -> Path:
    if path is not None:
        p = Path(path)
        return p if p.is_absolute() else (BASE_DIR / p)
    try:
        cfg = config if config is not None else load_yaml_config()
        rel = (cfg or {}).get("lightrag", {}).get("thesaurus_file")
        if rel:
            cand = Path(rel) if Path(rel).is_absolute() else (BASE_DIR / rel)
            if cand.exists():
                return cand
    except Exception:
        pass
    return THESAURUS_PATH

# Tập hợp thực thể chung chung bị cấm khi trích xuất node đồ thị (tránh tạo super-hub nhiễu)
FORBIDDEN_ENTITIES: Set[str] = {
    "sinh viên", "người học", "nhà trường", "trường",
    "năm học", "học kỳ", "quy định", "văn bản",
    "thời gian", "kết quả", "tài liệu", "điều khoản",
    "giáo viên", "giảng viên", "lớp học", "bài thi",
    "hội đồng", "ban", "phòng", "khoa",
}

# Mở rộng danh sách từ chung chung không dùng làm thực thể hạt giống khi phân tích câu hỏi UET
EXTENDED_FORBIDDEN_ENTITIES: Set[str] = set(FORBIDDEN_ENTITIES) | {
    "uet",
    "đhcn",
    "đại học công nghệ",
    "trường đại học công nghệ",
    "trường đh công nghệ",
    "trường đại học công nghệ - đhqghn",
    "trường đại học công nghệ - đhqghn (uet)",
    "đhqghn",
    "vnu",
    "đại học quốc gia hà nội",
    "đại học quốc gia hà nội (đhqghn)",
    "em",
    "mình",
    "bạn",
    "ngành",
    "môn",
    "môn học",
    "học phần",
    "điểm",
    "thông tin",
    "công nghệ",
    "quy chế",
    "thông báo",
    "hướng dẫn",
    "điều kiện",
    "yêu cầu",
    "bao nhiêu",
    "như thế nào",
    "năm",
}

# Bổ sung bí danh tương thích hai chiều với schema của GraphIndex (src/graph/graph.py)
_EXTRA_BUILTIN_ALIASES: Dict[str, str] = {
    "khoa công nghệ thông tin (fit)": "Khoa Công nghệ thông tin",
    "khoa điện tử viễn thông (fet)": "Khoa Điện tử Viễn thông",
    "khoa cơ học kỹ thuật và tự động hóa (fema)": "Khoa Cơ học kỹ thuật và Tự động hóa",
    "khoa vật lý kỹ thuật và công nghệ nano (fepn)": "Khoa Vật lý kỹ thuật và Công nghệ Nano",
    "khoa công nghệ nông nghiệp (fat)": "Khoa Công nghệ Nông nghiệp",
    "khoa công nghệ xây dựng - giao thông": "Khoa Công nghệ Xây dựng – Giao thông",
    "khoa công nghệ xây dựng - giao thông (fce)": "Khoa Công nghệ Xây dựng – Giao thông",
    "viện trí tuệ nhân tạo (iai)": "Viện Trí tuệ nhân tạo",
    "khoa hàng không vũ trụ (sae)": "Viện Công nghệ Hàng không Vũ trụ",
    "khoa hàng không vũ trụ": "Viện Công nghệ Hàng không Vũ trụ",
    "điểm trung bình chung (gpa)": "Điểm trung bình chung tích lũy",
    "điểm trung bình chung": "Điểm trung bình chung tích lũy",
    "điểm tbc": "Điểm trung bình chung tích lũy",
    "vstep.3-5": "Chứng chỉ VSTEP",
    "vstep 3-5": "Chứng chỉ VSTEP",
    "vnu-test": "Chứng chỉ VSTEP",
    "toefl itp": "Chứng chỉ TOEFL",
    "toefl ibt": "Chứng chỉ TOEFL",
    "ielts": "Chứng chỉ IELTS",
    "toeic": "Chứng chỉ TOEIC",
    "toefl": "Chứng chỉ TOEFL",
    "vstep": "Chứng chỉ VSTEP",
    "aptis": "Chứng chỉ Aptis",
    "jlpt": "Chứng chỉ tiếng Nhật JLPT",
}


def is_forbidden_entity(term: str, allow_school_names: bool = False) -> bool:
    """
    Kiểm tra xem một cụm từ có phải thực thể chung chung gây nhiễu đồ thị hay không.
    - allow_school_names=False (mặc định cho Query Seed): chặn cả 'UET', 'ĐHQGHN', 'sinh viên'...
    - allow_school_names=True (dành cho Graph Indexing): chỉ chặn danh từ chung ('sinh viên', 'năm học'...).
    """
    clean = re.sub(r"^\[[^\]]+\]\s*", "", (term or "").strip()).strip().lower()
    if len(clean) < 2 or clean.isdigit():
        return True
    target_set = FORBIDDEN_ENTITIES if allow_school_names else EXTENDED_FORBIDDEN_ENTITIES
    return clean in target_set


# Mã định danh các Khoa / Viện chuyên môn độc lập tại UET (dùng chống nhiễu liên khoa)
KNOWN_FACULTY_KEYS: Set[str] = {"fit", "fet", "fema", "fepn", "fat", "fce", "iai", "sae"}

_FACULTY_DETECT_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("fit", re.compile(r"\b(?:khoa\s+công\s+nghệ\s+thông\s+tin|khoa\s+cntt|fit)\b", re.IGNORECASE)),
    ("fet", re.compile(r"\b(?:khoa\s+điện\s+tử[\s\-–]+viễn\s+thông|khoa\s+đtvt|khoa\s+đt[\-\s]?vt|fet)\b", re.IGNORECASE)),
    ("fema", re.compile(r"\b(?:khoa\s+cơ\s+học\s+kỹ\s+thuật|cơ\s+học\s+kỹ\s+thuật\s+và\s+tự\s+động\s+hóa|fema)\b", re.IGNORECASE)),
    ("fepn", re.compile(r"\b(?:khoa\s+vật\s+lý\s+kỹ\s+thuật|vật\s+lý\s+kỹ\s+thuật\s+và\s+công\s+nghệ\s+nano|vlkt\s*&\s*cnnn|fepn)\b", re.IGNORECASE)),
    ("fat", re.compile(r"\b(?:khoa\s+công\s+nghệ\s+nông\s+nghiệp|fat)\b", re.IGNORECASE)),
    ("fce", re.compile(r"\b(?:khoa\s+công\s+nghệ\s+xây\s+dựng|xây\s+dựng[\s\-–]+giao\s+thông|fce)\b", re.IGNORECASE)),
    ("iai", re.compile(r"\b(?:viện\s+trí\s+tuệ\s+nhân\s+tạo|iai)\b", re.IGNORECASE)),
    ("sae", re.compile(r"\b(?:viện\s+công\s+nghệ\s+hàng\s+không\s+vũ\s+trụ|khoa\s+hàng\s+không\s+vũ\s+trụ|sae)\b", re.IGNORECASE)),
)


def detect_query_faculties(text: str) -> Set[str]:
    """
    Nhận diện tập mã Khoa/Viện chuyên môn được nêu đích danh trong câu hỏi hoặc đoạn tóm tắt.
    Chỉ khớp tên Khoa/Viện hoặc mã viết tắt chuẩn (FIT, FET, FEMA, FEPN, FAT, FCE, IAI, SAE),
    không bắt nhầm tên ngành chung chung (ví dụ 'Trí tuệ nhân tạo', 'Công nghệ sinh học').
    """
    if not text or not text.strip():
        return set()
    detected: Set[str] = set()
    for fac_key, rx in _FACULTY_DETECT_PATTERNS:
        if rx.search(text):
            detected.add(fac_key)
    return detected


class AcademicThesaurus:
    """
    Bộ từ điển chuẩn hóa thực thể học vụ UET - O(1) Lookup.

    Cung cấp:
      - normalize(term): Alias -> Canonical Name
      - get_aliases(canonical): Canonical Name -> Danh sách aliases
      - get_category(canonical): Lấy loại thực thể (ORGANIZATION, ACADEMIC_METRIC...)
      - get_description(canonical): Lấy mô tả thực thể
      - get_all_canonicals(): Toàn bộ tên chuẩn hóa trong từ điển
      - normalize_text(text): Chuẩn hóa toàn bộ aliases trong đoạn văn bản
    """

    def __init__(self, path: Optional[Path | str] = None, config: Optional[dict] = None):
        # alias (lowercase) -> canonical_name
        self._alias_to_canonical: Dict[str, str] = {}
        # canonical_name -> toàn bộ metadata entry
        self._canonical_metadata: Dict[str, dict] = {}
        # canonical_name -> danh sách aliases gốc (lowercase)
        self._canonical_to_aliases: Dict[str, List[str]] = {}

        resolved_path = _resolve_thesaurus_path(path=path, config=config)
        self._load(resolved_path)
        logger.info(
            f"Đã nạp Academic Thesaurus: {len(self._canonical_metadata)} thực thể chuẩn, "
            f"{len(self._alias_to_canonical)} aliases."
        )

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"Không tìm thấy từ điển tại: {path}. Bộ chuẩn hóa sẽ bỏ qua.")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("academic_thesaurus", [])
        canonical_lowers: Dict[str, str] = {}

        # Bước 1: Đăng ký toàn bộ tên chính tắc trước để bảo vệ khỏi bị alias mục khác ghi đè
        for item in items:
            canonical: str = item.get("canonical_name", "").strip()
            if not canonical:
                continue
            self._canonical_metadata[canonical] = item
            canon_low = canonical.lower()
            canonical_lowers[canon_low] = canonical
            self._alias_to_canonical[canon_low] = canonical

        # Bước 2: Đăng ký toàn bộ aliases (không cho phép ghi đè lên tên chính tắc của thực thể khác)
        for item in items:
            canonical = item.get("canonical_name", "").strip()
            if not canonical:
                continue
            aliases_lower: List[str] = []
            for alias in item.get("aliases", []):
                alias_clean = alias.strip().lower()
                if not alias_clean:
                    continue
                if alias_clean in canonical_lowers and canonical_lowers[alias_clean] != canonical:
                    logger.warning(
                        f"Bỏ qua alias trùng tên chính tắc khác: '{alias_clean}' "
                        f"(thuộc '{canonical}', trùng canonical '{canonical_lowers[alias_clean]}')."
                    )
                    continue
                self._alias_to_canonical[alias_clean] = canonical
                aliases_lower.append(alias_clean)

            self._canonical_to_aliases[canonical] = aliases_lower

        for extra_alias, target_canonical in _EXTRA_BUILTIN_ALIASES.items():
            if target_canonical in self._canonical_metadata and extra_alias not in self._alias_to_canonical:
                self._alias_to_canonical[extra_alias] = target_canonical
                self._canonical_to_aliases.setdefault(target_canonical, []).append(extra_alias)

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def normalize(self, term: str) -> str:
        """
        Chuẩn hóa một thuật ngữ về Canonical Name.
        Nếu không tìm thấy trong từ điển, trả về nguyên gốc term.

        Ví dụ:
            normalize("đrl") -> "Điểm rèn luyện"
            normalize("fit") -> "Khoa Công nghệ thông tin"
            normalize("từ không có trong từ điển") -> "từ không có trong từ điển"
        """
        if not term or not term.strip():
            return term
        return self._alias_to_canonical.get(term.strip().lower(), term)

    def normalize_all(self, terms: List[str]) -> List[str]:
        """Chuẩn hóa toàn bộ danh sách thuật ngữ."""
        return [self.normalize(t) for t in terms]

    def is_known(self, term: str) -> bool:
        """Kiểm tra xem term có nằm trong từ điển (kể cả alias) hay không."""
        return term.strip().lower() in self._alias_to_canonical

    def get_canonical(self, term: str) -> Optional[str]:
        """
        Lấy Canonical Name nếu có, None nếu không có trong từ điển.
        Khác normalize(): trả về None thay vì giữ nguyên term.
        """
        return self._alias_to_canonical.get(term.strip().lower())

    def get_aliases(self, canonical: str) -> List[str]:
        """Lấy danh sách aliases (lowercase) của một Canonical Name."""
        return self._canonical_to_aliases.get(canonical, [])

    def get_category(self, canonical: str) -> Optional[str]:
        """Lấy category của Canonical Name (ORGANIZATION, ACADEMIC_METRIC, ...)."""
        entry = self._canonical_metadata.get(canonical)
        return entry.get("category") if entry else None

    def get_description(self, canonical: str) -> Optional[str]:
        """Lấy mô tả ngắn của Canonical Name."""
        entry = self._canonical_metadata.get(canonical)
        return entry.get("description") if entry else None

    def get_all_canonicals(self) -> List[str]:
        """Trả về toàn bộ danh sách Canonical Names trong từ điển."""
        return list(self._canonical_metadata.keys())

    def get_canonicals_by_category(self, category: str) -> List[str]:
        """Lấy tất cả Canonical Names thuộc một category cụ thể."""
        return [
            name
            for name, meta in self._canonical_metadata.items()
            if meta.get("category") == category
        ]

    def normalize_entity_name(self, raw_name: str) -> Tuple[str, bool]:
        """
        Chuẩn hóa tên thực thể do LLM trích xuất ra.
        Trả về (canonical_name, was_normalized):
          - was_normalized=True: đã được ánh xạ về tên chuẩn hóa trong từ điển
          - was_normalized=False: giữ nguyên (không có trong từ điển)

        Dùng trong pipeline xây dựng Knowledge Graph để quyết định
        có cần thực hiện Vector Blocking (Embedding Similarity) hay không.
        """
        canonical = self.get_canonical(raw_name)
        if canonical:
            return canonical, True
        return raw_name, False

    def find_entities_in_text(self, text: str, max_entities: int = 15) -> List[str]:
        """
        Quét nhanh nội dung văn bản để nhận diện các thực thể có xuất hiện (qua canonical hoặc alias).
        Sử dụng thuật toán N-gram sliding window (từ 5 từ xuống 1 từ) trên bảng băm O(1).

        Args:
            text: Văn bản cần quét (ví dụ: nội dung của 1 Semantic Unit).
            max_entities: Số lượng thực thể chuẩn tối đa cần lấy ra.

        Returns:
            Danh sách các canonical_name xuất hiện trong text, giữ thứ tự xuất hiện.
        """
        if not text or not text.strip():
            return []

        clean_text = text.lower()
        # Tách từ giữ lại dấu gạch ngang, dấu cộng (ví dụ: c+, vnu-uet)
        words = re.findall(r'[\w\+\-\./]+', clean_text)
        if not words:
            return []

        found: List[str] = []
        found_set: Set[str] = set()

        max_n = min(5, len(words))
        for n in range(max_n, 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                canonical = self._alias_to_canonical.get(phrase)
                if canonical and canonical not in found_set:
                    found_set.add(canonical)
                    found.append(canonical)
                    if len(found) >= max_entities:
                        return found

        return found

    def build_dynamic_entity_context(self, text: str, max_entities: int = 12) -> str:
        """
        Tạo ngữ cảnh thực thể ĐỘNG (Dynamic Contextual Filtering):
        Chỉ trích xuất và gợi ý các thực thể THỰC SỰ XUẤT HIỆN trong đoạn văn bản của Semantic Unit.
        Tiết kiệm ~95% token so với việc gửi toàn bộ danh sách 169 thực thể.
        """
        matched = self.find_entities_in_text(text, max_entities=max_entities)
        if not matched:
            return (
                "=== QUY TẮC CHUẨN HÓA THỰC THỂ ===\n"
                "Sử dụng tên gọi thuật ngữ chuẩn xác theo ngữ cảnh văn bản. "
                "Hệ thống sẽ tự động chuẩn hóa các bí danh/từ viết tắt sau khi trích xuất."
            )

        lines = [
            "=== DANH SÁCH THỰC THỂ CHUẨN CÓ MẶT TRONG ĐOẠN NÀY (BẮT BUỘC DÙNG ĐÚNG CÁC TÊN NÀY NẾU CÓ TRÍCH XUẤT): ==="
        ]
        for name in matched:
            cat = self.get_category(name) or "THỰC THỂ"
            lines.append(f"• [{cat}] {name}")

        return "\n".join(lines)

    def build_pass1_entity_context(self) -> str:
        """
        Ngữ cảnh thực thể rút gọn dành riêng cho Chặng 1 (Document-level):
        Chỉ gửi danh sách các đơn vị tổ chức (ORGANIZATION) và hệ đào tạo (PROGRAM_AND_DEGREE),
        giúp xác định cơ quan ban hành và phạm vi áp dụng mà không tốn token cho các điều kiện chi tiết.
        """
        lines = ["=== DANH SÁCH ĐƠN VỊ & CƠ QUAN CHUẨN HÓA (CHẶNG 1) ==="]
        for cat_key in ["ORGANIZATION", "PROGRAM_AND_DEGREE"]:
            members = self.get_canonicals_by_category(cat_key)
            if members:
                lines.append(f"• {' | '.join(members[:15])}")
        return "\n".join(lines)

    def build_entity_context_for_prompt(self) -> str:
        """
        Tạo danh sách các thực thể chuẩn để nhúng vào Prompt trích xuất LightRAG.
        Dùng như một 'schema hint' để LLM ưu tiên dùng tên chuẩn khi trích xuất.
        """
        lines = ["=== DANH SÁCH THỰC THỂ CHUẨN HÓA UET (ƯU TIÊN DÙNG CÁC TÊN NÀY) ==="]
        categories = {
            "ORGANIZATION": "Đơn vị tổ chức",
            "ACADEMIC_METRIC": "Chỉ số học tập",
            "ACADEMIC_STATUS": "Trạng thái học vụ",
            "PROCEDURE": "Quy trình thủ tục",
            "CERTIFICATE_AND_STANDARD": "Chứng chỉ & Chuẩn đầu ra",
            "SCHOLARSHIP_AND_POLICY": "Học bổng & Chính sách",
            "PROGRAM_AND_DEGREE": "Chương trình & Văn bằng",
        }
        for cat_key, cat_label in categories.items():
            members = self.get_canonicals_by_category(cat_key)
            if members:
                lines.append(f"\n[{cat_label}]: {' | '.join(members)}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._canonical_metadata)

    def __repr__(self) -> str:
        return (
            f"AcademicThesaurus("
            f"entities={len(self._canonical_metadata)}, "
            f"aliases={len(self._alias_to_canonical)})"
        )


# Singleton toàn cục để dùng chung trong toàn hệ thống
_global_thesaurus: Optional[AcademicThesaurus] = None


def get_thesaurus() -> AcademicThesaurus:
    """Lấy instance singleton của AcademicThesaurus (nạp 1 lần, dùng mãi mãi)."""
    global _global_thesaurus
    if _global_thesaurus is None:
        _global_thesaurus = AcademicThesaurus()
    return _global_thesaurus


if __name__ == "__main__":
    # Test nhanh
    t = get_thesaurus()
    print(repr(t))
    print()

    # Test normalize
    test_cases = [
        "đrl", "drl", "CPA", "clc", "fit", "kkht", "sv5t",
        "vstep b1", "khóa luận tốt nghiệp", "từ không trong từ điển"
    ]
    print("=== Kiểm tra normalize() ===")
    for term in test_cases:
        normalized = t.normalize(term)
        status = "✅ Chuẩn hóa" if t.is_known(term) else "⚪ Giữ nguyên"
        print(f"  {status}: '{term}' -> '{normalized}'")

    print("\n=== Danh sách thực thể chuẩn hóa trong Prompt ===")
    print(t.build_entity_context_for_prompt())

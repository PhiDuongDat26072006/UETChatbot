"""
src/graph/adaptive_extractor.py - Bộ gom cụm ngữ nghĩa thích ứng (Adaptive Semantic Extractor).

Thay thế cách cắt chunk vụn 800 ký tự bằng cách gom cụm theo đơn vị ngữ nghĩa tự nhiên:
  - LEGAL_ARTICLE:    Văn bản pháp quy -> gom theo từng ĐIỀU (trọn vẹn tất cả các Khoản)
  - HEADING_SECTION:  Bài viết/Tin tức -> gom theo Heading Markdown (#, ##, ###)
  - TABLE_UNIT:       Bảng biểu -> đóng gói nguyên bảng thành 1 node meta
  - PARAGRAPH_UNIT:   Văn bản tự sự -> gom theo đoạn văn bản hoàn chỉnh (\n\n)

Mục tiêu: Giảm từ 13.598 chunks vụn xuống ~800-1.200 Semantic Units toàn vẹn ngữ nghĩa,
tiết kiệm >90% lượt gọi LLM API khi nạp đồ thị LightRAG.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import ProcessedData
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("graph.adaptive_extractor")

# ---------------------------------------------------------------------------
# CÁC LOẠI SEMANTIC UNIT
# ---------------------------------------------------------------------------
UNIT_TYPE_LEGAL    = "LEGAL_ARTICLE"     # Văn bản pháp quy: Điều/Khoản
UNIT_TYPE_HEADING  = "HEADING_SECTION"   # Bài viết/Giới thiệu: theo Heading
UNIT_TYPE_TABLE    = "TABLE_UNIT"        # Bảng biểu
UNIT_TYPE_PARA     = "PARAGRAPH_UNIT"    # Văn bản tự sự: theo đoạn văn

# Giới hạn kích thước Semantic Unit gửi vào LLM (tokens xấp xỉ ký tự / 2.5)
MAX_UNIT_CHARS = 3000     # ~1.200 tokens - đủ chứa 1 Điều hoàn chỉnh
MIN_UNIT_CHARS = 50       # Bỏ qua unit quá ngắn, chỉ chứa tiêu đề trống


@dataclass
class SemanticUnit:
    """
    Đơn vị ngữ nghĩa trọn vẹn - đầu vào chuẩn cho LightRAG Entity Extractor.
    Mỗi SemanticUnit tương ứng với 1 lần gọi LLM để trích xuất thực thể.
    """
    unit_id: str                        # Hash định danh duy nhất
    document_id: str                    # ID của ProcessedData cha
    document_title: str                 # Tiêu đề văn bản gốc
    source_uri: str                     # URL/Path gốc của tài liệu
    breadcrumb: str                     # Đường dẫn phân cấp: "Chương I > Điều 5: ..."
    namespaced_id: str                  # Định danh Namespace: "[QĐ-5115] Điều 5: ..."
    unit_type: str                      # LEGAL_ARTICLE / HEADING_SECTION / TABLE_UNIT / PARAGRAPH_UNIT
    content: str                        # Toàn bộ nội dung của đơn vị ngữ nghĩa
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_llm_prompt_context(self) -> str:
        """Tạo chuỗi ngữ cảnh chuẩn để nạp vào Prompt trích xuất thực thể."""
        return (
            f"=== THÔNG TIN VĂN BẢN ===\n"
            f"Văn bản gốc: {self.document_title}\n"
            f"Vị trí trong văn bản: {self.breadcrumb}\n"
            f"Định danh: {self.namespaced_id}\n"
            f"Loại nội dung: {self.unit_type}\n\n"
            f"=== NỘI DUNG ===\n"
            f"{self.content}"
        )


# ---------------------------------------------------------------------------
# BỘ NHẬN DẠNG LOẠI VĂN BẢN
# ---------------------------------------------------------------------------

class DocumentTypeDetector:
    """Nhận dạng loại tài liệu để chọn chiến lược gom cụm phù hợp."""

    # Regex nhận diện văn bản pháp quy (Điều/Khoản/Chương)
    _LEGAL_PATTERN = re.compile(
        r'(?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+|(?:Điều|ĐIỀU)\s+\d+[\.\:\s]',
        re.IGNORECASE,
    )
    # Regex nhận diện bảng Markdown (ít nhất 3 cột, có dấu |)
    _TABLE_PATTERN = re.compile(r'^\|.+\|.+\|', re.MULTILINE)
    # Regex nhận diện Heading Markdown
    _HEADING_PATTERN = re.compile(r'^#{1,4}\s+\S', re.MULTILINE)

    @classmethod
    def detect(cls, doc: ProcessedData) -> str:
        """
        Phân loại tài liệu:
          - LEGAL: Có nhiều Điều/Chương -> LEGAL_ARTICLE mode
          - TABLE: Phần lớn là bảng Markdown -> TABLE_UNIT mode
          - HEADING: Có Heading Markdown rõ ràng -> HEADING_SECTION mode
          - PARAGRAPH: Văn bản tự sự thông thường -> PARAGRAPH_UNIT mode
        """
        content = doc.content

        # 1. Ưu tiên nhận dạng văn bản pháp quy (Điều/Khoản)
        legal_hits = len(cls._LEGAL_PATTERN.findall(content))
        if legal_hits >= 3:
            return "LEGAL"

        # 2. Nhận dạng bảng biểu chiếm đa số
        table_hits = len(cls._TABLE_PATTERN.findall(content))
        total_lines = max(1, content.count('\n'))
        if table_hits / total_lines > 0.3:
            return "TABLE"

        # 3. Nhận dạng bài viết có Heading Markdown
        heading_hits = len(cls._HEADING_PATTERN.findall(content))
        if heading_hits >= 2:
            return "HEADING"

        # 4. Mặc định: văn bản tự sự
        return "PARAGRAPH"


# ---------------------------------------------------------------------------
# CÁC CHIẾN LƯỢC GOM CỤM THEO LOẠI VĂN BẢN
# ---------------------------------------------------------------------------

def _split_oversized_text(text: str, max_chars: int) -> List[str]:
    """
    Chia văn bản vượt quá max_chars thành các phần nhỏ bảo toàn ranh giới đoạn (\\n\\n) hoặc dòng (\\n).
    """
    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    raw_blocks = [p.strip() for p in clean.split("\n\n") if p.strip()]
    atomic_parts: List[str] = []
    for block in raw_blocks:
        if len(block) <= max_chars:
            atomic_parts.append(block)
        else:
            line_buf = ""
            for ln in block.splitlines():
                ln_s = ln.strip()
                if not ln_s:
                    continue
                if len(line_buf) + len(ln_s) + 1 > max_chars and line_buf:
                    atomic_parts.append(line_buf)
                    line_buf = ln_s[:max_chars]
                else:
                    line_buf = f"{line_buf}\n{ln_s}".strip() if line_buf else ln_s[:max_chars]
            if line_buf:
                atomic_parts.append(line_buf)

    chunks: List[str] = []
    buf = ""
    for part in atomic_parts:
        if len(buf) + len(part) + 2 > max_chars and buf:
            chunks.append(buf)
            buf = part
        else:
            buf = f"{buf}\n\n{part}".strip() if buf else part
    if buf:
        chunks.append(buf)
    return chunks


class LegalArticleExtractor:
    """
    Chiến lược gom cụm văn bản pháp quy: Gom trọn vẹn từng ĐIỀU.
    Tái sử dụng logic _split_into_sections() từ UETChunker đã có sẵn.
    """

    def __init__(self, max_unit_chars: int = MAX_UNIT_CHARS, min_unit_chars: int = MIN_UNIT_CHARS):
        self.max_unit_chars = max(200, int(max_unit_chars))
        self.min_unit_chars = max(1, int(min_unit_chars))

    @staticmethod
    def _extract_heading(line: str) -> Optional[Tuple[int, str]]:
        """Nhận diện tiêu đề pháp quy. Trả về (level, clean_title) hoặc None."""
        clean = line.strip()
        if not clean:
            return None
        # Markdown headings
        m = re.match(r'^(#{1,4})\s+(.+)$', clean)
        if m:
            return len(m.group(1)), m.group(2).strip()
        # Chương
        m = re.match(r'^(Chương\s+[IVXLCDM\d]+[\s\S]*?)$', clean, re.IGNORECASE)
        if m:
            return 1, m.group(1).strip()
        # Mục
        m = re.match(r'^(Mục\s+\d+[\s\S]*?)$', clean, re.IGNORECASE)
        if m:
            return 2, m.group(1).strip()
        # Điều
        m = re.match(r'^(Điều\s+\d+[\.\:\s][\s\S]*?)$', clean, re.IGNORECASE)
        if m:
            return 3, m.group(1).strip()
        return None

    def extract_sections(
        self, doc: ProcessedData
    ) -> List[Tuple[List[str], str]]:
        """
        Chia văn bản thành các section theo phân cấp [Tên tài liệu > Chương > Điều].
        Trả về danh sách (breadcrumbs: List[str], content: str).
        """
        lines = doc.content.splitlines()
        current_headers: Dict[int, str] = {}
        current_lines: List[str] = []
        sections: List[Tuple[List[str], str]] = []

        for line in lines:
            h = self._extract_heading(line)
            if h:
                level, title = h
                # Flush section hiện tại
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    is_only_heading = (
                        len(sec_text.splitlines()) <= 1
                        and self._extract_heading(sec_text) is not None
                    )
                    if sec_text and not is_only_heading:
                        crumbs = [doc.title] if doc.title else []
                        for lvl in sorted(current_headers.keys()):
                            crumbs.append(current_headers[lvl])
                        sections.append((crumbs, sec_text))
                    current_lines = []
                # Xóa cấp sâu hơn hoặc bằng cấp hiện tại
                for k in list(current_headers.keys()):
                    if k >= level:
                        del current_headers[k]
                current_headers[level] = title
            current_lines.append(line)

        # Flush cuối
        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                crumbs = [doc.title] if doc.title else []
                for lvl in sorted(current_headers.keys()):
                    crumbs.append(current_headers[lvl])
                sections.append((crumbs, sec_text))

        if not sections and doc.content.strip():
            sections.append(([doc.title] if doc.title else [], doc.content.strip()))

        return sections

    def build_units(self, doc: ProcessedData) -> List[SemanticUnit]:
        sections = self.extract_sections(doc)
        units: List[SemanticUnit] = []

        for crumbs, text in sections:
            if len(text.strip()) < self.min_unit_chars:
                continue

            if len(text) > self.max_unit_chars:
                sub_chunks = _split_oversized_text(text, self.max_unit_chars)
                for sub_idx, chunk_txt in enumerate(sub_chunks, 1):
                    if len(chunk_txt) < self.min_unit_chars:
                        continue
                    sub_crumbs = crumbs + [f"(phần {sub_idx})"] if len(sub_chunks) > 1 else crumbs
                    units.append(
                        _make_unit(doc, sub_crumbs, chunk_txt, UNIT_TYPE_LEGAL, len(units))
                    )
            else:
                units.append(_make_unit(doc, crumbs, text, UNIT_TYPE_LEGAL, len(units)))

        return units


class HeadingSectionExtractor:
    """
    Chiến lược gom cụm bài viết giới thiệu/tin tức: Gom theo Heading Markdown.
    Bảo toàn trọn vẹn danh sách nhân sự, tiêu chí, nội dung mục và cây phân cấp H1 > H2 > H3.
    """

    _HEADING_RE = re.compile(r'^(#{1,4})\s+(.+)$')

    def __init__(self, max_unit_chars: int = MAX_UNIT_CHARS, min_unit_chars: int = MIN_UNIT_CHARS):
        self.max_unit_chars = max(200, int(max_unit_chars))
        self.min_unit_chars = max(1, int(min_unit_chars))

    def build_units(self, doc: ProcessedData) -> List[SemanticUnit]:
        lines = doc.content.splitlines()
        sections: List[Tuple[List[str], str]] = []
        current_headers: Dict[int, str] = {}
        current_lines: List[str] = []

        def _current_crumbs() -> List[str]:
            crumbs = [doc.title] if doc.title else []
            for lvl in sorted(current_headers.keys()):
                crumbs.append(current_headers[lvl])
            return crumbs

        def flush(content_lines: List[str]):
            text = "\n".join(content_lines).strip()
            if len(text) >= self.min_unit_chars:
                sections.append((_current_crumbs(), text))

        for line in lines:
            m = self._HEADING_RE.match(line.strip())
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                flush(current_lines)
                current_lines = [line]
                # Chỉ xóa các cấp sâu hơn hoặc bằng cấp hiện tại, giữ nguyên Heading cha cấp cao hơn
                for k in list(current_headers.keys()):
                    if k >= level:
                        del current_headers[k]
                current_headers[level] = title
            else:
                current_lines.append(line)

        flush(current_lines)

        # Build SemanticUnits từ sections
        units: List[SemanticUnit] = []
        for crumbs, text in sections:
            if len(text) > self.max_unit_chars:
                sub_chunks = _split_oversized_text(text, self.max_unit_chars)
                for sub_idx, chunk_txt in enumerate(sub_chunks, 1):
                    if len(chunk_txt) < self.min_unit_chars:
                        continue
                    sub_crumbs = crumbs + [f"(phần {sub_idx})"] if len(sub_chunks) > 1 else crumbs
                    units.append(_make_unit(doc, sub_crumbs, chunk_txt, UNIT_TYPE_HEADING, len(units)))
            else:
                units.append(_make_unit(doc, crumbs, text, UNIT_TYPE_HEADING, len(units)))

        return units


class TableUnitExtractor:
    """
    Chiến lược gom cụm bảng biểu: Đóng gói bảng kèm ngữ cảnh cột, tự động cắt theo nhóm dòng
    (giữ nguyên dòng tiêu đề cột) nếu bảng vượt quá max_unit_chars.
    """

    _TABLE_ROW_RE = re.compile(r'^\|.+\|')
    _TABLE_SEP_RE = re.compile(r'^\|\s*[:\-]+')

    def __init__(self, max_unit_chars: int = MAX_UNIT_CHARS, min_unit_chars: int = MIN_UNIT_CHARS):
        self.max_unit_chars = max(200, int(max_unit_chars))
        self.min_unit_chars = max(1, int(min_unit_chars))

    def build_units(self, doc: ProcessedData) -> List[SemanticUnit]:
        lines = doc.content.splitlines()
        units: List[SemanticUnit] = []
        buffer_pre: List[str] = []     # Nội dung trước bảng (tiêu đề, mô tả)
        buffer_table: List[str] = []   # Các dòng thuộc bảng
        in_table = False

        def flush_table(pre: List[str], table: List[str]):
            pre_text = "\n".join(pre).strip()
            table_text = "\n".join(table).strip()
            if not table_text:
                return

            # Lấy tên bảng từ dòng trước (tiêu đề hoặc văn bản giới thiệu)
            table_heading = "Bảng dữ liệu"
            for line in reversed(pre):
                stripped = line.strip()
                if stripped:
                    table_heading = stripped.lstrip("#").strip()[:160]
                    break

            base_crumbs = [doc.title, table_heading] if doc.title else [table_heading]
            combined = (pre_text + "\n\n" + table_text).strip() if pre_text else table_text
            if len(combined) <= self.max_unit_chars or len(table) <= 2:
                units.append(_make_unit(doc, base_crumbs, combined, UNIT_TYPE_TABLE, len(units)))
                return

            # Bảng quá dài: giữ lại tiêu đề bảng + dòng header cột Markdown cho từng lát cắt dòng
            header_cnt = 2 if (len(table) >= 2 and self._TABLE_SEP_RE.match(table[1].strip())) else 1
            header_block = "\n".join(table[:header_cnt])
            data_rows = table[header_cnt:]
            short_pre = pre_text[-400:].strip() if len(pre_text) > 400 else pre_text
            prefix = f"{short_pre}\n\n{header_block}" if short_pre else header_block
            row_budget = max(200, self.max_unit_chars - len(prefix) - 2)

            slices: List[str] = []
            row_buf: List[str] = []
            curr_chars = 0
            for r in data_rows:
                if row_buf and curr_chars + len(r) + 1 > row_budget:
                    slices.append(prefix + "\n" + "\n".join(row_buf))
                    row_buf = [r]
                    curr_chars = len(r)
                else:
                    row_buf.append(r)
                    curr_chars += len(r) + 1
            if row_buf:
                slices.append(prefix + "\n" + "\n".join(row_buf))

            for s_idx, slice_txt in enumerate(slices, 1):
                sub_crumbs = base_crumbs + [f"(phần {s_idx})"] if len(slices) > 1 else base_crumbs
                units.append(_make_unit(doc, sub_crumbs, slice_txt, UNIT_TYPE_TABLE, len(units)))

        for line in lines:
            if self._TABLE_ROW_RE.match(line.strip()):
                if not in_table:
                    in_table = True
                buffer_table.append(line)
            else:
                if in_table:
                    flush_table(buffer_pre, buffer_table)
                    buffer_pre = []
                    buffer_table = []
                    in_table = False
                buffer_pre.append(line)

        # Flush cuối
        if in_table and buffer_table:
            flush_table(buffer_pre, buffer_table)
        elif buffer_pre:
            remaining = "\n".join(buffer_pre).strip()
            if len(remaining) >= self.min_unit_chars:
                crumbs = [doc.title] if doc.title else []
                for chunk_txt in _split_oversized_text(remaining, self.max_unit_chars):
                    if len(chunk_txt) >= self.min_unit_chars:
                        units.append(_make_unit(doc, crumbs, chunk_txt, UNIT_TYPE_PARA, len(units)))

        return units


class ParagraphUnitExtractor:
    """
    Chiến lược gom cụm văn bản tự sự: Gom theo đoạn văn bản hoàn chỉnh (\\n\\n).
    Đảm bảo không cắt đôi câu hoặc danh sách và không vượt quá max_unit_chars.
    """

    def __init__(self, max_unit_chars: int = MAX_UNIT_CHARS, min_unit_chars: int = MIN_UNIT_CHARS):
        self.max_unit_chars = max(200, int(max_unit_chars))
        self.min_unit_chars = max(1, int(min_unit_chars))

    def build_units(self, doc: ProcessedData) -> List[SemanticUnit]:
        units: List[SemanticUnit] = []
        crumbs = [doc.title] if doc.title else []
        sub_chunks = _split_oversized_text(doc.content, self.max_unit_chars)
        for sub_idx, chunk_txt in enumerate(sub_chunks, 1):
            if len(chunk_txt) >= self.min_unit_chars:
                sub_crumbs = crumbs + [f"(phần {sub_idx})"] if len(sub_chunks) > 1 else crumbs
                units.append(_make_unit(doc, sub_crumbs, chunk_txt, UNIT_TYPE_PARA, len(units)))
        return units


# ---------------------------------------------------------------------------
# HELPER: Tạo SemanticUnit chuẩn hóa
# ---------------------------------------------------------------------------

def _make_unit(
    doc: ProcessedData,
    breadcrumbs: List[str],
    content: str,
    unit_type: str,
    idx: int,
) -> SemanticUnit:
    """Factory tạo SemanticUnit với ID hash và Namespace chuẩn."""
    crumb_str = " > ".join(breadcrumbs) if breadcrumbs else doc.title or "Tài liệu UET"

    # Tên ngắn gọn nhất (phần cuối breadcrumb)
    article_name = breadcrumbs[-1] if breadcrumbs else doc.title or f"Phần {idx + 1}"

    # Tên Namespace: [Tên tài liệu ngắn] Điều/Phần X
    doc_short = (doc.title or "UET")[:60]
    namespaced_id = f"[{doc_short}] {article_name}"

    # Hash định danh duy nhất từ nội dung
    unit_hash = hashlib.md5(
        f"{doc.id}::{crumb_str}::{idx}".encode("utf-8")
    ).hexdigest()[:12]
    unit_id = f"{unit_type.lower()}_{unit_hash}"

    source_uri = doc.metadata.get("source_uri", doc.metadata.get("url", ""))

    return SemanticUnit(
        unit_id=unit_id,
        document_id=doc.id,
        document_title=doc.title or "Tài liệu UET",
        source_uri=source_uri,
        breadcrumb=crumb_str,
        namespaced_id=namespaced_id,
        unit_type=unit_type,
        content=content,
        metadata={
            "site": doc.metadata.get("site", ""),
            "faculty": doc.metadata.get("faculty", ""),
            "published_at": doc.metadata.get("published_at", ""),
            "resource_type": doc.metadata.get("resource_type", ""),
            "unit_index": idx,
        },
    )


# ---------------------------------------------------------------------------
# BỘ ĐIỀU PHỐI CHÍNH: AdaptiveSemanticExtractor
# ---------------------------------------------------------------------------

class AdaptiveSemanticExtractor:
    """
    Bộ điều phối chính: Tự động nhận diện loại tài liệu và
    áp dụng chiến lược gom cụm ngữ nghĩa tương ứng.

    Giảm từ 13.598 chunks nhỏ xuống ~800-1.200 Semantic Units trọn vẹn,
    phục vụ pipeline trích xuất đồ thị tri thức LightRAG.

    Cách dùng:
        extractor = AdaptiveSemanticExtractor()
        units = extractor.extract(doc)
        # units là List[SemanticUnit]
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_unit_chars: Optional[int] = None,
        min_unit_chars: Optional[int] = None,
    ):
        ext_cfg: Dict[str, Any] = {}
        try:
            raw_cfg = config if config is not None else load_yaml_config()
            ext_cfg = (raw_cfg or {}).get("lightrag", {}).get("extractor", {}) or {}
        except Exception:
            ext_cfg = {}

        self.max_unit_chars = int(
            max_unit_chars if max_unit_chars is not None else ext_cfg.get("max_unit_chars", MAX_UNIT_CHARS)
        )
        self.min_unit_chars = int(
            min_unit_chars if min_unit_chars is not None else ext_cfg.get("min_unit_chars", MIN_UNIT_CHARS)
        )
        self._legal_extractor     = LegalArticleExtractor(self.max_unit_chars, self.min_unit_chars)
        self._heading_extractor   = HeadingSectionExtractor(self.max_unit_chars, self.min_unit_chars)
        self._table_extractor     = TableUnitExtractor(self.max_unit_chars, self.min_unit_chars)
        self._paragraph_extractor = ParagraphUnitExtractor(self.max_unit_chars, self.min_unit_chars)

    def extract(self, doc: ProcessedData) -> List[SemanticUnit]:
        """
        Nhận diện loại tài liệu và tạo ra danh sách SemanticUnit tương ứng.

        Args:
            doc: ProcessedData đã qua bước Preprocessing (tiền xử lý).

        Returns:
            List[SemanticUnit]: Các đơn vị ngữ nghĩa trọn vẹn, sẵn sàng cho LLM trích xuất.
        """
        if not doc.content or not doc.content.strip():
            logger.warning(f"Tài liệu rỗng: {doc.id} - {doc.title}")
            return []

        doc_type = DocumentTypeDetector.detect(doc)

        logger.debug(f"[{doc_type}] {doc.title} ({len(doc.content)} ký tự)")

        if doc_type == "LEGAL":
            units = self._legal_extractor.build_units(doc)
        elif doc_type == "TABLE":
            units = self._table_extractor.build_units(doc)
        elif doc_type == "HEADING":
            units = self._heading_extractor.build_units(doc)
        else:
            units = self._paragraph_extractor.build_units(doc)

        # Lọc bỏ unit quá ngắn
        units = [u for u in units if u.char_count >= self.min_unit_chars]

        logger.info(
            f"[{doc_type}] '{doc.title}': "
            f"{len(doc.content)} ký tự -> {len(units)} Semantic Units"
        )
        return units

    def extract_batch(self, docs: List[ProcessedData]) -> List[SemanticUnit]:
        """
        Xử lý hàng loạt tài liệu và tổng hợp toàn bộ Semantic Units.

        Args:
            docs: Danh sách ProcessedData cần xử lý.

        Returns:
            List[SemanticUnit]: Tổng hợp tất cả Semantic Units từ toàn bộ tài liệu.
        """
        all_units: List[SemanticUnit] = []
        type_stats: Dict[str, int] = {
            UNIT_TYPE_LEGAL: 0,
            UNIT_TYPE_HEADING: 0,
            UNIT_TYPE_TABLE: 0,
            UNIT_TYPE_PARA: 0,
        }

        for doc in docs:
            units = self.extract(doc)
            all_units.extend(units)
            for u in units:
                type_stats[u.unit_type] = type_stats.get(u.unit_type, 0) + 1

        logger.info(
            f"Tổng hợp {len(docs)} tài liệu -> {len(all_units)} Semantic Units | "
            f"LEGAL={type_stats[UNIT_TYPE_LEGAL]}, "
            f"HEADING={type_stats[UNIT_TYPE_HEADING]}, "
            f"TABLE={type_stats[UNIT_TYPE_TABLE]}, "
            f"PARA={type_stats[UNIT_TYPE_PARA]}"
        )
        return all_units

    def save_units_jsonl(
        self,
        units: List[SemanticUnit],
        output_path: Path,
    ) -> int:
        """
        Lưu danh sách SemanticUnit ra file JSONL để kiểm tra và debug.

        Args:
            units: Danh sách SemanticUnit cần lưu.
            output_path: Đường dẫn file JSONL đầu ra.

        Returns:
            Số lượng units đã lưu.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        count = 0
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for unit in units:
                    f.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, output_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise
        logger.info(f"Đã lưu {count} Semantic Units vào: {output_path}")
        return count

    @staticmethod
    def load_units_jsonl(input_path: Path) -> List[SemanticUnit]:
        """Nạp lại SemanticUnit từ file JSONL đã lưu trước đó."""
        units: List[SemanticUnit] = []
        if not input_path.exists():
            logger.warning(f"Không tìm thấy file: {input_path}")
            return units
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    units.append(SemanticUnit(**data))
        return units


if __name__ == "__main__":
    # Test nhanh với mock document
    import uuid

    extractor = AdaptiveSemanticExtractor()

    # Mock tài liệu quy chế pháp quy
    legal_doc = ProcessedData(
        id=str(uuid.uuid4()),
        raw_data_id=None,
        title="Quy chế đào tạo đại học ĐHQGHN",
        content="""
Chương I. QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng
1. Quy chế này quy định về đào tạo đại học theo hệ thống tín chỉ tại Đại học Quốc gia Hà Nội.
2. Quy chế này áp dụng đối với sinh viên đại học chính quy của Đại học Quốc gia Hà Nội.

Điều 2. Hình thức đào tạo và chương trình đào tạo
1. Đào tạo đại học tại ĐHQGHN được thực hiện theo hình thức chính quy tập trung.
2. Chương trình đào tạo xác định chuẩn đầu ra, khối lượng học tập tối thiểu.

Chương II. TỔ CHỨC ĐÀO TẠO

Điều 10. Học kỳ và năm học
1. Năm học có hai học kỳ chính, mỗi học kỳ có ít nhất 15 tuần học.
2. Ngoài hai học kỳ chính, có thể tổ chức học kỳ phụ.
""".strip(),
        metadata={"source_uri": "https://uet.vnu.edu.vn/quy-che-dao-tao", "site": "UET"},
    )

    units = extractor.extract(legal_doc)
    print(f"\n[TEST LEGAL] -> {len(units)} units:")
    for u in units:
        print(f"  [{u.unit_type}] {u.breadcrumb[:80]} | {u.char_count} ký tự")

    # Mock bài viết giới thiệu Khoa (HEADING mode)
    heading_doc = ProcessedData(
        id=str(uuid.uuid4()),
        raw_data_id=None,
        title="Giới thiệu Khoa Công nghệ thông tin",
        content="""
# Giới thiệu chung
Khoa Công nghệ thông tin (FIT) được thành lập năm 2004 thuộc Trường ĐH Công nghệ – ĐHQGHN.

## Đội ngũ giảng viên
Khoa có hơn 50 giảng viên cơ hữu, trong đó có 15 GS và PGS, 30 TS.

## Chương trình đào tạo
Khoa đào tạo ngành Khoa học máy tính, Công nghệ thông tin (Chuẩn và CLC).
""".strip(),
        metadata={"source_uri": "https://fit.uet.vnu.edu.vn/gioi-thieu", "site": "FIT", "faculty": "fit"},
    )

    units2 = extractor.extract(heading_doc)
    print(f"\n[TEST HEADING] -> {len(units2)} units:")
    for u in units2:
        print(f"  [{u.unit_type}] {u.breadcrumb[:80]} | {u.char_count} ký tự")

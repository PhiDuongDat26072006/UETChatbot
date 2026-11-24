"""
src/chunking/chunker.py - Phụ trách quy trình phân đoạn văn bản (Text Chunking).
Thành viên phụ trách: Người phân đoạn dữ liệu.
Nhiệm vụ: Cắt các tài liệu ProcessedData thành các đoạn DataChunk nhỏ hợp lý kèm overlap và metadata.
Hỗ trợ:
1. Thứ tự Separators chuẩn hóa & Regex cho cấu trúc quy chế đào tạo Việt Nam (Chương, Mục, Điều, Khoản).
2. Bảo vệ ranh giới từ (Word Boundary Protection) trong cơ chế gối đầu Overlap.
3. Breadcrumb Context Injection theo phân cấp văn bản [Tên tài liệu > Chương > Điều].
4. Xử lý và làm giàu bảng dữ liệu Markdown theo hàng (Tabular serialization).
5. Lọc và gộp chunk mồ côi / chunk rác bằng min_chunk_size và lọc dòng vô nghĩa.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from src.base import BaseChunker, ProcessedData, DataChunk
from src.utils.helpers import get_logger, load_yaml_config
from src.ingestion.config.faculties import faculty_storage_name, find_faculty

logger = get_logger("chunking")


class UETChunker(BaseChunker):
    """
    Bộ phân đoạn văn bản thông minh (Structure-Aware Recursive Chunker) cho UET Chatbot.
    Tự động chia nhỏ tài liệu ProcessedData thành các đối tượng DataChunk có kích thước
    xấp xỉ `chunk_size` với độ gối đầu `chunk_overlap`, bảo tồn ngữ cảnh văn bản và gắn metadata.
    Đồng bộ tham số trực tiếp từ config.yaml.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        separators: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Tự động nạp cấu hình từ config.yaml nếu không truyền trực tiếp
        cfg = config.get("chunking", {}) if config else load_yaml_config().get("chunking", {})

        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else cfg.get("chunk_size", 800)
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else cfg.get("chunk_overlap", 150)
        )
        # min_chunk_size tự động thích ứng với chunk_size nhỏ trong test
        raw_min = min_chunk_size if min_chunk_size is not None else cfg.get("min_chunk_size", 100)
        self.min_chunk_size = min(raw_min, max(20, self.chunk_size // 4))

        # Đồng bộ separators trực tiếp từ config.yaml hoặc tham số truyền vào
        if separators is not None:
            self.separators = list(separators)
        else:
            self.separators = list(cfg.get("separators", []))
            if "" not in self.separators:
                self.separators.append("")

    # =========================================================================
    # XỬ LÝ BẢNG BIỂU MARKDOWN (MARKDOWN TABLE HANDLING)
    # =========================================================================

    @staticmethod
    def _table_cells(line: str) -> List[str]:
        """Tách các ô của một dòng Markdown."""
        value = line.strip()
        if not value.startswith('|'):
            return []
        return [cell.strip() for cell in value.strip('|').split('|')]

    @staticmethod
    def _parse_markdown_table(text: str) -> Optional[tuple[str, List[str], List[str], str]]:
        """Tìm bảng Markdown ở bất kỳ vị trí nào trong nội dung."""
        lines = text.splitlines()
        for start, line in enumerate(lines):
            headers = UETChunker._table_cells(line)
            if len(headers) < 2 or not any(headers):
                continue

            # Chuẩn hóa các header trống thành 'Cột {i+1}'
            headers = [h if h else f"Cột {i + 1}" for i, h in enumerate(headers)]

            rows = []
            end = start + 1
            while end < len(lines):
                current_line = lines[end]
                if not current_line.strip():
                    end += 1
                    continue
                row_cells = UETChunker._table_cells(current_line)
                if len(row_cells) != len(headers):
                    break
                if all(cell and set(cell) <= {'-', ':'} for cell in row_cells):
                    end += 1
                    continue
                rows.append(current_line.strip())
                end += 1

            if rows:
                before = "\n".join(lines[:start]).strip()
                after = "\n".join(lines[end:]).strip()
                return before, headers, rows, after
        return None

    @staticmethod
    def _extract_table_title(before: str, document_title: str) -> str:
        """Lấy tiêu đề gần nhất đứng trước bảng (ưu tiên Markdown heading), nếu có."""
        for line in reversed(before.splitlines()):
            clean = line.strip()
            if clean.startswith("#"):
                return clean.lstrip("#").strip()[:160]
            candidate = clean.lstrip('#').strip()
            if candidate and len(candidate) <= 160:
                return candidate
        return document_title or "Bảng dữ liệu"

    def _chunk_markdown_table(
        self,
        doc: ProcessedData,
        table_title: str,
        headers: List[str],
        rows: List[str],
    ) -> List[DataChunk]:
        """Làm giàu từng hàng bằng tiêu đề bảng/cột rồi chia theo cửa sổ trượt."""
        prefix = f"[{doc.title}]\n" if doc.title else ""
        injected_title = f"[META: {table_title}]\n"
        chunk_prefix = prefix + injected_title
        budget = max(1, self.chunk_size - len(chunk_prefix))
        enriched_rows = []
        for row in rows:
            values = self._table_cells(row)
            if len(values) < len(headers):
                values = values + [""] * (len(headers) - len(values))
            pairs = []
            for index, (column, value) in enumerate(zip(headers, values)):
                label = column or f"Cột {index + 1}"
                pairs.append(f"{label}: {value}")
            enriched_rows.append("- " + " | ".join(pairs))

        groups: List[str] = []
        start = 0
        while start < len(enriched_rows):
            end = start
            body_len = 0
            while end < len(enriched_rows):
                row_len = len(enriched_rows[end])
                if end > start and body_len + 1 + row_len > budget:
                    break
                body_len += row_len + (1 if end > start else 0)
                end += 1

            groups.append("\n".join(enriched_rows[start:end]))
            if end >= len(enriched_rows):
                break

            # Chọn các hàng cuối làm phần overlap cho cửa sổ kế tiếp
            next_start = end
            overlap_len = 0
            while next_start > start + 1 and overlap_len < self.chunk_overlap:
                next_start -= 1
                overlap_len += len(enriched_rows[next_start]) + 1
            while next_start < end and (
                sum(len(row) for row in enriched_rows[next_start:end + 1])
                + (end - next_start)
                > budget
            ):
                next_start += 1
            start = max(start + 1, next_start)

        result = []
        total = len(groups)
        for idx, body in enumerate(groups):
            enriched = chunk_prefix + body
            meta = dict(doc.metadata or {})
            meta.update({
                "title": doc.title,
                "document_id": doc.id,
                "chunk_index": idx,
                "total_chunks": total,
                "char_count": len(enriched),
                "data_type": "tabular",
                "table_title": table_title,
            })
            result.append(DataChunk(document_id=doc.id, text=enriched, chunk_index=idx, metadata=meta))
        return result

    # =========================================================================
    # BƯỚC 1 & 3: HEADING & BREADCRUMB CONTEXT INJECTION
    # =========================================================================

    @staticmethod
    def _extract_heading(line: str) -> Optional[Tuple[int, str]]:
        """
        Nhận diện tiêu đề Markdown hoặc tiêu đề phân cấp quy chế đào tạo Việt Nam.
        Trả về (level, clean_title) nếu là tiêu đề, ngược lại trả về None.
        Level:
          1: Markdown # hoặc Chương
          2: Markdown ## hoặc Mục
          3: Markdown ### hoặc Điều
          4: Markdown ####
        """
        clean = line.strip()
        if not clean:
            return None

        # 1. Markdown headers (#, ##, ###, ####)
        if clean.startswith("#"):
            match = re.match(r"^(#{1,6})\s+(.+)$", clean)
            if match:
                hashes, title = match.groups()
                return len(hashes), title.strip()

        # 2. Chương (văn bản hành chính / quy chế)
        match_chuong = re.match(r"^(Chương\s+[IVXLCDM\d]+[:\.\s\-–—]?\s*.*)$", clean, re.IGNORECASE)
        if match_chuong:
            return 1, match_chuong.group(1).strip()

        # 3. Mục
        match_muc = re.match(r"^(Mục\s+\d+[:\.\s\-–—]?\s*.*)$", clean, re.IGNORECASE)
        if match_muc:
            return 2, match_muc.group(1).strip()

        # 4. Điều (quy chế đào tạo)
        match_dieu = re.match(r"^(Điều\s+\d+[\.:\s\-–—]?\s*.*)$", clean, re.IGNORECASE)
        if match_dieu:
            return 3, match_dieu.group(1).strip()

        return None

    def _split_into_sections(self, doc_title: str, text: str) -> List[Tuple[List[str], str]]:
        """
        Chia văn bản thành các section theo phân cấp tiêu đề (Heading & Điều/Khoản)
        đồng thời theo dõi ngăn xếp breadcrumbs [Tên tài liệu > Chương > Điều].
        """
        lines = text.splitlines()
        current_headers: Dict[int, str] = {}
        current_lines: List[str] = []
        sections: List[Tuple[List[str], str]] = []

        for line in lines:
            h = self._extract_heading(line)
            if h:
                level, title = h
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    # Không tạo section riêng nếu đoạn chỉ là 1 dòng tiêu đề đứng trơ trọi
                    is_just_heading = len(sec_text.splitlines()) <= 1 and self._extract_heading(sec_text) is not None
                    if sec_text and not is_just_heading:
                        crumbs = [doc_title] if doc_title else []
                        for lvl in sorted(current_headers.keys()):
                            crumbs.append(current_headers[lvl])
                        sections.append((crumbs, sec_text))
                    current_lines = []

                # Xóa các cấp tiêu đề sâu hơn hoặc bằng cấp hiện tại
                keys_to_remove = [k for k in current_headers if k >= level]
                for k in keys_to_remove:
                    del current_headers[k]
                current_headers[level] = title

            current_lines.append(line)

        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            is_just_heading = len(sec_text.splitlines()) <= 1 and self._extract_heading(sec_text) is not None
            if sec_text and not is_just_heading:
                crumbs = [doc_title] if doc_title else []
                for lvl in sorted(current_headers.keys()):
                    crumbs.append(current_headers[lvl])
                sections.append((crumbs, sec_text))

        # Nếu không tìm thấy heading nào, toàn bộ văn bản là 1 section duy nhất
        if not sections and text.strip():
            sections.append(([doc_title] if doc_title else [], text.strip()))

        return sections

    # =========================================================================
    # BƯỚC 1 & 2: RECURSIVE TEXT SPLITTING & WORD-BOUNDARY OVERLAP
    # =========================================================================

    def _split_text(self, text: str, separators: List[str], max_size: Optional[int] = None) -> List[str]:
        """
        Chia đệ quy một chuỗi văn bản dựa theo danh sách phân cấp ký tự phân tách (separators).
        Hỗ trợ cả ký tự thuần lẫn Regex có lookahead để bảo vệ cấu trúc heading.
        """
        if not text:
            return []

        target_max_size = max_size if max_size is not None else self.chunk_size

        # 1. Tìm ký tự phân tách phù hợp xuất hiện đầu tiên trong separators
        chosen_separator = ""
        next_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                chosen_separator = ""
                break
            if sep.startswith("regex:"):
                pattern = sep[6:]
                if re.search(pattern, text):
                    chosen_separator = sep
                    next_separators = separators[i + 1 :]
                    break
            elif sep in text:
                chosen_separator = sep
                next_separators = separators[i + 1 :]
                break

        # 2. Tách văn bản theo chosen_separator
        is_lookahead = False
        is_heading_prefix = chosen_separator in ["### ", "## ", "# "]

        if chosen_separator.startswith("regex:"):
            pattern = chosen_separator[6:]
            is_lookahead = "(?=" in pattern
            raw_splits = [p for p in re.split(pattern, text) if p]
        elif is_heading_prefix:
            h_tag = chosen_separator.strip()
            pattern = rf"(?<=\n)(?={h_tag}\s)|(?=^{h_tag}\s)"
            raw_splits = [p for p in re.split(pattern, text) if p]
            is_lookahead = True
        elif chosen_separator != "":
            raw_splits = text.split(chosen_separator)
        else:
            raw_splits = list(text)

        # 3. Duyệt qua từng mảnh nhỏ để kiểm tra xem có vượt quá target_max_size không
        final_splits: List[str] = []
        for idx, part in enumerate(raw_splits):
            if not part:
                continue

            if is_lookahead or chosen_separator in ["", " "]:
                piece = part
            else:
                piece = (
                    part
                    if (idx == len(raw_splits) - 1 or part.endswith(chosen_separator))
                    else (part + chosen_separator)
                )

            if len(piece) <= target_max_size:
                final_splits.append(piece)
            else:
                if next_separators:
                    sub_splits = self._split_text(piece, next_separators, max_size=target_max_size)
                    final_splits.extend(sub_splits)
                else:
                    step = max(1, target_max_size - self.chunk_overlap)
                    for start_idx in range(0, len(piece), step):
                        final_splits.append(piece[start_idx : start_idx + target_max_size])

        return final_splits

    def _merge_splits(
        self,
        splits: List[str],
        max_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Gom nhóm các mảnh văn bản nhỏ thành các đoạn chunk hoàn chỉnh có độ dài <= `max_size`
        và duy trì độ gối đầu `chunk_overlap` an toàn theo ranh giới từ (Word Boundary Protection).
        """
        target_max_size = max_size if max_size is not None else self.chunk_size
        target_overlap = overlap if overlap is not None else self.chunk_overlap

        merged_chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece)

            if current_length + piece_len > target_max_size and current_chunk:
                chunk_str = "".join(current_chunk).strip()
                if chunk_str:
                    merged_chunks.append(chunk_str)

                overlap_splits: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= target_overlap:
                        overlap_splits.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break

                # BƯỚC 2: Fallback Overlap an toàn theo ranh giới từ (Word Boundary Protection)
                if not overlap_splits and target_overlap > 0 and current_chunk:
                    last_item = current_chunk[-1]
                    desired_start = max(0, len(last_item) - target_overlap)
                    if (
                        desired_start > 0
                        and not last_item[desired_start - 1].isspace()
                        and not last_item[desired_start].isspace()
                    ):
                        space_pos = last_item.rfind(" ", max(0, desired_start - 15), desired_start)
                        if space_pos != -1:
                            start_idx = space_pos + 1
                        else:
                            space_pos_fwd = last_item.find(
                                " ", desired_start, min(len(last_item), desired_start + 15)
                            )
                            start_idx = space_pos_fwd + 1 if space_pos_fwd != -1 else desired_start
                    else:
                        start_idx = desired_start

                    tail_overlap = last_item[start_idx:]
                    if tail_overlap:
                        overlap_splits.append(tail_overlap)

                current_chunk = overlap_splits
                current_length = sum(len(x) for x in current_chunk)

            current_chunk.append(piece)
            current_length += piece_len

        if current_chunk:
            final_str = "".join(current_chunk).strip()
            if final_str:
                merged_chunks.append(final_str)

        return merged_chunks

    # =========================================================================
    # BƯỚC 5: LỌC VÀ GỘP CHUNK MỒ CÔI (ORPHAN / NOISE CHUNKS FILTER)
    # =========================================================================

    def _filter_and_merge_small_chunks(self, chunks: List[DataChunk]) -> List[DataChunk]:
        """
        Lọc bỏ các chunk rác không chứa ký tự chữ/số, làm sạch dòng vô nghĩa,
        và gộp các chunk mồ côi cùng heading có độ dài nhỏ hơn `min_chunk_size` vào chunk liền trước.
        """
        if not chunks:
            return []

        cleaned_chunks: List[DataChunk] = []
        for chunk in chunks:
            # Loại bỏ các dòng rác không có chữ hoặc số ở đầu/cuối thân chunk
            text = chunk.text
            if text.startswith("[") and "]\n" in text:
                header_part, body_part = text.split("]\n", 1)
                lines = [ln for ln in body_part.splitlines() if any(c.isalnum() for c in ln)]
                if not lines:
                    continue
                chunk.text = f"{header_part}]\n" + "\n".join(lines).strip()
            else:
                lines = [ln for ln in text.splitlines() if any(c.isalnum() for c in ln)]
                if not lines:
                    continue
                chunk.text = "\n".join(lines).strip()

            chunk.metadata["char_count"] = len(chunk.text)
            cleaned_chunks.append(chunk)

        if len(cleaned_chunks) <= 1:
            return cleaned_chunks

        merged: List[DataChunk] = []
        for chunk in cleaned_chunks:
            body = chunk.text
            if body.startswith("[") and "]\n" in body:
                body = body.split("]\n", 1)[1]

            if not any(c.isalnum() for c in body):
                continue

            # Chỉ gộp nếu chunk trước đó cùng heading và tổng kích thước hợp lý
            if len(body.strip()) < self.min_chunk_size and merged:
                prev_chunk = merged[-1]
                same_heading = prev_chunk.metadata.get("heading") == chunk.metadata.get("heading")
                if same_heading and prev_chunk.metadata.get("data_type") != "tabular":
                    combined = prev_chunk.text + "\n" + body.strip()
                    if len(combined) <= int(self.chunk_size * 1.3):
                        prev_chunk.text = combined
                        prev_chunk.metadata["char_count"] = len(combined)
                        continue

            merged.append(chunk)

        total = len(merged)
        for idx, c in enumerate(merged):
            c.chunk_index = idx
            c.metadata["chunk_index"] = idx
            c.metadata["total_chunks"] = total

        return merged

    def _chunk_plain_segment(self, doc: ProcessedData, text: str) -> List[DataChunk]:
        """
        Chia một đoạn văn bản theo phân cấp Section / Breadcrumbs và Context Injection.
        """
        if not text.strip():
            return []

        sections = self._split_into_sections(doc.title, text)
        result: List[DataChunk] = []

        for crumbs, sec_text in sections:
            if not sec_text.strip():
                continue

            if len(crumbs) > 1:
                header_prefix = f"[{' > '.join(crumbs)}]\n"
                heading_name = crumbs[-1]
                breadcrumb_str = " > ".join(crumbs)
            elif crumbs and crumbs[0]:
                header_prefix = f"[{crumbs[0]}]\n"
                heading_name = crumbs[0]
                breadcrumb_str = crumbs[0]
            else:
                header_prefix = ""
                heading_name = doc.title
                breadcrumb_str = doc.title

            header_len = len(header_prefix)
            body_budget = max(100, self.chunk_size - header_len)

            if len(sec_text) <= body_budget:
                segments = [sec_text.strip()]
            else:
                raw_splits = self._split_text(sec_text, self.separators, max_size=body_budget)
                segments = self._merge_splits(raw_splits, max_size=body_budget)

            for segment in segments:
                enriched = (
                    header_prefix + segment
                    if header_prefix and not segment.startswith(header_prefix)
                    else segment
                )
                meta = dict(doc.metadata or {})
                meta.update({
                    "title": doc.title,
                    "document_id": doc.id,
                    "breadcrumb": breadcrumb_str,
                    "heading": heading_name,
                    "char_count": len(enriched),
                    "data_type": "text",
                })
                result.append(DataChunk(
                    document_id=doc.id,
                    text=enriched,
                    chunk_index=len(result),
                    metadata=meta,
                ))

        return result

    def chunk(self, doc: ProcessedData) -> List[DataChunk]:
        """
        Chia một tài liệu ProcessedData thành danh sách các đoạn DataChunk kèm metadata chi tiết.
        Bảo tồn cấu trúc bảng biểu, tiêm breadcrumb tiêu đề, gối đầu an toàn và lọc chunk rác.
        """
        logger.info(f"Bắt đầu phân đoạn tài liệu: '{doc.title}' (ID: {doc.id})")

        raw_text = doc.content.strip()
        if not raw_text:
            logger.warning(f"Tài liệu ID '{doc.id}' không có nội dung văn bản.")
            return []

        # Xử lý các bảng Markdown trong nội dung (nếu có)
        table = self._parse_markdown_table(raw_text)
        if table is not None:
            chunks: List[DataChunk] = []
            remaining = raw_text
            while remaining:
                table = self._parse_markdown_table(remaining)
                if table is None:
                    chunks.extend(self._chunk_plain_segment(doc, remaining))
                    break

                before, headers, rows, after = table
                chunks.extend(self._chunk_plain_segment(doc, before))
                table_title = self._extract_table_title(before, doc.title)
                chunks.extend(self._chunk_markdown_table(doc, table_title, headers, rows))

                if after == remaining:
                    break
                remaining = after

            chunks = self._filter_and_merge_small_chunks(chunks)
            logger.info(f"Hoàn tất phân đoạn tài liệu có bảng '{doc.title}': Đã tạo {len(chunks)} chunks.")
            return chunks

        # Kiểm tra cấu trúc phân cấp tiêu đề
        sections = self._split_into_sections(doc.title, raw_text)

        # Nếu độ dài toàn bộ tài liệu nhỏ hơn hoặc bằng chunk_size và không có cấu trúc phân cấp -> 1 chunk duy nhất
        if len(sections) <= 1 and len(raw_text) <= self.chunk_size:
            chunk_metadata = dict(doc.metadata) if doc.metadata else {}
            enriched_text = (
                f"[{doc.title}]\n{raw_text}"
                if doc.title and not raw_text.startswith(f"[{doc.title}]")
                else raw_text
            )
            chunk_metadata.update({
                "title": doc.title,
                "document_id": doc.id,
                "chunk_index": 0,
                "total_chunks": 1,
                "breadcrumb": doc.title,
                "heading": doc.title,
                "char_count": len(enriched_text),
                "data_type": "text",
            })
            return [
                DataChunk(
                    document_id=doc.id,
                    text=enriched_text,
                    chunk_index=0,
                    metadata=chunk_metadata,
                )
            ]

        # Phân đoạn văn bản bình thường với Breadcrumbs & Word-boundary Overlap
        chunks = self._chunk_plain_segment(doc, raw_text)

        # BƯỚC 5: Lọc bỏ rác và gộp chunk mồ côi (< min_chunk_size)
        chunks = self._filter_and_merge_small_chunks(chunks)

        logger.info(f"Hoàn tất phân đoạn tài liệu '{doc.title}': Đã tạo {len(chunks)} chunks.")
        return chunks

    @staticmethod
    def _extract_faculty_code(meta: Dict[str, Any]) -> str:
        """Trích xuất mã khoa chuẩn hóa (fit, fat, fce, fema, fepn, fet, iai, sae, uet...) từ metadata."""
        raw_unit = str(meta.get("unit") or meta.get("faculty") or "").strip().lower()
        if raw_unit:
            clean_unit = "".join(c for c in raw_unit if c.isalnum() or c in "_-")
            if clean_unit:
                return clean_unit

        raw_domain = str(meta.get("domain") or "").strip().lower()
        if raw_domain:
            parts = raw_domain.split(".")
            if "uet" in parts:
                idx = parts.index("uet")
                if idx > 0:
                    candidate = parts[idx - 1]
                    return parts[idx - 2] if candidate == "www" and idx > 1 else candidate
                return "uet"
            elif parts and parts[0]:
                return parts[0]

        source = str(meta.get("source") or meta.get("source_url_or_path") or "").strip()
        if source:
            if "://" in source:
                try:
                    from urllib.parse import urlparse
                    parts = urlparse(source).netloc.lower().split(".")
                    if "uet" in parts:
                        idx = parts.index("uet")
                        if idx > 0:
                            candidate = parts[idx - 1]
                            return parts[idx - 2] if candidate == "www" and idx > 1 else candidate
                        return "uet"
                    elif parts and parts[0]:
                        return parts[0]
                except Exception:
                    pass
            for part in source.replace("\\", "/").split("/"):
                if "uet." in part.lower():
                    p_split = part.lower().split(".")
                    if "uet" in p_split:
                        idx = p_split.index("uet")
                        if idx > 0:
                            candidate = p_split[idx - 1]
                            return p_split[idx - 2] if candidate == "www" and idx > 1 else candidate
                        return "uet"

        return "general"

    def chunk_batch(self, docs: List[ProcessedData], append: bool = True) -> List[DataChunk]:
        """
        Chia chunk hàng loạt tài liệu và lưu kết quả vào thư mục data/chunked_data.
        Mỗi khoa (faculty/domain) sẽ được lưu vào một file jsonl riêng ({faculty}_chunks.jsonl).
        """
        all_chunks: List[DataChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))

        chunks_by_faculty = defaultdict(list)
        for chunk in all_chunks:
            target = find_faculty(chunk.metadata.get("unit", ""))
            faculty = faculty_storage_name(target["faculty_id"]) if target else chunk.metadata.get("domain", "other")
            chunks_by_faculty[faculty].append(chunk)

        base_dir = Path(__file__).resolve().parent.parent.parent
        chunk_data_dir = base_dir / "data" / "chunked_data"
        chunk_data_dir.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        for faculty, faculty_chunks in chunks_by_faculty.items():
            safe_faculty_name = "".join(c for c in faculty if c.isalnum() or c in "_-") or "general"
            file_path = chunk_data_dir / f"{safe_faculty_name}_chunks.jsonl"

            try:
                with open(file_path, mode, encoding="utf-8") as f:
                    for chunk in faculty_chunks:
                        try:
                            json_str = chunk.model_dump_json()
                        except AttributeError:
                            json_str = chunk.json()
                        f.write(json_str + "\n")
                logger.info(f"Đã lưu {len(faculty_chunks)} chunks của khoa '{faculty}' vào file: {file_path}")
            except Exception as e:
                logger.error(f"Lỗi khi lưu chunks của khoa '{faculty}' vào file {file_path}: {e}")

        return all_chunks

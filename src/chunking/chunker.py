"""
src/chunking/chunker.py - Phụ trách quy trình phân đoạn văn bản (Text Chunking).
Thành viên phụ trách: Người phân đoạn dữ liệu.
Nhiệm vụ: Cắt các tài liệu ProcessedData thành các đoạn DataChunk nhỏ hợp lý kèm overlap và metadata.
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
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

        # Đồng bộ separators từ config.yaml hoặc tham số truyền vào
        if separators is not None:
            self.separators = separators
        elif "separators" in cfg and cfg["separators"]:
            self.separators = list(cfg["separators"])
            if "" not in self.separators:
                self.separators.append("")
        else:
            # Danh sách phân cấp ký tự phân tách mặc định
            self.separators = [
                "\n\n",
                "\n",
                "### ",
                "## ",
                "# ",
                ". ",
                "; ",
                " ",
                "",
            ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Chia đệ quy một chuỗi văn bản dựa theo danh sách phân cấp ký tự phân tách (separators).
        """
        if not text:
            return []

        # 1. Tìm ký tự phân tách phù hợp xuất hiện đầu tiên trong separators
        chosen_separator = ""
        next_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                chosen_separator = ""
                break
            if sep in text:
                chosen_separator = sep
                next_separators = separators[i + 1 :]
                break

        # 2. Tách văn bản theo chosen_separator
        if chosen_separator != "":
            raw_splits = text.split(chosen_separator)
        else:
            raw_splits = list(text)

        # 3. Duyệt qua từng mảnh nhỏ để kiểm tra xem có vượt quá chunk_size không
        final_splits: List[str] = []
        for idx, part in enumerate(raw_splits):
            if not part:
                continue

            # Thêm lại separator vào cuối đoạn ngoại trừ trường hợp tách theo từng ký tự hoặc rỗng
            if chosen_separator not in ["", " "]:
                piece = (
                    part
                    if (idx == len(raw_splits) - 1 or part.endswith(chosen_separator))
                    else (part + chosen_separator)
                )
            else:
                piece = part

            if len(piece) <= self.chunk_size:
                final_splits.append(piece)
            else:
                # Nếu đoạn vẫn lớn hơn chunk_size và còn separators nhỏ hơn -> Tách đệ quy
                if next_separators:
                    sub_splits = self._split_text(piece, next_separators)
                    final_splits.extend(sub_splits)
                else:
                    # Nếu hết separators mà đoạn vẫn quá dài -> Cắt cứng theo chunk_size
                    step = max(1, self.chunk_size - self.chunk_overlap)
                    for start_idx in range(0, len(piece), step):
                        final_splits.append(piece[start_idx : start_idx + self.chunk_size])

        return final_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Gom nhóm các mảnh văn bản nhỏ thành các đoạn chunk hoàn chỉnh có độ dài <= `chunk_size`
        và duy trì độ gối đầu `chunk_overlap` giữa 2 chunk liên tiếp.
        """
        merged_chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece)

            # Nếu thêm piece mới vượt quá chunk_size -> Đóng chunk hiện tại
            if current_length + piece_len > self.chunk_size and current_chunk:
                chunk_str = "".join(current_chunk).strip()
                if chunk_str:
                    merged_chunks.append(chunk_str)

                # Giữ lại các mảnh cuối để làm overlap cho chunk kế tiếp
                overlap_splits: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_splits.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break

                # Cách xử lý dự phòng khi bị mất phần gối đầu:
                # Nếu mảnh cuối cùng dài hơn chunk_overlap khiến overlap_splits bị rỗng,
                # thay vì để mất 100% độ gối đầu, ta cắt lấy phần đuôi của mảnh cuối cùng
                if not overlap_splits and self.chunk_overlap > 0 and current_chunk:
                    last_item = current_chunk[-1]
                    tail_overlap = last_item[-self.chunk_overlap:]
                    if tail_overlap:
                        overlap_splits.append(tail_overlap)

                current_chunk = overlap_splits
                current_length = sum(len(x) for x in current_chunk)

            current_chunk.append(piece)
            current_length += piece_len

        # Đóng chunk cuối cùng nếu còn
        if current_chunk:
            final_str = "".join(current_chunk).strip()
            if final_str:
                merged_chunks.append(final_str)

        return merged_chunks

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
            if len(headers) < 2 or not all(headers):
                continue

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
        """Lấy tiêu đề gần nhất đứng trước bảng, nếu có."""
        for line in reversed(before.splitlines()):
            candidate = line.strip().lstrip('#').strip()
            if candidate:
                return candidate[:160]
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
                # Hàng quá dài vẫn được giữ nguyên như một chunk riêng.
                body_len += row_len + (1 if end > start else 0)
                end += 1

            groups.append("\n".join(enriched_rows[start:end]))
            if end >= len(enriched_rows):
                break

            # Chọn các hàng cuối làm phần overlap cho cửa sổ kế tiếp.
            next_start = end
            overlap_len = 0
            while next_start > start + 1 and overlap_len < self.chunk_overlap:
                next_start -= 1
                overlap_len += len(enriched_rows[next_start]) + 1
            # Giảm overlap nếu nó không chừa chỗ cho ít nhất một hàng mới.
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
            meta.update({"title": doc.title, "document_id": doc.id, "chunk_index": idx, "total_chunks": total, "char_count": len(enriched), "data_type": "tabular"})
            result.append(DataChunk(document_id=doc.id, text=enriched, chunk_index=idx, metadata=meta))
        return result

    def _chunk_plain_segment(self, doc: ProcessedData, text: str) -> List[DataChunk]:
        """Chia một đoạn văn bản theo cách thông thường của chunker."""
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            segments = [text.strip()]
        else:
            segments = self._merge_splits(self._split_text(text, self.separators))

        result = []
        for idx, segment in enumerate(segments):
            meta = dict(doc.metadata or {})
            meta.update({"title": doc.title, "document_id": doc.id,
                         "chunk_index": idx, "total_chunks": len(segments),
                         "char_count": len(segment), "data_type": "text"})
            enriched = f"[{doc.title}]\n{segment}" if doc.title else segment
            result.append(DataChunk(document_id=doc.id, text=enriched,
                                    chunk_index=idx, metadata=meta))
        return result
    def chunk(self, doc: ProcessedData) -> List[DataChunk]:
        """
        Chia một tài liệu ProcessedData thành danh sách các đoạn DataChunk kèm metadata chi tiết.
        :param doc: Đối tượng ProcessedData đã làm sạch
        :return: Danh sách các đối tượng DataChunk
        """
        logger.info(f"Bắt đầu phân đoạn tài liệu: '{doc.title}' (ID: {doc.id})")

        raw_text = doc.content.strip()
        if not raw_text:
            logger.warning(f"Tài liệu ID '{doc.id}' không có nội dung văn bản.")
            return []

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

                # Ngăn vòng lặp vô hạn nếu dữ liệu lỗi khiến bộ phân tích không tiến lên.
                if after == remaining:
                    break
                remaining = after

            total = len(chunks)
            for idx, chunk in enumerate(chunks):
                chunk.chunk_index = idx
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_chunks"] = total
            return chunks

        # Nếu độ dài toàn bộ tài liệu nhỏ hơn hoặc bằng chunk_size -> Trả về 1 chunk duy nhất
        if len(raw_text) <= self.chunk_size:
            chunk_metadata = dict(doc.metadata) if doc.metadata else {}
            chunk_metadata.update({
                "title": doc.title,
                "document_id": doc.id,
                "chunk_index": 0,
                "total_chunks": 1,
                "char_count": len(raw_text),
            })
            enriched_text = (
                f"[{doc.title}]\n{raw_text}"
                if doc.title and not raw_text.startswith(f"[{doc.title}]")
                else raw_text
            )
            return [
                DataChunk(
                    document_id=doc.id,
                    text=enriched_text,
                    chunk_index=0,
                    metadata=chunk_metadata,
                )
            ]

        # 1. Tách văn bản thành các mảnh đệ quy theo phân cấp separators
        raw_splits = self._split_text(raw_text, self.separators)

        # 2. Gom mảnh kèm độ gối đầu (overlap)
        text_segments = self._merge_splits(raw_splits)

        # 3. Tạo danh sách đối tượng DataChunk và đính kèm metadata
        data_chunks: List[DataChunk] = []
        total_chunks = len(text_segments)

        for idx, segment in enumerate(text_segments):
            chunk_meta = dict(doc.metadata) if doc.metadata else {}
            chunk_meta.update({
                "title": doc.title,
                "document_id": doc.id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "char_count": len(segment),
            })

            # Gắn tiêu đề tài liệu vào đầu mỗi chunk để bảo toàn ngữ cảnh.
            enriched_text = (
                f"[{doc.title}]\n{segment}"
                if doc.title and not segment.startswith(f"[{doc.title}]")
                else segment
            )

            chunk_obj = DataChunk(
                document_id=doc.id,
                text=enriched_text,
                chunk_index=idx,
                metadata=chunk_meta,
            )
            data_chunks.append(chunk_obj)

        logger.info(f"Hoàn tất phân đoạn tài liệu '{doc.title}': Đã tạo {total_chunks} chunks.")
        return data_chunks

    @staticmethod
    def _extract_faculty_code(meta: Dict[str, Any]) -> str:
        """Trích xuất mã khoa chuẩn hóa (fit, fat, fce, fema, fepn, fet, iai, sae, uet...) từ metadata."""
        # 1. unit hoặc faculty
        raw_unit = str(meta.get("unit") or meta.get("faculty") or "").strip().lower()
        if raw_unit:
            clean_unit = "".join(c for c in raw_unit if c.isalnum() or c in "_-")
            if clean_unit:
                return clean_unit

        # 2. Tên miền (ví dụ: fit.uet.vnu.edu.vn, uet.edu.vn...)
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

        # 3. URL nguồn (source)
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

    def chunk_batch(self, docs: List[ProcessedData]) -> List[DataChunk]:
        """
        Chia chunk hàng loạt tài liệu và lưu kết quả vào thư mục data/chunked_data.
        Mỗi khoa (faculty/domain) sẽ được lưu vào một file jsonl riêng ({faculty}_chunks.jsonl).
        """
        all_chunks: List[DataChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))

        # Phân loại chunk theo từng khoa dựa vào metadata
        chunks_by_faculty = defaultdict(list)
        for chunk in all_chunks:
            # Không gộp các khoa khác nhau vào cùng tệp chỉ vì chúng dùng chung máy chủ.
            target = find_faculty(chunk.metadata.get("unit", ""))
            faculty = faculty_storage_name(target["faculty_id"]) if target else chunk.metadata.get("domain", "other")
            chunks_by_faculty[faculty].append(chunk)

        # Đảm bảo thư mục lưu trữ data/chunked_data tồn tại
        base_dir = Path(__file__).resolve().parent.parent.parent
        chunk_data_dir = base_dir / "data" / "chunked_data"
        chunk_data_dir.mkdir(parents=True, exist_ok=True)

        # Lưu mỗi khoa vào một file jsonl riêng
        for faculty, faculty_chunks in chunks_by_faculty.items():
            safe_faculty_name = "".join(c for c in faculty if c.isalnum() or c in "_-") or "general"
            file_path = chunk_data_dir / f"{safe_faculty_name}_chunks.jsonl"

            try:
                with open(file_path, "a", encoding="utf-8") as f:
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



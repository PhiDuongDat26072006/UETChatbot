"""
src/chunking/chunker.py - Phụ trách quy trình phân đoạn văn bản (Text Chunking).
Thành viên phụ trách: Người phân đoạn dữ liệu.
Nhiệm vụ: Cắt các tài liệu ProcessedData thành các đoạn DataChunk nhỏ hợp lý kèm overlap và metadata.
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Optional
from src.base import BaseChunker, ProcessedData, DataChunk
from src.utils.helpers import get_logger
from src.ingestion.config.faculties import faculty_storage_name, find_faculty

logger = get_logger("chunking")


class UETChunker(BaseChunker):
    """
    Bộ phân đoạn văn bản thông minh (Structure-Aware Recursive Chunker) cho UET Chatbot.
    Tự động chia nhỏ tài liệu ProcessedData thành các đối tượng DataChunk có kích thước
    xấp xỉ `chunk_size` với độ gối đầu `chunk_overlap`, bảo tồn ngữ cảnh văn bản và gắn metadata.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Danh sách phân cấp ký tự phân tách: từ cấu trúc lớn (Chương, Điều, Đoạn) đến nhỏ (Dòng, Câu, Từ)
        self.separators = separators or [
            "\n\n",
            "\n",
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
            return [
                DataChunk(
                    document_id=doc.id,
                    text=raw_text,
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

            chunk_obj = DataChunk(
                document_id=doc.id,
                text=segment,
                chunk_index=idx,
                metadata=chunk_meta,
            )
            data_chunks.append(chunk_obj)

        logger.info(f"Hoàn tất phân đoạn tài liệu '{doc.title}': Đã tạo {total_chunks} chunks.")
        return data_chunks

    def chunk_batch(self, docs: List[ProcessedData]) -> List[DataChunk]:
        """
        Chia chunk hàng loạt tài liệu và lưu kết quả vào thư mục chunk_data.
        Mỗi khoa (domain) sẽ được lưu vào một file jsonl riêng.
        """
        all_chunks: List[DataChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))

        # Phân loại chunk theo từng khoa dựa vào domain (hoặc category/source)
        chunks_by_faculty = defaultdict(list)
        for chunk in all_chunks:
            # A shared request host must not merge distinct faculty output files.
            target = find_faculty(chunk.metadata.get("unit", ""))
            faculty = faculty_storage_name(target["faculty_id"]) if target else chunk.metadata.get("domain", "other")
            chunks_by_faculty[faculty].append(chunk)

        # Đảm bảo thư mục lưu trữ chunk_data tồn tại
        base_dir = Path(__file__).resolve().parent.parent.parent
        chunk_data_dir = base_dir / "chunk_data"
        chunk_data_dir.mkdir(parents=True, exist_ok=True)

        # Lưu mỗi khoa vào một file jsonl riêng
        for faculty, faculty_chunks in chunks_by_faculty.items():
            # Xử lý tên file an toàn
            safe_faculty_name = "".join(c for c in faculty if c.isalnum() or c in "._-")
            file_path = chunk_data_dir / f"{safe_faculty_name}_chunks.jsonl"
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
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



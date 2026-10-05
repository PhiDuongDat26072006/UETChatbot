"""
src/retrieval/bm25.py - Bộ máy tìm kiếm từ khóa BM25 (Lexical Search).
Tối ưu hóa cho ngữ liệu học vụ Tiếng Việt của UET (hỗ trợ Unigram, Bigram, số hiệu văn bản, mã học phần).
Hoạt động độc lập không phụ thuộc thư viện ngoài, xử lý an toàn khi kho dữ liệu rỗng hoặc chưa đầy đủ.
"""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.base import DataChunk, RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("bm25")


class BM25Index:
    """
    Chỉ mục tìm kiếm từ khóa BM25+ (Best Matching 25 Plus).
    BM25+ khắc phục nhược điểm của BM25 thông thường trên các văn bản dài (như quy chế, điều khoản)
    nhờ tham số delta (mặc định 1.0) để đảm bảo mọi đoạn có chứa từ khóa đều nhận điểm tích cực.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        """
        :param k1: Hệ số bão hòa tần suất từ (Term Frequency Saturation)
        :param b: Hệ số điều chỉnh độ dài văn bản (Length Normalization)
        :param delta: Hằng số cộng thêm của BM25+ (tránh phạt quá mức văn bản dài)
        """
        self.k1 = k1
        self.b = b
        self.delta = delta

        self.corpus_chunks: List[DataChunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self._is_indexed: bool = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tách từ và sinh n-gram cho tiếng Việt và mã quy chế:
        - Giữ nguyên chữ cái tiếng Việt, chữ số, dấu gạch nối (mã học phần INT1008, QH-2022, QĐ-ĐHQGHN).
        - Sinh unigram và bigram từ liên tiếp để bắt từ ghép tiếng Việt (ví dụ: 'học bổng', 'tốt nghiệp').
        """
        if not text:
            return []

        # Chuẩn hóa khoảng trắng và chuyển chữ thường
        clean_text = text.lower().strip()
        # Tách từ đơn dựa trên ký tự từ và gạch nối
        words = re.findall(r"[\w\-]+", clean_text)
        if not words:
            return []

        tokens: List[str] = []
        # 1. Unigram
        for w in words:
            if len(w) > 0:
                tokens.append(w)

        # 2. Bigram (cho từ ghép tiếng Việt 2 âm tiết: "học_bổng", "tốt_nghiệp", "điểm_rèn")
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if len(w1) > 1 and len(w2) > 1:
                tokens.append(f"{w1}_{w2}")

        return tokens

    def index_chunks(self, chunks: List[DataChunk]) -> int:
        """
        Lập chỉ mục BM25 từ danh sách DataChunk.
        :param chunks: Danh sách các DataChunk
        :return: Số lượng chunk đã lập chỉ mục
        """
        if not chunks:
            logger.warning("BM25Index nhận danh sách chunks rỗng. Không có dữ liệu để lập chỉ mục.")
            self.clear()
            return 0

        self.corpus_chunks = list(chunks)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        total_tokens = 0
        num_docs = len(self.corpus_chunks)

        for chunk in self.corpus_chunks:
            # Ghép cả title (nếu có trong metadata) và text để tăng trọng số tìm kiếm theo tiêu đề
            title = chunk.metadata.get("title", "") if chunk.metadata else ""
            full_text = f"{title} {chunk.text}" if title else chunk.text
            tokens = self.tokenize(full_text)

            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            # Đếm tần suất từ trong chunk
            tf: Dict[str, int] = {}
            seen_tokens: Set[str] = set()
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
                if token not in seen_tokens:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen_tokens.add(token)

            self.doc_term_freqs.append(tf)

        self.avg_doc_len = (total_tokens / num_docs) if num_docs > 0 else 0.0

        # Tính toán IDF theo chuẩn Robertson-Spärck Jones có làm mịn (smoothing)
        self.idf = {}
        for token, df in self.doc_freqs.items():
            # Công thức IDF làm mịn: ln((N - n + 0.5) / (n + 0.5) + 1)
            idf_score = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[token] = max(0.05, idf_score)

        self._is_indexed = True
        logger.info(f"Đã lập chỉ mục BM25 thành công cho {num_docs} chunks (tổng {len(self.doc_freqs)} từ khóa).")
        return num_docs

    def search(self, query: str, top_k: int = 100) -> List[RetrievedContext]:
        """
        Tìm kiếm BM25 cho một câu truy vấn văn bản.
        :param query: Câu hỏi dạng text của người dùng
        :param top_k: Số lượng văn bản cần lấy (ví dụ 100)
        :return: Danh sách RetrievedContext sắp xếp theo điểm BM25 giảm dần
        """
        if not self._is_indexed or not self.corpus_chunks:
            logger.debug("BM25Index chưa có dữ liệu hoặc rỗng. Trả về danh sách rỗng.")
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Chỉ tính toán các từ khóa có trong từ điển
        valid_query_tokens = [t for t in query_tokens if t in self.doc_freqs]
        if not valid_query_tokens:
            logger.debug(f"Không tìm thấy từ khóa nào của query '{query}' trong chỉ mục BM25.")
            return []

        scores: List[Tuple[float, int]] = []
        avg_len = self.avg_doc_len if self.avg_doc_len > 0 else 1.0

        for idx, tf_dict in enumerate(self.doc_term_freqs):
            doc_len = self.doc_lengths[idx]
            len_norm = 1.0 - self.b + self.b * (doc_len / avg_len)
            doc_score = 0.0

            for token in valid_query_tokens:
                if token in tf_dict:
                    freq = tf_dict[token]
                    idf_val = self.idf.get(token, 0.0)
                    # Công thức BM25+
                    tf_component = (freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm)
                    doc_score += idf_val * (tf_component + self.delta)

            if doc_score > 0.0:
                scores.append((doc_score, idx))

        if not scores:
            return []

        # Sắp xếp giảm dần theo điểm BM25
        scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scores[:top_k]

        max_score = top_candidates[0][0] if top_candidates and top_candidates[0][0] > 0 else 1.0

        results: List[RetrievedContext] = []
        for rank, (raw_score, doc_idx) in enumerate(top_candidates, 1):
            chunk = self.corpus_chunks[doc_idx]
            # Chuẩn hóa điểm BM25 về khoảng (0, 1] để đồng bộ giao diện
            normalized_score = round(min(1.0, max(0.01, raw_score / max_score)), 4)

            # Metadata enriched với thông tin điểm BM25 gốc
            enriched_meta = dict(chunk.metadata) if chunk.metadata else {}
            enriched_meta["bm25_raw_score"] = round(raw_score, 4)
            enriched_meta["retrieval_branch"] = "bm25"

            results.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=normalized_score,
                    rank=rank,
                    metadata=enriched_meta,
                )
            )

        return results

    def count(self) -> int:
        """Số lượng chunk đang được chỉ mục."""
        return len(self.corpus_chunks)

    def clear(self) -> None:
        """Xóa sạch chỉ mục BM25."""
        self.corpus_chunks.clear()
        self.doc_lengths.clear()
        self.doc_term_freqs.clear()
        self.doc_freqs.clear()
        self.idf.clear()
        self.avg_doc_len = 0.0
        self._is_indexed = False

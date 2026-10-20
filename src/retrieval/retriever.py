"""
src/retrieval/retriever.py - Phụ trách quy trình truy xuất ngữ cảnh (Retrieval Pipeline).
Thành viên phụ trách: Người làm Retrieval & Similarity Search.

Quy trình chuẩn hóa từ câu hỏi đến K chunks:
    User Query -> [Query Text & Query Embedding]
               -> [BM25 Search & Vector Search]
               -> [Hybrid Search Fusion - RRF: ~100 candidate chunks]
               -> [Reranker: 20 ranked chunks]
               -> [Top-K Selection: K chunks (mặc định K=5)]
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import (
    BaseEmbeddingModel,
    BaseVectorStore,
    DataChunk,
    RetrievedContext,
    UserQuery,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.reranker import BaseReranker, RerankerFactory
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("retrieval")


class UETRetriever:
    """
    Bộ điều phối truy xuất ngữ cảnh hoàn chỉnh (Hybrid Retrieval & Reranking Engine).
    Kết nối song song BM25 Search và Vector Search, dung hợp RRF, Rerank và trích xuất Top-K.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo UETRetriever.
        :param embedding_model: Mô hình nhúng câu hỏi kế thừa BaseEmbeddingModel
        :param vector_store: Cơ sở dữ liệu vector kế thừa BaseVectorStore
        :param reranker: Mô hình Reranker (nếu None sẽ tự động khởi tạo qua RerankerFactory)
        :param config: Dict cấu hình tham số tùy chỉnh (hoặc tự đọc từ config.yaml)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        # Đọc cấu hình từ config.yaml nếu không truyền trực tiếp
        raw_config = config or load_yaml_config()
        retrieval_cfg = raw_config.get("retrieval", {})

        # Tham số theo chuẩn Pipeline
        self.bm25_top_k: int = retrieval_cfg.get("bm25_top_k", 100)
        self.vector_top_k: int = retrieval_cfg.get("vector_top_k", 100)
        self.candidate_pool_size: int = retrieval_cfg.get("candidate_pool_size", 100)  # ~100 candidate chunks
        self.rerank_top_n: int = retrieval_cfg.get("rerank_top_n", 20)                 # 20 ranked chunks
        self.default_top_k: int = retrieval_cfg.get("final_top_k", 5)                  # K chunks (ví dụ 5)
        self.rrf_k: int = retrieval_cfg.get("rrf_k", 60)
        self.bm25_weight: float = retrieval_cfg.get("bm25_weight", 0.4)
        self.vector_weight: float = retrieval_cfg.get("vector_weight", 0.6)
        self.default_score_threshold: Optional[float] = retrieval_cfg.get("score_threshold", None)

        # Khởi tạo BM25 Index
        self.bm25_index = BM25Index()

        # Khởi tạo Reranker
        reranker_type = retrieval_cfg.get("reranker_type", "auto")
        reranker_model = retrieval_cfg.get("reranker_model", None)
        self.reranker = reranker or RerankerFactory.create(reranker_type=reranker_type, model_name=reranker_model)

        # Đồng bộ dữ liệu hiện có từ VectorStore sang BM25 Index (xử lý an toàn nếu dữ liệu chưa sẵn sàng)
        self._sync_bm25_from_vector_store()

    def _sync_bm25_from_vector_store(self) -> int:
        """
        Nạp BM25Index từ file chỉ mục database/bm25_index.pkl.
        Nếu không tìm thấy file, thông báo lỗi và yêu cầu chạy pipeline tạo chỉ mục, không fallback.
        """
        # Hỗ trợ mock / in-memory VectorStore trong Unit Test
        if hasattr(self.vector_store, "_chunks") and not hasattr(self.vector_store, "collection"):
            if self.vector_store._chunks:
                return self.bm25_index.index_chunks(list(self.vector_store._chunks))
            self.bm25_index.clear()
            return 0

        # Xác định đường dẫn file bm25_index.pkl
        bm25_path = BASE_DIR / "database" / "bm25_index.pkl"

        if not bm25_path or not bm25_path.exists():
            logger.error(
                f"Không tìm thấy file chỉ mục BM25 tại '{bm25_path}'. "
                "Vui lòng chạy lệnh tạo chỉ mục: 'python main.py --ingest --from chunked --to indexed_data'."
            )
            return 0

        try:
            from src.indexing.sparse import BM25Index as SparseBM25Index
            self.bm25_index = SparseBM25Index.load(bm25_path)
            count = getattr(self.bm25_index, "count", lambda: len(getattr(self.bm25_index, "chunks", [])))()
            logger.info(f"Đã nạp thành công {count} chunks vào BM25Index từ file '{bm25_path}'.")
            return count
        except Exception as e:
            logger.error(f"Không tìm thấy/Lỗi khi đọc file chỉ mục BM25 '{bm25_path}': {e}")
            return 0

    def index_chunks(self, chunks: List[DataChunk]) -> int:
        """
        Cung cấp giao diện cho phép nạp thêm hoặc cập nhật chỉ mục BM25 từ bên ngoài.
        :param chunks: Danh sách DataChunk mới
        :return: Số lượng chunk đã lập chỉ mục
        """
        return self.bm25_index.index_chunks(chunks)

    def retrieve(
        self,
        query: UserQuery,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Quy trình truy xuất ngữ cảnh đa tầng (Hybrid Multi-Stage Retrieval):
        1. BM25 Search (từ khóa) & Vector Search (ngữ nghĩa) song song.
        2. Hợp nhất Hybrid (RRF) -> ~100 candidate chunks.
        3. Reranker (Cross-Scoring) -> 20 ranked chunks.
        4. Top-K Selection -> K chunks (ví dụ K=5) trả về cho LLM.

        :param query: Câu hỏi người dùng (UserQuery)
        :param top_k: Số lượng văn bản liên quan cuối cùng cần lấy (mặc định 5)
        :param score_threshold: Ngưỡng điểm tin cậy tối thiểu (tùy chọn)
        :param filters: Bộ lọc metadata (ví dụ lọc theo khoa, năm học)
        :return: Danh sách RetrievedContext đã xếp hạng
        """
        query_text = (query.query_text or "").strip()
        if not query_text:
            logger.warning("Truy vấn người dùng rỗng. Trả về kết quả rỗng.")
            return []

        logger.info(f"Bắt đầu quy trình Hybrid Retrieval cho: '{query_text}'")

        # Đảm bảo BM25 Index đã nạp dữ liệu nếu ban đầu chưa có
        if self.bm25_index.count() == 0:
            self._sync_bm25_from_vector_store()

        effective_top_k = top_k or self.default_top_k
        active_filters = filters or query.filters

        # =====================================================================
        # BƯỚC 1: SONG SONG BM25 SEARCH VÀ VECTOR SEARCH
        # =====================================================================
        # 1.1. Nhánh BM25 Search
        bm25_results: List[RetrievedContext] = []
        try:
            bm25_results = self.bm25_index.search(query=query_text, top_k=self.bm25_top_k)
            logger.debug(f"[BM25] Thu được {len(bm25_results)} kết quả.")
        except Exception as e:
            logger.warning(f"Lỗi nhánh BM25 Search: {e}. Vẫn tiếp tục với Vector Search.")

        # 1.2. Nhánh Query Embedding & Vector Search
        vector_results: List[RetrievedContext] = []
        try:
            query_vector = self.embedding_model.embed_query(query)
            vector_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=self.vector_top_k,
                filters=active_filters,
            )
            logger.debug(f"[Vector Search] Thu được {len(vector_results)} kết quả.")
        except Exception as e:
            logger.warning(f"Lỗi nhánh Vector Search: {e}. Vẫn tiếp tục với BM25 Search.")

        # =====================================================================
        # BƯỚC 2: HYBRID SEARCH FUSION -> ~100 CANDIDATE CHUNKS
        # =====================================================================
        candidate_chunks = HybridSearcher.reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            top_n=self.candidate_pool_size,
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
        )

        if not candidate_chunks:
            logger.warning("Không tìm thấy đoạn văn bản nào từ cả 2 nhánh BM25 và Vector Search.")
            return []

        # =====================================================================
        # BƯỚC 3: RERANKER -> 20 RANKED CHUNKS
        # =====================================================================
        try:
            ranked_chunks = self.reranker.rerank(
                query=query_text,
                candidates=candidate_chunks,
                top_n=self.rerank_top_n,
            )
        except Exception as e:
            logger.warning(f"Lỗi khi chạy Reranker ({e}). Sử dụng thứ tự từ Hybrid Search.")
            ranked_chunks = candidate_chunks[:self.rerank_top_n]

        # =====================================================================
        # BƯỚC 4: TOP-K SELECTION -> K CHUNKS (MẶC ĐỊNH 5)
        # =====================================================================
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold

        # Lọc theo ngưỡng tin cậy nếu có
        filtered_chunks: List[RetrievedContext] = []
        if threshold is not None:
            filtered_chunks = [c for c in ranked_chunks if c.similarity_score >= threshold]
            if not filtered_chunks and ranked_chunks:
                # Nếu lọc quá chặt dẫn đến không còn chunk nào, giữ lại 1 chunk có điểm cao nhất
                logger.info(f"Không có chunk nào vượt qua ngưỡng {threshold}, giữ lại top 1 chunk tốt nhất.")
                filtered_chunks = [ranked_chunks[0]]
        else:
            filtered_chunks = ranked_chunks

        # Chọn top_k chunks cuối cùng
        final_contexts = filtered_chunks[:effective_top_k]

        # Chuẩn hóa lại thứ hạng từ 1 đến K cho danh sách trả về
        for i, ctx in enumerate(final_contexts, 1):
            ctx.rank = i

        logger.info(
            f"Hoàn thành Retrieval: từ {len(candidate_chunks)} candidates -> "
            f"{len(ranked_chunks)} reranked -> chọn ra {len(final_contexts)} chunks cuối cùng."
        )
        return final_contexts


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n🔍 ĐANG KIỂM THỬ ĐƠN VỊ MODULE RETRIEVAL (SRC/RETRIEVAL)...")

    # Mô hình Embedding giả lập phục vụ unit test độc lập
    class DummyEmbedder(BaseEmbeddingModel):
        @property
        def dimension(self) -> int:
            return 384

        @property
        def model_name(self) -> str:
            return "dummy-test-embedder"

        def embed_texts(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * self.dimension for _ in texts]

    # Vector Store giả lập
    class DummyStore(BaseVectorStore):
        def __init__(self):
            self.mock_chunks = [
                DataChunk(
                    chunk_id="chunk_test_1",
                    text="Sinh viên UET có điểm GPA từ 3.60 trở lên và điểm rèn luyện xuất sắc được nhận học bổng xuất sắc.",
                    metadata={"title": "Quy chế học bổng UET", "source": "quy_che_hoc_bong.pdf"},
                ),
                DataChunk(
                    chunk_id="chunk_test_2",
                    text="Điều kiện xét tốt nghiệp: tích lũy đủ số tín chỉ, GPA từ 2.00 trở lên, chuẩn đầu ra ngoại ngữ B2.",
                    metadata={"title": "Quy định tốt nghiệp", "source": "quy_che_tot_nghiep.pdf"},
                ),
                DataChunk(
                    chunk_id="chunk_test_3",
                    text="Sinh viên bị cảnh báo học tập nếu điểm trung bình chung học kỳ dưới 0.80 đối với học kỳ 1.",
                    metadata={"title": "Quy chế cảnh báo học vụ", "source": "canh_bao_hoc_vu.pdf"},
                ),
            ]

        def store(self, chunks, vectors=None) -> int:
            self.mock_chunks.extend(chunks)
            return len(chunks)

        def search(self, query_vector, top_k: int = 5, filters=None) -> List[RetrievedContext]:
            return [
                RetrievedContext(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    similarity_score=0.88 - (i * 0.1),
                    rank=i + 1,
                    metadata=c.metadata,
                )
                for i, c in enumerate(self.mock_chunks[:top_k])
            ]

        def count(self) -> int:
            return len(self.mock_chunks)

        def clear(self) -> None:
            self.mock_chunks.clear()

    # Khởi tạo UETRetriever
    store = DummyStore()
    retriever = UETRetriever(embedding_model=DummyEmbedder(), vector_store=store)
    retriever.index_chunks(store.mock_chunks)

    # Thử nghiệm câu hỏi
    test_q = UserQuery(query_text="học bổng xuất sắc UET cần điều kiện GPA bao nhiêu?")
    results = retriever.retrieve(test_q, top_k=2)

    print(f"\nCâu hỏi: {test_q.query_text}")
    print(f"Số kết quả lấy về: {len(results)}")
    for r in results:
        print(f" - [Rank {r.rank} | Score: {r.similarity_score}] {r.metadata.get('title')}: {r.text[:80]}...")

    assert len(results) > 0, "Lỗi: Không tìm thấy kết quả nào!"
    print("\n✅ KIỂM THỬ MODULE RETRIEVAL THÀNH CÔNG RỰC RỠ!\n")

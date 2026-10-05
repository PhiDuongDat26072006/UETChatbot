"""
tests/test_retrieval.py - Bộ kiểm thử đơn vị toàn diện cho phân hệ Retrieval.
Kiểm tra từng thành phần độc lập:
1. BM25Index (lập chỉ mục, tokenization tiếng Việt, tìm kiếm từ khóa, dữ liệu rỗng)
2. HybridSearcher (RRF hợp nhất, deduplication, danh sách khuyết thiếu)
3. Reranker (Heuristic, RerankerFactory)
4. UETRetriever End-to-End (BM25 + Vector -> Hybrid -> Rerank -> Top-K)
5. Xử lý an toàn khi kho dữ liệu chưa đầy đủ hoặc rỗng (Partial / Empty Data Handling)
"""

import unittest
from typing import List, Optional
from src.base import (
    BaseEmbeddingModel,
    BaseVectorStore,
    DataChunk,
    RetrievedContext,
    UserQuery,
)
from src.retrieval import (
    BM25Index,
    HybridSearcher,
    HeuristicReranker,
    RerankerFactory,
    UETRetriever,
)


class MockEmbeddingModel(BaseEmbeddingModel):
    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "mock-embedding-model"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.05 * (i + 1)] * self.dimension for i, _ in enumerate(texts)]


class MockVectorStore(BaseVectorStore):
    def __init__(self, chunks: Optional[List[DataChunk]] = None):
        self._chunks = list(chunks) if chunks else []

    def store(self, chunks: List[DataChunk], vectors=None) -> int:
        self._chunks.extend(chunks)
        return len(chunks)

    def search(self, query_vector, top_k: int = 5, filters=None) -> List[RetrievedContext]:
        results = []
        for i, c in enumerate(self._chunks[:top_k]):
            results.append(
                RetrievedContext(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    similarity_score=round(0.85 - (i * 0.05), 4),
                    rank=i + 1,
                    metadata=c.metadata,
                )
            )
        return results

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


class TestRetrievalPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            DataChunk(
                chunk_id="chunk_scholarship",
                text="Quy định xét cấp học bổng khuyến khích học tập: GPA >= 3.60 và điểm rèn luyện xuất sắc.",
                metadata={"title": "Quy chế học bổng", "doc_type": "decision"},
            ),
            DataChunk(
                chunk_id="chunk_graduation",
                text="Điều kiện tốt nghiệp UET: Tích lũy đủ tín chỉ, GPA >= 2.00, chuẩn tiếng Anh B2.",
                metadata={"title": "Quy chế tốt nghiệp", "doc_type": "regulation"},
            ),
            DataChunk(
                chunk_id="chunk_warning",
                text="Cảnh báo học vụ: Điểm trung bình học kỳ dưới 0.80 cho học kỳ đầu hoặc nợ quá 24 tín chỉ.",
                metadata={"title": "Cảnh báo học tập", "doc_type": "handbook"},
            ),
        ]

    def test_bm25_tokenization_and_search(self):
        """1. Kiểm tra BM25 tokenization tiếng Việt và tìm kiếm từ khóa."""
        index = BM25Index()
        # Test tokenization
        tokens = index.tokenize("Học bổng khuyến khích học tập UET")
        self.assertIn("học", tokens)
        self.assertIn("bổng", tokens)
        self.assertIn("học_bổng", tokens)  # Bigram tiếng Việt

        # Test index chunks
        count = index.index_chunks(self.sample_chunks)
        self.assertEqual(count, 3)

        # Test tìm kiếm
        results = index.search("học bổng xuất sắc", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk_scholarship")
        self.assertGreater(results[0].similarity_score, 0.0)

    def test_bm25_empty_query_and_empty_corpus(self):
        """2. Kiểm tra BM25 xử lý an toàn khi query hoặc kho dữ liệu rỗng."""
        index = BM25Index()
        # Chưa index gì
        self.assertEqual(index.search("bất kỳ câu nào"), [])

        # Index xong nhưng query rỗng
        index.index_chunks(self.sample_chunks)
        self.assertEqual(index.search(""), [])
        self.assertEqual(index.search("   "), [])

    def test_hybrid_searcher_rrf(self):
        """3. Kiểm tra Reciprocal Rank Fusion (RRF) kết hợp 2 nhánh."""
        bm25_res = [
            RetrievedContext(chunk_id="c1", text="Đoạn văn 1", similarity_score=0.9, rank=1),
            RetrievedContext(chunk_id="c2", text="Đoạn văn 2", similarity_score=0.7, rank=2),
        ]
        vec_res = [
            RetrievedContext(chunk_id="c2", text="Đoạn văn 2", similarity_score=0.88, rank=1),
            RetrievedContext(chunk_id="c3", text="Đoạn văn 3", similarity_score=0.65, rank=2),
        ]

        fused = HybridSearcher.reciprocal_rank_fusion(bm25_res, vec_res, top_n=10)
        self.assertEqual(len(fused), 3)  # c1, c2, c3 (c2 được deduplicate)

        # c2 xuất hiện ở cả 2 nhánh nên phải có điểm cao nhất
        self.assertEqual(fused[0].chunk_id, "c2")
        self.assertEqual(fused[0].rank, 1)

    def test_hybrid_searcher_edge_cases(self):
        """4. Kiểm tra Hybrid search khi 1 trong 2 nhánh hoặc cả 2 đều rỗng."""
        bm25_res = [RetrievedContext(chunk_id="c1", text="Text 1", similarity_score=0.8, rank=1)]
        empty_res = []

        # Chỉ có BM25
        res_bm25_only = HybridSearcher.reciprocal_rank_fusion(bm25_res, empty_res)
        self.assertEqual(len(res_bm25_only), 1)

        # Chỉ có Vector
        res_vec_only = HybridSearcher.reciprocal_rank_fusion(empty_res, bm25_res)
        self.assertEqual(len(res_vec_only), 1)

        # Cả 2 đều rỗng
        res_both_empty = HybridSearcher.reciprocal_rank_fusion(empty_res, empty_res)
        self.assertEqual(len(res_both_empty), 0)

    def test_heuristic_reranker(self):
        """5. Kiểm tra Heuristic Reranker tái xếp hạng."""
        reranker = HeuristicReranker()
        candidates = [
            RetrievedContext(chunk_id="c1", text="Quy định học phí học kỳ mới", similarity_score=0.5, rank=1),
            RetrievedContext(chunk_id="c2", text="Học bổng khuyến khích học tập xuất sắc", similarity_score=0.45, rank=2),
        ]
        reranked = reranker.rerank(query="học bổng xuất sắc", candidates=candidates, top_n=2)
        self.assertEqual(len(reranked), 2)
        # c2 có độ khớp từ khóa câu hỏi cao hơn nhiều nên được đẩy lên rank 1
        self.assertEqual(reranked[0].chunk_id, "c2")

    def test_reranker_factory(self):
        """6. Kiểm tra RerankerFactory khởi tạo backend hợp lệ."""
        reranker = RerankerFactory.create(reranker_type="heuristic")
        self.assertIsInstance(reranker, HeuristicReranker)

        auto_reranker = RerankerFactory.create(reranker_type="auto")
        self.assertIsNotNone(auto_reranker)

    def test_uet_retriever_full_flow(self):
        """7. Kiểm tra UETRetriever toàn bộ quy trình: Query -> K chunks."""
        embedder = MockEmbeddingModel()
        store = MockVectorStore(self.sample_chunks)
        retriever = UETRetriever(embedding_model=embedder, vector_store=store)
        retriever.index_chunks(self.sample_chunks)

        query = UserQuery(query_text="Điều kiện xét học bổng khuyến khích học tập")
        results = retriever.retrieve(query, top_k=2)

        self.assertLessEqual(len(results), 2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].chunk_id, "chunk_scholarship")
        self.assertIn("retrieval_method", results[0].metadata)

    def test_partial_data_handling(self):
        """8. Kiểm tra khi dữ liệu chưa đầy đủ hoặc rỗng (theo yêu cầu người dùng)."""
        embedder = MockEmbeddingModel()
        empty_store = MockVectorStore(chunks=[])
        retriever = UETRetriever(embedding_model=embedder, vector_store=empty_store)

        # Chạy truy vấn trên store rỗng -> Không được văng exception, trả về []
        query = UserQuery(query_text="Câu hỏi khi chưa có dữ liệu")
        results = retriever.retrieve(query, top_k=5)
        self.assertEqual(results, [])

        # Nạp dữ liệu một phần sau đó
        single_chunk = [
            DataChunk(chunk_id="chunk_single", text="Dữ liệu nạp thêm 1 phần", metadata={})
        ]
        retriever.index_chunks(single_chunk)
        empty_store.store(single_chunk)

        results_after = retriever.retrieve(query, top_k=5)
        self.assertGreaterEqual(len(results_after), 1)
        self.assertEqual(results_after[0].chunk_id, "chunk_single")


if __name__ == "__main__":
    unittest.main()

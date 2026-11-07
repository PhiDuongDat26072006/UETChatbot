"""
tests/test_mmr.py - Bộ kiểm thử đơn vị cho thuật toán Maximal Marginal Relevance (MMR).
Kiểm tra:
1. Khử trùng lặp ngữ nghĩa (Near-duplicate suppression & Diversity optimization)
2. Giá trị biên của lambda (lambda = 1.0 vs lambda = 0.0)
3. Các trường hợp ngoại lệ (Danh sách rỗng, N <= top_n, vector 0)
4. Tích hợp MMR trong UETRetriever và cơ chế Fallback logging khi thiếu vector trong database
"""

import unittest
import numpy as np
from typing import Dict, List, Optional

from src.base import (
    BaseEmbeddingModel,
    BaseVectorStore,
    DataChunk,
    RetrievedContext,
    UserQuery,
)
from src.retrieval.mmr import MaximalMarginalRelevance
from src.retrieval.retriever import UETRetriever


class TestMaximalMarginalRelevance(unittest.TestCase):

    def setUp(self):
        # Query vector: Quan tâm đồng thời cả khía cạnh 0 và khía cạnh 1
        self.query_vec = [1.0, 1.0, 0.0]

        # 5 ứng viên:
        # c1: Bao phủ khía cạnh 0
        # c2: Trùng lặp ngữ nghĩa với c1 (gần như giống hệt c1)
        # c3: Bao phủ khía cạnh 1 (chủ đề mới quan trọng)
        # c4: Trùng lặp ngữ nghĩa với c3 (gần như giống hệt c3)
        # c5: Bao phủ khía cạnh 2 (độc lập)
        self.candidates = [
            RetrievedContext(chunk_id="c1", text="Quy định học bổng xuất sắc GPA >= 3.6", similarity_score=0.95),
            RetrievedContext(chunk_id="c2", text="Tiêu chuẩn học bổng khuyến khích GPA >= 3.6", similarity_score=0.94),
            RetrievedContext(chunk_id="c3", text="Điều kiện xét tốt nghiệp chuẩn tiếng Anh B2", similarity_score=0.95),
            RetrievedContext(chunk_id="c4", text="Quy định chuẩn đầu ra ngoại ngữ tốt nghiệp B2", similarity_score=0.94),
            RetrievedContext(chunk_id="c5", text="Cảnh báo học tập khi nợ quá 24 tín chỉ", similarity_score=0.50),
        ]
        self.candidate_vecs = [
            [1.0, 0.0, 0.0],       # c1: Aspect 0
            [0.999, 0.01, 0.0],    # c2: Trùng lặp với c1
            [0.0, 1.0, 0.0],       # c3: Aspect 1
            [0.0, 0.999, 0.01],    # c4: Trùng lặp với c3
            [0.0, 0.0, 1.0],       # c5: Aspect 2
        ]

    def test_mmr_suppresses_redundancy(self):
        """1. Kiểm tra MMR loại bỏ các đoạn trùng lặp, ưu tiên chọn các khía cạnh đa dạng."""
        # Lấy top 3 ứng viên với lambda = 0.5 (cân bằng đều Relevance và Diversity)
        results = MaximalMarginalRelevance.select(
            query_vector=self.query_vec,
            candidates=self.candidates,
            candidate_vectors=self.candidate_vecs,
            lambda_mult=0.5,
            top_n=3,
        )

        self.assertEqual(len(results), 3)
        selected_ids = [r.chunk_id for r in results]

        # Khía cạnh 0 (c1, c2): MMR chỉ chọn duy nhất 1 đại diện tốt nhất, loại bỏ đoạn trùng lặp còn lại
        aspect_0_count = sum(1 for cid in ["c1", "c2"] if cid in selected_ids)
        self.assertEqual(aspect_0_count, 1, f"MMR phải chọn đúng 1 đại diện cho khía cạnh 0, thực tế: {selected_ids}")

        # Khía cạnh 1 (c3, c4): MMR chỉ chọn duy nhất 1 đại diện tốt nhất, loại bỏ đoạn trùng lặp còn lại
        aspect_1_count = sum(1 for cid in ["c3", "c4"] if cid in selected_ids)
        self.assertEqual(aspect_1_count, 1, f"MMR phải chọn đúng 1 đại diện cho khía cạnh 1, thực tế: {selected_ids}")

        # Khía cạnh 2 (c5): Đoạn độc lập, không bị trùng lặp -> được chọn vào top 3
        self.assertIn("c5", selected_ids, f"c5 đại diện cho khía cạnh độc lập phải có trong top 3: {selected_ids}")

    def test_lambda_extremes(self):
        """2. Kiểm tra giá trị biên: lambda = 1.0 (thuần Relevance) vs lambda = 0.0 (thuần Diversity)."""
        # Khi lambda = 1.0: hoàn toàn theo query similarity
        # c1 và c3 có similarity cao nhất với query [1, 1, 0], tiếp theo là c2, c4, c5
        res_rel = MaximalMarginalRelevance.select(
            query_vector=self.query_vec,
            candidates=self.candidates,
            candidate_vectors=self.candidate_vecs,
            lambda_mult=1.0,
            top_n=3,
        )
        selected_rel_ids = [r.chunk_id for r in res_rel]
        # c5 có similarity thấp nhất (0.0) nên không thể lọt vào top 3 khi lambda = 1.0
        self.assertNotIn("c5", selected_rel_ids)

        # Khi lambda = 0.0: sau chunk đầu tiên, ưu tiên chọn các vector trực giao/xa nhất
        res_div = MaximalMarginalRelevance.select(
            query_vector=self.query_vec,
            candidates=self.candidates,
            candidate_vectors=self.candidate_vecs,
            lambda_mult=0.0,
            top_n=3,
        )
        selected_div_ids = [r.chunk_id for r in res_div]
        # c5 trực giao với cả c1 và c3 nên khi tối đa hóa diversity sẽ được chọn
        self.assertIn("c5", selected_div_ids)

    def test_edge_cases(self):
        """3. Kiểm tra các trường hợp ngoại lệ: danh sách rỗng, N <= top_n, vector 0."""
        # Danh sách rỗng
        empty_res = MaximalMarginalRelevance.select(
            query_vector=self.query_vec,
            candidates=[],
            candidate_vectors=[],
            top_n=3,
        )
        self.assertEqual(empty_res, [])

        # Số ứng viên nhỏ hơn hoặc bằng top_n -> trả về nguyên vẹn
        small_res = MaximalMarginalRelevance.select(
            query_vector=self.query_vec,
            candidates=self.candidates[:2],
            candidate_vectors=self.candidate_vecs[:2],
            top_n=5,
        )
        self.assertEqual(len(small_res), 2)

        # Vector 0 (không gây lỗi ZeroDivisionError)
        zero_vecs = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        safe_res = MaximalMarginalRelevance.select(
            query_vector=[0.0, 0.0, 0.0],
            candidates=self.candidates[:2],
            candidate_vectors=zero_vecs,
            top_n=2,
        )
        self.assertEqual(len(safe_res), 2)


class MockEmbeddingWithCount(BaseEmbeddingModel):
    def __init__(self, dim: int = 16):
        self._dim = dim
        self.embed_texts_calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "mock-embedder-test"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self.embed_texts_calls += len(texts)
        return [[0.1 * (i + 1)] * self._dim for i, _ in enumerate(texts)]


class MockVectorStoreWithEmbeddingLookup(BaseVectorStore):
    def __init__(self, chunks: List[DataChunk]):
        self.chunks = list(chunks)
        self.stored_embeddings: Dict[str, List[float]] = {
            c.chunk_id: [0.1 * (i + 1)] * 16 for i, c in enumerate(chunks)
        }

    def store(self, chunks, vectors=None) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def search(self, query_vector, top_k: int = 5, filters=None) -> List[RetrievedContext]:
        return [
            RetrievedContext(
                chunk_id=c.chunk_id,
                text=c.text,
                similarity_score=round(0.9 - (i * 0.01), 4),
                rank=i + 1,
                metadata=c.metadata,
            )
            for i, c in enumerate(self.chunks[:top_k])
        ]

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.stored_embeddings.clear()

    def get_embeddings_by_ids(self, chunk_ids: List[str]) -> Dict[str, List[float]]:
        return {cid: self.stored_embeddings[cid] for cid in chunk_ids if cid in self.stored_embeddings}


class MockBM25SearchOnly:
    """Mock BM25Index nhẹ cho unit test, tránh xung đột đường dẫn tiếng Việt của underthesea pycrfsuite."""
    def __init__(self, chunks: List[DataChunk]):
        self.chunks = chunks

    def search(self, query: str, top_k: int = 50) -> List[RetrievedContext]:
        return [
            RetrievedContext(
                chunk_id=c.chunk_id,
                text=c.text,
                similarity_score=round(10.0 - (i * 0.1), 2),
                rank=i + 1,
                metadata=c.metadata,
            )
            for i, c in enumerate(self.chunks[:top_k])
        ]


class TestRetrieverMMRIntegration(unittest.TestCase):

    def test_mmr_integration_and_fallback_logging(self):
        """4. Kiểm thử tích hợp MMR trong UETRetriever và kiểm tra log khi kích hoạt Fallback."""
        # Tạo 50 chunks giả lập
        mock_chunks = [
            DataChunk(
                chunk_id=f"chunk_{i}",
                text=f"Văn bản quy chế đào tạo điều {i} về học vụ UET.",
                metadata={"title": f"Quy định điều {i}"},
            )
            for i in range(50)
        ]

        embedder = MockEmbeddingWithCount(dim=16)
        store = MockVectorStoreWithEmbeddingLookup(chunks=mock_chunks)
        mock_bm25 = MockBM25SearchOnly(chunks=mock_chunks)

        # Giả lập làm mất vector của 3 chunk trong database để kích hoạt Fallback
        for missing_id in ["chunk_5", "chunk_10", "chunk_15"]:
            if missing_id in store.stored_embeddings:
                del store.stored_embeddings[missing_id]

        cfg = {
            "retrieval": {
                "bm25_top_k": 50,
                "vector_top_k": 50,
                "candidate_pool_size": 50,
                "enable_mmr": True,
                "mmr_lambda": 0.6,
                "mmr_top_n": 25,  # Từ 50 candidates chọn lọc 25 qua MMR
                "rerank_top_n": 10,
                "final_top_k": 5,
                "reranker_type": "heuristic",
            }
        }

        retriever = UETRetriever(
            embedding_model=embedder,
            vector_store=store,
            bm25_index=mock_bm25,  # Inject mock BM25
            config=cfg,
        )

        # Sử dụng assertLogs để kiểm tra BẮT BUỘC có log warning về [MMR Fallback]
        with self.assertLogs("retrieval", level="WARNING") as cm:
            results = retriever.retrieve(UserQuery(query_text="học vụ UET quy định điều 5"), top_k=5)

        # 1. Kết quả trả về đủ 5 chunks
        self.assertEqual(len(results), 5)

        # 2. Kiểm tra log warning [MMR Fallback] xuất hiện
        fallback_logged = any("[MMR Fallback]" in msg for msg in cm.output)
        self.assertTrue(
            fallback_logged,
            f"Không tìm thấy log '[MMR Fallback]' trong output: {cm.output}"
        )

        # 3. Kiểm tra embed_texts được gọi để sinh vector bổ sung cho các chunk thiếu
        self.assertGreater(embedder.embed_texts_calls, 0)


if __name__ == "__main__":
    unittest.main()

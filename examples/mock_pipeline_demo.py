"""
========================================================================================
VÍ DỤ MINH HỌA CÁCH LÀM VIỆC SONG SONG VỚI BASE.PY
File này mô phỏng cách từng thành viên trong nhóm viết code kế thừa từ base.py
và cách ghép nối chúng thành một hệ thống hoàn chỉnh.
========================================================================================
"""

import sys
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được base.py từ thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Optional
from base import (
    DataSource,
    RawData,
    ProcessedData,
    DataChunk,
    EmbeddedVector,
    UserQuery,
    EmbeddedQueryVector,
    RetrievedContext,
    AugmentedPrompt,
    Response,
    BaseDataCrawler,
    BasePreprocessor,
    BaseChunker,
    BaseEmbeddingModel,
    BaseVectorStore,
    BasePromptAugmenter,
    BaseLLM,
    BaseIngestionPipeline,
    BaseRAGPipeline,
)


# ========================================================================================
# BẠN A PHỤ TRÁCH: Data Crawling (Kế thừa BaseDataCrawler)
# ========================================================================================
class MockUETCrawer(BaseDataCrawler):
    def crawl(self, source: DataSource) -> List[RawData]:
        print(f"[Bạn A - Crawl]: Đang cào dữ liệu từ {source.uri}...")
        # Giả lập cào được 1 bài viết quy chế đào tạo
        return [
            RawData(
                source_uri=source.uri,
                title="Quy định xét học bổng khuyến khích học tập UET",
                content="<p>Sinh viên đạt điểm GPA từ 3.6 trở lên và điểm rèn luyện xuất sắc được xét học bổng loại Xuất sắc.</p>",
                raw_metadata={"author": "Phòng Đào Tạo", "date": "2026-09-01"},
            )
        ]


# ========================================================================================
# BẠN B PHỤ TRÁCH: Pre-Processing (Kế thừa BasePreprocessor)
# ========================================================================================
class MockPreprocessor(BasePreprocessor):
    def process(self, raw: RawData) -> ProcessedData:
        print(f"[Bạn B - Preprocess]: Đang làm sạch nội dung cho bài viết '{raw.title}'...")
        # Giả lập bóc tách HTML, làm sạch text
        cleaned_text = raw.content.replace("<p>", "").replace("</p>", "").strip()
        return ProcessedData(
            raw_data_id=raw.id,
            title=raw.title or "Văn bản đào tạo UET",
            content=cleaned_text,
            metadata={"source": raw.source_uri, "loai_van_ban": "Học bổng"},
        )


# ========================================================================================
# BẠN C PHỤ TRÁCH: Chunking (Kế thừa BaseChunker)
# ========================================================================================
class MockChunker(BaseChunker):
    def chunk(self, doc: ProcessedData) -> List[DataChunk]:
        print(f"[Bạn C - Chunking]: Đang cắt nhỏ văn bản '{doc.title}'...")
        # Giả lập chia thành các chunk nhỏ
        return [
            DataChunk(
                document_id=doc.id,
                text=doc.content,
                chunk_index=0,
                metadata={"title": doc.title, **doc.metadata},
            )
        ]


# ========================================================================================
# BẠN D PHỤ TRÁCH: Embedding & Vector Store (Kế thừa BaseEmbeddingModel & BaseVectorStore)
# ========================================================================================
class MockEmbeddingModel(BaseEmbeddingModel):
    @property
    def dimension(self) -> int:
        # 384 chiều khớp với mô hình all-MiniLM-L6-v2 của ChromaDB
        return 384

    @property
    def model_name(self) -> str:
        return "all-MiniLM-L6-v2-mock"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Giả lập vector embedding 384 chiều cho việc kiểm thử
        return [[0.01 * (i + 1) for i in range(384)] for _ in texts]


class MockVectorStore(BaseVectorStore):
    """
    Vector Store có khả năng tự động kết nối tới kho dữ liệu thật vector_db/
    (ChromaDB 452 chunks) nếu tồn tại, hoặc fallback sang in-memory mock.
    """
    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "uet_knowledge_base"):
        self._db: List[DataChunk] = []
        self.persist_dir = persist_dir or (BASE_DIR / "database" / "vector_db" if (BASE_DIR / "database" / "vector_db").exists() else BASE_DIR / "vector_db")
        self.collection_name = collection_name
        self._chroma_col = None

        # Kiểm tra và kết nối ChromaDB cục bộ nếu thư mục vector_db tồn tại
        if (self.persist_dir / "chroma.sqlite3").exists():
            try:
                import chromadb
                import chromadb.utils.embedding_functions as ef
                client = chromadb.PersistentClient(path=str(self.persist_dir))
                fn = ef.DefaultEmbeddingFunction()
                self._chroma_col = client.get_collection(self.collection_name, embedding_function=fn)
                print(f"[Bạn D - Store]: ĐÃ KẾT NỐI ChromaDB THẬT tại '{self.persist_dir.name}' "
                      f"(Tìm thấy {self._chroma_col.count()} vectors trong collection '{self.collection_name}')")
            except Exception as e:
                print(f"[Bạn D - Store]: Không thể đọc ChromaDB ({e}), dùng in-memory mock.")

    def store(self, chunks: List[DataChunk], vectors: Optional[List[EmbeddedVector]] = None) -> int:
        print(f"[Bạn D - Store]: Đang lưu {len(chunks)} chunks vào bộ nhớ...")
        for chunk in chunks:
            self._db.append(chunk)
        return len(chunks)

    def search(self, query_vector: EmbeddedQueryVector, top_k: int = 5, filters=None) -> List[RetrievedContext]:
        print(f"[Bạn D - Search]: Đang tìm kiếm độ tương đồng với câu hỏi: '{query_vector.query_text}'...")
        
        # ƯU TIÊN 1: Tra cứu trên 452 embedded vector thật của ChromaDB nếu có
        if self._chroma_col is not None and self._chroma_col.count() > 0:
            try:
                res = self._chroma_col.query(query_texts=[query_vector.query_text], n_results=top_k)
                contexts = []
                if res and res.get("documents") and len(res["documents"][0]) > 0:
                    for i in range(len(res["documents"][0])):
                        doc_text = res["documents"][0][i]
                        meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                        cid = res["ids"][0][i] if res.get("ids") else f"chunk_{i}"
                        dist = res["distances"][0][i] if res.get("distances") else 0.5
                        # Chuyển đổi distance L2 sang similarity score [0.0, 1.0]
                        score = round(max(0.1, 1.0 - (dist / 2.0)), 2)
                        contexts.append(
                            RetrievedContext(
                                chunk_id=cid,
                                text=doc_text,
                                similarity_score=score,
                                rank=i + 1,
                                metadata=meta,
                            )
                        )
                    print(f"[Bạn D - Search]: Đã tìm thấy {len(contexts)} văn bản thật từ vector_db!")
                    return contexts
            except Exception as err:
                print(f"[Bạn D - Search]: Lỗi truy vấn ChromaDB ({err}), fallback về in-memory.")

        # ƯU TIÊN 2: Fallback in-memory
        contexts = []
        for i, chunk in enumerate(self._db[:top_k]):
            contexts.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=0.95,
                    rank=i + 1,
                    metadata=chunk.metadata,
                )
            )
        return contexts

    def count(self) -> int:
        if self._chroma_col is not None:
            return self._chroma_col.count()
        return len(self._db)

    def clear(self) -> None:
        self._db.clear()


# ========================================================================================
# BẠN E PHỤ TRÁCH: Prompt Augmentation & LLM API (Kế thừa BasePromptAugmenter & BaseLLM)
# ========================================================================================
class MockPromptAugmenter(BasePromptAugmenter):
    def augment(self, query: UserQuery, contexts: List[RetrievedContext]) -> AugmentedPrompt:
        print(f"[Bạn E - Augment]: Ghép câu hỏi và {len(contexts)} đoạn ngữ cảnh vào Prompt...")
        context_str = "\n".join([f"- {c.text}" for c in contexts])
        system_instruction = (
            "Bạn là trợ lý AI tư vấn quy chế đào tạo trường ĐH Công nghệ - ĐHQGHN (UET). "
            "Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp."
        )
        formatted = f"{system_instruction}\n\nNgữ cảnh:\n{context_str}\n\nCâu hỏi: {query.query_text}\nTrả lời:"
        return AugmentedPrompt(
            user_query=query,
            contexts=contexts,
            system_instruction=system_instruction,
            formatted_prompt=formatted,
        )


class MockLLM(BaseLLM):
    @property
    def model_name(self) -> str:
        return "gemini-2.5-flash-mock"

    def generate(self, prompt: AugmentedPrompt) -> Response:
        print(f"[Bạn E - LLM]: Đang gọi API mô hình {self.model_name}...")
        answer = "Theo quy định UET, sinh viên cần đạt GPA từ 3.6 trở lên và rèn luyện xuất sắc để được học bổng xuất sắc."
        return Response(
            query_id=prompt.user_query.query_id,
            answer=answer,
            sources=prompt.contexts,
            model_name=self.model_name,
            latency_seconds=0.45,
        )


# ========================================================================================
# CHẠY THỬ NGHIỆM HỆ THỐNG GHÉP NỐI (PIPELINE DEMO)
# ========================================================================================
if __name__ == "__main__":
    print("=== 1. KHỞI TẠO CÁC MODULE DO CÁC THÀNH VIÊN VIẾT ===")
    crawler = MockUETCrawer()
    preprocessor = MockPreprocessor()
    chunker = MockChunker()
    embedding = MockEmbeddingModel()
    vector_store = MockVectorStore()
    prompt_augmenter = MockPromptAugmenter()
    llm = MockLLM()

    print("\n=== 2. CHẠY OFFLINE PIPELINE (INGESTION & INDEXING) ===")
    ingestion_pipeline = BaseIngestionPipeline(
        crawler=crawler,
        preprocessor=preprocessor,
        chunker=chunker,
        embedding_model=embedding,
        vector_store=vector_store,
    )

    sources = [DataSource(source_type="web", uri="https://uet.vnu.edu.vn/hoc-bong")]
    ingest_result = ingestion_pipeline.run(sources)
    print("Kết quả Ingestion:", ingest_result)

    print("\n=== 3. CHẠY ONLINE PIPELINE (RETRIEVAL & GENERATION) ===")
    rag_pipeline = BaseRAGPipeline(
        embedding_model=embedding,
        vector_store=vector_store,
        prompt_augmenter=prompt_augmenter,
        llm=llm,
    )

    user_query = UserQuery(query_text="Điều kiện để nhận học bổng xuất sắc tại UET là gì?")
    response = rag_pipeline.query(user_query)

    print("\n" + "=" * 50)
    print("KẾT QUẢ PHẢN HỒI CHO SINH VIÊN:")
    print("Câu hỏi:", user_query.query_text)
    print("Trả lời:", response.answer)
    print("Nguồn tham khảo:")
    for src in response.sources:
        print(f" - [Độ khớp {src.similarity_score}]: {src.text} (Nguồn: {src.metadata.get('title')})")
    print("=" * 50)

"""
========================================================================================
KIẾN TRÚC NỀN TẢNG UET CHATBOT (BASE ARCHITECTURE)
Phục vụ phân chia công việc & phát triển song song cho nhóm dự án UET Chatbot.
Mỗi thành viên cài đặt (implement) một hoặc nhiều lớp trừu tượng (Abstract Base Classes)
dưới đây nhưng bắt buộc phải tuân thủ đúng định dạng Input / Output (Data Models).
========================================================================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional
from pydantic import BaseModel, Field
import uuid


# ========================================================================================
# 1. DATA MODELS (MIDDLE DATA / INPUT / OUTPUT THEO SƠ ĐỒ)
# ========================================================================================

class DataSource(BaseModel):
    """
    [Input - Khối tam giác xanh]
    Nguồn dữ liệu đầu vào cần thu thập (Website UET, trang Đào tạo, thư mục tài liệu PDF...).
    """
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID định danh nguồn")
    source_type: str = Field(..., description="Loại nguồn: 'web', 'file', 'api', 'directory'...")
    uri: str = Field(..., description="Đường dẫn: URL (https://uet.vnu.edu.vn/...) hoặc đường dẫn file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata bổ sung: khoa, ngày bắt đầu, độ sâu crawl...")


class RawData(BaseModel):
    """
    [Middle Data - Khối chữ nhật hồng 'raw data']
    Dữ liệu thô thu thập được từ bước Data Crawling trước khi làm sạch.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Mã định danh dữ liệu thô")
    content: str = Field(..., description="Nội dung thô (HTML, text nguyên bản, mã Markdown...)")
    source_uri: str = Field(..., description="Nguồn gốc tài liệu (URL bài viết, tên file gốc...)")
    title: Optional[str] = Field(default=None, description="Tiêu đề bài viết hoặc tên văn bản quy chế")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata thô: ngày đăng, người đăng, tags...")
    created_at: datetime = Field(default_factory=datetime.now, description="Thời điểm thu thập")


class ProcessedData(BaseModel):
    """
    [Middle Data - Khối chữ nhật hồng 'processed data']
    Dữ liệu văn bản đã qua tiền xử lý: loại bỏ thẻ HTML thừa, làm sạch, chuẩn hóa Tiếng Việt.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Mã dữ liệu đã xử lý")
    raw_data_id: Optional[str] = Field(default=None, description="Tham chiếu tới ID của RawData tương ứng")
    title: str = Field(..., description="Tiêu đề chuẩn hóa của bài viết / quy chế")
    content: str = Field(..., description="Nội dung văn bản sạch, chuẩn hóa ngữ pháp / chính tả")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata chuẩn: khoa_ban, nam_hoc, he_dao_tao (chính quy, CLC), so_hieu_van_ban...",
    )


class DataChunk(BaseModel):
    """
    [Middle Data - Khối chữ nhật hồng 'data chunk']
    Đoạn văn bản sau khi chia nhỏ (Chunking) từ ProcessedData kèm metadata hỗ trợ truy vết.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID định danh chunk")
    document_id: Optional[str] = Field(default=None, description="ID của ProcessedData cha")
    text: str = Field(..., description="Nội dung văn bản của đoạn chunk")
    chunk_index: int = Field(default=0, description="Thứ tự của chunk trong văn bản gốc")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata chi tiết: source, title, page_number, chapter, article (Điều/Khoản)...",
    )


class EmbeddedVector(BaseModel):
    """
    [Middle Data - Khối chữ nhật hồng 'Embedded Vector']
    Vector nhúng nhiều chiều được tạo từ một DataChunk qua mô hình Embedding.
    """
    chunk_id: str = Field(..., description="ID của chunk được nhúng vector")
    vector: List[float] = Field(..., description="Mảng số thực biểu diễn vector embedding")
    dimension: int = Field(..., description="Số chiều của vector (ví dụ 384, 768, 1536)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata đi kèm vector để filter khi search")


class UserQuery(BaseModel):
    """
    [Input - Khối tam giác vàng 'User query']
    Câu hỏi và thông tin truy vấn từ người dùng (sinh viên, giảng viên).
    """
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Mã truy vấn")
    query_text: str = Field(..., description="Nội dung câu hỏi của người dùng")
    session_id: Optional[str] = Field(default=None, description="Mã phiên hội thoại (để theo dõi ngữ cảnh)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Bộ lọc truy vấn (ví dụ lọc theo năm học, hệ đào tạo)")


class EmbeddedQueryVector(BaseModel):
    """
    [Middle Data - Khối chữ nhật cam 'Embedded query vector']
    Vector nhúng nhiều chiều của câu hỏi UserQuery.
    """
    query_id: str = Field(..., description="Mã truy vấn tương ứng")
    query_text: str = Field(..., description="Câu hỏi gốc")
    vector: List[float] = Field(..., description="Mảng số thực biểu diễn vector của câu hỏi")
    dimension: int = Field(..., description="Số chiều của vector")


class RetrievedContext(BaseModel):
    """
    [Middle Data - Khối chữ nhật cam 'Retrieved contexts']
    Đoạn ngữ cảnh trích xuất từ Vector Database có độ liên quan cao nhất với câu hỏi.
    """
    chunk_id: str = Field(..., description="ID của chunk được tìm thấy")
    text: str = Field(..., description="Nội dung trích đoạn văn bản")
    similarity_score: float = Field(..., description="Điểm số tương đồng (Cosine similarity / Distance)")
    rank: int = Field(default=1, description="Thứ hạng liên quan (1 là liên quan nhất)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata gốc: tên văn bản, link nguồn, trang...")


class AugmentedPrompt(BaseModel):
    """
    [Middle Data - Khối chữ nhật cam 'Augmented Prompt']
    Prompt hoàn chỉnh sau khi ghép User Query + Retrieved Contexts + Chỉ dẫn hệ thống (System Prompt).
    """
    user_query: UserQuery = Field(..., description="Câu hỏi ban đầu của người dùng")
    contexts: List[RetrievedContext] = Field(..., description="Danh sách các đoạn ngữ cảnh trích xuất")
    system_instruction: str = Field(..., description="Chỉ dẫn phong cách trả lời (vai trò chuyên viên đào tạo UET...)")
    formatted_prompt: str = Field(..., description="Toàn bộ chuỗi prompt chuẩn bị gửi cho LLM")


class Response(BaseModel):
    """
    [Output - Khối tròn xanh 'Response']
    Câu trả lời cuối cùng gửi về cho người dùng kèm trích dẫn nguồn.
    """
    query_id: str = Field(..., description="ID câu hỏi tương ứng")
    answer: str = Field(..., description="Nội dung câu trả lời do LLM sinh ra")
    sources: List[RetrievedContext] = Field(default_factory=list, description="Danh sách các nguồn trích dẫn dẫn chứng")
    model_name: Optional[str] = Field(default=None, description="Tên mô hình LLM đã dùng")
    latency_seconds: Optional[float] = Field(default=None, description="Thời gian xử lý tính bằng giây")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Thông tin phụ: token usage, debug info...")


# ========================================================================================
# 2. ABSTRACT PROCESS CLASSES (CÁC QUY TRÌNH - GIAO DIỆN TRỪU TƯỢNG)
# ========================================================================================

class BaseDataCrawler(ABC):
    """
    [Process - Khối xanh 'Data Crawling']
    Quy trình cào dữ liệu từ Data Source (Web UET, Portal sinh viên, hoặc tệp tài liệu).
    Thành viên phụ trách: Người thu thập dữ liệu (Crawler).
    """

    @abstractmethod
    def crawl(self, source: DataSource) -> List[RawData]:
        """
        Cào dữ liệu từ 1 nguồn cụ thể.
        :param source: Nguồn dữ liệu (URL, thư mục, tệp...)
        :return: Danh sách các đối tượng RawData thô.
        """
        pass

    def crawl_all(self, sources: List[DataSource]) -> List[RawData]:
        """Cào từ danh sách nhiều nguồn."""
        results: List[RawData] = []
        for src in sources:
            results.extend(self.crawl(src))
        return results


class BasePreprocessor(ABC):
    """
    [Process - Khối xanh 'Pre-Processing']
    Quy trình làm sạch dữ liệu thô (loại bỏ HTML, banner, script, chuẩn hóa tiếng Việt).
    Thành viên phụ trách: Người xử lý & chuẩn hóa dữ liệu.
    """

    @abstractmethod
    def process(self, raw: RawData) -> ProcessedData:
        """
        Tiền xử lý một tài liệu thô thành tài liệu sạch.
        :param raw: Dữ liệu thô ban đầu
        :return: Dữ liệu văn bản đã chuẩn hóa
        """
        pass

    def process_batch(self, raw_list: List[RawData]) -> List[ProcessedData]:
        """Xử lý hàng loạt tài liệu thô."""
        return [self.process(raw) for raw in raw_list]


class BaseChunker(ABC):
    """
    [Process - Khối xanh 'Chunking']
    Quy trình cắt nhỏ văn bản thành các đoạn (chunks) hợp lý kèm overlap và metadata.
    Thành viên phụ trách: Người phân đoạn văn bản.
    """

    @abstractmethod
    def chunk(self, doc: ProcessedData) -> List[DataChunk]:
        """
        Chia một tài liệu ProcessedData thành danh sách các DataChunk.
        :param doc: Tài liệu đã qua tiền xử lý
        :return: Danh sách các DataChunk kèm metadata
        """
        pass

    def chunk_batch(self, docs: List[ProcessedData]) -> List[DataChunk]:
        """Chia chunk hàng loạt tài liệu."""
        all_chunks: List[DataChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        return all_chunks


class BaseEmbeddingModel(ABC):
    """
    [Pre-trained Model - Khối thoi hồng 'Embedding model']
    Mô hình nhúng chuyển đổi văn bản thành vector biểu diễn ngữ nghĩa.
    Thành viên phụ trách: Người quản lý mô hình Embedding (Gemini Embedding, BGE, MiniLM...).
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Số chiều của vector embedding (ví dụ: 384, 768, 1536)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Tên của mô hình embedding đang sử dụng."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Nhúng một danh sách chuỗi văn bản thành danh sách vector."""
        pass

    def embed_chunks(self, chunks: List[DataChunk]) -> List[EmbeddedVector]:
        """
        Nhúng danh sách DataChunk thành các EmbeddedVector tương ứng.
        :param chunks: Danh sách chunk văn bản
        :return: Danh sách đối tượng EmbeddedVector
        """
        texts = [c.text for c in chunks]
        vectors = self.embed_texts(texts)
        return [
            EmbeddedVector(
                chunk_id=chunk.chunk_id,
                vector=vec,
                dimension=self.dimension,
                metadata=chunk.metadata,
            )
            for chunk, vec in zip(chunks, vectors)
        ]

    def embed_query(self, query: UserQuery) -> EmbeddedQueryVector:
        """
        Nhúng câu hỏi của người dùng thành EmbeddedQueryVector.
        :param query: Đối tượng câu hỏi UserQuery
        :return: Đối tượng EmbeddedQueryVector
        """
        vectors = self.embed_texts([query.query_text])
        return EmbeddedQueryVector(
            query_id=query.query_id,
            query_text=query.query_text,
            vector=vectors[0],
            dimension=self.dimension,
        )


class BaseVectorStore(ABC):
    """
    [Process: Store & Similarity searching process - Khối xanh & Khối trụ 'Vector Database']
    Quản lý lưu trữ và tìm kiếm vector tương đồng (ChromaDB, FAISS, Qdrant...).
    Thành viên phụ trách: Người quản lý cơ sở dữ liệu Vector.
    """

    @abstractmethod
    def store(
        self,
        chunks: List[DataChunk],
        vectors: Optional[List[EmbeddedVector]] = None,
    ) -> int:
        """
        Lưu danh sách chunks (và vectors nếu có) vào Vector Database.
        :param chunks: Danh sách các DataChunk
        :param vectors: Danh sách vector tương ứng (tùy chọn nếu DB tự nhúng)
        :return: Số lượng chunks đã lưu thành công
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: EmbeddedQueryVector,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Tìm kiếm top_k đoạn văn bản có độ tương đồng cao nhất với câu hỏi.
        :param query_vector: Vector của câu hỏi
        :param top_k: Số lượng văn bản liên quan cần lấy
        :param filters: Bộ lọc metadata nếu có
        :return: Danh sách RetrievedContext sắp xếp theo độ tương đồng giảm dần
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Trả về tổng số chunk đang có trong vector database."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Xóa sạch dữ liệu trong vector database để lập chỉ mục lại."""
        pass


class BasePromptAugmenter(ABC):
    """
    [Process - Khối tạo 'Augmented Prompt']
    Quy trình kết hợp User Query và Retrieved Contexts thành một Prompt hoàn chỉnh cho LLM.
    Thành viên phụ trách: Prompt Engineer / RAG Designer.
    """

    @abstractmethod
    def augment(
        self,
        query: UserQuery,
        contexts: List[RetrievedContext],
    ) -> AugmentedPrompt:
        """
        Tạo AugmentedPrompt từ câu hỏi và ngữ cảnh trích xuất.
        :param query: Câu hỏi người dùng
        :param contexts: Danh sách ngữ cảnh liên quan tìm được từ vector DB
        :return: Đối tượng AugmentedPrompt
        """
        pass


class BaseLLM(ABC):
    """
    [Pre-trained Model / Process - Khối thoi hồng 'LLM' & mũi tên 'API calling']
    Giao diện gọi mô hình ngôn ngữ lớn (Google Gemini, OpenAI, Claude, hoặc mô hình cục bộ).
    Thành viên phụ trách: Người tích hợp LLM & Sinh phản hồi.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Tên mô hình LLM (ví dụ 'gemini-2.5-flash', 'gpt-4o'...)."""
        pass

    @abstractmethod
    def generate(self, prompt: AugmentedPrompt) -> Response:
        """
        Gọi LLM để sinh câu trả lời hoàn chỉnh từ AugmentedPrompt.
        :param prompt: Prompt đã gắn ngữ cảnh và câu hỏi
        :return: Đối tượng Response trả về người dùng
        """
        pass

    def generate_stream(self, prompt: AugmentedPrompt) -> Generator[str, None, None]:
        """
        (Tùy chọn) Sinh phản hồi dạng luồng (streaming) cho giao diện người dùng mượt mà.
        """
        # Mặc định gọi hàm sinh nguyên khối nếu class con chưa cài đặt streaming
        yield self.generate(prompt).answer


# ========================================================================================
# 3. PIPELINE ORCHESTRATORS (BỘ ĐIỀU PHỐI GHÉP NỐI TOÀN BỘ HỆ THỐNG)
# ========================================================================================

class BaseIngestionPipeline(ABC):
    """
    [Toàn bộ Khối trên của Sơ đồ: Data Ingestion & Indexing Pipeline]
    Điều phối luồng: Data Source -> Crawling -> Pre-processing -> Chunking -> Embedding -> Vector DB.
    """

    def __init__(
        self,
        crawler: BaseDataCrawler,
        preprocessor: BasePreprocessor,
        chunker: BaseChunker,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
    ):
        self.crawler = crawler
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def run(self, sources: List[DataSource], clear_existing: bool = False) -> Dict[str, Any]:
        """
        Chạy toàn bộ quy trình Ingestion.
        """
        if clear_existing:
            self.vector_store.clear()

        # 1. Data Crawling
        raw_data_list = self.crawler.crawl_all(sources)

        # 2. Pre-Processing
        processed_data_list = self.preprocessor.process_batch(raw_data_list)

        # 3. Chunking
        chunks = self.chunker.chunk_batch(processed_data_list)

        # 4. Embedding
        vectors = self.embedding_model.embed_chunks(chunks)

        # 5. Store vào Vector DB
        saved_count = self.vector_store.store(chunks, vectors)

        return {
            "status": "success",
            "sources_count": len(sources),
            "raw_docs_count": len(raw_data_list),
            "processed_docs_count": len(processed_data_list),
            "chunks_count": len(chunks),
            "saved_in_db": saved_count,
            "total_in_db": self.vector_store.count(),
        }


class BaseRAGPipeline(ABC):
    """
    [Toàn bộ Khối dưới của Sơ đồ: Retrieval & Generation Pipeline]
    Điều phối luồng: User Query -> Query Embedding -> Similarity Search -> Augmented Prompt -> LLM API -> Response.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        prompt_augmenter: BasePromptAugmenter,
        llm: BaseLLM,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.prompt_augmenter = prompt_augmenter
        self.llm = llm

    def query(self, query: UserQuery, top_k: int = 5) -> Response:
        """
        Thực thi quy trình trả lời câu hỏi:
        1. Nhúng câu hỏi thành vector (Embedding)
        2. Tìm kiếm các đoạn liên quan trong Vector DB (Similarity searching process)
        3. Ghép bối cảnh và câu hỏi vào Prompt (Augmented Prompt)
        4. Gọi LLM sinh câu trả lời (API calling -> Response)
        """
        # 1. Embedding query
        query_vector = self.embedding_model.embed_query(query)

        # 2. Similarity search
        contexts = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=query.filters,
        )

        # 3. Prompt Augmentation
        augmented_prompt = self.prompt_augmenter.augment(query=query, contexts=contexts)

        # 4. Call LLM
        response = self.llm.generate(augmented_prompt)
        return response


# ========================================================================================
# 4. EVALUATION FRAMEWORK (HỆ THỐNG ĐÁNH GIÁ CHẤT LƯỢNG RAG)
# ========================================================================================

class EvaluationSample(BaseModel):
    """
    Một mẫu dữ liệu kiểm thử (Benchmark QA sample) để đánh giá RAG.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID câu hỏi đánh giá")
    query: str = Field(..., description="Câu hỏi kiểm thử")
    ground_truth: Optional[str] = Field(default=None, description="Câu trả lời chuẩn mực (Ground Truth)")
    expected_source: Optional[str] = Field(default=None, description="Tên văn bản hoặc điều khoản chuẩn")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata bổ sung: độ khó, chủ đề...")


class EvaluationResult(BaseModel):
    """
    Kết quả đánh giá trên 1 mẫu thử nghiệm (Single sample evaluation).
    """
    sample_id: str
    query: str
    generated_answer: str
    ground_truth: Optional[str] = None
    retrieved_contexts: List[RetrievedContext] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Điểm các chỉ số (thang 0.0 - 1.0): faithfulness, answer_relevance, context_precision...",
    )
    feedback: Optional[str] = Field(default=None, description="Nhận xét chi tiết hoặc giải thích điểm số")


class EvaluationReport(BaseModel):
    """
    Báo cáo tổng hợp kết quả đánh giá trên toàn bộ tập dữ liệu (Benchmark Report).
    """
    total_samples: int
    average_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Điểm trung bình các chỉ số: faithfulness, answer_relevance, context_recall...",
    )
    results: List[EvaluationResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseEvaluator(ABC):
    """
    [Evaluation Process]
    Giao diện trừu tượng cho việc đánh giá chất lượng hệ thống RAG.
    Thành viên phụ trách: Người đánh giá chất lượng (RAG Evaluator / QA Tester).
    """

    @abstractmethod
    def evaluate_sample(
        self,
        sample: EvaluationSample,
        response: Response,
    ) -> EvaluationResult:
        """
        Đánh giá câu trả lời của 1 câu hỏi cụ thể theo các tiêu chí:
        1. Context Relevance (Ngữ cảnh trích xuất có đúng trọng tâm câu hỏi?)
        2. Faithfulness / Groundedness (Câu trả lời có bám sát ngữ cảnh hay bị ảo giác?)
        3. Answer Relevance (Câu trả lời có thực sự giải đáp câu hỏi?)
        4. Context Recall (So với Ground Truth, ngữ cảnh có chứa đủ thông tin?)
        """
        pass

    def evaluate_pipeline(
        self,
        pipeline: BaseRAGPipeline,
        dataset: List[EvaluationSample],
        top_k: int = 5,
    ) -> EvaluationReport:
        """
        Chạy đánh giá tự động trên toàn bộ tập benchmark dataset.
        """
        results: List[EvaluationResult] = []
        metrics_sum: Dict[str, float] = {}

        for sample in dataset:
            user_query = UserQuery(query_text=sample.query)
            response = pipeline.query(user_query, top_k=top_k)
            result = self.evaluate_sample(sample, response)
            results.append(result)

            for metric_name, score in result.metrics.items():
                metrics_sum[metric_name] = metrics_sum.get(metric_name, 0.0) + score

        total = len(dataset)
        avg_metrics = {m: (s / total) for m, s in metrics_sum.items()} if total > 0 else {}

        return EvaluationReport(
            total_samples=total,
            average_metrics=avg_metrics,
            results=results,
        )


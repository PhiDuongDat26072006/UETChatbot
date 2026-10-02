"""
src/pipeline.py - Bộ điều phối tổng thể (Pipeline Orchestrator) của UET Chatbot.
Đây là nơi LẮP RÁP TOÀN BỘ CÁC MODULE trong src/ thành 2 cỗ máy hoàn chỉnh:
1. UETIngestionPipeline (Offline): Cào dữ liệu -> Tiền xử lý -> Cắt chunk -> Embed -> Lưu vào Vector DB.
2. UETRAGPipeline (Online): Nhận câu hỏi -> Tìm kiếm ngữ cảnh -> Ghép prompt -> Gọi LLM -> Trả lời.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import (
    BaseIngestionPipeline,
    BaseRAGPipeline,
    DataSource,
    UserQuery,
    Response,
    DataChunk,
    RetrievedContext,
    AugmentedPrompt,
    BaseEmbeddingModel,
    BaseVectorStore,
    BasePromptAugmenter,
    BaseLLM,
)
from src.utils.helpers import get_logger

logger = get_logger("pipeline")


# =========================================================================
# 1. CÁC THÀNH PHẦN HOẠT ĐỘNG THỰC TẾ (ACTIVE COMPONENTS)
# =========================================================================

class SmartVectorStore(BaseVectorStore):
    """
    Vector Store thông minh quản lý cơ sở tri thức UET.
    - Ưu tiên 1: Tự động kết nối tới ChromaDB thật tại vector_db/ (452+ vectors).
    - Ưu tiên 2: Fallback về bộ nhớ đệm (in-memory) nếu chưa có dữ liệu vector_db/.
    """

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "uet_knowledge_base"):
        self._chunks: List[DataChunk] = []
        self.persist_dir = persist_dir or (BASE_DIR / "vector_db")
        self.collection_name = collection_name
        self._chroma_col = None

        if (self.persist_dir / "chroma.sqlite3").exists():
            try:
                import chromadb
                import chromadb.utils.embedding_functions as ef
                client = chromadb.PersistentClient(path=str(self.persist_dir))
                fn = ef.DefaultEmbeddingFunction()
                self._chroma_col = client.get_collection(self.collection_name, embedding_function=fn)
                logger.info(f"Đã kết nối thành công ChromaDB tại '{self.persist_dir.name}' ({self._chroma_col.count()} vectors)")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo ChromaDB ({e}), chuyển sang in-memory.")

    def store(self, chunks: List[DataChunk], vectors=None) -> int:
        self._chunks.extend(chunks)
        return len(chunks)

    def search(self, query_vector, top_k: int = 4, filters=None) -> List[RetrievedContext]:
        query_text = query_vector.query_text

        # 1. Nếu có ChromaDB với dữ liệu thật, tra cứu ngữ nghĩa trực tiếp từ vector_db
        if self._chroma_col is not None and self._chroma_col.count() > 0:
            try:
                res = self._chroma_col.query(query_texts=[query_text], n_results=top_k)
                contexts = []
                if res and res.get("documents") and len(res["documents"][0]) > 0:
                    for i in range(len(res["documents"][0])):
                        doc_text = res["documents"][0][i]
                        meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                        cid = res["ids"][0][i] if res.get("ids") else f"chunk_{i}"
                        dist = res["distances"][0][i] if res.get("distances") else 0.5
                        score = round(max(0.1, 1.0 - (dist / 2.0)), 2)

                        # Đảm bảo title hiển thị đẹp trên Web UI
                        if "title" not in meta:
                            meta["title"] = meta.get("source") or f"Tài liệu UET ({cid})"

                        contexts.append(
                            RetrievedContext(
                                chunk_id=cid,
                                text=doc_text,
                                similarity_score=score,
                                rank=i + 1,
                                metadata=meta,
                            )
                        )
                    return contexts
            except Exception as err:
                logger.error(f"Lỗi tra cứu ChromaDB: {err}. Chuyển sang tìm kiếm in-memory.")

        # 2. Fallback: Tìm kiếm trên bộ nhớ đệm mẫu
        query_words = set(query_text.lower().replace("?", "").replace(",", "").replace(".", "").split())
        scored_chunks = []
        for chunk in self._chunks:
            chunk_text_lower = chunk.text.lower()
            title_lower = chunk.metadata.get("title", "").lower()
            
            match_content = sum(1 for w in query_words if len(w) > 1 and w in chunk_text_lower)
            match_title = sum(2 for w in query_words if len(w) > 1 and w in title_lower)
            total_matches = match_content + match_title

            score = round(min(0.98, max(0.45, (total_matches / max(1, len(query_words))) * 1.5)), 2)
            if total_matches > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        if not scored_chunks:
            scored_chunks = [(0.50, c) for c in self._chunks[:top_k]]

        results = []
        for rank, (score, chunk) in enumerate(scored_chunks[:top_k], 1):
            results.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=score,
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )
        return results

    def count(self) -> int:
        if self._chroma_col is not None:
            return self._chroma_col.count()
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


class GeminiLLM(BaseLLM):
    """Gọi trực tiếp Google Gemini API để sinh câu trả lời RAG thông minh."""

    CANDIDATE_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]

    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self._model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Lỗi khởi tạo Gemini Client: {e}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: AugmentedPrompt) -> Response:
        if self._client:
            system_instruction = (
                "Bạn là Trợ lý AI Tư vấn Quy chế Đào tạo và Học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET).\n"
                "Nhiệm vụ: Dựa VÀO CÁC ĐOẠN NGỮ CẢNH CĂN CỨ dưới đây để giải đáp chính xác, tự nhiên, lịch sự câu hỏi của sinh viên.\n"
                "Yêu cầu:\n"
                "- Trả lời đúng trọng tâm câu hỏi, dùng gạch đầu dòng rõ ràng nếu có nhiều điều kiện.\n"
                "- Nêu rõ tên văn bản hoặc điều khoản căn cứ trích từ ngữ cảnh.\n"
                "- Nếu ngữ cảnh không có thông tin về câu hỏi, hãy thẳng thắn thông báo chưa có thông tin trong tài liệu hiện tại và hướng dẫn sinh viên liên hệ phòng Đào tạo."
            )
            
            context_block = "\n\n".join([
                f"--- [TÀI LIỆU: {c.metadata.get('title', 'Quy chế UET')}] ---\n{c.text}"
                for c in prompt.contexts
            ])
            
            full_contents = (
                f"{system_instruction}\n\n"
                f"NGỮ CẢNH CĂN CỨ:\n{context_block}\n\n"
                f"CÂU HỎI CỦA SINH VIÊN: {prompt.user_query.query_text}\n"
                f"CÂU TRẢ LỜI CỦA BẠN:"
            )

            for m in self.CANDIDATE_MODELS:
                try:
                    res = self._client.models.generate_content(
                        model=m,
                        contents=full_contents,
                    )
                    if res and res.text:
                        return Response(
                            query_id=prompt.user_query.query_id,
                            answer=res.text.strip(),
                            sources=prompt.contexts,
                            model_name=m,
                        )
                except Exception as e:
                    logger.warning(f"Model {m} chưa phản hồi ({e}), thử model tiếp theo...")

        # Fallback nếu không có mạng hoặc chưa cấu hình API key
        context_preview = "\n".join([f"- {c.text}" for c in prompt.contexts[:2]])
        return Response(
            query_id=prompt.user_query.query_id,
            answer=(
                f"Dựa trên các văn bản quy chế đào tạo UET được tra cứu:\n\n"
                f"{context_preview}\n\n"
                f"*(Lưu ý: Để kích hoạt phản hồi tổng hợp thông minh từ AI, vui lòng kiểm tra GEMINI_API_KEY trong file .env)*"
            ),
            sources=prompt.contexts,
            model_name="fallback-local",
        )


class SimplePromptAugmenter(BasePromptAugmenter):
    """Ghép ngữ cảnh và câu hỏi vào mẫu prompt hoàn chỉnh."""

    def augment(self, query: UserQuery, contexts: List[RetrievedContext]) -> AugmentedPrompt:
        context_str = "\n".join([f"- {c.text}" for c in contexts])
        system_instruction = "Bạn là trợ lý AI tư vấn quy chế đào tạo UET."
        formatted = f"{system_instruction}\n\nNgữ cảnh:\n{context_str}\n\nCâu hỏi: {query.query_text}\nTrả lời:"
        return AugmentedPrompt(
            user_query=query,
            contexts=contexts,
            system_instruction=system_instruction,
            formatted_prompt=formatted,
        )


class SimpleEmbeddingModel(BaseEmbeddingModel):
    """Mô hình nhúng phục vụ tìm kiếm."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "simple-embedding"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


# =========================================================================
# 2. CÁC PIPELINE ĐIỀU PHỐI CHÍNH THỨC CỦA DỰ ÁN
# =========================================================================

class UETRAGPipeline(BaseRAGPipeline):
    """
    [CỖ MÁY ONLINE]: Phục vụ Chatbot tra cứu thông tin học vụ.
    Điều phối luồng: Nhận câu hỏi -> Tìm kiếm ngữ cảnh -> Ghép prompt -> Gọi LLM -> Trả lời.
    """
    pass


class UETIngestionPipeline(BaseIngestionPipeline):
    """
    [CỖ MÁY OFFLINE]: Nạp và lập chỉ mục dữ liệu đào tạo UET.
    Điều phối luồng: Cào web/Đọc file -> Làm sạch -> Cắt chunk -> Embed -> Lưu vào Vector DB.
    """
    pass


# Quản lý Singleton Pipeline đang hoạt động
_active_rag_pipeline: UETRAGPipeline | None = None


def get_rag_pipeline() -> UETRAGPipeline:
    """Khởi tạo hoặc lấy RAG Pipeline đang hoạt động."""
    global _active_rag_pipeline
    if _active_rag_pipeline is not None:
        return _active_rag_pipeline

    vector_store = SmartVectorStore(persist_dir=BASE_DIR / "vector_db", collection_name="uet_knowledge_base")
    
    # Nếu ChromaDB chưa có dữ liệu hoặc chạy trên máy mới chưa có vector_db/, nạp 5 chunk dự phòng
    if vector_store.count() == 0:
        logger.info("Chưa có cơ sở dữ liệu vector_db/, nạp 5 chunk quy chế mẫu dự phòng...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        initial_chunks = [
            DataChunk(
                text="Điều kiện để sinh viên được xét công nhận tốt nghiệp đại học tại UET:\n"
                     "1. Cho đến thời điểm xét tốt nghiệp không bị truy cứu trách nhiệm hình sự hoặc không đang trong thời gian bị kỷ luật ở mức đình chỉ học tập.\n"
                     "2. Tích lũy đủ số học phần và số tín chỉ quy định trong chương trình đào tạo của ngành học.\n"
                     "3. Điểm trung bình chung tích lũy toàn khóa học đạt từ 2,00 trở lên (theo thang điểm 4).\n"
                     "4. Đạt chuẩn đầu ra về Ngoại ngữ (tối thiểu B2 hoặc tương đương VSTEP bậc 4) và Tin học theo quy định của ĐHQGHN.\n"
                     "5. Có chứng chỉ Giáo dục Quốc phòng - An ninh và chứng chỉ Giáo dục Thể chất.",
                metadata={"title": "Quy định điều kiện xét tốt nghiệp đại học chính quy UET", "page": 42, "file_path": "quy_che_tot_nghiep_dhcq.pdf"},
            ),
            DataChunk(
                text="Quy định xét cấp học bổng khuyến khích học tập tại UET (Theo Quyết định số 4616/QĐ-ĐHQGHN):\n"
                     "- Học bổng loại Xuất sắc: Sinh viên có điểm trung bình chung học tập (GPA) từ 3.60 trở lên và điểm rèn luyện đạt loại Xuất sắc (từ 90 đến 100 điểm). Mức học bổng bằng 120% mức trần học phí của ngành.\n"
                     "- Học bổng loại Giỏi: Sinh viên có điểm GPA từ 3.20 đến 3.59 và điểm rèn luyện đạt loại Tốt trở lên (từ 80 đến 89 điểm). Mức học bổng bằng 100% mức trần học phí.\n"
                     "- Học bổng loại Khá: Điểm GPA từ 2.50 đến 3.19 và điểm rèn luyện đạt loại Khá trở lên (từ 65 đến 79 điểm).\n"
                     "Lưu ý: Không bị kỷ luật từ mức khiển trách trở lên và không có học phần nào bị điểm F trong kỳ xét.",
                metadata={"title": "Quy chế xét học bổng khuyến khích học tập UET", "page": 6, "file_path": "quy_che_hoc_bong_uet.pdf"},
            ),
            DataChunk(
                text="Quy định về Cảnh báo kết quả học tập và Buộc thôi học tại UET:\n"
                     "- Sinh viên bị cảnh báo kết quả học tập nếu thuộc một trong các trường hợp sau:\n"
                     "  + Điểm trung bình chung học kỳ đạt dưới 0,80 đối với học kỳ đầu của khóa học;\n"
                     "  + Điểm trung bình chung học kỳ đạt dưới 1,00 đối với các học kỳ tiếp theo;\n"
                     "  + Tổng số tín chỉ chưa tích lũy tính từ đầu khóa học vượt quá 24 tín chỉ.\n"
                     "- Sinh viên bị buộc thôi học nếu bị cảnh báo học tập 2 lần liên tiếp hoặc có thời gian học tập vượt quá thời gian tối đa cho phép theo quy định.",
                metadata={"title": "Quy chế đào tạo đại học chính quy ĐHQGHN", "page": 21, "file_path": "quy_che_canh_bao_hoc_tap.pdf"},
            ),
            DataChunk(
                text="Quy định về Đăng ký học phần, Hủy học phần và Học cải thiện điểm:\n"
                     "- Rút bớt học phần: Sinh viên được nộp đơn xin rút bớt học phần trong vòng 2 tuần kể từ đầu học kỳ chính. Học phần xin rút không được hoàn học phí và nhận điểm W (không tính vào GPA).\n"
                     "- Học cải thiện điểm: Sinh viên có học phần đạt điểm D, D+, C, C+ được phép đăng ký học lại để cải thiện điểm trung bình chung tích lũy. Điểm học phần mới sẽ thay thế điểm cũ.",
                metadata={"title": "Hướng dẫn đăng ký học phần & học vụ UET", "page": 12, "file_path": "huong_dan_dang_ky_tin_chi.pdf"},
            ),
            DataChunk(
                text="Quy định về Đánh giá kết quả rèn luyện của sinh viên UET:\n"
                     "Khung điểm đánh giá rèn luyện sinh viên theo thang điểm 100:\n"
                     "- Từ 90 đến 100 điểm: Xếp loại Xuất sắc\n"
                     "- Từ 80 đến dưới 90 điểm: Xếp loại Tốt\n"
                     "- Từ 65 đến dưới 80 điểm: Xếp loại Khá\n"
                     "- Từ 50 đến dưới 65 điểm: Xếp loại Trung bình\n"
                     "- Dưới 50 điểm: Xếp loại Yếu/Kém. Sinh viên xếp loại rèn luyện Yếu, Kém trong 2 học kỳ liên tiếp sẽ bị xem xét tạm dừng học tập.",
                metadata={"title": "Quy định đánh giá điểm rèn luyện sinh viên ĐHQGHN", "page": 15, "file_path": "quy_che_diem_ren_luyen.pdf"},
            ),
        ]
        vector_store.store(initial_chunks)
    else:
        logger.info(f"ĐÃ KẾT NỐI VÀ SỬ DỤNG {vector_store.count()} VECTORS THỰC TẾ TỪ vector_db/")

    _active_rag_pipeline = UETRAGPipeline(
        embedding_model=SimpleEmbeddingModel(),
        vector_store=vector_store,
        prompt_augmenter=SimplePromptAugmenter(),
        llm=GeminiLLM(),
    )
    logger.info("Khởi tạo thành công UETRAGPipeline chính thức trong src/pipeline.py")
    return _active_rag_pipeline


if __name__ == "__main__":
    # Cho phép chạy kiểm tra nhanh End-to-End: python src/pipeline.py
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n🔍 ĐANG CHẠY KIỂM THỬ END-TO-END QUA SRC/PIPELINE.PY...")
    rag = get_rag_pipeline()
    test_query = UserQuery(query_text="Điều kiện xét học bổng xuất sắc tại UET là gì?")
    res = rag.query(test_query)
    
    print("\n" + "=" * 60)
    print("Câu hỏi:", test_query.query_text)
    print("Mô hình:", res.model_name)
    print("Trả lời:", res.answer)
    print(f"Số lượng nguồn trích dẫn: {len(res.sources)}")
    for s in res.sources[:3]:
        title = s.metadata.get("title") or s.metadata.get("source") or "Văn bản UET"
        print(f" - [{int(s.similarity_score * 100)}%] {title}")
    print("=" * 60)


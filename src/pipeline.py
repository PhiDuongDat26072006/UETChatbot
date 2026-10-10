"""
src/pipeline.py - Bộ điều phối tổng thể (Pipeline Orchestrator) của UET Chatbot.
Đây là nơi LẮP RÁP TOÀN BỘ CÁC PHÂN HỆ trong src/ thành 2 cỗ máy RAG hoàn chỉnh:
1. UETIngestionPipeline (Offline): Cào dữ liệu -> Tiền xử lý -> Cắt chunk -> Embed -> Lưu vào Vector DB.
2. UETRAGPipeline (Online): Nhận câu hỏi -> Truy xuất ngữ cảnh (Hybrid RRF + Rerank) -> Ghép prompt -> Gọi LLM -> Trả lời.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Base contracts
from src.base import (
    BaseIngestionPipeline,
    BaseRAGPipeline,
    BaseDataCrawler,
    BasePreprocessor,
    BaseChunker,
    BaseEmbeddingModel,
    BaseVectorStore,
    BasePromptAugmenter,
    BaseLLM,
    DataSource,
    RawData,
    ProcessedData,
    DataChunk,
    RetrievedContext,
    AugmentedPrompt,
    UserQuery,
    Response,
)

import json
from src.utils.helpers import get_logger, Timer, load_yaml_config

# Concrete modules from team branches
from src.ingestion import UETDataLoader, clean_whitespace, DataManifestTracker
from src.chunking import UETChunker
from src.embeddings import UETEmbedder
from src.vectordb import UETVectorStore
from src.retrieval import UETRetriever
from src.prompts import UETPromptAugmenter, DEFAULT_UET_SYSTEM_PROMPT
from src.llm import UETLLMClient
from src.indexing.chunks import load_chunks_from_dir

logger = get_logger("pipeline")


# =========================================================================
# 1. TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSOR)
# =========================================================================

class UETPreprocessor(BasePreprocessor):
    """
    Quy trình tiền xử lý và làm sạch dữ liệu văn bản UET.
    Chuyển đổi RawData thành ProcessedData: loại bỏ khoảng trắng thừa, chuẩn hóa tiêu đề và metadata.
    """

    def process(self, raw: RawData) -> ProcessedData:
        clean_text = clean_whitespace(raw.content)
        title = (raw.title or "").strip()
        if not title:
            first_line = clean_text.split("\n")[0].strip() if clean_text else ""
            title = first_line[:120] if first_line else "Tài liệu đào tạo UET"

        metadata = dict(raw.raw_metadata or {})
        metadata.setdefault("source_uri", raw.source_uri)
        metadata.setdefault("title", title)

        return ProcessedData(
            raw_data_id=raw.id,
            title=title,
            content=clean_text,
            metadata=metadata,
        )


# =========================================================================
# 2. CỖ MÁY OFFLINE: INGESTION PIPELINE
# =========================================================================

class UETIngestionPipeline(BaseIngestionPipeline):
    """
    [CỖ MÁY OFFLINE]: Nạp và lập chỉ mục dữ liệu đào tạo UET.
    Điều phối luồng: Cào web/Đọc file -> Làm sạch -> Cắt chunk -> Embed -> Lưu vào Vector DB.
    Hỗ trợ điều phối chặng linh hoạt (Multi-Stage Ingestion) và chống xử lý trùng lặp (Deduplication).
    """

    STAGES: List[str] = ["raw", "processed", "chunked", "vectordb", "indexed_data"]

    def __init__(
        self,
        crawler: Optional[BaseDataCrawler] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        chunker: Optional[BaseChunker] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        manifest: Optional[DataManifestTracker] = None,
    ):
        emb = embedding_model or UETEmbedder()
        vs = vector_store or UETVectorStore(persist_dir=BASE_DIR / "database" / "vector_db", embedding_model=emb)
        super().__init__(
            crawler=crawler or UETDataLoader(),
            preprocessor=preprocessor or UETPreprocessor(),
            chunker=chunker or UETChunker(),
            embedding_model=emb,
            vector_store=vs,
        )
        self.manifest = manifest or DataManifestTracker(manifest_path=BASE_DIR / "data" / ".ingest_manifest.json")

    def run(self, sources: List[DataSource], clear_existing: bool = False) -> Dict[str, Any]:
        """
        Thực thi quy trình lập chỉ mục toàn bộ dữ liệu nguồn.
        """
        logger.info(f"Bắt đầu quy trình Ingestion với {len(sources)} nguồn dữ liệu...")
        with Timer() as timer:
            from src.indexing import UETIndexer

            raw_docs = self.crawler.crawl_all(sources)
            processed_docs = self.preprocessor.process_batch(raw_docs)
            chunks = self.chunker.chunk_batch(processed_docs)
            indexed = UETIndexer(vector_store=self.vector_store,
                                 embedding_model=self.embedding_model).index_chunks(
                                     chunks, clear_existing=clear_existing)
            result = {"status": indexed["status"], "sources_count": len(sources),
                      "raw_docs_count": len(raw_docs), "processed_docs_count": len(processed_docs),
                      "chunks_count": len(chunks), "saved_in_db": indexed["indexed_chunks"],
                      "total_in_db": indexed["total_in_db"]}
        result["latency_seconds"] = round(timer.elapsed, 2)
        logger.info(
            f"Hoàn thành Ingestion trong {result['latency_seconds']}s: "
            f"{result['raw_docs_count']} văn bản thô -> {result['chunks_count']} chunks -> "
            f"Tổng hiện có trong DB: {result['total_in_db']}"
        )
        return result

    def _run_raw_to_processed(self) -> Dict[str, int]:
        """Chuyển đổi dữ liệu từ raw -> processed."""
        raw_dir = BASE_DIR / "data" / "raw_data"
        processed_dir = BASE_DIR / "data" / "processed_data"
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_file = processed_dir / "ingested_processed.jsonl"

        raw_source = DataSource(source_type="directory", uri=str(raw_dir))
        raw_docs = self.crawler.crawl(raw_source)

        processed_count = 0
        skipped_count = 0

        with open(out_file, "a", encoding="utf-8") as f_out:
            for raw in raw_docs:
                doc_hash = DataManifestTracker.compute_hash(raw.content, raw.title)
                if self.manifest.is_processed(raw.id, "processed", doc_hash):
                    skipped_count += 1
                    continue

                proc = self.preprocessor.process(raw)
                record = {
                    "id": proc.id,
                    "title": proc.title,
                    "content": proc.content,
                    "metadata": proc.metadata,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.manifest.mark_processed(raw.id, "processed", doc_hash, title=proc.title)
                processed_count += 1

        self.manifest.save()
        return {"processed": processed_count, "skipped": skipped_count}

    def _run_processed_to_chunked(self) -> Dict[str, int]:
        """Chuyển đổi dữ liệu từ processed -> chunked."""
        processed_dir = BASE_DIR / "data" / "processed_data"
        chunked_dir = BASE_DIR / "data" / "chunked_data"
        chunked_dir.mkdir(parents=True, exist_ok=True)
        out_file = chunked_dir / "staged_chunks.jsonl"

        jsonl_files = sorted(processed_dir.rglob("*.jsonl"))
        processed_docs = 0
        skipped_docs = 0
        total_chunks = 0

        with open(out_file, "a", encoding="utf-8") as f_out:
            for jf in jsonl_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line:
                                continue

                            data = json.loads(line)
                            doc_id = str(data.get("id") or data.get("source_url_or_path") or "")
                            content = data.get("content") or ""
                            title = data.get("title") or ""
                            doc_hash = DataManifestTracker.compute_hash(content, title)

                            if self.manifest.is_processed(doc_id, "chunked", doc_hash):
                                skipped_docs += 1
                                continue

                            meta = data.get("metadata") or {}
                            if "source_url_or_path" in data and "source" not in meta:
                                meta["source"] = data["source_url_or_path"]
                            if title and "title" not in meta:
                                meta["title"] = title

                            proc_doc = ProcessedData(
                                id=doc_id,
                                title=title,
                                content=content,
                                metadata=meta,
                            )
                            chunks = self.chunker.chunk(proc_doc)
                            for c in chunks:
                                c_record = {
                                    "chunk_id": c.chunk_id,
                                    "document_id": c.document_id,
                                    "text": c.text,
                                    "chunk_index": c.chunk_index,
                                    "metadata": c.metadata,
                                }
                                f_out.write(json.dumps(c_record, ensure_ascii=False) + "\n")
                                total_chunks += 1

                            self.manifest.mark_processed(
                                doc_id, "chunked", doc_hash, title=title, extra_meta={"chunk_count": len(chunks)}
                            )
                            processed_docs += 1
                except Exception as err:
                    logger.warning(f"Lỗi khi đọc file {jf}: {err}")

        self.manifest.save()
        return {"processed_docs": processed_docs, "skipped_docs": skipped_docs, "chunks_created": total_chunks}

    def _run_chunked_to_vectordb(
        self, batch_size: int = 500
    ) -> Dict[str, int]:
        """Chuyển đổi dữ liệu từ chunked -> vectordb (nhúng vector và lưu vào ChromaDB)."""
        chunked_dir = BASE_DIR / "data" / "chunked_data"
        jsonl_files = sorted(chunked_dir.glob("*.jsonl"))

        total_indexed = 0
        skipped_chunks = 0
        batch: List[DataChunk] = []

        for jf in jsonl_files:
            try:
                with open(jf, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue

                        item = json.loads(line)
                        cid = str(item.get("chunk_id") or "")
                        text = item.get("text") or ""
                        doc_id = item.get("document_id")
                        c_idx = item.get("chunk_index", 0)
                        meta = item.get("metadata") or {}

                        if self.manifest.is_processed(cid, "vectordb"):
                            skipped_chunks += 1
                            continue

                        chunk_obj = DataChunk(
                            chunk_id=cid,
                            document_id=doc_id,
                            text=text,
                            chunk_index=c_idx,
                            metadata=meta,
                        )
                        batch.append(chunk_obj)

                        if len(batch) >= batch_size:
                            vectors = self.embedding_model.embed_chunks(batch)
                            self.vector_store.store(batch, vectors)
                            for c in batch:
                                self.manifest.mark_processed(c.chunk_id, "vectordb", content_hash=c.chunk_id)
                            total_indexed += len(batch)
                            logger.info(f"Đã nhúng & lưu {total_indexed} chunks vào Vector DB...")
                            batch.clear()
            except Exception as err:
                logger.warning(f"Lỗi khi xử lý file chunk {jf}: {err}")

        if batch:
            vectors = self.embedding_model.embed_chunks(batch)
            self.vector_store.store(batch, vectors)
            for c in batch:
                self.manifest.mark_processed(c.chunk_id, "vectordb", content_hash=c.chunk_id)
            total_indexed += len(batch)
            batch.clear()

        self.manifest.save()
        return {"chunks_indexed": total_indexed, "skipped_chunks": skipped_chunks, "total_in_db": self.vector_store.count()}

    def _run_chunked_to_indexed_data(self) -> Dict[str, Any]:
        """
        Lập chỉ mục BM25 trực tiếp từ data/chunked_data/*.jsonl và lưu ra database/bm25_index.pkl.
        Được thực hiện bởi phân hệ Lập chỉ mục (src.indexing), tách biệt hoàn toàn khỏi phân hệ Retrieval.
        """
        from src.indexing.chunks import load_chunks_from_dir
        from src.indexing.sparse import BM25Index

        chunked_dir = BASE_DIR / "data" / "chunked_data"
        chunks = load_chunks_from_dir(chunked_dir)

        bm25 = BM25Index(chunks)
        out_path = BASE_DIR / "database" / "bm25_index.pkl"
        bm25.save(out_path)

        for chunk in chunks:
            self.manifest.mark_processed(
                chunk.chunk_id,
                "indexed_data",
                content_hash=self.manifest.compute_hash(chunk.text),
            )
        self.manifest.save()

        return {
            "chunks_indexed": bm25.count(),
            "sparse_file": str(out_path),
            "status": "success",
        }

    def run_stages(
        self,
        from_stage: str = "raw",
        to_stage: str = "vectordb",
        batch_size: int = 500,
    ) -> Dict[str, Any]:
        """
        Kích hoạt chạy luồng dữ liệu từ một giai đoạn đến một giai đoạn cụ thể.
        Tự động điều phối các bước trung gian hợp lệ:
        - Tuyến Vector DB: raw -> processed -> chunked -> vectordb
        - Tuyến Sparse Index: raw -> processed -> chunked -> indexed_data
        """
        from_stage = from_stage.lower().strip()
        to_stage = to_stage.lower().strip()

        if from_stage not in self.STAGES or to_stage not in self.STAGES:
            raise ValueError(
                f"Giai đoạn không hợp lệ. Các giai đoạn được hỗ trợ: {self.STAGES}. "
                f"Nhận được: from='{from_stage}', to='{to_stage}'"
            )

        stage_order = {"raw": 0, "processed": 1, "chunked": 2, "vectordb": 3, "indexed_data": 3}
        if stage_order[from_stage] >= stage_order[to_stage]:
            raise ValueError(
                f"from_stage '{from_stage}' phải đứng trước to_stage '{to_stage}'. "
                f"Thứ tự hợp lệ: raw -> processed -> chunked -> vectordb / indexed_data"
            )

        results: Dict[str, Any] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "transitions": {},
        }

        with Timer() as timer:
            # Chặng 1: raw -> processed
            if from_stage == "raw" and to_stage in ["processed", "chunked", "vectordb", "indexed_data"]:
                logger.info("▶ BẮT ĐẦU CHẶNG: raw -> processed")
                res_1 = self._run_raw_to_processed()
                results["transitions"]["raw_to_processed"] = res_1
                logger.info(f"✔ Hoàn thành raw -> processed: {res_1}")

            # Chặng 2: processed -> chunked
            if from_stage in ["raw", "processed"] and to_stage in ["chunked", "vectordb", "indexed_data"]:
                logger.info("▶ BẮT ĐẦU CHẶNG: processed -> chunked")
                res_2 = self._run_processed_to_chunked()
                results["transitions"]["processed_to_chunked"] = res_2
                logger.info(f"✔ Hoàn thành processed -> chunked: {res_2}")

            # Chặng 3a: chunked -> vectordb
            if from_stage in ["raw", "processed", "chunked"] and to_stage == "vectordb":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> vectordb")
                res_3 = self._run_chunked_to_vectordb(batch_size=batch_size)
                results["transitions"]["chunked_to_vectordb"] = res_3
                logger.info(f"✔ Hoàn thành chunked -> vectordb: {res_3}")

            # Chặng 3b: chunked -> indexed_data
            if from_stage in ["raw", "processed", "chunked"] and to_stage == "indexed_data":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> indexed_data (BM25 Sparse Index)")
                res_4 = self._run_chunked_to_indexed_data()
                results["transitions"]["chunked_to_indexed_data"] = res_4
                logger.info(f"✔ Hoàn thành chunked -> indexed_data: {res_4}")

        results["latency_seconds"] = round(timer.elapsed, 2)
        logger.info(f"Hoàn thành toàn bộ tiến trình từ '{from_stage}' đến '{to_stage}' trong {results['latency_seconds']}s")
        return results

    def get_stage_stats(self) -> Dict[str, Any]:
        """Thống kê tổng số lượng dữ liệu hiện có ở từng giai đoạn."""
        raw_dir = BASE_DIR / "data" / "raw_data"
        processed_dir = BASE_DIR / "data" / "processed_data"
        chunked_dir = BASE_DIR / "data" / "chunked_data"
        bm25_file = BASE_DIR / "database" / "bm25_index.pkl"

        raw_files = len(list(raw_dir.rglob("*.*"))) if raw_dir.exists() else 0

        proc_docs = 0
        if processed_dir.exists():
            for f in processed_dir.rglob("*.jsonl"):
                try:
                    with open(f, encoding="utf-8") as fp:
                        proc_docs += sum(1 for line in fp if line.strip())
                except Exception:
                    pass

        chunk_count = 0
        if chunked_dir.exists():
            for f in chunked_dir.glob("*.jsonl"):
                try:
                    with open(f, encoding="utf-8") as fp:
                        chunk_count += sum(1 for line in fp if line.strip())
                except Exception:
                    pass

        db_count = self.vector_store.count()
        bm25_exists = bm25_file.exists()
        bm25_size_mb = round(bm25_file.stat().st_size / (1024 * 1024), 2) if bm25_exists else 0.0

        return {
            "stages": {
                "raw": {"path": str(raw_dir.relative_to(BASE_DIR)), "file_count": raw_files},
                "processed": {"path": str(processed_dir.relative_to(BASE_DIR)), "document_count": proc_docs},
                "chunked": {"path": str(chunked_dir.relative_to(BASE_DIR)), "chunk_count": chunk_count},
                "vectordb": {"path": "database/vector_db", "vector_count": db_count},
                "indexed_data": {"path": "database/bm25_index.pkl", "exists": bm25_exists, "size_mb": bm25_size_mb},
            },
            "manifest_tracking": self.manifest.get_stats(),
        }


# =========================================================================
# 3. CỖ MÁY ONLINE: RAG PIPELINE
# =========================================================================

class UETRAGPipeline(BaseRAGPipeline):
    """
    [CỖ MÁY ONLINE]: Phục vụ Chatbot tra cứu thông tin học vụ.
    Điều phối luồng: Nhận câu hỏi -> Truy xuất ngữ cảnh (UETRetriever: BM25 + Vector -> Hybrid RRF -> Rerank -> Top-K)
                   -> Ghép prompt (UETPromptAugmenter) -> Gọi LLM (UETLLMClient) -> Trả lời.
    """

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        prompt_augmenter: Optional[BasePromptAugmenter] = None,
        llm: Optional[BaseLLM] = None,
        retriever: Optional[UETRetriever] = None,
    ):
        emb = embedding_model or UETEmbedder()
        vs = vector_store or UETVectorStore(persist_dir=BASE_DIR / "database" / "vector_db", embedding_model=emb)
        pa = prompt_augmenter or UETPromptAugmenter()
        client_llm = llm or UETLLMClient()

        super().__init__(
            embedding_model=emb,
            vector_store=vs,
            prompt_augmenter=pa,
            llm=client_llm,
        )

        # UETRetriever điều phối quy trình BM25 + Vector + Hybrid RRF + Reranker
        self.retriever = retriever or UETRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

    def query(self, query: UserQuery, top_k: int = 5) -> Response:
        """
        Thực thi quy trình trả lời câu hỏi:
        1. Gọi UETRetriever: BM25 + Vector -> Hybrid RRF (~100 chunks) -> Rerank (20 chunks) -> Top-K
        2. Ghép ngữ cảnh vào Prompt (AugmentedPrompt) theo chuẩn BaseRAGPipeline
        3. Gọi LLM sinh câu trả lời
        """
        with Timer() as timer:
            # Bước 1: Truy xuất ngữ cảnh bằng UETRetriever
            contexts = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=query.filters,
            )

            # Bước 2 & 3: Ghép prompt và gọi LLM
            augmented_prompt = self.prompt_augmenter.augment(query=query, contexts=contexts)
            response = self.llm.generate(augmented_prompt)

        response.latency_seconds = round(timer.elapsed, 3)
        return response

    def query_stream(self, query: UserQuery, top_k: int = 5) -> Generator[str, None, None]:
        """
        Sinh phản hồi dạng luồng (streaming) cho giao diện Web/Chatbot.
        """
        contexts = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=query.filters,
        )
        augmented_prompt = self.prompt_augmenter.augment(query=query, contexts=contexts)
        yield from self.llm.generate_stream(augmented_prompt)

# =========================================================================
# 4. KHỞI TẠO PIPELINES (FACTORY & SINGLETON)
# =========================================================================

_active_rag_pipeline: Optional[UETRAGPipeline] = None
_active_ingestion_pipeline: Optional[UETIngestionPipeline] = None


def get_rag_pipeline() -> UETRAGPipeline:
    """Khởi tạo hoặc lấy RAG Pipeline đang hoạt động."""
    global _active_rag_pipeline
    if _active_rag_pipeline is not None:
        return _active_rag_pipeline

    embedding_model = UETEmbedder()
    vector_store = UETVectorStore(persist_dir=BASE_DIR / "database" / "vector_db", embedding_model=embedding_model)

    # Nếu ChromaDB chưa có dữ liệu hoặc chạy lần đầu, nạp 5 chunk quy chế mẫu dự phòng
    if vector_store.count() == 0:
        logger.info("Chưa có cơ sở dữ liệu database/vector_db/, nạp 5 chunk quy chế mẫu dự phòng...")
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
        logger.info(f"Đã kết nối và sử dụng {vector_store.count()} vectors thực tế từ database/vector_db/")

    prompt_augmenter = UETPromptAugmenter()
    llm = UETLLMClient()
    retriever = UETRetriever(embedding_model=embedding_model, vector_store=vector_store)

    _active_rag_pipeline = UETRAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        prompt_augmenter=prompt_augmenter,
        llm=llm,
        retriever=retriever,
    )
    logger.info("Khởi tạo thành công UETRAGPipeline tích hợp đầy đủ trong src/pipeline.py")
    return _active_rag_pipeline


def get_ingestion_pipeline() -> UETIngestionPipeline:
    """Khởi tạo hoặc lấy Ingestion Pipeline đang hoạt động."""
    global _active_ingestion_pipeline
    if _active_ingestion_pipeline is not None:
        return _active_ingestion_pipeline

    _active_ingestion_pipeline = UETIngestionPipeline()
    return _active_ingestion_pipeline


def get_retriever() -> UETRetriever:
    """
    [Phân hệ Retrieval]: Khởi tạo và cung cấp UETRetriever điều phối tìm kiếm đa tầng.
    """
    embedding_model = UETEmbedder()
    vector_store = UETVectorStore(persist_dir=BASE_DIR / "database" / "vector_db", embedding_model=embedding_model)
    return UETRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


def run_ingest_cli(from_stage: str = "raw", to_stage: str = "vectordb"):
    """Khởi chạy quy trình điều phối và nạp dữ liệu đa chặng với báo cáo kết quả trên terminal."""
    print("\n" + "=" * 65)
    print(f"📦 BẮT ĐẦU ĐIỀU PHỐI DỮ LIỆU ĐA CHẶNG: [{from_stage.upper()}] ──> [{to_stage.upper()}]")
    print("=" * 65)

    pipeline = get_ingestion_pipeline()
    try:
        res = pipeline.run_stages(from_stage=from_stage, to_stage=to_stage)
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình điều phối dữ liệu: {e}")
        return

    print("\n" + "=" * 65)
    print("📊 BÁO CÁO KẾT QUẢ ĐIỀU PHỐI DỮ LIỆU (INGESTION REPORT)")
    print("=" * 65)
    transitions = res.get("transitions", {})
    if "raw_to_processed" in transitions:
        t = transitions["raw_to_processed"]
        print(f"  • Chặng [raw ──> processed]    : {t.get('processed', 0)} đã xử lý, {t.get('skipped', 0)} bỏ qua (trùng lặp)")
    if "processed_to_chunked" in transitions:
        t = transitions["processed_to_chunked"]
        print(f"  • Chặng [processed ──> chunked]: {t.get('processed_docs', 0)} tài liệu bóc tách -> {t.get('chunks_created', 0)} chunks mới, {t.get('skipped_docs', 0)} bỏ qua")
    if "chunked_to_vectordb" in transitions:
        t = transitions["chunked_to_vectordb"]
        print(f"  • Chặng [chunked ──> vectordb]     : {t.get('chunks_indexed', 0)} chunks nhúng mới, {t.get('skipped_chunks', 0)} bỏ qua (đã có trong DB)")
        print(f"    Tổng số vectors hiện có trong ChromaDB: {t.get('total_in_db', 0)}")
    if "chunked_to_indexed_data" in transitions:
        t = transitions["chunked_to_indexed_data"]
        print(f"  • Chặng [chunked ──> indexed_data] : {t.get('chunks_indexed', 0)} chunks đã lập chỉ mục BM25")
        print(f"    File lưu trữ BM25 index: {t.get('sparse_file')}")

    print(f"\n⏱️  Tổng thời gian thực thi: {res.get('latency_seconds', 0)}s")
    print("=" * 65 + "\n")


def run_ingest_status_cli():
    """Xem báo cáo thống kê số lượng dữ liệu ở từng chặng hiển thị trên terminal."""
    pipeline = get_ingestion_pipeline()
    stats = pipeline.get_stage_stats()
    stages = stats.get("stages", {})
    manifest = stats.get("manifest_tracking", {})

    print("\n" + "=" * 65)
    print("📊 THỐNG KÊ TRẠNG THÁI DỮ LIỆU CÁC CHẶNG (INGESTION STAGES)")
    print("=" * 65)
    print(f"  1. Giai đoạn [RAW]         : {stages.get('raw', {}).get('file_count', 0)} tệp/thư mục ({stages.get('raw', {}).get('path')})")
    print(f"  2. Giai đoạn [PROCESSED]   : {stages.get('processed', {}).get('document_count', 0)} tài liệu sạch ({stages.get('processed', {}).get('path')})")
    print(f"  3. Giai đoạn [CHUNKED]     : {stages.get('chunked', {}).get('chunk_count', 0)} chunks ({stages.get('chunked', {}).get('path')})")
    print(f"  4. Giai đoạn [VECTORDB]    : {stages.get('vectordb', {}).get('vector_count', 0)} vectors ({stages.get('vectordb', {}).get('path')})")
    idx_info = stages.get("indexed_data", {})
    status_str = f"Sẵn sàng ({idx_info.get('size_mb')} MB)" if idx_info.get("exists") else "Chưa tạo"
    print(f"  5. Giai đoạn [INDEXED_DATA]: {status_str} ({idx_info.get('path')})")

    print("\n📋 Sổ cái theo dõi chống trùng lặp (Manifest Tracker):")
    print(f"  • Tổng tài liệu đang theo dõi : {manifest.get('total_tracked_documents', 0)}")
    comp = manifest.get("stages_completed", {})
    print(f"  • Hoàn thành chặng processed  : {comp.get('processed', 0)}")
    print(f"  • Hoàn thành chặng chunked    : {comp.get('chunked', 0)}")
    print(f"  • Hoàn thành chặng vectordb   : {comp.get('vectordb', 0)}")
    print(f"  • Hoàn thành chặng indexed_data: {comp.get('indexed_data', 0)}")
    print(f"  • Lần cập nhật gần nhất       : {manifest.get('last_updated') or 'Chưa có'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    # Chạy kiểm thử End-to-End: python src/pipeline.py
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
    print(f"Thời gian: {res.latency_seconds}s")
    print("Trả lời:", res.answer)
    print(f"Số lượng nguồn trích dẫn: {len(res.sources)}")
    for s in res.sources[:3]:
        title = s.metadata.get("title") or s.metadata.get("source") or "Văn bản UET"
        print(f" - [{int(s.similarity_score * 100)}%] {title}")
    print("=" * 60)

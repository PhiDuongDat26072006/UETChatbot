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
from collections import defaultdict
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

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
from src.prompts import UETPromptAugmenter
from src.llm import UETLLMClient
from src.indexing.chunks import load_chunks_from_dir

if TYPE_CHECKING:
    from src.retrieval import UETRetriever

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

        metadata = dict(raw.metadata or {})
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

    STAGES: List[str] = ["raw", "processed", "chunked", "vectordb", "indexed_data", "graph_data", "lightrag"]

    def __init__(
        self,
        crawler: Optional[BaseDataCrawler] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        chunker: Optional[BaseChunker] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        manifest: Optional[DataManifestTracker] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or load_yaml_config()
        chunk_cfg = cfg.get("chunking", {})

        # Nạp trực tiếp tham số từ config.yaml cho UETChunker
        active_chunker = chunker or UETChunker(
            chunk_size=chunk_cfg.get("chunk_size"),
            chunk_overlap=chunk_cfg.get("chunk_overlap"),
            separators=chunk_cfg.get("separators"),
            config=cfg,
        )

        self._embedding_model: Optional[BaseEmbeddingModel] = embedding_model
        self._vector_store: Optional[BaseVectorStore] = vector_store
        super().__init__(
            crawler=crawler or UETDataLoader(),
            preprocessor=preprocessor or UETPreprocessor(),
            chunker=active_chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        self.config = cfg
        self.manifest = manifest or DataManifestTracker(manifest_path=BASE_DIR / "data" / ".ingest_manifest.json")

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        """Khởi tạo lazy mô hình embedding để tránh tải bge-m3 ở các chặng không dùng vector."""
        if self._embedding_model is None:
            self._embedding_model = UETEmbedder()
            if self._vector_store is not None and getattr(self._vector_store, "embedding_model", None) is None:
                self._vector_store.embedding_model = self._embedding_model
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, value: Optional[BaseEmbeddingModel]) -> None:
        self._embedding_model = value

    @property
    def vector_store(self) -> BaseVectorStore:
        """Khởi tạo lazy VectorStore khi thực sự cần truy cập ChromaDB."""
        if self._vector_store is None:
            self._vector_store = UETVectorStore(
                persist_dir=BASE_DIR / "database" / "vector_db",
                embedding_model=self._embedding_model,
            )
        return self._vector_store

    @vector_store.setter
    def vector_store(self, value: Optional[BaseVectorStore]) -> None:
        self._vector_store = value

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

    @staticmethod
    def _extract_faculty_code(data: Dict[str, Any], meta: Dict[str, Any], file_path: Optional[Path] = None) -> str:
        """
        Trích xuất mã khoa chuẩn hóa (fit, fat, fce, fema, fepn, fet, iai, sae, uet...)
        từ unit, domain, URL nguồn hoặc tên file.
        """
        # 1. Thử lấy từ unit (e.g. FIT, FAT, FEPN, FEMA, FET, SAE, IAI, FCE, UET...)
        raw_unit = str(data.get("unit") or meta.get("unit") or meta.get("faculty") or "").strip().lower()
        if raw_unit:
            clean_unit = "".join(c for c in raw_unit if c.isalnum() or c in "_-")
            if clean_unit:
                return clean_unit

        # 2. Thử lấy từ domain (e.g. fit.uet.vnu.edu.vn, uet.edu.vn...)
        raw_domain = str(data.get("domain") or meta.get("domain") or "").strip().lower()
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

        # 3. Thử trích xuất từ URL hoặc đường dẫn file cục bộ trong source
        source = str(data.get("source_url_or_path") or meta.get("source") or "").strip()
        if source:
            if "://" in source:
                try:
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

        # 4. Thử lấy từ tên file đang đọc (ví dụ: fit.uet.vnu.edu.vn.jsonl -> fit)
        if file_path is not None:
            stem = file_path.stem.lower()
            if "uet." in stem:
                parts = stem.split(".")
                if "uet" in parts:
                    idx = parts.index("uet")
                    if idx > 0:
                        candidate = parts[idx - 1]
                        return parts[idx - 2] if candidate == "www" and idx > 1 else candidate
                    return "uet"
            elif stem and stem != "ingested_processed":
                return stem

        return "general"

    def _run_processed_to_chunked(self) -> Dict[str, Any]:
        """
        Chuyển đổi dữ liệu từ processed -> chunked.
        Phân loại và lưu chunks theo từng khoa vào các file riêng biệt:
        data/chunked_data/{faculty}_chunks.jsonl (ví dụ: fit_chunks.jsonl, fat_chunks.jsonl...).
        """
        processed_dir = BASE_DIR / "data" / "processed_data"
        chunked_dir = BASE_DIR / "data" / "chunked_data"
        chunked_dir.mkdir(parents=True, exist_ok=True)

        jsonl_files = sorted(processed_dir.rglob("*.jsonl"))
        processed_docs = 0
        skipped_docs = 0
        total_chunks = 0
        chunks_by_faculty_count: Dict[str, int] = defaultdict(int)

        file_handles: Dict[str, Any] = {}
        try:
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
                            if "domain" in data and "domain" not in meta:
                                meta["domain"] = data["domain"]
                            if "unit" in data and "unit" not in meta:
                                meta["unit"] = data["unit"]
                            if "category" in data and "category" not in meta:
                                meta["category"] = data["category"]

                            faculty_code = self._extract_faculty_code(data, meta, jf)
                            meta["faculty"] = faculty_code

                            proc_doc = ProcessedData(
                                id=doc_id,
                                title=title,
                                content=content,
                                metadata=meta,
                            )
                            chunks = self.chunker.chunk(proc_doc)

                            if faculty_code not in file_handles:
                                out_file = chunked_dir / f"{faculty_code}_chunks.jsonl"
                                file_handles[faculty_code] = open(out_file, "a", encoding="utf-8")

                            f_out = file_handles[faculty_code]
                            for c in chunks:
                                c.metadata["faculty"] = faculty_code
                                c_record = {
                                    "chunk_id": c.chunk_id,
                                    "document_id": c.document_id,
                                    "text": c.text,
                                    "chunk_index": c.chunk_index,
                                    "metadata": c.metadata,
                                }
                                f_out.write(json.dumps(c_record, ensure_ascii=False) + "\n")
                                total_chunks += 1
                                chunks_by_faculty_count[faculty_code] += 1

                            self.manifest.mark_processed(
                                doc_id,
                                "chunked",
                                doc_hash,
                                title=title,
                                extra_meta={"chunk_count": len(chunks), "faculty": faculty_code},
                            )
                            processed_docs += 1
                except Exception as err:
                    logger.warning(f"Lỗi khi đọc file {jf}: {err}")
        finally:
            for fh in file_handles.values():
                try:
                    fh.close()
                except Exception:
                    pass

        self.manifest.save()
        logger.info(
            f"Hoàn thành bóc tách processed -> chunked: Đã tạo {total_chunks} chunks cho {processed_docs} tài liệu "
            f"chia thành các file khoa: {dict(chunks_by_faculty_count)}."
        )
        return {
            "processed_docs": processed_docs,
            "skipped_docs": skipped_docs,
            "chunks_created": total_chunks,
            "faculties": dict(chunks_by_faculty_count),
        }

    def _run_chunked_to_vectordb(
        self, batch_size: int = 500, clear_existing: bool = False
    ) -> Dict[str, int]:
        """Chuyển đổi dữ liệu từ chunked -> vectordb (nhúng vector và lưu vào ChromaDB)."""
        if clear_existing:
            logger.info("⚠️  --clear được bật: Xóa sạch toàn bộ dữ liệu cũ trong ChromaDB và manifest stage 'vectordb'...")
            self.vector_store.clear()
            self.manifest.clear_stage("vectordb")
            self.manifest.save()
            logger.info("✅ Đã xóa sạch ChromaDB và manifest stage 'vectordb'.")
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

    def _run_chunked_to_indexed_data(self, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Lập chỉ mục BM25 trực tiếp từ data/chunked_data/*.jsonl và lưu ra database/bm25_index.pkl.
        Được thực hiện bởi phân hệ Lập chỉ mục (src.indexing), tách biệt hoàn toàn khỏi phân hệ Retrieval.
        """
        from src.indexing import UETIndexer

        chunked_dir = BASE_DIR / "data" / "chunked_data"
        chunks = load_chunks_from_dir(chunked_dir)

        out_path = BASE_DIR / "database" / "bm25_index.pkl"
        if clear_existing and out_path.exists():
            out_path.unlink()

        indexer = UETIndexer(
            vector_store=self.vector_store,
            sparse_path=out_path,
        )
        result = indexer.index_chunks(chunks, clear_existing=clear_existing)

        # Chỉ cập nhật manifest khi kiểm tra và lưu BM25 thành công.
        for chunk in chunks:
            self.manifest.mark_processed(
                chunk.chunk_id,
                "indexed_data",
                content_hash=self.manifest.compute_hash(chunk.text),
            )
        self.manifest.save()

        # Tự động đồng bộ lại graph_index.pkl nếu bật cờ auto_sync_on_reindex và đã có graph_index cũ
        graph_cfg = self.config.get("graph", {})
        graph_rel = self.config.get("paths", {}).get("graph_index_file", "database/graph_index.pkl")
        graph_out = Path(graph_rel) if Path(graph_rel).is_absolute() else (BASE_DIR / graph_rel)
        if bool(graph_cfg.get("auto_sync_on_reindex", True)) and graph_out.exists() and chunks:
            logger.info("🔄 Phát hiện cập nhật indexed_data -> Tự động đồng bộ lại Unified GraphIndex (0 LLM)...")
            try:
                self._run_chunked_to_graph_data(clear_existing=True)
            except Exception as sync_err:
                logger.warning(f"Không thể tự động đồng bộ graph_index.pkl: {sync_err}")

        return {
            "chunks_indexed": result["indexed_chunks"],
            "sparse_file": str(out_path),
            "status": result["status"],
        }

    def _run_chunked_to_graph_data(
        self,
        clear_existing: bool = False,
        domain: Optional[str] = None,
        max_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Xây dựng chỉ mục Unified GraphRAG (hợp nhất LightRAG graph.json + entity_chroma + Regex chunks)
        và lưu ra database/graph_index.pkl. Hỗ trợ fallback đọc chunks từ bm25_index.pkl khi chunked_data trống.
        """
        from src.graph.graph import GraphIndex
        from src.indexing.sparse import BM25Index

        paths_cfg = self.config.get("paths", {})
        graph_cfg = self.config.get("graph", {})
        lightrag_cfg = self.config.get("lightrag", {})

        chunked_dir = BASE_DIR / "data" / "chunked_data"
        effective_max_chunks = max_chunks if max_chunks is not None else graph_cfg.get("max_chunks")
        try:
            chunks = load_chunks_from_dir(
                chunked_dir,
                domain=domain,
                max_chunks=effective_max_chunks,
            )
        except (FileNotFoundError, ValueError):
            chunks = []

        # Fallback: Nạp danh sách DataChunk trực tiếp từ database/bm25_index.pkl nếu data/chunked_data trống
        if not chunks:
            bm25_rel = paths_cfg.get("bm25_index_file", "database/bm25_index.pkl")
            bm25_path = Path(bm25_rel) if Path(bm25_rel).is_absolute() else (BASE_DIR / bm25_rel)
            if bm25_path.exists():
                logger.info(f"Thư mục chunked_data trống; nạp danh sách chunks từ '{bm25_path}'...")
                bm25_idx = BM25Index.load(bm25_path)
                loaded_chunks = list(getattr(bm25_idx, "chunks", []) or [])
                if domain and domain.lower().strip() != "all":
                    dom_low = domain.lower().strip()
                    loaded_chunks = [
                        c for c in loaded_chunks
                        if dom_low in str((c.metadata or {}).get("faculty") or (c.metadata or {}).get("domain") or "").lower()
                    ]
                if effective_max_chunks is not None:
                    loaded_chunks = loaded_chunks[: int(effective_max_chunks)]
                chunks = loaded_chunks

        out_path = Path(paths_cfg.get("graph_index_file", "database/graph_index.pkl"))
        if not out_path.is_absolute():
            out_path = BASE_DIR / out_path

        cache_path = Path(paths_cfg.get("graph_llm_cache_file", "database/graph_llm_cache.json"))
        if not cache_path.is_absolute():
            cache_path = BASE_DIR / cache_path

        # Phát hiện kho tri thức LightRAG (graph.json + semantic_units_cache.jsonl + entity_chroma) nếu có
        merge_lightrag = bool(graph_cfg.get("merge_lightrag_kg", True)) and domain is None and effective_max_chunks is None
        lr_graph_path: Optional[Path] = None
        lr_units_path: Optional[Path] = None
        lr_chroma_dir: Optional[Path] = None
        if merge_lightrag:
            cand_lr = Path(lightrag_cfg.get("graph_path", "database/lightrag_db/graph.json"))
            if not cand_lr.is_absolute():
                cand_lr = BASE_DIR / cand_lr
            if cand_lr.exists():
                lr_graph_path = cand_lr
                cand_units = Path(lightrag_cfg.get("semantic_units_cache", "database/lightrag_db/semantic_units_cache.jsonl"))
                if not cand_units.is_absolute():
                    cand_units = BASE_DIR / cand_units
                if cand_units.exists():
                    lr_units_path = cand_units
                cand_chroma = Path(lightrag_cfg.get("entity_vector_dir", "database/lightrag_db/entity_chroma"))
                if not cand_chroma.is_absolute():
                    cand_chroma = BASE_DIR / cand_chroma
                if (cand_chroma / "chroma.sqlite3").exists():
                    lr_chroma_dir = cand_chroma

        if clear_existing and out_path.exists():
            out_path.unlink()

        enable_llm = bool(graph_cfg.get("enable_llm_extraction", False))
        llm_provider = str(graph_cfg.get("llm_provider", "openrouter"))
        llm_client = UETLLMClient(provider=llm_provider) if enable_llm else None

        active_store = self.vector_store
        # Nếu cần nhúng 11 cụm quan hệ chuẩn hoặc vector dự phòng, truyền embedding_model khi có LightRAG merge
        fallback_emb = (
            self.embedding_model
            if (lr_graph_path is not None or active_store.count() == 0)
            else self._embedding_model
        )

        graph_index = GraphIndex(
            chunks=chunks,
            embedding_model=fallback_emb,
            vector_store=active_store,
            llm_client=llm_client,
            enable_llm_extraction=enable_llm,
            llm_provider=llm_provider,
            llm_model=graph_cfg.get("llm_model", "tencent/hy4-flash"),
            llm_domains=graph_cfg.get("llm_domains"),
            max_llm_chunks=graph_cfg.get("max_llm_chunks"),
            embed_batch_size=int(graph_cfg.get("embed_batch_size", 256)),
            cache_path=cache_path,
            lightrag_graph_path=lr_graph_path,
            semantic_units_path=lr_units_path,
            entity_chroma_dir=lr_chroma_dir,
        )
        graph_index.save(out_path)

        for chunk in chunks:
            self.manifest.mark_processed(
                chunk.chunk_id,
                "graph_data",
                content_hash=self.manifest.compute_hash(chunk.text),
            )
        self.manifest.save()

        return {
            "chunks_indexed": graph_index.count(),
            "entities_count": graph_index.entity_count(),
            "triples_count": graph_index.triple_count(),
            "lightrag_nodes_merged": getattr(graph_index, "lightrag_nodes_merged", 0),
            "lightrag_edges_merged": getattr(graph_index, "lightrag_edges_merged", 0),
            "llm_calls": graph_index.llm_calls_count,
            "graph_file": str(out_path),
            "status": "success" if chunks else "empty",
        }

    def run_stages(
        self,
        from_stage: str = "raw",
        to_stage: str = "vectordb",
        batch_size: int = 500,
        clear: bool = False,
        clear_existing: Optional[bool] = None,
        domain: Optional[str] = None,
        max_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Kích hoạt chạy luồng dữ liệu từ một giai đoạn đến một giai đoạn cụ thể.
        Tự động điều phối các bước trung gian hợp lệ:
        - Tuyến Vector DB: raw -> processed -> chunked -> vectordb
        - Tuyến Sparse Index: raw -> processed -> chunked -> indexed_data

        Nếu clear=True: xóa toàn bộ dữ liệu cũ (chunked_data, vectordb, BM25 index, manifest)
        trước khi chạy lại, đảm bảo nạp mới hoàn toàn.
        """
        if clear_existing is not None:
            clear = clear or clear_existing

        import shutil

        from_stage = from_stage.lower().strip()
        to_stage = to_stage.lower().strip()

        if from_stage not in self.STAGES or to_stage not in self.STAGES:
            raise ValueError(
                f"Giai đoạn không hợp lệ. Các giai đoạn được hỗ trợ: {self.STAGES}. "
                f"Nhận được: from='{from_stage}', to='{to_stage}'"
            )

        if to_stage in ("indexed_data", "graph_data") and from_stage in ["raw", "processed"]:
            raise ValueError(
                "Để đảm bảo tính toàn vẹn và đồng bộ ID giữa ChromaDB, BM25 và GraphRAG, "
                "vui lòng thực hiện theo 2 bước tuần tự:\n"
                f"  • Bước 1: python main.py --ingest --from {from_stage} --to vectordb --clear\n"
                f"  • Bước 2: python main.py --ingest --from chunked --to {to_stage} --clear"
            )

        stage_order = {
            "raw": 0,
            "processed": 1,
            "chunked": 2,
            "vectordb": 3,
            "indexed_data": 3,
            "graph_data": 3,
            "lightrag": 3,
        }
        if stage_order[from_stage] >= stage_order[to_stage]:
            raise ValueError(
                f"from_stage '{from_stage}' phải đứng trước to_stage '{to_stage}'. "
                f"Thứ tự hợp lệ: raw -> processed -> chunked -> vectordb / indexed_data / graph_data / lightrag"
            )

        # ---- XÓA DỮ LIỆU CŨ nếu --clear được bật ----
        if clear:
            logger.info("🗑️ [--clear] Bắt đầu xóa dữ liệu cũ trước khi chạy lại...")
            stages_to_reset = set()

            # 1. Reset processed nếu chạy từ raw
            if from_stage == "raw":
                stages_to_reset.add("processed")

            # 2. Xóa các file trong folder chunked_data CHỈ KHI thực sự đi qua chặng processed -> chunked
            if from_stage in ["raw", "processed"] and to_stage in ["chunked", "vectordb"]:
                chunked_dir = BASE_DIR / "data" / "chunked_data"
                if chunked_dir.exists():
                    shutil.rmtree(chunked_dir)
                    chunked_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("✅ Đã dọn sạch các file trong folder data/chunked_data")
                stages_to_reset.add("chunked")

            # 3. Xóa toàn bộ vectors trong ChromaDB nếu đích là vectordb
            if to_stage == "vectordb":
                try:
                    self.vector_store.clear()
                    logger.info("✅ Đã xóa toàn bộ vectors trong ChromaDB")
                except Exception as e:
                    logger.warning(f"⚠️ Không thể xóa vectordb: {e}")
                stages_to_reset.add("vectordb")

            # 4. Xóa file BM25 index cũ nếu đích là indexed_data
            if to_stage == "indexed_data":
                bm25_path = BASE_DIR / "database" / "bm25_index.pkl"
                if bm25_path.exists():
                    bm25_path.unlink()
                    logger.info("✅ Đã xóa file BM25 index cũ (database/bm25_index.pkl)")
                stages_to_reset.add("indexed_data")

            # 4b. Xóa file GraphRAG index cũ nếu đích là graph_data
            if to_stage == "graph_data":
                graph_rel = self.config.get("paths", {}).get("graph_index_file", "database/graph_index.pkl")
                graph_path = Path(graph_rel) if Path(graph_rel).is_absolute() else (BASE_DIR / graph_rel)
                if graph_path.exists():
                    graph_path.unlink()
                    logger.info(f"✅ Đã xóa file GraphRAG index cũ ({graph_rel})")
                stages_to_reset.add("graph_data")

            # 5. Xóa Knowledge Graph cũ nếu đích là lightrag
            if to_stage == "lightrag":
                lr_cfg = self.config.get("lightrag", {})
                lr_graph_rel = lr_cfg.get("graph_path", "database/lightrag_db/graph.json")
                lr_cache_rel = lr_cfg.get("semantic_units_cache", "database/lightrag_db/semantic_units_cache.jsonl")
                lr_graph_path = Path(lr_graph_rel) if Path(lr_graph_rel).is_absolute() else (BASE_DIR / lr_graph_rel)
                lr_bak_path = lr_graph_path.with_name(lr_graph_path.name + ".bak")
                cache_path = Path(lr_cache_rel) if Path(lr_cache_rel).is_absolute() else (BASE_DIR / lr_cache_rel)
                if lr_graph_path.exists():
                    lr_graph_path.unlink()
                    logger.info(f"✅ Đã xóa file đồ thị cũ ({lr_graph_rel})")
                if lr_bak_path.exists():
                    lr_bak_path.unlink()
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"✅ Đã xóa semantic units cache cũ ({lr_cache_rel})")
                stages_to_reset.add("lightrag")

            # 6. Reset manifest entries cho các chặng tương ứng
            if stages_to_reset:
                self.manifest.reset_stages(list(stages_to_reset))
                logger.info(f"✅ Đã reset manifest tracking cho các chặng: {list(stages_to_reset)}")


        results: Dict[str, Any] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "transitions": {},
        }

        with Timer() as timer:
            # Chặng 1: raw -> processed
            if from_stage == "raw" and to_stage in ["processed", "chunked", "vectordb", "lightrag"]:
                logger.info("▶ BẮT ĐẦU CHẶNG: raw -> processed")
                res_1 = self._run_raw_to_processed()
                results["transitions"]["raw_to_processed"] = res_1
                logger.info(f"✔ Hoàn thành raw -> processed: {res_1}")

            # Chặng 2: processed -> chunked (chỉ cần cho RAG truyền thống: vectordb, indexed_data)
            if from_stage in ["raw", "processed"] and to_stage in ["chunked", "vectordb"]:
                logger.info("▶ BẮT ĐẦU CHẶNG: processed -> chunked")
                res_2 = self._run_processed_to_chunked()
                results["transitions"]["processed_to_chunked"] = res_2
                logger.info(f"✔ Hoàn thành processed -> chunked: {res_2}")

            # Chặng 3a: chunked -> vectordb (RAG truyền thống)
            if from_stage in ["raw", "processed", "chunked"] and to_stage == "vectordb":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> vectordb")
                res_3 = self._run_chunked_to_vectordb(batch_size=batch_size, clear_existing=clear)
                results["transitions"]["chunked_to_vectordb"] = res_3
                logger.info(f"✔ Hoàn thành chunked -> vectordb: {res_3}")

            # Chặng 3b: chunked -> indexed_data (BM25 RAG truyền thống)
            if from_stage == "chunked" and to_stage == "indexed_data":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> indexed_data (BM25 Sparse Index)")
                res_4 = self._run_chunked_to_indexed_data(clear_existing=clear)
                results["transitions"]["chunked_to_indexed_data"] = res_4
                logger.info(f"✔ Hoàn thành chunked -> indexed_data: {res_4}")

            # Chặng 3c: chunked -> graph_data
            if from_stage == "chunked" and to_stage == "graph_data":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> graph_data (GraphRAG Vector Index)")
                res_5 = self._run_chunked_to_graph_data(
                    clear_existing=clear,
                    domain=domain,
                    max_chunks=max_chunks,
                )
                results["transitions"]["chunked_to_graph_data"] = res_5
                logger.info(f"✔ Hoàn thành chunked -> graph_data: {res_5}")

            # Chặng 3d: processed -> lightrag (Xây dựng Knowledge Graph qua Semantic Units)
            if from_stage in ["raw", "processed", "chunked"] and to_stage == "lightrag":
                logger.info("▶ BẮT ĐẦU CHẶNG: processed -> lightrag (Xây dựng Knowledge Graph qua Semantic Units)")
                res_lr = self._run_processed_to_lightrag(clear_existing=clear)
                results["transitions"]["processed_to_lightrag"] = res_lr
                logger.info(f"✔ Hoàn thành processed -> lightrag: {res_lr}")

        results["latency_seconds"] = round(timer.elapsed, 2)
        logger.info(f"Hoàn thành toàn bộ tiến trình từ '{from_stage}' đến '{to_stage}' trong {results['latency_seconds']}s")
        return results

    def _load_processed_docs(self) -> List[ProcessedData]:
        """
        Nạp danh sách tài liệu sạch từ data/processed_data/*.jsonl.
        Đảm bảo nguyên vẹn cấu trúc văn bản (không bị cắt vụn 800 ký tự)
        để AdaptiveSemanticExtractor phân tách thành các Semantic Units hoàn chỉnh.
        """
        processed_dir = BASE_DIR / "data" / "processed_data"
        docs: List[ProcessedData] = []
        if processed_dir.exists():
            for f in sorted(processed_dir.glob("*.jsonl")):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        for line in fp:
                            if line.strip():
                                docs.append(ProcessedData(**json.loads(line)))
                except Exception as e:
                    logger.warning(f"Lỗi khi đọc file {f}: {e}")

        # Fallback: nếu processed_data trống nhưng có chunked_data
        if not docs:
            chunked_dir = BASE_DIR / "data" / "chunked_data"
            if chunked_dir.exists():
                from collections import defaultdict
                from src.indexing.chunks import load_chunks_from_dir

                chunks = load_chunks_from_dir(chunked_dir)
                if chunks:
                    logger.info(f"Tái tạo ProcessedData từ {len(chunks)} chunks trong {chunked_dir}...")
                    doc_map = defaultdict(list)
                    for c in chunks:
                        doc_map[c.document_id].append(c)

                    MAX_DOC_CHARS = 60_000
                    for doc_id, doc_chunks in doc_map.items():
                        doc_chunks.sort(key=lambda x: getattr(x, "chunk_index", 0))
                        merged_text = "\n\n".join(c.text for c in doc_chunks)
                        if len(merged_text) > MAX_DOC_CHARS:
                            merged_text = merged_text[:MAX_DOC_CHARS]

                        first_chunk = doc_chunks[0]
                        meta = dict(first_chunk.metadata or {})
                        title = meta.get("title") or meta.get("heading") or "Tài liệu đào tạo UET"
                        docs.append(
                            ProcessedData(
                                id=doc_id,
                                raw_data_id=doc_id,
                                title=title,
                                content=merged_text,
                                metadata=meta,
                            )
                        )
        return docs

    def _run_processed_to_lightrag(self, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Khởi chạy quy trình trích xuất và xây dựng Knowledge Graph LightRAG:
        1. Nạp trực tiếp các tài liệu sạch nguyên bản (ProcessedData) từ data/processed_data/*.jsonl.
        2. Tự động bóc tách thành Semantic Units (Điều/Khoản, Bảng, Heading) qua AdaptiveSemanticExtractor.
        3. Khởi chạy 2-Pass Extraction (Chặng 1 Root Entities, Chặng 2 Semantic Units).
        4. Vector hóa và lưu trữ Knowledge Graph.
        """
        from src.graph.lightrag_indexer import LightRAGIndexer

        docs = self._load_processed_docs()
        if not docs:
            logger.warning("Không tìm thấy tài liệu sạch nào trong data/processed_data. Bỏ qua nạp LightRAG.")
            return {"status": "skipped", "reason": "no_processed_docs_found"}

        logger.info(f"Đã nạp {len(docs)} tài liệu sạch. Khởi tạo pipeline LightRAG với Semantic Units...")

        cfg = self.config.get("lightrag", {})
        indexer_cfg = cfg.get("indexer", {})
        indexer = LightRAGIndexer(
            llm_provider=indexer_cfg.get("llm_provider"),
            delay_between_calls=indexer_cfg.get("delay_between_calls", 0.5),
            max_pass2_retries=indexer_cfg.get("max_pass2_retries", 2),
            max_workers=indexer_cfg.get("max_workers", 5),
            header_max_chars=indexer_cfg.get("header_max_chars", 2000),
            config=self.config,
        )
        return indexer.run_full_pipeline(
            docs=docs,
            checkpoint_every=indexer_cfg.get("checkpoint_every", 10),
            clear_existing=clear_existing,
            max_workers=indexer_cfg.get("max_workers", 5),
        )


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
        graph_rel = self.config.get("paths", {}).get("graph_index_file", "database/graph_index.pkl")
        graph_file = Path(graph_rel) if Path(graph_rel).is_absolute() else (BASE_DIR / graph_rel)
        graph_exists = graph_file.exists()
        graph_size_mb = round(graph_file.stat().st_size / (1024 * 1024), 2) if graph_exists else 0.0

        lr_rel = self.config.get("lightrag", {}).get("graph_path", "database/lightrag_db/graph.json")
        lr_file = Path(lr_rel) if Path(lr_rel).is_absolute() else (BASE_DIR / lr_rel)
        lr_exists = lr_file.exists()
        lr_size_mb = round(lr_file.stat().st_size / (1024 * 1024), 2) if lr_exists else 0.0
        ent_count, rel_count = 0, 0
        if lr_exists:
            try:
                with open(lr_file, "r", encoding="utf-8") as gfp:
                    gdata = json.load(gfp)
                    ent_count = len(gdata.get("entities") or gdata.get("nodes") or [])
                    rel_count = len(gdata.get("relations") or gdata.get("edges") or [])
            except Exception:
                pass

        return {
            "stages": {
                "raw": {"path": str(raw_dir.relative_to(BASE_DIR)), "file_count": raw_files},
                "processed": {"path": str(processed_dir.relative_to(BASE_DIR)), "document_count": proc_docs},
                "chunked": {"path": str(chunked_dir.relative_to(BASE_DIR)), "chunk_count": chunk_count},
                "vectordb": {"path": "database/vector_db", "vector_count": db_count},
                "indexed_data": {"path": "database/bm25_index.pkl", "exists": bm25_exists, "size_mb": bm25_size_mb},
                "graph_data": {"path": str(graph_rel), "exists": graph_exists, "size_mb": graph_size_mb},
                "lightrag": {
                    "path": "database/lightrag_db/graph.json",
                    "exists": lr_exists,
                    "size_mb": lr_size_mb,
                    "entity_count": ent_count,
                    "relation_count": rel_count,
                },
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
        if retriever is None:
            from src.retrieval import UETRetriever
            retriever = UETRetriever(
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
            )
        self.retriever = retriever

    def query(self, query: UserQuery, top_k: Optional[int] = None, engine: Optional[str] = None) -> Response:
        """
        Thực thi quy trình trả lời câu hỏi:
        1. Gọi UETRetriever: Hỗ trợ động cơ vector, lightrag, hybrid hoặc auto
        2. Ghép ngữ cảnh vào Prompt (AugmentedPrompt) theo chuẩn BaseRAGPipeline
        3. Gọi LLM sinh câu trả lời
        """
        k = top_k if top_k is not None else getattr(self.retriever, "default_top_k", 10)
        with Timer() as timer:
            # Bước 1: Truy xuất ngữ cảnh bằng UETRetriever (hỗ trợ chọn engine)
            contexts = self.retriever.retrieve(
                query=query,
                top_k=k,
                filters=query.filters,
                engine=engine,
            )

            # Bước 2 & 3: Ghép prompt và gọi LLM
            augmented_prompt = self.prompt_augmenter.augment(query=query, contexts=contexts)
            response = self.llm.generate(augmented_prompt)

        response.latency_seconds = round(timer.elapsed, 3)
        return response

    def query_stream(self, query: UserQuery, top_k: Optional[int] = None, engine: Optional[str] = None) -> Generator[str, None, None]:
        """
        Sinh phản hồi dạng luồng (streaming) cho giao diện Web/Chatbot.
        """
        k = top_k if top_k is not None else getattr(self.retriever, "default_top_k", 10)
        contexts = self.retriever.retrieve(
            query=query,
            top_k=k,
            filters=query.filters,
            engine=engine,
        )
        augmented_prompt = self.prompt_augmenter.augment(query=query, contexts=contexts)
        yield from self.llm.generate_stream(augmented_prompt)

# =========================================================================
# 4. KHỞI TẠO PIPELINES (FACTORY & SINGLETON)
# =========================================================================

_active_rag_pipeline: Optional[UETRAGPipeline] = None
_active_agentic_rag_pipeline: Optional[Any] = None
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
        logger.info("Chưa có cơ sở dữ liệu vector_db/, nạp 5 chunk quy chế mẫu dự phòng...")
    else:
        logger.info(f"Đã kết nối và sử dụng {vector_store.count()} vectors thực tế từ database/vector_db/")

    from src.retrieval import UETRetriever

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


def get_agentic_rag_pipeline() -> Any:
    """Khởi tạo hoặc lấy Agentic RAG Pipeline (LangGraph State Machine) đang hoạt động."""
    global _active_agentic_rag_pipeline
    if _active_agentic_rag_pipeline is not None:
        return _active_agentic_rag_pipeline

    from src.agent import UETAgenticRAGPipeline

    embedding_model = UETEmbedder()
    vector_store = UETVectorStore(persist_dir=BASE_DIR / "database" / "vector_db", embedding_model=embedding_model)

    from src.retrieval import UETRetriever

    prompt_augmenter = UETPromptAugmenter()
    llm = UETLLMClient()
    retriever = UETRetriever(embedding_model=embedding_model, vector_store=vector_store)

    _active_agentic_rag_pipeline = UETAgenticRAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        prompt_augmenter=prompt_augmenter,
        llm=llm,
        retriever=retriever,
    )
    logger.info("Khởi tạo thành công UETAgenticRAGPipeline tích hợp LangGraph trong src/pipeline.py")
    return _active_agentic_rag_pipeline


def get_active_pipeline(mode: Optional[str] = None) -> Any:
    """
    Lấy pipeline theo tham số mode, biến môi trường RAG_MODE hoặc config.yaml ('modular' hoặc 'agentic').
    Mặc định: 'modular' (Hybrid RAG); chỉ bật Agentic khi có cờ --agentic hoặc cấu hình rõ.
    """
    env_mode = os.environ.get("RAG_MODE")
    cfg = load_yaml_config()
    chosen_mode = (mode or env_mode or cfg.get("agent", {}).get("mode") or "modular").strip().lower()
    if chosen_mode == "agentic":
        return get_agentic_rag_pipeline()
    return get_rag_pipeline()


def describe_pipeline_mode(pipeline: Any) -> str:
    """Trả về nhãn chế độ đang chạy: Hybrid / Graph + Hybrid / Agentic + Hybrid / Graph + Hybrid + Agentic."""
    is_agentic = hasattr(pipeline, "agent_graph")
    retriever = getattr(pipeline, "retriever", None)
    enable_graph = bool(getattr(retriever, "enable_graph", False))
    graph_loaded = getattr(retriever, "graph_index", None) is not None and (
        getattr(retriever.graph_index, "count", lambda: 0)() > 0
    )
    parts: List[str] = []
    if enable_graph:
        parts.append("Graph" if graph_loaded else "Graph (chưa có graph_index.pkl)")
    parts.append("Hybrid")
    if is_agentic:
        parts.append("Agentic")
    return " + ".join(parts)


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
    from src.retrieval import UETRetriever

    embedding_model = UETEmbedder()
    vector_store = UETVectorStore(persist_dir=BASE_DIR / "database" / "vector_db", embedding_model=embedding_model)
    return UETRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


def run_ingest_cli(
    from_stage: str = "raw",
    to_stage: str = "vectordb",
    clear: bool = False,
    clear_existing: Optional[bool] = None,
    domain: Optional[str] = None,
    max_chunks: Optional[int] = None,
):
    if clear_existing is not None:
        clear = clear or clear_existing
    """Khởi chạy quy trình điều phối và nạp dữ liệu đa chặng với báo cáo kết quả trên terminal."""
    print("\n" + "=" * 65)
    if clear:
        print("🗑️  [--clear] Đã bật cờ làm mới: Dữ liệu cũ sẽ bị xóa và nạp lại toàn bộ!")
    print(f"📦 BẮT ĐẦU ĐIỀU PHỐI DỮ LIỆU ĐA CHẶNG: [{from_stage.upper()}] ──> [{to_stage.upper()}]")
    if domain or max_chunks:
        print(f"   • Phạm vi thử nghiệm: domain={domain or 'ALL'}, max_chunks={max_chunks or 'ALL'}")
    print("=" * 65)

    pipeline = get_ingestion_pipeline()
    try:
        res = pipeline.run_stages(
            from_stage=from_stage,
            to_stage=to_stage,
            clear=clear,
            clear_existing=clear,
            domain=domain,
            max_chunks=max_chunks,
        )
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
        faculties = t.get("faculties", {})
        faculties_str = f" ({len(faculties)} file khoa: {', '.join(faculties.keys())})" if faculties else ""
        print(f"  • Chặng [processed ──> chunked]: {t.get('processed_docs', 0)} tài liệu bóc tách -> {t.get('chunks_created', 0)} chunks mới{faculties_str}, {t.get('skipped_docs', 0)} bỏ qua")
    if "chunked_to_vectordb" in transitions:
        t = transitions["chunked_to_vectordb"]
        print(f"  • Chặng [chunked ──> vectordb]     : {t.get('chunks_indexed', 0)} chunks nhúng mới, {t.get('skipped_chunks', 0)} bỏ qua (đã có trong DB)")
        print(f"    Tổng số vectors hiện có trong ChromaDB: {t.get('total_in_db', 0)}")
    if "chunked_to_indexed_data" in transitions:
        t = transitions["chunked_to_indexed_data"]
        print(f"  • Chặng [chunked ──> indexed_data] : {t.get('chunks_indexed', 0)} chunks đã lập chỉ mục BM25")
        print(f"    File lưu trữ BM25 index: {t.get('sparse_file')}")
    if "chunked_to_graph_data" in transitions:
        t = transitions["chunked_to_graph_data"]
        lr_info_str = (
            f", LightRAG merged: +{t.get('lightrag_nodes_merged', 0)} nodes / +{t.get('lightrag_edges_merged', 0)} edges"
            if t.get("lightrag_nodes_merged") or t.get("lightrag_edges_merged")
            else ""
        )
        print(
            f"  • Chặng [chunked ──> graph_data]   : {t.get('chunks_indexed', 0)} chunks -> "
            f"{t.get('entities_count', 0)} entities, {t.get('triples_count', 0)} triples "
            f"(OpenRouter LLM calls: {t.get('llm_calls', 0)}{lr_info_str})"
        )
        print(f"    File lưu trữ Graph index: {t.get('graph_file')}")
    t_lr = transitions.get("processed_to_lightrag") or transitions.get("chunked_to_lightrag")
    if t_lr:
        lr_nodes = t_lr.get("final_nodes", t_lr.get("total_entities", 0))
        lr_edges = t_lr.get("final_edges", t_lr.get("total_relations", 0))
        lr_file = t_lr.get("graph_path") or t_lr.get("graph_file")
        status_note = " (tạm dừng sớm do lỗi LLM liên tiếp)" if t_lr.get("aborted_due_to_llm_errors") else ""
        print(f"  • Chặng [processed ──> lightrag]   : Đã xây dựng Knowledge Graph{status_note}")
        print(f"    Thực thể: {lr_nodes}, Quan hệ: {lr_edges}, File: {lr_file}")

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
    graph_info = stages.get("graph_data", {})
    g_status_str = f"Sẵn sàng ({graph_info.get('size_mb')} MB)" if graph_info.get("exists") else "Chưa tạo"
    print(f"  6. Giai đoạn [GRAPH_DATA]  : {g_status_str} ({graph_info.get('path')})")

    lr_info = stages.get("lightrag", {})
    if lr_info.get("exists"):
        lr_status = f"Sẵn sàng ({lr_info.get('entity_count')} entities, {lr_info.get('relation_count')} relations, {lr_info.get('size_mb')} MB)"
    else:
        lr_status = "Chưa tạo (Sẵn sàng khởi chạy với: python main.py --ingest --to lightrag)"
    print(f"  7. Giai đoạn [LIGHTRAG]    : {lr_status} ({lr_info.get('path')})")

    print("\n📋 Sổ cái theo dõi chống trùng lặp (Manifest Tracker):")
    print(f"  • Tổng tài liệu đang theo dõi : {manifest.get('total_tracked_documents', 0)}")
    comp = manifest.get("stages_completed", {})
    print(f"  • Hoàn thành chặng processed  : {comp.get('processed', 0)}")
    print(f"  • Hoàn thành chặng chunked    : {comp.get('chunked', 0)}")
    print(f"  • Hoàn thành chặng vectordb   : {comp.get('vectordb', 0)}")
    print(f"  • Hoàn thành chặng indexed_data: {comp.get('indexed_data', 0)}")
    print(f"  • Hoàn thành chặng graph_data  : {comp.get('graph_data', 0)}")
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

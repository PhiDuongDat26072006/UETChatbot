"""Multi-stage ingestion pipeline orchestration for UET Chatbot."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.base import (
    BaseChunker,
    BaseDataCrawler,
    BaseEmbeddingModel,
    BaseIngestionPipeline,
    BasePreprocessor,
    BaseVectorStore,
    DataChunk,
    DataSource,
    ProcessedData,
)
from src.chunking import UETChunker
from src.embeddings import UETEmbedder
from src.indexing.chunks import load_chunks_from_dir
from src.ingestion import DataManifestTracker, UETDataLoader
from src.llm import UETLLMClient
from src.pipelines.helpers import (
    UETPreprocessor,
    extract_faculty_code,
    get_pipeline_attr,
    load_processed_docs,
)
from src.utils.helpers import (
    BASE_DIR,
    Timer,
    count_nonempty_lines,
    get_logger,
    load_yaml_config,
    resolve_project_path,
)
from src.vectordb import UETVectorStore

logger = get_logger("pipeline")


class UETIngestionPipeline(BaseIngestionPipeline):
    """
    [CỖ MÁY OFFLINE]: Nạp và lập chỉ mục dữ liệu đào tạo UET.
    Điều phối luồng: Cào web/Đọc file -> Làm sạch -> Cắt chunk -> Embed -> Lưu vào Vector DB.
    Hỗ trợ điều phối chặng linh hoạt (Multi-Stage Ingestion) và chống xử lý trùng lặp (Deduplication).
    """

    STAGES: List[str] = ["raw", "processed", "chunked", "vectordb", "indexed_data", "graph_data", "lightrag"]
    _STAGE_ORDER: Dict[str, int] = {
        "raw": 0,
        "processed": 1,
        "chunked": 2,
        "vectordb": 3,
        "indexed_data": 3,
        "graph_data": 3,
        "lightrag": 3,
    }

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
        self.manifest = manifest or DataManifestTracker(
            manifest_path=self._base_dir() / "data" / ".ingest_manifest.json"
        )

    @staticmethod
    def _base_dir() -> Path:
        return Path(get_pipeline_attr("BASE_DIR", BASE_DIR))

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        """Khởi tạo lazy mô hình embedding để tránh tải bge-m3 ở các chặng không dùng vector."""
        if self._embedding_model is None:
            embedder_cls = get_pipeline_attr("UETEmbedder", UETEmbedder)
            self._embedding_model = embedder_cls()
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
            store_cls = get_pipeline_attr("UETVectorStore", UETVectorStore)
            self._vector_store = store_cls(
                persist_dir=self._base_dir() / "database" / "vector_db",
                embedding_model=self._embedding_model,
            )
        return self._vector_store

    @vector_store.setter
    def vector_store(self, value: Optional[BaseVectorStore]) -> None:
        self._vector_store = value

    @staticmethod
    def _extract_faculty_code(
        data: Dict[str, Any],
        meta: Dict[str, Any],
        file_path: Optional[Path] = None,
    ) -> str:
        """Delegate faculty extraction for backward compatibility with existing callers."""
        return extract_faculty_code(data, meta, file_path)

    def run(self, sources: List[DataSource], clear_existing: bool = False) -> Dict[str, Any]:
        """Thực thi quy trình lập chỉ mục toàn bộ dữ liệu nguồn."""
        logger.info(f"Bắt đầu quy trình Ingestion với {len(sources)} nguồn dữ liệu...")
        with Timer() as timer:
            from src.indexing import UETIndexer

            raw_docs = self.crawler.crawl_all(sources)
            processed_docs = self.preprocessor.process_batch(raw_docs)
            chunks = self.chunker.chunk_batch(processed_docs)
            indexed = UETIndexer(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
            ).index_chunks(chunks, clear_existing=clear_existing)
            result: Dict[str, Any] = {
                "status": indexed["status"],
                "sources_count": len(sources),
                "raw_docs_count": len(raw_docs),
                "processed_docs_count": len(processed_docs),
                "chunks_count": len(chunks),
                "saved_in_db": indexed["indexed_chunks"],
                "total_in_db": indexed["total_in_db"],
            }
        result["latency_seconds"] = round(timer.elapsed, 2)
        logger.info(
            f"Hoàn thành Ingestion trong {result['latency_seconds']}s: "
            f"{result['raw_docs_count']} văn bản thô -> {result['chunks_count']} chunks -> "
            f"Tổng hiện có trong DB: {result['total_in_db']}"
        )
        return result

    def _run_raw_to_processed(self) -> Dict[str, int]:
        """Chuyển đổi dữ liệu từ raw -> processed."""
        base_dir = self._base_dir()
        raw_dir = base_dir / "data" / "raw_data"
        processed_dir = base_dir / "data" / "processed_data"
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_file = processed_dir / "ingested_processed.jsonl"

        raw_docs = self.crawler.crawl(DataSource(source_type="directory", uri=str(raw_dir)))
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

    def _run_processed_to_chunked(self) -> Dict[str, Any]:
        """
        Chuyển đổi dữ liệu từ processed -> chunked.
        Phân loại và lưu chunks theo từng khoa vào các file riêng biệt:
        data/chunked_data/{faculty}_chunks.jsonl (ví dụ: fit_chunks.jsonl, fat_chunks.jsonl...).
        """
        base_dir = self._base_dir()
        processed_dir = base_dir / "data" / "processed_data"
        chunked_dir = base_dir / "data" / "chunked_data"
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

                            meta = dict(data.get("metadata") or {})
                            if "source_url_or_path" in data and "source" not in meta:
                                meta["source"] = data["source_url_or_path"]
                            if title and "title" not in meta:
                                meta["title"] = title
                            for field in ("domain", "unit", "category"):
                                if field in data and field not in meta:
                                    meta[field] = data[field]

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
                            for chunk in chunks:
                                chunk.metadata["faculty"] = faculty_code
                                c_record = {
                                    "chunk_id": chunk.chunk_id,
                                    "document_id": chunk.document_id,
                                    "text": chunk.text,
                                    "chunk_index": chunk.chunk_index,
                                    "metadata": chunk.metadata,
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

    def _flush_vectordb_batch(self, batch: List[DataChunk]) -> int:
        """Embed, store, and record a batch of DataChunks in ChromaDB and the manifest."""
        if not batch:
            return 0
        vectors = self.embedding_model.embed_chunks(batch)
        self.vector_store.store(batch, vectors)
        for chunk in batch:
            self.manifest.mark_processed(chunk.chunk_id, "vectordb", content_hash=chunk.chunk_id)
        count = len(batch)
        batch.clear()
        return count

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

        chunked_dir = self._base_dir() / "data" / "chunked_data"
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
                        if self.manifest.is_processed(cid, "vectordb"):
                            skipped_chunks += 1
                            continue

                        batch.append(
                            DataChunk(
                                chunk_id=cid,
                                document_id=item.get("document_id"),
                                text=item.get("text") or "",
                                chunk_index=item.get("chunk_index", 0),
                                metadata=item.get("metadata") or {},
                            )
                        )

                        if len(batch) >= batch_size:
                            total_indexed += self._flush_vectordb_batch(batch)
                            logger.info(f"Đã nhúng & lưu {total_indexed} chunks vào Vector DB...")
            except Exception as err:
                logger.warning(f"Lỗi khi xử lý file chunk {jf}: {err}")

        total_indexed += self._flush_vectordb_batch(batch)
        self.manifest.save()
        return {
            "chunks_indexed": total_indexed,
            "skipped_chunks": skipped_chunks,
            "total_in_db": self.vector_store.count(),
        }

    def _run_chunked_to_indexed_data(self, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Lập chỉ mục BM25 trực tiếp từ data/chunked_data/*.jsonl và lưu ra database/bm25_index.pkl.
        Được thực hiện bởi phân hệ Lập chỉ mục (src.indexing), tách biệt hoàn toàn khỏi phân hệ Retrieval.
        """
        from src.indexing import UETIndexer

        base_dir = self._base_dir()
        chunked_dir = base_dir / "data" / "chunked_data"
        chunks = load_chunks_from_dir(chunked_dir)

        out_path = base_dir / "database" / "bm25_index.pkl"
        if clear_existing and out_path.exists():
            out_path.unlink()

        indexer = UETIndexer(vector_store=self.vector_store, sparse_path=out_path)
        result = indexer.index_chunks(chunks, clear_existing=clear_existing)

        for chunk in chunks:
            self.manifest.mark_processed(
                chunk.chunk_id,
                "indexed_data",
                content_hash=self.manifest.compute_hash(chunk.text),
            )
        self.manifest.save()

        graph_cfg = self.config.get("graph", {})
        graph_rel = self.config.get("paths", {}).get("graph_index_file", "database/graph_index.pkl")
        graph_out = resolve_project_path(graph_rel, base_dir)
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

    def _load_chunks_for_graph(
        self,
        base_dir: Path,
        domain: Optional[str],
        max_chunks: Optional[int],
    ) -> List[DataChunk]:
        """Load chunks from chunked_data, falling back to bm25_index.pkl if chunked_data is empty."""
        from src.indexing.sparse import BM25Index

        chunked_dir = base_dir / "data" / "chunked_data"
        try:
            chunks = load_chunks_from_dir(chunked_dir, domain=domain, max_chunks=max_chunks)
        except (FileNotFoundError, ValueError):
            chunks = []

        if chunks:
            return chunks

        bm25_rel = self.config.get("paths", {}).get("bm25_index_file", "database/bm25_index.pkl")
        bm25_path = resolve_project_path(bm25_rel, base_dir)
        if not bm25_path.exists():
            return []

        logger.info(f"Thư mục chunked_data trống; nạp danh sách chunks từ '{bm25_path}'...")
        bm25_idx = BM25Index.load(bm25_path)
        loaded_chunks = list(getattr(bm25_idx, "chunks", []) or [])
        if domain and domain.lower().strip() != "all":
            dom_low = domain.lower().strip()
            loaded_chunks = [
                c
                for c in loaded_chunks
                if dom_low
                in str((c.metadata or {}).get("faculty") or (c.metadata or {}).get("domain") or "").lower()
            ]
        if max_chunks is not None:
            loaded_chunks = loaded_chunks[: int(max_chunks)]
        return loaded_chunks

    def _discover_lightrag_artifacts(
        self,
        base_dir: Path,
        domain: Optional[str],
        effective_max_chunks: Optional[int],
    ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Locate LightRAG graph.json, semantic_units_cache.jsonl, and entity_chroma if merging is enabled."""
        graph_cfg = self.config.get("graph", {})
        lightrag_cfg = self.config.get("lightrag", {})
        merge_lightrag = (
            bool(graph_cfg.get("merge_lightrag_kg", True))
            and domain is None
            and effective_max_chunks is None
        )
        if not merge_lightrag:
            return None, None, None

        cand_lr = resolve_project_path(
            lightrag_cfg.get("graph_path", "database/lightrag_db/graph.json"), base_dir
        )
        if not cand_lr.exists():
            return None, None, None

        cand_units = resolve_project_path(
            lightrag_cfg.get("semantic_units_cache", "database/lightrag_db/semantic_units_cache.jsonl"),
            base_dir,
        )
        cand_chroma = resolve_project_path(
            lightrag_cfg.get("entity_vector_dir", "database/lightrag_db/entity_chroma"), base_dir
        )
        return (
            cand_lr,
            cand_units if cand_units.exists() else None,
            cand_chroma if (cand_chroma / "chroma.sqlite3").exists() else None,
        )

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

        base_dir = self._base_dir()
        paths_cfg = self.config.get("paths", {})
        graph_cfg = self.config.get("graph", {})

        effective_max_chunks = max_chunks if max_chunks is not None else graph_cfg.get("max_chunks")
        chunks = self._load_chunks_for_graph(base_dir, domain, effective_max_chunks)

        out_path = resolve_project_path(paths_cfg.get("graph_index_file", "database/graph_index.pkl"), base_dir)
        cache_path = resolve_project_path(
            paths_cfg.get("graph_llm_cache_file", "database/graph_llm_cache.json"), base_dir
        )

        lr_graph_path, lr_units_path, lr_chroma_dir = self._discover_lightrag_artifacts(
            base_dir, domain, effective_max_chunks
        )

        if clear_existing and out_path.exists():
            out_path.unlink()

        enable_llm = bool(graph_cfg.get("enable_llm_extraction", False))
        llm_provider = str(graph_cfg.get("llm_provider", "openrouter"))
        llm_client = UETLLMClient(provider=llm_provider) if enable_llm else None

        active_store = self.vector_store
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

    def _validate_stages(self, from_stage: str, to_stage: str) -> None:
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

        if self._STAGE_ORDER[from_stage] >= self._STAGE_ORDER[to_stage]:
            raise ValueError(
                f"from_stage '{from_stage}' phải đứng trước to_stage '{to_stage}'. "
                f"Thứ tự hợp lệ: raw -> processed -> chunked -> vectordb / indexed_data / graph_data / lightrag"
            )

    def _clear_target_stages(self, from_stage: str, to_stage: str) -> None:
        base_dir = self._base_dir()
        logger.info("🗑️ [--clear] Bắt đầu xóa dữ liệu cũ trước khi chạy lại...")
        stages_to_reset = set()

        if from_stage == "raw":
            stages_to_reset.add("processed")

        # Chỉ xóa chunked_data khi thực sự đi qua chặng processed -> chunked (không xóa khi to_stage == 'lightrag').
        if from_stage in ["raw", "processed"] and to_stage in ["chunked", "vectordb"]:
            chunked_dir = base_dir / "data" / "chunked_data"
            if chunked_dir.exists():
                shutil.rmtree(chunked_dir)
                chunked_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✅ Đã dọn sạch các file trong folder data/chunked_data")
            stages_to_reset.add("chunked")

        if to_stage == "vectordb":
            try:
                self.vector_store.clear()
                logger.info("✅ Đã xóa toàn bộ vectors trong ChromaDB")
            except Exception as err:
                logger.warning(f"⚠️ Không thể xóa vectordb: {err}")
            stages_to_reset.add("vectordb")

        if to_stage == "indexed_data":
            bm25_path = base_dir / "database" / "bm25_index.pkl"
            if bm25_path.exists():
                bm25_path.unlink()
                logger.info("✅ Đã xóa file BM25 index cũ (database/bm25_index.pkl)")
            stages_to_reset.add("indexed_data")

        if to_stage == "graph_data":
            graph_rel = self.config.get("paths", {}).get("graph_index_file", "database/graph_index.pkl")
            graph_path = resolve_project_path(graph_rel, base_dir)
            if graph_path.exists():
                graph_path.unlink()
                logger.info(f"✅ Đã xóa file GraphRAG index cũ ({graph_rel})")
            stages_to_reset.add("graph_data")

        if to_stage == "lightrag":
            lr_cfg = self.config.get("lightrag", {})
            lr_graph_rel = lr_cfg.get("graph_path", "database/lightrag_db/graph.json")
            lr_cache_rel = lr_cfg.get("semantic_units_cache", "database/lightrag_db/semantic_units_cache.jsonl")
            lr_graph_path = resolve_project_path(lr_graph_rel, base_dir)
            lr_bak_path = lr_graph_path.with_name(lr_graph_path.name + ".bak")
            cache_path = resolve_project_path(lr_cache_rel, base_dir)
            if lr_graph_path.exists():
                lr_graph_path.unlink()
                logger.info(f"✅ Đã xóa file đồ thị cũ ({lr_graph_rel})")
            if lr_bak_path.exists():
                lr_bak_path.unlink()
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"✅ Đã xóa semantic units cache cũ ({lr_cache_rel})")
            stages_to_reset.add("lightrag")

        if stages_to_reset:
            self.manifest.reset_stages(list(stages_to_reset))
            logger.info(f"✅ Đã reset manifest tracking cho các chặng: {list(stages_to_reset)}")

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
        Tự động điều phối các bước trung gian hợp lệ.
        """
        if clear_existing is not None:
            clear = clear or clear_existing

        from_stage = from_stage.lower().strip()
        to_stage = to_stage.lower().strip()
        self._validate_stages(from_stage, to_stage)

        if clear:
            self._clear_target_stages(from_stage, to_stage)

        results: Dict[str, Any] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "transitions": {},
        }

        with Timer() as timer:
            if from_stage == "raw" and to_stage in ["processed", "chunked", "vectordb", "lightrag"]:
                logger.info("▶ BẮT ĐẦU CHẶNG: raw -> processed")
                res_1 = self._run_raw_to_processed()
                results["transitions"]["raw_to_processed"] = res_1
                logger.info(f"✔ Hoàn thành raw -> processed: {res_1}")

            if from_stage in ["raw", "processed"] and to_stage in ["chunked", "vectordb"]:
                logger.info("▶ BẮT ĐẦU CHẶNG: processed -> chunked")
                res_2 = self._run_processed_to_chunked()
                results["transitions"]["processed_to_chunked"] = res_2
                logger.info(f"✔ Hoàn thành processed -> chunked: {res_2}")

            if from_stage in ["raw", "processed", "chunked"] and to_stage == "vectordb":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> vectordb")
                res_3 = self._run_chunked_to_vectordb(batch_size=batch_size, clear_existing=clear)
                results["transitions"]["chunked_to_vectordb"] = res_3
                logger.info(f"✔ Hoàn thành chunked -> vectordb: {res_3}")

            if from_stage == "chunked" and to_stage == "indexed_data":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> indexed_data (BM25 Sparse Index)")
                res_4 = self._run_chunked_to_indexed_data(clear_existing=clear)
                results["transitions"]["chunked_to_indexed_data"] = res_4
                logger.info(f"✔ Hoàn thành chunked -> indexed_data: {res_4}")

            if from_stage == "chunked" and to_stage == "graph_data":
                logger.info("▶ BẮT ĐẦU CHẶNG: chunked -> graph_data (GraphRAG Vector Index)")
                res_5 = self._run_chunked_to_graph_data(
                    clear_existing=clear,
                    domain=domain,
                    max_chunks=max_chunks,
                )
                results["transitions"]["chunked_to_graph_data"] = res_5
                logger.info(f"✔ Hoàn thành chunked -> graph_data: {res_5}")

            if from_stage in ["raw", "processed", "chunked"] and to_stage == "lightrag":
                logger.info("▶ BẮT ĐẦU CHẶNG: processed -> lightrag (Xây dựng Knowledge Graph qua Semantic Units)")
                res_lr = self._run_processed_to_lightrag(clear_existing=clear)
                results["transitions"]["processed_to_lightrag"] = res_lr
                logger.info(f"✔ Hoàn thành processed -> lightrag: {res_lr}")

        results["latency_seconds"] = round(timer.elapsed, 2)
        logger.info(f"Hoàn thành toàn bộ tiến trình từ '{from_stage}' đến '{to_stage}' trong {results['latency_seconds']}s")
        return results

    def _load_processed_docs(self) -> List[ProcessedData]:
        """Delegate processed document loading using the active base directory."""
        return load_processed_docs(self._base_dir())

    def _run_processed_to_lightrag(self, clear_existing: bool = False) -> Dict[str, Any]:
        """Khởi chạy quy trình trích xuất và xây dựng Knowledge Graph LightRAG."""
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
        base_dir = self._base_dir()
        raw_dir = base_dir / "data" / "raw_data"
        processed_dir = base_dir / "data" / "processed_data"
        chunked_dir = base_dir / "data" / "chunked_data"
        bm25_file = base_dir / "database" / "bm25_index.pkl"

        raw_files = len(list(raw_dir.rglob("*.*"))) if raw_dir.exists() else 0
        proc_docs = sum(count_nonempty_lines(f) for f in processed_dir.rglob("*.jsonl")) if processed_dir.exists() else 0
        chunk_count = sum(count_nonempty_lines(f) for f in chunked_dir.glob("*.jsonl")) if chunked_dir.exists() else 0

        db_count = self.vector_store.count()
        bm25_exists = bm25_file.exists()
        bm25_size_mb = round(bm25_file.stat().st_size / (1024 * 1024), 2) if bm25_exists else 0.0

        graph_rel = self.config.get("paths", {}).get("graph_index_file", "database/graph_index.pkl")
        graph_file = resolve_project_path(graph_rel, base_dir)
        graph_exists = graph_file.exists()
        graph_size_mb = round(graph_file.stat().st_size / (1024 * 1024), 2) if graph_exists else 0.0

        lr_rel = self.config.get("lightrag", {}).get("graph_path", "database/lightrag_db/graph.json")
        lr_file = resolve_project_path(lr_rel, base_dir)
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
                "raw": {"path": str(raw_dir.relative_to(base_dir)), "file_count": raw_files},
                "processed": {"path": str(processed_dir.relative_to(base_dir)), "document_count": proc_docs},
                "chunked": {"path": str(chunked_dir.relative_to(base_dir)), "chunk_count": chunk_count},
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

"""
src/graph/lightrag_indexer.py - Cỗ máy Offline nạp đồ thị tri thức LightRAG.

⚠️  LƯU Ý QUAN TRỌNG:
    Module này CHỈ định nghĩa các class và hàm. KHÔNG tự động chạy khi import.
    Việc xây dựng toàn bộ Knowledge Graph cho dữ liệu thực phải được kích hoạt
    thủ công qua lệnh CLI: python main.py --ingest --to lightrag [--clear]
    và phải có sự phê duyệt của quản trị viên dự án.

Kiến trúc Hierarchical 2-Pass Extraction:
    - CHẶNG 1 (Document-level): Trích xuất Root Entities từ phần mở đầu/mục lục văn bản.
    - CHẶNG 2 (Unit-level): Trích xuất thực thể chi tiết từng Semantic Unit,
      nối cạnh về Root Entities đã có từ Chặng 1.

Chống trùng lặp:
    - AcademicThesaurus: Chuẩn hóa alias/viết tắt -> Canonical Name (O(1)).
    - EntityRegistry: In-memory registry ngăn tạo node trùng lặp.
    - Graph Persistence: Lưu trữ bằng NetworkX + JSON (không cần Neo4j).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent

from src.utils.helpers import get_ascii_safe_chroma_path, get_logger, load_yaml_config
from src.graph.thesaurus_utils import (
    AcademicThesaurus,
    FORBIDDEN_ENTITIES,
    get_thesaurus,
    is_forbidden_entity,
)
from src.graph.adaptive_extractor import SemanticUnit, AdaptiveSemanticExtractor
from src.graph.lightrag_schema import (
    ALLOWED_RELATION_TYPES,
    DEFAULT_ENTITY_VECTOR_DIR,
    DEFAULT_GRAPH_PATH,
    VALID_ENTITY_CATEGORIES,
    EntityRegistry,
    LLMResponseParser,
    PromptBuilder,
)
from src.graph.lightrag_storage import (
    load_lightrag_graph,
    populate_from_graph_data,
    save_lightrag_graph,
    vectorize_lightrag_entities,
)

logger = get_logger("graph.lightrag_indexer")


# ---------------------------------------------------------------------------
# LIGHTRAG INDEXER: Cỗ máy chính
# ---------------------------------------------------------------------------

class LightRAGIndexer:
    """
    Cỗ máy Offline xây dựng Knowledge Graph cho UET Chatbot LightRAG.

    Pipeline Hierarchical 2-Pass:
      CHẶNG 1: Scan doc header -> Root Entities
      CHẶNG 2: Từng Semantic Unit -> chi tiết Entities + Edges -> gộp vào Registry

    ⚠️  KHÔNG gọi run() hoặc run_full_pipeline() mà không có sự phê duyệt.
    Chỉ nên dùng các phương thức riêng lẻ để kiểm thử (PoC, test nhỏ).

    Cách khởi chạy ĐÚNG (khi đã được phê duyệt):
        python main.py --ingest --to lightrag [--clear]
    """

    def __init__(
        self,
        graph_path: Optional[Path] = None,
        entity_vector_dir: Optional[Path] = None,
        llm_provider: Optional[str] = None,
        delay_between_calls: Optional[float] = None,
        max_pass2_retries: Optional[int] = None,
        max_workers: Optional[int] = None,
        header_max_chars: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config if config is not None else load_yaml_config()
        lr_cfg = (cfg or {}).get("lightrag", {})
        idx_cfg = lr_cfg.get("indexer", {})

        cfg_graph_rel = lr_cfg.get("graph_path")
        resolved_graph_path = (
            Path(graph_path)
            if graph_path is not None
            else (
                Path(cfg_graph_rel) if Path(cfg_graph_rel).is_absolute() else (BASE_DIR / cfg_graph_rel)
                if cfg_graph_rel
                else DEFAULT_GRAPH_PATH
            )
        )
        cfg_vec_rel = lr_cfg.get("entity_vector_dir")
        resolved_vec_dir = (
            Path(entity_vector_dir)
            if entity_vector_dir is not None
            else (
                Path(cfg_vec_rel) if Path(cfg_vec_rel).is_absolute() else (BASE_DIR / cfg_vec_rel)
                if cfg_vec_rel
                else DEFAULT_ENTITY_VECTOR_DIR
            )
        )

        self.graph_path = resolved_graph_path
        self.entity_vector_dir = resolved_vec_dir
        self.delay = float(
            delay_between_calls
            if delay_between_calls is not None
            else idx_cfg.get("delay_between_calls", 0.5)
        )
        self.max_retries = int(
            max_pass2_retries
            if max_pass2_retries is not None
            else idx_cfg.get("max_pass2_retries", 2)
        )
        self.max_workers = max(
            1,
            int(max_workers if max_workers is not None else idx_cfg.get("max_workers", 5)),
        )
        self.header_max_chars = max(
            200,
            int(header_max_chars if header_max_chars is not None else idx_cfg.get("header_max_chars", 2000)),
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Khởi tạo hạ tầng
        self.thesaurus = get_thesaurus()
        self.extractor = AdaptiveSemanticExtractor(config=cfg)
        self.registry = EntityRegistry()
        self.prompt_builder = PromptBuilder(self.thesaurus)
        self.parser = LLMResponseParser()

        # Lưu trữ đồ thị (nodes + edges riêng để dễ serialize)
        self._nodes: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []
        self._doc_summaries: Dict[str, Dict[str, Any]] = {}

        # Thống kê phiên nạp
        self._stats = {
            "docs_processed": 0,
            "docs_failed": 0,
            "units_processed": 0,
            "units_failed": 0,
            "llm_calls": 0,
            "llm_errors": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "rate_limit_waits": 0,
        }
        self._consecutive_llm_failures = 0

        # Đường dẫn cache Semantic Units
        cache_rel = lr_cfg.get("semantic_units_cache", "database/lightrag_db/semantic_units_cache.jsonl")
        self.semantic_units_cache_path = (
            Path(cache_rel) if Path(cache_rel).is_absolute() else (BASE_DIR / cache_rel)
        )
        self.semantic_units_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_units: List[SemanticUnit] = []
        self._cached_unit_ids: Set[str] = set()

        # Khởi tạo LLM Client
        self._llm = self._init_llm(llm_provider, config=cfg)

        # Tạo thư mục lưu trữ
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.entity_vector_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _init_llm(provider: Optional[str], config: Optional[Dict[str, Any]] = None):
        """Khởi tạo LLM Client tương thích với UETLLMClient có sẵn."""
        try:
            from src.llm.llm_client import UETLLMClient
            cfg = config if config is not None else load_yaml_config()
            effective_provider = (
                provider
                or cfg.get("lightrag", {}).get("indexer", {}).get("llm_provider")
                or cfg.get("llm", {}).get("provider", "openrouter")
            )
            llm = UETLLMClient(provider=effective_provider, enable_cache=False)
            logger.info(f"Khởi tạo LLM Client: provider={effective_provider}")
            return llm
        except Exception as e:
            logger.error(f"Không thể khởi tạo LLM Client: {e}")
            return None

    def _provider_has_keys(self, provider_name: str) -> bool:
        """Kiểm tra provider có cấu hình API key hợp lệ trước khi gọi để tránh spam cảnh báo."""
        key_attr_map = {
            "openrouter": "openrouter_keys",
            "groq": "groq_keys",
            "gemini": "gemini_keys",
            "sambanova": "sambanova_keys",
        }
        attr = key_attr_map.get(provider_name)
        if not attr or not hasattr(self._llm, attr):
            return True
        keys = getattr(self._llm, attr, None)
        return bool(keys)

    def _call_llm_raw(self, prompt_text: str) -> Optional[str]:
        """
        Gọi LLM API và trả về text thô (thread-safe).
        Tự động bỏ qua provider chưa cấu hình key, fallback giữa các provider khả dụng
        (OpenRouter -> Groq -> Gemini -> SambaNova), và retry có backoff theo max_retries.
        """
        if self._stop_event.is_set():
            return None

        if not self._llm:
            logger.error("LLM Client chưa được khởi tạo.")
            with self._lock:
                self._consecutive_llm_failures += 1
            return None

        # Danh sách callers ưu tiên (lọc sẵn provider có API key)
        providers_order = [self._llm.provider]
        for p in ["openrouter", "groq", "gemini", "sambanova"]:
            if p not in providers_order:
                providers_order.append(p)

        caller_map = {
            "openrouter": self._llm._call_openrouter,
            "groq": self._llm._call_groq,
            "gemini": self._llm._call_gemini,
            "sambanova": self._llm._call_sambanova,
        }

        active_providers = [
            p for p in providers_order
            if p in caller_map and self._provider_has_keys(p)
        ]
        if not active_providers:
            logger.error("Không có LLM provider nào được cấu hình API key hợp lệ.")
            with self._lock:
                self._stats["llm_errors"] += 1
                self._consecutive_llm_failures += 1
            return None

        max_attempts = max(1, int(self.max_retries) + 1)
        for attempt in range(max_attempts):
            if self._stop_event.is_set():
                return None

            for p_name in active_providers:
                if self._stop_event.is_set():
                    return None
                caller = caller_map[p_name]
                try:
                    text, model_name, _ = caller(prompt_text)
                    if text and text.strip():
                        with self._lock:
                            self._stats["llm_calls"] += 1
                            self._consecutive_llm_failures = 0
                        if self.delay > 0 and not self._stop_event.is_set():
                            time.sleep(self.delay)  # Rate limit delay trên từng luồng
                        return text
                except Exception as e:
                    logger.warning(f"[{p_name}] Gặp sự cố: {e}. Đổi ngay sang provider tiếp theo...")
                    continue

            if attempt + 1 < max_attempts and not self._stop_event.is_set():
                backoff = min(2.0 * (attempt + 1), 6.0)
                with self._lock:
                    self._stats["rate_limit_waits"] += 1
                logger.warning(
                    f"Tất cả provider khả dụng đều lỗi (lần {attempt + 1}/{max_attempts}). "
                    f"Nghỉ {backoff:.1f}s rồi thử lại..."
                )
                time.sleep(backoff)

        with self._lock:
            self._stats["llm_errors"] += 1
            self._consecutive_llm_failures += 1
        return None

    # -------------------------------------------------------------------------
    # CHẶNG 1: Trích xuất Root Entities từ phần mở đầu văn bản
    # -------------------------------------------------------------------------

    def _get_consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_llm_failures

    def _extract_pass1_data(
        self,
        doc_title: str,
        doc_content: str,
        header_max_chars: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """Trích xuất Chặng 1 thuần (thread-safe, không thay đổi Registry dùng chung)."""
        eff_header_chars = int(
            header_max_chars if header_max_chars is not None else self.header_max_chars
        )
        header = doc_content[:eff_header_chars]
        prompt = self.prompt_builder.build_pass1_prompt(header, doc_title)
        raw = self._call_llm_raw(prompt)
        llm_ok = bool(raw)

        fallback_ctx = {
            "document_title": doc_title,
            "document_code": None,
            "namespace_prefix": f"[{doc_title[:40]}]",
            "root_entities": [],
        }
        if not raw:
            logger.warning(f"Chặng 1 không có phản hồi cho: {doc_title}")
            return fallback_ctx, None, [], False

        parsed = self.parser.parse_pass1(raw)
        if not parsed:
            logger.warning(f"Chặng 1 parse thất bại cho: {doc_title}")
            return fallback_ctx, None, [], llm_ok

        doc_code = parsed.get("document_code") or doc_title[:40]
        namespace_prefix = f"[{doc_code}]"

        summary_key = str(doc_id).strip() if (doc_id and str(doc_id).strip()) else doc_title
        pass1_unit_id = f"pass1_{hashlib.md5(summary_key.encode('utf-8')).hexdigest()[:8]}"

        root_entity_names: List[str] = []
        root_nodes: List[Dict[str, Any]] = []
        seen_roots: Set[str] = set()
        for ent in parsed.get("root_entities", []):
            raw_name = str(ent.get("canonical_name", "")).strip()
            if not raw_name or is_forbidden_entity(raw_name, allow_school_names=True):
                continue
            name = self.thesaurus.normalize(raw_name)
            if not name or len(name) < 2 or is_forbidden_entity(name, allow_school_names=True):
                continue
            low_name = name.lower()
            if low_name in seen_roots:
                continue
            seen_roots.add(low_name)
            root_entity_names.append(name)
            root_nodes.append({
                "canonical": name,
                "entity_type": ent.get("entity_type", "VAN_BAN"),
                "description": parsed.get("scope", ""),
                "source_unit_id": pass1_unit_id,
                "aliases": self.thesaurus.get_aliases(name),
            })

        doc_summary = {
            "doc_id": summary_key,
            "title": parsed.get("document_title") or doc_title,
            "document_code": doc_code,
            "document_type": parsed.get("document_type", "LEGAL"),
            "issuing_body": parsed.get("issuing_body", ""),
            "summary": parsed.get("scope", "") or parsed.get("document_title", ""),
            "key_entities": root_entity_names,
        }

        logger.info(
            f"Chặng 1 '{doc_title}': namespace={namespace_prefix}, "
            f"{len(root_nodes)} root entities"
        )
        return {**parsed, "namespace_prefix": namespace_prefix}, doc_summary, root_nodes, True

    def _register_pass1_result(
        self,
        summary_key: str,
        doc_summary: Optional[Dict[str, Any]],
        root_nodes: List[Dict[str, Any]],
    ) -> None:
        """Gộp kết quả Chặng 1 vào trạng thái đồ thị dùng chung (phải gọi dưới self._lock hoặc main thread)."""
        if doc_summary is not None:
            self._doc_summaries[summary_key] = doc_summary
        for node in root_nodes:
            before_count = self.registry.node_count()
            self.registry.register_node(
                canonical=node["canonical"],
                entity_type=node["entity_type"],
                description=node["description"],
                source_unit_id=node["source_unit_id"],
                aliases=node["aliases"],
            )
            if self.registry.node_count() > before_count:
                self._stats["nodes_created"] += 1

    def run_pass1(
        self,
        doc_title: str,
        doc_content: str,
        header_max_chars: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Chặng 1: Gửi phần mở đầu văn bản cho LLM để lấy Root Entities.
        Chỉ chạy 1 lần cho mỗi tài liệu.
        """
        summary_key = str(doc_id).strip() if (doc_id and str(doc_id).strip()) else doc_title
        root_ctx, doc_summary, root_nodes, _ = self._extract_pass1_data(
            doc_title, doc_content, header_max_chars=header_max_chars, doc_id=doc_id
        )
        with self._lock:
            self._register_pass1_result(summary_key, doc_summary, root_nodes)
        return root_ctx

    # -------------------------------------------------------------------------
    # CHẶNG 2: Trích xuất chi tiết từng Semantic Unit
    # -------------------------------------------------------------------------

    def _extract_pass2_unit_data(
        self,
        unit: SemanticUnit,
        root_context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """Trích xuất Chặng 2 thuần cho 1 Semantic Unit (thread-safe, không sửa Registry)."""
        namespace_prefix = root_context.get("namespace_prefix", f"[{unit.document_title[:40]}]")
        root_entities = root_context.get("root_entities", [])

        prompt = self.prompt_builder.build_pass2_prompt(unit, root_entities, namespace_prefix)
        raw = self._call_llm_raw(prompt)
        if not raw:
            logger.warning(f"Chặng 2 không có phản hồi: {unit.unit_id}")
            return [], [], False

        entities, relationships = self.parser.parse_pass2(
            raw=raw,
            thesaurus=self.thesaurus,
            namespace_prefix=namespace_prefix,
            unit_id=unit.unit_id,
        )
        return entities, relationships, True

    def _register_pass2_result(
        self,
        unit_id: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        llm_ok: bool,
    ) -> Tuple[int, int]:
        """Gộp thực thể/cạnh của 1 unit vào Registry (phải gọi dưới self._lock hoặc main thread)."""
        if not llm_ok:
            self._stats["units_failed"] += 1
            return 0, 0

        nodes_added = 0
        edges_added = 0
        for ent in entities:
            before_count = self.registry.node_count()
            self.registry.register_node(
                canonical=ent["canonical_name"],
                entity_type=ent["entity_type"],
                description=ent["description"],
                source_unit_id=unit_id,
                aliases=self.thesaurus.get_aliases(ent["canonical_name"]),
            )
            if self.registry.node_count() > before_count:
                nodes_added += 1

        for rel in relationships:
            edge = self.registry.register_edge(
                source=rel["source"],
                target=rel["target"],
                relation_type=rel["relation_type"],
                evidence_text=rel["evidence"],
                source_unit_id=unit_id,
            )
            if edge is not None:
                self._edges.append(edge)
                edges_added += 1

        self._stats["units_processed"] += 1
        self._stats["nodes_created"] += nodes_added
        self._stats["edges_created"] += edges_added
        return nodes_added, edges_added

    def run_pass2_unit(
        self,
        unit: SemanticUnit,
        root_context: Dict[str, Any],
    ) -> Tuple[int, int]:
        """
        Chặng 2: Trích xuất thực thể và quan hệ từ 1 Semantic Unit.
        Gộp kết quả vào EntityRegistry (chống trùng lặp tự động).
        """
        entities, relationships, llm_ok = self._extract_pass2_unit_data(unit, root_context)
        with self._lock:
            return self._register_pass2_result(unit.unit_id, entities, relationships, llm_ok)

    # -------------------------------------------------------------------------
    # PIPELINE TỔNG: Chạy cho 1 tài liệu (PoC / Test lẻ & Worker song song)
    # -------------------------------------------------------------------------

    def _add_cached_units(self, units: List[SemanticUnit]) -> None:
        """Thêm Semantic Units vào cache với cơ chế chống trùng lặp theo unit_id."""
        for u in units:
            uid = getattr(u, "unit_id", None)
            if uid and uid in self._cached_unit_ids:
                continue
            self._cached_units.append(u)
            if uid:
                self._cached_unit_ids.add(uid)

    def _extract_document_payload(
        self,
        doc,  # ProcessedData
        max_consecutive_failures: Optional[int] = None,
        unit_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Chạy trích xuất 2 chặng cho 1 tài liệu ở chế độ cô lập (thread-safe).
        Trả về payload thuần để gộp tuần tự vào Registry trên main thread / dưới self._lock.
        """
        logger.info(f"Bắt đầu xử lý: '{doc.title}' ({len(doc.content)} ký tự)")

        doc_id = str(getattr(doc, "id", "") or "").strip()
        summary_key = doc_id if doc_id else doc.title

        if self._stop_event.is_set():
            return {
                "doc_id": summary_key,
                "doc_title": doc.title,
                "doc_summary": None,
                "root_nodes": [],
                "pass1_ok": False,
                "units": [],
                "unit_results": [],
                "units_succeeded": 0,
                "units_failed": 0,
            }

        root_context, doc_summary, root_nodes, pass1_ok = self._extract_pass1_data(
            doc.title,
            doc.content,
            header_max_chars=self.header_max_chars,
            doc_id=doc_id or None,
        )
        if (
            max_consecutive_failures is not None
            and self._get_consecutive_failures() >= max_consecutive_failures
        ):
            self._stop_event.set()

        units = self.extractor.extract(doc)
        logger.info(f"  -> {len(units)} Semantic Units")

        unit_results: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], bool]] = []
        units_succeeded = 0
        units_failed = 0

        eff_unit_workers = max(1, int(unit_workers))
        if eff_unit_workers > 1 and len(units) > 1 and not self._stop_event.is_set():
            executor = ThreadPoolExecutor(max_workers=min(eff_unit_workers, len(units)))
            futures = {}
            try:
                for unit in units:
                    if self._stop_event.is_set():
                        units_failed += 1
                        continue
                    futures[executor.submit(self._extract_pass2_unit_data, unit, root_context)] = unit

                for fut in as_completed(futures):
                    unit = futures[fut]
                    entities, relationships, llm_ok = fut.result()
                    unit_results.append((unit.unit_id, entities, relationships, llm_ok))
                    if llm_ok:
                        units_succeeded += 1
                    else:
                        units_failed += 1
                        if (
                            max_consecutive_failures is not None
                            and self._get_consecutive_failures() >= max_consecutive_failures
                        ):
                            self._stop_event.set()
            except KeyboardInterrupt:
                self._stop_event.set()
                for fut in futures:
                    fut.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=not self._stop_event.is_set(), cancel_futures=self._stop_event.is_set())
        else:
            for i, unit in enumerate(units, 1):
                if self._stop_event.is_set() or (
                    max_consecutive_failures is not None
                    and self._get_consecutive_failures() >= max_consecutive_failures
                ):
                    self._stop_event.set()
                    remaining = len(units) - i + 1
                    units_failed += remaining
                    logger.error(
                        f"🛑 Circuit breaker kích hoạt giữa tài liệu '{doc.title}' "
                        f"({self._get_consecutive_failures()} lỗi LLM liên tiếp). Bỏ qua {remaining} units còn lại."
                    )
                    break
                logger.debug(f"  [{i}/{len(units)}] {unit.breadcrumb[:60]}")
                entities, relationships, llm_ok = self._extract_pass2_unit_data(unit, root_context)
                unit_results.append((unit.unit_id, entities, relationships, llm_ok))
                if llm_ok:
                    units_succeeded += 1
                else:
                    units_failed += 1
                    if (
                        max_consecutive_failures is not None
                        and self._get_consecutive_failures() >= max_consecutive_failures
                    ):
                        self._stop_event.set()

        return {
            "doc_id": summary_key,
            "doc_title": doc.title,
            "doc_summary": doc_summary,
            "root_nodes": root_nodes,
            "pass1_ok": pass1_ok,
            "units": units,
            "unit_results": unit_results,
            "units_succeeded": units_succeeded,
            "units_failed": units_failed,
        }

    def _merge_document_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gộp payload trích xuất của 1 tài liệu vào đồ thị chung (thread-safe qua self._lock)."""
        doc_title = payload["doc_title"]
        summary_key = payload.get("doc_id") or doc_title
        pass1_ok = payload["pass1_ok"]
        units = payload["units"]
        units_succeeded = payload["units_succeeded"]
        units_failed = payload["units_failed"]

        with self._lock:
            self._register_pass1_result(
                summary_key, payload["doc_summary"], payload["root_nodes"]
            )

            total_nodes = 0
            total_edges = 0
            for unit_id, entities, relationships, llm_ok in payload["unit_results"]:
                n, e = self._register_pass2_result(unit_id, entities, relationships, llm_ok)
                total_nodes += n
                total_edges += e

            doc_fully_succeeded = pass1_ok and units_failed == 0 and len(units) > 0
            if pass1_ok and units_failed == 0 and len(units) == 0:
                doc_fully_succeeded = True

            if doc_fully_succeeded:
                self._add_cached_units(units)
                self._stats["docs_processed"] += 1
            else:
                self._stats["docs_failed"] += 1
                logger.warning(
                    f"⚠️  Tài liệu '{doc_title}' chưa trích xuất trọn vẹn "
                    f"(pass1_ok={pass1_ok}, units_ok={units_succeeded}/{len(units)}, units_failed={units_failed}). "
                    f"KHÔNG lưu vào cache hoàn thành để tự động chạy bù ở lần resume tiếp theo."
                )

            self._nodes = self.registry.all_nodes()
            node_total = self.registry.node_count()
            edge_total = len(self._edges)

        logger.info(f"Hoàn thành '{doc_title}': +{total_nodes} nodes, +{total_edges} edges")
        return {
            "document": doc_title,
            "units_processed": units_succeeded,
            "units_failed": units_failed,
            "pass1_ok": pass1_ok,
            "fully_succeeded": doc_fully_succeeded,
            "nodes_added": total_nodes,
            "edges_added": total_edges,
            "total_nodes_in_graph": node_total,
            "total_edges_in_graph": edge_total,
        }

    def process_single_document(
        self,
        doc,  # ProcessedData
        max_consecutive_failures: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý 1 tài liệu hoàn chỉnh qua 2 chặng:
          1. Chặng 1: Lấy Root Entities.
          2. Chặng 2: Lấy chi tiết từng Semantic Unit.

        Chỉ đánh dấu tài liệu vào semantic_units_cache khi KHÔNG có unit nào lỗi gọi LLM,
        đảm bảo các tài liệu gặp sự cố API sẽ tự động được chạy bù ở lần resume sau.
        """
        payload = self._extract_document_payload(
            doc, max_consecutive_failures=max_consecutive_failures
        )
        return self._merge_document_payload(payload)

    # -------------------------------------------------------------------------
    # PIPELINE TỔNG: Chạy cho toàn bộ dữ liệu (PHẢI CÓ PHÊ DUYỆT)
    # -------------------------------------------------------------------------

    def run_full_pipeline(
        self,
        docs,            # List[ProcessedData]
        checkpoint_every: int = 10,
        clear_existing: bool = False,
        max_consecutive_failures: int = 10,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        ⚠️  PHƯƠNG THỨC NÀY PHẢI CÓ SỰ PHÊ DUYỆT CỦA QUẢN TRỊ VIÊN TRƯỚC KHI CHẠY.
        ⚠️  Sẽ tiêu tốn nhiều lượt gọi LLM API.

        Chạy pipeline hoàn chỉnh cho toàn bộ danh sách tài liệu:
          - Nạp đồ thị checkpoint hiện tại (nếu có, có tự động fallback từ .bak).
          - Hỗ trợ chạy đa luồng an toàn (workers gọi LLM song song, main thread gộp vào Registry).
          - Lưu checkpoint atomic sau mỗi `checkpoint_every` tài liệu mới xử lý.
          - Circuit breaker tự dừng an toàn & lưu checkpoint nếu LLM lỗi liên tiếp.
          - Bắt KeyboardInterrupt (Ctrl+C) để lưu checkpoint trước khi thoát, không hỏng file.
          - Lưu đồ thị cuối cùng và vector hóa thực thể.

        Kích hoạt qua: python main.py --ingest --to lightrag [--clear]
        """
        self._stop_event.clear()
        with self._lock:
            self._consecutive_llm_failures = 0

        if clear_existing:
            logger.info("⚠️  --clear: Xóa đồ thị cũ và bắt đầu lại từ đầu.")
            self._nodes = []
            self._edges = []
            self._doc_summaries = {}
            self.registry = EntityRegistry()
            self._cached_units = []
            self._cached_unit_ids = set()
            # Xóa file cũ
            for p in (
                self.graph_path,
                self.graph_path.with_name(self.graph_path.name + ".bak"),
                self.semantic_units_cache_path,
            ):
                if p.exists():
                    p.unlink()
            if (self.entity_vector_dir / "chroma.sqlite3").exists():
                try:
                    import chromadb
                    from chromadb.config import Settings

                    client = chromadb.PersistentClient(
                        path=get_ascii_safe_chroma_path(self.entity_vector_dir),
                        settings=Settings(anonymized_telemetry=False),
                    )
                    client.delete_collection(name="lightrag_entity_collection")
                    logger.info("✅ Đã làm sạch collection 'lightrag_entity_collection' khi chạy --clear.")
                except Exception:
                    pass
        else:
            # Nạp checkpoint đồ thị hiện có (nếu có)
            self._load_graph()

        processed_doc_ids = set()
        if not clear_existing and self._cached_units:
            for u in self._cached_units:
                d_id = getattr(u, "document_id", None) or getattr(u, "doc_id", None)
                if d_id:
                    processed_doc_ids.add(d_id)
        if processed_doc_ids:
            logger.info(f"🔄 Checkpoint: Đã tìm thấy {len(processed_doc_ids)} tài liệu đã hoàn thành. Sẽ tự động bỏ qua và tiếp tục từ tài liệu chưa xử lý!")

        pending_docs = [
            doc for doc in docs
            if clear_existing or doc.id not in processed_doc_ids
        ]
        eff_workers = max(1, int(max_workers if max_workers is not None else self.max_workers))
        failure_limit = max_consecutive_failures if max_consecutive_failures > 0 else None

        total = len(docs)
        pending_total = len(pending_docs)
        logger.info(
            f"🚀 Bắt đầu nạp {pending_total}/{total} tài liệu cần xử lý vào Knowledge Graph LightRAG "
            f"(workers={eff_workers})..."
        )

        newly_processed = 0
        aborted_due_to_llm_errors = False
        interrupted = False

        try:
            if eff_workers > 1 and pending_total > 1:
                # Xếp tài liệu dài nhất chạy trước (LPT) để các luồng kết thúc đều nhau
                ordered_docs = sorted(
                    pending_docs,
                    key=lambda d: len(getattr(d, "content", "") or ""),
                    reverse=True,
                )
                pool_size = min(eff_workers, pending_total)
                executor = ThreadPoolExecutor(max_workers=pool_size)
                future_to_doc = {}
                try:
                    for doc in ordered_docs:
                        fut = executor.submit(
                            self._extract_document_payload,
                            doc,
                            failure_limit,
                            1,
                        )
                        future_to_doc[fut] = doc

                    for fut in as_completed(future_to_doc):
                        doc = future_to_doc[fut]
                        payload = fut.result()
                        newly_processed += 1
                        logger.info(f"[{newly_processed}/{pending_total}] Gộp kết quả: {doc.title}")
                        self._merge_document_payload(payload)

                        if failure_limit is not None and self._get_consecutive_failures() >= failure_limit:
                            aborted_due_to_llm_errors = True
                            self._stop_event.set()
                            for pending_fut in future_to_doc:
                                pending_fut.cancel()
                            logger.error(
                                f"🛑 Circuit breaker: Đã xảy ra {self._get_consecutive_failures()} lỗi LLM liên tiếp. "
                                f"Tạm dừng an toàn và lưu checkpoint!"
                            )
                            break

                        if checkpoint_every > 0 and newly_processed % checkpoint_every == 0:
                            self.save_graph()
                            logger.info(
                                f"  Checkpoint saved ({newly_processed}/{pending_total}): "
                                f"{self.registry.node_count()} nodes, {len(self._edges)} edges"
                            )
                except KeyboardInterrupt:
                    interrupted = True
                    self._stop_event.set()
                    for pending_fut in future_to_doc:
                        pending_fut.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    executor.shutdown(wait=not self._stop_event.is_set(), cancel_futures=self._stop_event.is_set())
            else:
                unit_workers = eff_workers if (eff_workers > 1 and pending_total == 1) else 1
                for idx, doc in enumerate(pending_docs, 1):
                    if failure_limit is not None and self._get_consecutive_failures() >= failure_limit:
                        aborted_due_to_llm_errors = True
                        self._stop_event.set()
                        logger.error(
                            f"🛑 Circuit breaker: Đã xảy ra {self._get_consecutive_failures()} lỗi LLM liên tiếp. "
                            f"Tạm dừng an toàn và lưu checkpoint để tránh tạo dữ liệu rỗng!"
                        )
                        break

                    logger.info(f"[{idx}/{pending_total}] Xử lý: {doc.title}")
                    payload = self._extract_document_payload(
                        doc,
                        max_consecutive_failures=failure_limit,
                        unit_workers=unit_workers,
                    )
                    self._merge_document_payload(payload)
                    newly_processed += 1

                    if failure_limit is not None and self._get_consecutive_failures() >= failure_limit:
                        aborted_due_to_llm_errors = True
                        self._stop_event.set()
                        logger.error(
                            f"🛑 Circuit breaker: Đã xảy ra {self._get_consecutive_failures()} lỗi LLM liên tiếp. "
                            f"Tạm dừng an toàn và lưu checkpoint!"
                        )
                        break

                    if checkpoint_every > 0 and newly_processed % checkpoint_every == 0:
                        self.save_graph()
                        logger.info(f"  Checkpoint saved: {self.registry.node_count()} nodes, {len(self._edges)} edges")
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("⚠️  Nhận tín hiệu dừng (KeyboardInterrupt). Đang lưu checkpoint an toàn trước khi thoát...")
            self.save_graph()
            raise

        # Lưu đồ thị cuối cùng (atomic)
        self.save_graph()

        # Chỉ vector hóa toàn bộ khi chạy xong bình thường (không bị dừng sớm do outage)
        if not aborted_due_to_llm_errors and not interrupted:
            self.vectorize_entities(force_reembed=clear_existing)
        else:
            logger.warning(
                "⚠️  Bỏ qua bước vector hóa do phiên nạp dừng sớm. "
                "Hãy kiểm tra lại LLM API rồi chạy lại lệnh resume để hoàn tất phần còn lại."
            )

        summary = {
            **self._stats,
            "final_nodes": self.registry.node_count(),
            "final_edges": len(self._edges),
            "graph_path": str(self.graph_path),
            "aborted_due_to_llm_errors": aborted_due_to_llm_errors,
            "workers": eff_workers,
        }
        logger.info(f"✅ Hoàn thành! {summary}")
        return summary

    # -------------------------------------------------------------------------
    # LƯU TRỮ & NẠP ĐỒ THỊ (delegated to lightrag_storage.py)
    # -------------------------------------------------------------------------

    def save_graph(self) -> None:
        """Lưu đồ thị (nodes + edges) ra file JSON theo cơ chế atomic write + sao lưu .bak."""
        save_lightrag_graph(self)

    def _populate_from_graph_data(self, data: Dict[str, Any]) -> None:
        """Khôi phục trạng thái in-memory (Registry, edges, summaries, stats) từ dict JSON."""
        populate_from_graph_data(self, data)

    def _load_graph(self) -> bool:
        """
        Nạp đồ thị từ file JSON (checkpoint).
        Tự động thử bản dự phòng .bak nếu graph.json bị hỏng.
        Nếu file tồn tại nhưng cả bản chính lẫn .bak đều hỏng -> ném RuntimeError
        thay vì âm thầm chạy lại từ đầu và ghi đè dữ liệu.
        """
        return load_lightrag_graph(self)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Trả về thống kê đồ thị hiện tại (không cần nạp file)."""
        return {
            "nodes": self.registry.node_count(),
            "edges": len(self._edges),
            "graph_path": str(self.graph_path),
            "graph_exists": self.graph_path.exists(),
            "session_stats": self._stats,
        }

    # -------------------------------------------------------------------------
    # VECTOR HÓA THỰC THỂ (phục vụ Retrieval)
    # -------------------------------------------------------------------------

    def vectorize_entities(
        self,
        force_reembed: bool = False,
        chroma_dir: Optional[Path] = None,
    ) -> int:
        """Vector hóa tên + mô tả của các Node trong đồ thị vào ChromaDB."""
        return vectorize_lightrag_entities(
            self,
            force_reembed=force_reembed,
            chroma_dir=chroma_dir,
        )


if __name__ == "__main__":
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    print("LightRAGIndexer module loaded. Các class sẵn sàng sử dụng.")
    print("Để chạy PoC với 1 văn bản: LightRAGIndexer().process_single_document(doc)")
    print("Để chạy toàn pipeline (CẦN PHÊ DUYỆT): python main.py --ingest --to lightrag")

"""
src/graph/lightrag_storage.py - Persistence (atomic save/load with .bak recovery)
and incremental ChromaDB entity vectorization for LightRAGIndexer.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.graph.lightrag_schema import EntityRegistry
from src.utils.helpers import get_logger

logger = get_logger("graph.lightrag_indexer")


def save_lightrag_graph(indexer: Any) -> None:
    """Lưu đồ thị (nodes + edges) ra file JSON theo cơ chế atomic write + sao lưu .bak."""
    indexer._nodes = indexer.registry.all_nodes()
    graph_data = {
        "meta": {
            "created_by": "LightRAGIndexer",
            "node_count": len(indexer._nodes),
            "edge_count": len(indexer._edges),
            "summary_count": len(indexer._doc_summaries),
            "stats": indexer._stats,
        },
        "nodes": indexer._nodes,
        "edges": indexer._edges,
        "entities": {node["canonical_name"]: node for node in indexer._nodes},
        "relations": indexer._edges,
        "document_summaries": indexer._doc_summaries,
    }
    indexer.graph_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = indexer.graph_path.with_name(indexer.graph_path.name + ".tmp")
    bak_path = indexer.graph_path.with_name(indexer.graph_path.name + ".bak")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Sao lưu bản cũ hợp lệ sang .bak trước khi thay thế nguyên tử
        if indexer.graph_path.exists() and indexer.graph_path.stat().st_size > 0:
            try:
                shutil.copy2(indexer.graph_path, bak_path)
            except OSError as bak_err:
                logger.warning(f"Không thể tạo file sao lưu {bak_path}: {bak_err}")

        os.replace(tmp_path, indexer.graph_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    # Lưu cache Semantic Units ra file JSONL (atomic) phục vụ tra cứu và resume
    if indexer._cached_units:
        indexer.extractor.save_units_jsonl(
            indexer._cached_units, indexer.semantic_units_cache_path
        )

    logger.info(
        f"Đã lưu đồ thị: {len(indexer._nodes)} nodes, {len(indexer._edges)} edges, "
        f"{len(indexer._doc_summaries)} summaries -> {indexer.graph_path}"
    )


def populate_from_graph_data(indexer: Any, data: Dict[str, Any]) -> None:
    """Khôi phục trạng thái in-memory (Registry, edges, summaries, stats) từ dict JSON."""
    indexer.registry = EntityRegistry()
    indexer._edges = data.get("edges", []) or data.get("relations", [])
    indexer._doc_summaries = data.get("document_summaries", {})
    if "meta" in data and "stats" in data["meta"]:
        indexer._stats = {**indexer._stats, **data["meta"]["stats"]}
    nodes = data.get("nodes", [])
    if not nodes and "entities" in data:
        if isinstance(data["entities"], dict):
            nodes = list(data["entities"].values())
        elif isinstance(data["entities"], list):
            nodes = data["entities"]
    for node in nodes:
        canonical = node.get("canonical_name", "")
        if canonical:
            indexer.registry._nodes[canonical.lower()] = node
    dedup_edges: List[Dict[str, Any]] = []
    for edge in indexer._edges:
        rel_t = edge.get("relation_type") or edge.get("predicate", "liên_quan")
        edge_key = f"{edge['source'].lower()}::{rel_t}::{edge['target'].lower()}"
        if edge_key in indexer.registry._edges_by_key:
            existing = indexer.registry._edges_by_key[edge_key]
            for uid in edge.get("source_unit_ids", ()) or ():
                if uid and uid not in existing.setdefault("source_unit_ids", []):
                    existing["source_unit_ids"].append(uid)
            for ev in edge.get("evidence_texts", ()) or ():
                if (
                    ev
                    and ev not in existing.setdefault("evidence_texts", [])
                    and len(existing["evidence_texts"]) < 5
                ):
                    existing["evidence_texts"].append(ev)
            continue
        indexer.registry._edge_key_set.add(edge_key)
        indexer.registry._edges_by_key[edge_key] = edge
        dedup_edges.append(edge)
    indexer._edges = dedup_edges
    indexer._nodes = list(indexer.registry._nodes.values())


def load_lightrag_graph(indexer: Any) -> bool:
    """
    Nạp đồ thị từ file JSON (checkpoint).
    Tự động thử bản dự phòng .bak nếu graph.json bị hỏng.
    Nếu file tồn tại nhưng cả bản chính lẫn .bak đều hỏng -> ném RuntimeError
    thay vì âm thầm chạy lại từ đầu và ghi đè dữ liệu.
    """
    bak_path = indexer.graph_path.with_name(indexer.graph_path.name + ".bak")
    if not indexer.graph_path.exists() and not bak_path.exists():
        logger.info("Không tìm thấy đồ thị cũ, bắt đầu từ đầu.")
        return False

    candidates = []
    if indexer.graph_path.exists():
        candidates.append(indexer.graph_path)
    if bak_path.exists():
        candidates.append(bak_path)

    last_err: Optional[Exception] = None
    loaded_from: Optional[Path] = None
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            indexer._populate_from_graph_data(data)
            loaded_from = path
            break
        except Exception as e:
            last_err = e
            logger.error(f"Lỗi nạp đồ thị từ {path}: {e}")

    if loaded_from is None:
        raise RuntimeError(
            f"File checkpoint đồ thị '{indexer.graph_path}' bị hỏng và không có bản .bak hợp lệ "
            f"(lỗi cuối: {last_err}). Hãy khôi phục file hoặc dùng cờ --clear nếu muốn tạo lại từ đầu."
        )

    if loaded_from == bak_path:
        logger.warning(f"⚠️  Đã tự động khôi phục đồ thị từ bản sao lưu {bak_path}!")

    # Khôi phục cache Semantic Units nếu có
    if indexer.semantic_units_cache_path.exists():
        indexer._cached_units = indexer.extractor.load_units_jsonl(
            indexer.semantic_units_cache_path
        )
        indexer._cached_unit_ids = {
            u.unit_id for u in indexer._cached_units if getattr(u, "unit_id", None)
        }

    logger.info(
        f"Đã nạp đồ thị checkpoint ({loaded_from.name}): "
        f"{indexer.registry.node_count()} nodes, {len(indexer._edges)} edges"
    )
    return True


def vectorize_lightrag_entities(
    indexer: Any,
    force_reembed: bool = False,
    chroma_dir: Optional[Path] = None,
) -> int:
    """
    Vector hóa tên + mô tả của các Node trong đồ thị bằng BGE-M3 ONNX.
    Lưu vào ChromaDB collection riêng: lightrag_entity_collection.

    Tối ưu gia tăng (Incremental Embedding):
      - Mặc định (force_reembed=False): Kiểm tra collection hiện có, BỎ QUA các entity
        đã có vector với cùng nội dung văn bản; nếu chỉ đổi metadata (vd: thêm source_unit_ids)
        thì chỉ cập nhật metadata mà KHÔNG gọi lại model embedding.
      - Chỉ nhúng lại toàn bộ khi force_reembed=True (vd: cờ --clear).

    Returns:
        Số lượng entity mới/thay đổi đã được vector hóa trong phiên này.
    """
    nodes = indexer.registry.all_nodes()
    if not nodes:
        logger.warning("Không có node nào để vector hóa.")
        return 0

    try:
        import chromadb
        from chromadb.config import Settings

        # Mở ChromaDB collection dành riêng cho LightRAG Entity trước khi tải model nặng
        target_chroma_dir = Path(chroma_dir) if chroma_dir else indexer.entity_vector_dir
        target_chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(target_chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name="lightrag_entity_collection",
            metadata={"hnsw:space": "cosine"},
        )

        existing_map: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        if not force_reembed:
            try:
                existing = collection.get(include=["documents", "metadatas"])
                ex_ids = existing.get("ids") or []
                ex_docs = existing.get("documents") or []
                ex_metas = existing.get("metadatas") or []
                for eid, edoc, emeta in zip(ex_ids, ex_docs, ex_metas):
                    existing_map[eid] = (edoc or "", emeta or {})
            except Exception as read_err:
                logger.warning(
                    f"Không thể đọc entity hiện có từ Chroma ({read_err}); sẽ nhúng toàn bộ."
                )

        items_to_embed: List[Tuple[str, str, Dict[str, Any]]] = []
        meta_only_updates: List[Tuple[str, Dict[str, Any]]] = []
        skipped_unchanged = 0

        for node in nodes:
            name = node["canonical_name"]
            desc = " ".join(node.get("descriptions", []))[:300]
            text_to_embed = f"{name}. {desc}".strip()
            nid = node["id"]
            meta = {
                "canonical_name": name,
                "entity_type": node.get("entity_type", "OTHER"),
                "source_unit_ids": json.dumps(node.get("source_unit_ids", [])),
            }

            if not force_reembed and nid in existing_map:
                old_doc, old_meta = existing_map[nid]
                if old_doc == text_to_embed:
                    if old_meta == meta:
                        skipped_unchanged += 1
                    else:
                        meta_only_updates.append((nid, meta))
                    continue

            items_to_embed.append((nid, text_to_embed, meta))

        # Cập nhật các node chỉ thay đổi metadata (không cần nhúng lại vector)
        if meta_only_updates:
            batch_size = 500
            for i in range(0, len(meta_only_updates), batch_size):
                sub = meta_only_updates[i : i + batch_size]
                collection.update(
                    ids=[x[0] for x in sub],
                    metadatas=[x[1] for x in sub],
                )

        logger.info(
            f"📊 Incremental Entity Embedding: tổng={len(nodes)}, "
            f"bỏ qua không đổi={skipped_unchanged}, "
            f"chỉ cập nhật metadata={len(meta_only_updates)}, "
            f"cần nhúng mới/thay đổi={len(items_to_embed)}"
        )

        if not items_to_embed:
            logger.info("✅ Không có entity mới/thay đổi nào cần nhúng thêm.")
            return 0

        from src.embeddings.embedder import UETEmbedder
        embedder = UETEmbedder()

        batch_size = 100
        count = 0

        for i in range(0, len(items_to_embed), batch_size):
            batch = items_to_embed[i : i + batch_size]
            ids = [x[0] for x in batch]
            texts = [x[1] for x in batch]
            metadatas = [x[2] for x in batch]

            # Embedding batch bằng BGE-M3 ONNX (tối ưu hóa batch inference)
            if hasattr(embedder, "embed_texts"):
                vectors = embedder.embed_texts(texts)
            elif hasattr(embedder, "embed_batch"):
                vectors = embedder.embed_batch(texts)
            else:
                vectors = []
                for text in texts:
                    vec = embedder.embed_query(text)
                    vectors.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))

            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            count += len(batch)
            logger.info(f"  Vector hóa {count}/{len(items_to_embed)} entities mới...")

        logger.info(
            f"✅ Đã vector hóa {count} entities mới/thay đổi vào lightrag_entity_collection "
            f"(tổng trong collection: {collection.count()})"
        )
        return count

    except Exception as e:
        logger.error(f"Lỗi vector hóa entities: {e}")
        return 0

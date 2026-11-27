"""Offline GraphIndex construction, LightRAG KG merging, and vector matrix derivation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.graph.entity_utils import (
    _ENTITY_TYPE_WEIGHTS,
    _LIGHTRAG_CATEGORY_MAP,
    _RELATION_SEED_PHRASES,
    _graph_tokenize,
    _normalize_relation,
    _strip_chunk_header_for_match,
    canonicalize_lightrag_entity_name,
    normalize_entity_id,
)
from src.graph.hybrid_extractor import HybridGraphExtractor
from src.graph.thesaurus_utils import is_forbidden_entity
from src.utils.helpers import get_logger

logger = get_logger("graph.bridge")


def l2_normalize_array(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def l2_normalize(vectors: List[List[float]]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return l2_normalize_array(arr)


def embed_in_batches(index: Any, texts: List[str]) -> Optional[np.ndarray]:
    if not texts or index.embedding_model is None:
        return None
    all_vecs: List[List[float]] = []
    for start in range(0, len(texts), index.embed_batch_size):
        batch = texts[start : start + index.embed_batch_size]
        all_vecs.extend(index.embedding_model.embed_texts(batch))
    return l2_normalize(all_vecs)


def build_doc_neighbors(index: Any) -> None:
    doc_groups: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for idx, chunk in enumerate(index.chunks):
        doc_id = chunk.document_id or (chunk.metadata or {}).get("document_id")
        if not doc_id:
            continue
        c_idx = chunk.chunk_index if chunk.chunk_index is not None else (chunk.metadata or {}).get("chunk_index", idx)
        doc_groups[str(doc_id)].append((int(c_idx), idx, chunk.chunk_id))

    neighbors: Dict[str, List[str]] = defaultdict(list)
    for entries in doc_groups.values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda item: (item[0], item[1]))
        for pos, (_, _, cid) in enumerate(entries):
            if pos > 0:
                neighbors[cid].append(entries[pos - 1][2])
            if pos + 1 < len(entries):
                neighbors[cid].append(entries[pos + 1][2])
    index.doc_chunk_neighbors = dict(neighbors)


def load_lightrag_entity_chroma_vectors(entity_chroma_dir: Optional[Path]) -> Dict[str, np.ndarray]:
    """Đọc trực tiếp kho vector thực thể BGE-M3 chuẩn (Tên + Định nghĩa) từ entity_chroma sang RAM."""
    if entity_chroma_dir is None:
        return {}
    sqlite_path = Path(entity_chroma_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return {}
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(entity_chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(name="lightrag_entity_collection")
        total_ents = col.count()
        if total_ents == 0:
            return {}

        vec_map: Dict[str, np.ndarray] = {}
        page_size = 2500
        for offset in range(0, total_ents, page_size):
            fetched = col.get(limit=page_size, offset=offset, include=["embeddings", "metadatas"])
            embs = fetched.get("embeddings")
            metas = fetched.get("metadatas") or []
            if embs is None or len(embs) == 0:
                continue
            for emb, meta in zip(embs, metas):
                if emb is None or not isinstance(meta, dict):
                    continue
                raw_name = str(meta.get("canonical_name") or "").strip()
                if not raw_name:
                    continue
                arr = np.asarray(emb, dtype=np.float32)
                if arr.ndim != 1 or arr.shape[0] == 0:
                    continue
                norm = float(np.linalg.norm(arr))
                if norm > 0.0:
                    arr = arr / norm
                canon = canonicalize_lightrag_entity_name(raw_name)
                eid = normalize_entity_id(canon) if canon else ""
                if eid:
                    vec_map[eid] = arr
        logger.info(f"Đã nạp {len(vec_map)} vector thực thể BGE-M3 chuẩn từ '{entity_chroma_dir}'.")
        return vec_map
    except Exception as exc:
        logger.warning(f"Không thể đọc vector từ entity_chroma '{entity_chroma_dir}': {exc}")
        return {}


def derive_vectors_from_chunks(
    index: Any,
    chunk_vectors: Dict[str, List[float]],
    precomputed_entity_vecs: Optional[Dict[str, np.ndarray]] = None,
) -> bool:
    """Tổng hợp ma trận NumPy cho cả Entities và Triples."""
    valid_vecs: Dict[str, np.ndarray] = {}
    dim: Optional[int] = None
    if precomputed_entity_vecs:
        first_ent_vec = next(iter(precomputed_entity_vecs.values()), None)
        if first_ent_vec is not None:
            dim = int(first_ent_vec.shape[0])

    for cid, vec in (chunk_vectors or {}).items():
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] == 0:
            continue
        if dim is None:
            dim = int(arr.shape[0])
        if arr.shape[0] == dim:
            valid_vecs[cid] = arr

    if (not valid_vecs and not precomputed_entity_vecs) or dim is None:
        return False

    rel_vec_map: Dict[str, np.ndarray] = {}
    if index.embedding_model is not None and index.triple_records:
        try:
            rel_keys = list(_RELATION_SEED_PHRASES.keys())
            rel_phrases = [_RELATION_SEED_PHRASES[k] for k in rel_keys]
            rel_embs = embed_in_batches(index, rel_phrases)
            if rel_embs is not None and rel_embs.shape[1] == dim:
                for k, r_vec in zip(rel_keys, rel_embs):
                    rel_vec_map[k] = r_vec
        except Exception:
            rel_vec_map = {}

    entity_vec_map: Dict[str, np.ndarray] = {}
    direct_ent_hits = 0
    if index.entity_ids:
        ent_mat = np.zeros((len(index.entity_ids), dim), dtype=np.float32)
        for idx, eid in enumerate(index.entity_ids):
            direct_v = precomputed_entity_vecs.get(eid) if precomputed_entity_vecs else None
            if direct_v is not None and direct_v.shape[0] == dim:
                ent_mat[idx] = direct_v
                entity_vec_map[eid] = direct_v
                direct_ent_hits += 1
                continue

            cids = index.graph.nodes[eid].get("chunks", ())
            acc = np.zeros(dim, dtype=np.float32)
            cnt = 0
            for cid in cids:
                v = valid_vecs.get(cid)
                if v is not None:
                    acc += v
                    cnt += 1
            if cnt > 0:
                acc /= float(cnt)
                ent_mat[idx] = acc
                entity_vec_map[eid] = acc
        index.entity_vectors = l2_normalize_array(ent_mat)

    if index.triple_records:
        tr_mat = np.zeros((len(index.triple_records), dim), dtype=np.float32)
        for idx, tr in enumerate(index.triple_records):
            acc = np.zeros(dim, dtype=np.float32)
            w_sum = 0.0
            h_vec = entity_vec_map.get(tr["head_id"])
            if h_vec is not None:
                acc += 0.38 * h_vec
                w_sum += 0.38
            t_vec = entity_vec_map.get(tr["tail_id"])
            if t_vec is not None:
                acc += 0.38 * t_vec
                w_sum += 0.38
            r_vec = rel_vec_map.get(tr["relation"])
            if r_vec is not None:
                acc += 0.14 * r_vec
                w_sum += 0.14

            c_acc = np.zeros(dim, dtype=np.float32)
            c_cnt = 0
            for cid in tr["chunk_ids"]:
                v = valid_vecs.get(cid)
                if v is not None:
                    c_acc += v
                    c_cnt += 1
            if c_cnt > 0:
                acc += 0.10 * (c_acc / float(c_cnt))
                w_sum += 0.10

            if w_sum > 0.0:
                tr_mat[idx] = acc / w_sum
        index.triple_vectors = l2_normalize_array(tr_mat)

    logger.info(
        f"Đã dựng ma trận NumPy {dim} chiều cho {len(index.entity_ids)} entities "
        f"(entity_chroma chuẩn={direct_ent_hits}) và {len(index.triple_records)} triples."
    )
    return True


def embed_graph_items(
    index: Any,
    chunk_vectors: Optional[Dict[str, List[float]]] = None,
    entity_chroma_dir: Optional[Path] = None,
) -> None:
    precomputed_ent_vecs = load_lightrag_entity_chroma_vectors(entity_chroma_dir)
    resolved_chunk_vecs = chunk_vectors
    if (
        not resolved_chunk_vecs
        and index.vector_store is not None
        and hasattr(index.vector_store, "get_embeddings_by_ids")
    ):
        try:
            resolved_chunk_vecs = index.vector_store.get_embeddings_by_ids([c.chunk_id for c in index.chunks])
        except Exception as exc:
            logger.warning(f"Không thể lấy chunk vectors từ VectorStore ({exc}). Fallback sang embedding_model.")
            resolved_chunk_vecs = None

    if (resolved_chunk_vecs or precomputed_ent_vecs) and derive_vectors_from_chunks(
        index,
        resolved_chunk_vecs or {},
        precomputed_entity_vecs=precomputed_ent_vecs,
    ):
        return

    if index.embedding_model is None:
        return
    if index.entity_texts:
        index.entity_vectors = embed_in_batches(index, index.entity_texts)
    if index.triple_records:
        triple_texts = [tr["text"] for tr in index.triple_records]
        index.triple_vectors = embed_in_batches(index, triple_texts)


def ingest_lightrag_knowledge_graph(
    index: Any,
    graph_path: Path,
    units_path: Optional[Path],
    triple_key_to_idx: Dict[Tuple[str, str, str], int],
) -> None:
    """Hợp nhất tri thức 2-Pass từ LightRAG (graph.json + semantic_units_cache.jsonl) vào GraphIndex."""
    try:
        with graph_path.open("r", encoding="utf-8") as fp:
            lr_data = json.load(fp)
    except Exception as exc:
        logger.warning(f"Không thể đọc LightRAG graph '{graph_path}': {exc}")
        return

    raw_entities = lr_data.get("entities") or {}
    if isinstance(raw_entities, list):
        raw_entities = {n["canonical_name"]: n for n in raw_entities if "canonical_name" in n}
    elif not raw_entities and "nodes" in lr_data:
        raw_entities = {n["canonical_name"]: n for n in lr_data.get("nodes", []) if "canonical_name" in n}

    raw_relations = lr_data.get("relations") or lr_data.get("edges") or []
    raw_summaries = lr_data.get("document_summaries") or {}
    index.document_summaries = dict(raw_summaries) if isinstance(raw_summaries, dict) else {}

    doc_chunks_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    title_chunks_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    title_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
    orig_titles_to_chunks: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for chunk in index.chunks:
        meta = chunk.metadata or {}
        doc_id = str(chunk.document_id or meta.get("document_id") or meta.get("doc_id") or "").strip()
        raw_title = str(meta.get("title") or "").strip()
        title_low = raw_title.casefold()
        body_low = _strip_chunk_header_for_match(chunk.text)
        if doc_id:
            doc_chunks_map[doc_id].append((chunk.chunk_id, body_low))
        if raw_title:
            orig_titles_to_chunks[raw_title].append((chunk.chunk_id, body_low))
        if title_low:
            title_chunks_map[title_low].append((chunk.chunk_id, body_low))
            if doc_id:
                title_to_doc_ids[title_low].add(doc_id)

    for sum_key, sum_info in index.document_summaries.items():
        if not isinstance(sum_info, dict):
            continue
        explicit_did = str(sum_info.get("doc_id") or "").strip()
        if explicit_did and explicit_did in doc_chunks_map:
            index.doc_id_to_summary[explicit_did] = sum_info
        if str(sum_key).strip() in doc_chunks_map:
            index.doc_id_to_summary[str(sum_key).strip()] = sum_info
        t_low = str(sum_info.get("title") or sum_key).strip().casefold()
        for did in title_to_doc_ids.get(t_low, ()):
            index.doc_id_to_summary.setdefault(did, sum_info)
        sum_key_low = str(sum_key).strip().casefold()
        for did in title_to_doc_ids.get(sum_key_low, ()):
            index.doc_id_to_summary.setdefault(did, sum_info)

    unit_to_chunks: Dict[str, List[Tuple[str, str]]] = {}
    pass1_hash_to_chunks: Dict[str, List[Tuple[str, str]]] = {}
    for did, c_list in doc_chunks_map.items():
        p1_did = f"pass1_{hashlib.md5(did.encode('utf-8')).hexdigest()[:8]}"
        pass1_hash_to_chunks[p1_did] = c_list[:2]
    for raw_title, c_list in orig_titles_to_chunks.items():
        p1_title = f"pass1_{hashlib.md5(raw_title.encode('utf-8')).hexdigest()[:8]}"
        pass1_hash_to_chunks.setdefault(p1_title, c_list[:2])

    if units_path is None:
        default_units = graph_path.parent / "semantic_units_cache.jsonl"
        if default_units.exists():
            units_path = default_units

    seen_doc_titles: Dict[str, str] = {}
    if units_path is not None and units_path.exists():
        try:
            with units_path.open("r", encoding="utf-8") as ufp:
                for line in ufp:
                    line = line.strip()
                    if not line:
                        continue
                    u = json.loads(line)
                    uid = str(u.get("unit_id") or "")
                    did = str(u.get("document_id") or "")
                    d_title = str(u.get("document_title") or "")
                    if d_title and did:
                        seen_doc_titles[d_title] = did
                    cand_chunks = doc_chunks_map.get(did) or title_chunks_map.get(d_title.strip().casefold()) or []
                    if not uid or not cand_chunks:
                        continue
                    if len(cand_chunks) <= 2:
                        unit_to_chunks[uid] = cand_chunks
                        continue
                    u_norm = re.sub(r"\s+", " ", str(u.get("content") or "")).casefold()
                    matched_c: List[Tuple[str, str]] = []
                    for cid, c_body in cand_chunks:
                        if not c_body:
                            continue
                        probe_1 = c_body[: min(65, len(c_body))]
                        mid = len(c_body) // 2
                        probe_2 = c_body[mid : min(mid + 65, len(c_body))]
                        if (len(probe_1) >= 20 and probe_1 in u_norm) or (len(probe_2) >= 20 and probe_2 in u_norm):
                            matched_c.append((cid, c_body))
                    unit_to_chunks[uid] = matched_c if matched_c else cand_chunks[:3]
        except Exception as exc:
            logger.warning(f"Lỗi khi đọc semantic_units_cache '{units_path}': {exc}")

    for orig_title, did in seen_doc_titles.items():
        p1_id = f"pass1_{hashlib.md5(orig_title.encode('utf-8')).hexdigest()[:8]}"
        c_list = doc_chunks_map.get(did) or []
        if c_list:
            pass1_hash_to_chunks[p1_id] = c_list[:2]

    def _split_primary_secondary(
        unit_ids: Iterable[str],
        evidence_probes: List[str],
        name_probes: List[str],
    ) -> Tuple[Set[str], Set[str]]:
        pool: List[Tuple[str, str]] = []
        seen_c: Set[str] = set()
        for uid in unit_ids:
            cands = unit_to_chunks.get(uid) or pass1_hash_to_chunks.get(uid) or ()
            for cid, body_low in cands:
                if cid not in seen_c:
                    seen_c.add(cid)
                    pool.append((cid, body_low))
        if not pool:
            return set(), set()

        ev_clean = [re.sub(r"\s+", " ", p).strip().casefold() for p in evidence_probes if p and len(p.strip()) >= 12]
        nm_clean = [re.sub(r"\s+", " ", p).strip().casefold() for p in name_probes if p and len(p.strip()) >= 3]

        primary: Set[str] = set()
        secondary: Set[str] = set()
        for cid, body_low in pool:
            hit = False
            for ev in ev_clean:
                snippet = ev[: min(48, len(ev))]
                if snippet in body_low:
                    hit = True
                    break
            if not hit:
                for nm in nm_clean:
                    if nm in body_low:
                        hit = True
                        break
            if hit:
                primary.add(cid)
            else:
                secondary.add(cid)

        if not primary:
            for cid, _ in pool[:2]:
                primary.add(cid)
                secondary.discard(cid)

        if len(secondary) > 6:
            secondary = set(sorted(secondary)[:6])
        return primary, secondary

    raw_name_to_eid: Dict[str, str] = {}
    for raw_key, node_info in raw_entities.items():
        if not isinstance(node_info, dict):
            continue
        raw_canon = str(node_info.get("canonical_name") or raw_key).strip()
        raw_cat = str(node_info.get("entity_type") or node_info.get("category") or "CONCEPT").upper()
        mapped_type = _LIGHTRAG_CATEGORY_MAP.get(raw_cat, "CONCEPT")
        canon_name = canonicalize_lightrag_entity_name(raw_canon, mapped_type)
        if len(canon_name) < 2 or is_forbidden_entity(canon_name, allow_school_names=True):
            continue
        eid = normalize_entity_id(canon_name)
        if not eid:
            continue

        raw_name_to_eid[raw_canon.casefold()] = eid
        raw_name_to_eid[str(raw_key).strip().casefold()] = eid

        descs_raw = node_info.get("descriptions") or []
        if isinstance(descs_raw, str):
            descs = [descs_raw.strip()] if descs_raw.strip() else []
        else:
            descs = [str(d).strip() for d in descs_raw if str(d).strip()]

        stripped_probe = re.sub(r"^\[[^\]]+\]\s*", "", canon_name).strip()
        prim_cids, sec_cids = _split_primary_secondary(
            node_info.get("source_unit_ids") or (),
            evidence_probes=descs[:2],
            name_probes=[stripped_probe, raw_canon],
        )
        if not prim_cids and not index.graph.has_node(eid):
            continue

        if not index.graph.has_node(eid):
            index.graph.add_node(
                eid,
                name=canon_name,
                type=mapped_type,
                chunks=set(),
                secondary_chunks=set(),
                descriptions=[],
            )
            index.lightrag_nodes_merged += 1

        g_node = index.graph.nodes[eid]
        g_node["chunks"].update(prim_cids)
        g_node.setdefault("secondary_chunks", set()).update(sec_cids - g_node["chunks"])
        existing_descs: List[str] = g_node.setdefault("descriptions", [])
        for d in descs:
            if d not in existing_descs and len(existing_descs) < 3:
                existing_descs.append(d)

        for cid in prim_cids:
            ent_list = index.chunk_to_entities[cid]
            if eid not in ent_list:
                ent_list.append(eid)

    for rel_item in raw_relations:
        if not isinstance(rel_item, dict):
            continue
        src_raw = str(rel_item.get("source") or "").strip()
        tgt_raw = str(rel_item.get("target") or "").strip()
        if not src_raw or not tgt_raw:
            continue

        hid = raw_name_to_eid.get(src_raw.casefold())
        if not hid:
            h_canon = canonicalize_lightrag_entity_name(src_raw)
            if not h_canon or is_forbidden_entity(h_canon, allow_school_names=True):
                continue
            hid = normalize_entity_id(h_canon)
        else:
            h_canon = index.graph.nodes[hid]["name"] if index.graph.has_node(hid) else canonicalize_lightrag_entity_name(src_raw)

        tid = raw_name_to_eid.get(tgt_raw.casefold())
        if not tid:
            t_canon = canonicalize_lightrag_entity_name(tgt_raw)
            if not t_canon or is_forbidden_entity(t_canon, allow_school_names=True):
                continue
            tid = normalize_entity_id(t_canon)
        else:
            t_canon = index.graph.nodes[tid]["name"] if index.graph.has_node(tid) else canonicalize_lightrag_entity_name(tgt_raw)

        if not hid or not tid or hid == tid:
            continue

        raw_rel_type = str(rel_item.get("relation_type") or rel_item.get("predicate") or "LIÊN_QUAN_ĐẾN")
        norm_rel = _normalize_relation(raw_rel_type)

        ev_list = rel_item.get("evidence_texts") or []
        if isinstance(ev_list, str):
            ev_list = [ev_list]
        elif not ev_list and rel_item.get("evidence"):
            ev_list = [str(rel_item["evidence"])]
        best_evidence = re.sub(r"\s+", " ", str(ev_list[0]).strip())[:140] if ev_list else ""

        h_probe = re.sub(r"^\[[^\]]+\]\s*", "", h_canon).strip()
        t_probe = re.sub(r"^\[[^\]]+\]\s*", "", t_canon).strip()
        prim_cids, sec_cids = _split_primary_secondary(
            rel_item.get("source_unit_ids") or (),
            evidence_probes=[best_evidence] if best_evidence else [],
            name_probes=[h_probe, t_probe],
        )

        if not prim_cids:
            h_chunks = index.graph.nodes[hid].get("chunks", set()) if index.graph.has_node(hid) else set()
            t_chunks = index.graph.nodes[tid].get("chunks", set()) if index.graph.has_node(tid) else set()
            shared = h_chunks & t_chunks
            prim_cids = shared if shared else set(sorted(h_chunks)[:2] + sorted(t_chunks)[:2])
        if not prim_cids:
            continue

        if not index.graph.has_node(hid):
            index.graph.add_node(
                hid, name=h_canon, type="CONCEPT",
                chunks=set(prim_cids), secondary_chunks=set(sec_cids), descriptions=[],
            )
        else:
            index.graph.nodes[hid]["chunks"].update(prim_cids)

        if not index.graph.has_node(tid):
            index.graph.add_node(
                tid, name=t_canon, type="CONCEPT",
                chunks=set(prim_cids), secondary_chunks=set(sec_cids), descriptions=[],
            )
        else:
            index.graph.nodes[tid]["chunks"].update(prim_cids)

        if index.graph.has_edge(hid, tid):
            edge = index.graph[hid][tid]
            edge["relations"].add(norm_rel)
            edge["chunks"].update(prim_cids)
            edge["weight"] += 1.25
        else:
            index.graph.add_edge(
                hid, tid, relations={norm_rel}, chunks=set(prim_cids), weight=1.25,
            )
            index.lightrag_edges_merged += 1

        h_display = index.graph.nodes[hid]["name"]
        t_display = index.graph.nodes[tid]["name"]
        base_text = f"{h_display} -[{norm_rel}]-> {t_display}"
        rich_text = f"{base_text} (Căn cứ: {best_evidence})" if best_evidence else base_text

        t_sig = (hid, norm_rel, tid)
        if t_sig not in triple_key_to_idx:
            t_idx = len(index.triple_records)
            triple_key_to_idx[t_sig] = t_idx
            index.triple_records.append({
                "head_id": hid,
                "head_name": h_display,
                "relation": norm_rel,
                "tail_id": tid,
                "tail_name": t_display,
                "text": rich_text,
                "evidence": best_evidence,
                "chunk_ids": set(prim_cids),
                "secondary_chunk_ids": set(sec_cids - prim_cids),
            })
        else:
            t_idx = triple_key_to_idx[t_sig]
            rec = index.triple_records[t_idx]
            rec["chunk_ids"].update(prim_cids)
            rec.setdefault("secondary_chunk_ids", set()).update(sec_cids - rec["chunk_ids"])
            if best_evidence and not rec.get("evidence"):
                rec["evidence"] = best_evidence
                rec["text"] = rich_text

        for cid in prim_cids:
            if t_idx not in index.chunk_to_triples[cid]:
                index.chunk_to_triples[cid].append(t_idx)


def build_graph(
    index: Any,
    extractor: HybridGraphExtractor,
    lightrag_graph_path: Optional[Path] = None,
    semantic_units_path: Optional[Path] = None,
) -> None:
    triple_key_to_idx: Dict[Tuple[str, str, str], int] = {}

    for chunk in index.chunks:
        cid = chunk.chunk_id
        entities, triples = extractor.extract(chunk)
        chunk_ent_ids: List[str] = []

        for ent in entities:
            eid = ent["id"]
            if not index.graph.has_node(eid):
                index.graph.add_node(
                    eid,
                    name=ent["name"],
                    type=ent["type"],
                    chunks=set(),
                    secondary_chunks=set(),
                    descriptions=[],
                )
            index.graph.nodes[eid]["chunks"].add(cid)
            if eid not in chunk_ent_ids:
                chunk_ent_ids.append(eid)

        index.chunk_to_entities[cid] = chunk_ent_ids

        for tr in triples:
            hid = normalize_entity_id(tr["head"])
            tid = normalize_entity_id(tr["tail"])
            rel = tr["relation"]
            if not hid or not tid or hid == tid:
                continue

            if not index.graph.has_node(hid):
                index.graph.add_node(
                    hid, name=tr["head"], type=tr.get("head_type", "CONCEPT"),
                    chunks=set(), secondary_chunks=set(), descriptions=[],
                )
            if not index.graph.has_node(tid):
                index.graph.add_node(
                    tid, name=tr["tail"], type=tr.get("tail_type", "CONCEPT"),
                    chunks=set(), secondary_chunks=set(), descriptions=[],
                )

            index.graph.nodes[hid]["chunks"].add(cid)
            index.graph.nodes[tid]["chunks"].add(cid)

            if index.graph.has_edge(hid, tid):
                edge = index.graph[hid][tid]
                edge["relations"].add(rel)
                edge["chunks"].add(cid)
                edge["weight"] += 1.0
            else:
                index.graph.add_edge(hid, tid, relations={rel}, chunks={cid}, weight=1.0)

            t_sig = (hid, rel, tid)
            if t_sig not in triple_key_to_idx:
                t_idx = len(index.triple_records)
                triple_key_to_idx[t_sig] = t_idx
                index.triple_records.append({
                    "head_id": hid,
                    "head_name": index.graph.nodes[hid]["name"],
                    "relation": rel,
                    "tail_id": tid,
                    "tail_name": index.graph.nodes[tid]["name"],
                    "text": f"{index.graph.nodes[hid]['name']} -[{rel}]-> {index.graph.nodes[tid]['name']}",
                    "evidence": "",
                    "chunk_ids": {cid},
                    "secondary_chunk_ids": set(),
                })
            else:
                t_idx = triple_key_to_idx[t_sig]
                index.triple_records[t_idx]["chunk_ids"].add(cid)

            if t_idx not in index.chunk_to_triples[cid]:
                index.chunk_to_triples[cid].append(t_idx)

    if lightrag_graph_path is not None and Path(lightrag_graph_path).exists():
        ingest_lightrag_knowledge_graph(
            index,
            graph_path=Path(lightrag_graph_path),
            units_path=Path(semantic_units_path) if semantic_units_path else None,
            triple_key_to_idx=triple_key_to_idx,
        )

    index.entity_ids = list(index.graph.nodes())
    index.entity_texts = []
    for eid in index.entity_ids:
        nd = index.graph.nodes[eid]
        descs = nd.get("descriptions") or []
        desc_snippet = f" - {descs[0][:180]}" if descs else ""
        index.entity_texts.append(f"{nd['name']} ({nd['type']}){desc_snippet}")

    total_docs = max(1, len(index.chunks))
    idf_denom = max(math.log(total_docs + 1.0), 1e-6)
    index.entity_idf = {}
    for eid in index.entity_ids:
        node = index.graph.nodes[eid]
        df = max(1, len(node.get("chunks", ())))
        raw_idf = max(0.0, math.log((total_docs + 1.0) / (df + 0.5)) / idf_denom)
        type_w = _ENTITY_TYPE_WEIGHTS.get(str(node.get("type") or "CONCEPT"), 1.0)
        index.entity_idf[eid] = round(min(1.35, max(0.15, raw_idf * type_w)), 4)

    index.entity_idf_array = (
        np.asarray([index.entity_idf[eid] for eid in index.entity_ids], dtype=np.float32)
        if index.entity_ids
        else None
    )

    build_doc_neighbors(index)

    for eid in index.entity_ids:
        name = index.graph.nodes[eid]["name"]
        stripped_name = re.sub(r"^\[[^\]]+\]\s*", "", name).strip()
        toks = _graph_tokenize(name) | _graph_tokenize(stripped_name)
        index.entity_tokens[eid] = toks
        for tok in toks:
            index.token_to_entities[tok].append(eid)

    index.token_to_entities = dict(index.token_to_entities)
    index.chunk_to_entities = dict(index.chunk_to_entities)
    index.chunk_to_triples = dict(index.chunk_to_triples)

    logger.info(
        f"Đã dựng Unified Knowledge Graph: {len(index.chunks)} chunks -> "
        f"{len(index.entity_ids)} entities, {len(index.triple_records)} triples "
        f"(LightRAG merged: +{index.lightrag_nodes_merged} nodes, +{index.lightrag_edges_merged} edges, "
        f"{len(index.document_summaries)} summaries)."
    )

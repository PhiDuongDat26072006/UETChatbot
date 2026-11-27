"""Online GraphIndex search (spreading activation) and LightRAG subgraph chunk grounding."""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.base import RetrievedContext
from src.graph.entity_utils import (
    _ARTICLE_RE,
    _CERT_RE,
    _COURSE_CODE_RE,
    _DOC_CODE_RE,
    _RELATION_WEIGHTS,
    _graph_tokenize,
    _normalize_relation,
    canonicalize_entity_name,
    canonicalize_lightrag_entity_name,
    normalize_entity_id,
)
from src.graph.thesaurus_utils import (
    KNOWN_FACULTY_KEYS,
    detect_query_faculties,
    get_thesaurus,
    is_forbidden_entity,
)
from src.utils.helpers import get_logger

logger = get_logger("graph.bridge")


def search_graph_index(
    index: Any,
    query: str,
    query_vector: Optional[List[float]] = None,
    top_k: int = 50,
    seed_top_m: int = 10,
    max_hops: int = 2,
    hop_decay: float = 0.65,
    adjacent_alpha: float = 0.18,
    filters: Optional[Dict[str, Any]] = None,
    extracted_entities: Optional[List[str]] = None,
    high_level_themes: Optional[List[str]] = None,
    skip_thesaurus_scan: bool = False,
) -> List[RetrievedContext]:
    """
    Truy xuất trên Unified Graph Vector Index (PA1/Thesaurus/Regex + NumPy Vector + Spreading Activation).
    """
    clean_query = (query or "").strip()
    if top_k <= 0 or not index.chunks or not index.entity_ids or not clean_query:
        return []

    q_vec_arr: Optional[np.ndarray] = None
    if query_vector is None and index.embedding_model is not None:
        try:
            vecs = index.embedding_model.embed_texts([clean_query])
            if vecs:
                query_vector = vecs[0]
        except Exception as exc:
            logger.warning(f"Không thể embed query trong GraphIndex.search: {exc}")

    if query_vector is not None:
        arr = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            q_vec_arr = arr / norm

    entity_scores: Dict[str, float] = defaultdict(float)
    triple_chunk_bonus: Dict[str, float] = defaultdict(float)
    activated_triple_indices: Dict[int, float] = {}
    entity_idf = getattr(index, "entity_idf", {})
    idf_arr = getattr(index, "entity_idf_array", None)
    ent_hits: Dict[str, int] = defaultdict(int)

    for rx in (_DOC_CODE_RE, _COURSE_CODE_RE, _ARTICLE_RE, _CERT_RE):
        for m in rx.finditer(clean_query):
            code_eid = normalize_entity_id(canonicalize_entity_name(m.group(1)))
            if code_eid and index.graph.has_node(code_eid):
                idf = entity_idf.get(code_eid, 1.0)
                entity_scores[code_eid] += 1.35 * idf
                ent_hits[code_eid] += 2

    if not skip_thesaurus_scan:
        try:
            thesaurus = get_thesaurus()
            q_words = re.findall(r"[\w\+\-\./]+", clean_query.lower())
            covered: Set[int] = set()
            for n_gram in range(min(6, len(q_words)), 0, -1):
                for pos in range(len(q_words) - n_gram + 1):
                    span = set(range(pos, pos + n_gram))
                    if span & covered:
                        continue
                    phrase = " ".join(q_words[pos : pos + n_gram])
                    if is_forbidden_entity(phrase, allow_school_names=False):
                        continue
                    th_canon = thesaurus.get_canonical(phrase)
                    if th_canon and not is_forbidden_entity(th_canon, allow_school_names=False):
                        covered.update(span)
                        th_eid = normalize_entity_id(canonicalize_entity_name(th_canon))
                        if th_eid and index.graph.has_node(th_eid):
                            idf = entity_idf.get(th_eid, 1.0)
                            entity_scores[th_eid] += 1.15 * idf
                            ent_hits[th_eid] += 2
        except Exception:
            pass

    if extracted_entities:
        for rank_idx, raw_ent in enumerate(extracted_entities[:8]):
            if is_forbidden_entity(raw_ent, allow_school_names=False):
                continue
            canon_ent = canonicalize_lightrag_entity_name(raw_ent)
            if not canon_ent or is_forbidden_entity(canon_ent, allow_school_names=False):
                continue
            pa1_eid = normalize_entity_id(canon_ent)
            decay_w = max(0.80, 1.0 - rank_idx * 0.04)
            if pa1_eid and index.graph.has_node(pa1_eid):
                idf = entity_idf.get(pa1_eid, 1.0)
                entity_scores[pa1_eid] += 1.30 * decay_w * idf
                ent_hits[pa1_eid] += 2
            else:
                sub_toks = _graph_tokenize(canon_ent)
                if sub_toks:
                    cand_counts: Dict[str, int] = defaultdict(int)
                    for st in sub_toks:
                        for ceid in index.token_to_entities.get(st, ()):
                            cand_counts[ceid] += 1
                    for ceid, c_cnt in cand_counts.items():
                        denom = max(1, len(index.entity_tokens.get(ceid, ())))
                        if c_cnt / denom >= 0.75 and c_cnt >= min(2, len(sub_toks)):
                            idf = entity_idf.get(ceid, 1.0)
                            entity_scores[ceid] += 0.90 * decay_w * idf
                            ent_hits[ceid] += 1

    q_tokens = _graph_tokenize(clean_query)
    if q_tokens:
        lex_hits: Dict[str, int] = defaultdict(int)
        for q_tok in q_tokens:
            for eid in index.token_to_entities.get(q_tok, ()):
                lex_hits[eid] += 1
                ent_hits[eid] += 1
        for eid, hit_count in lex_hits.items():
            denom = max(1, len(index.entity_tokens.get(eid, ())))
            overlap_ratio = hit_count / denom
            idf = entity_idf.get(eid, 1.0)
            exact_rare_boost = 1.25 if (overlap_ratio >= 0.99 and idf >= 0.75) else 1.0
            lex_score = (0.45 * overlap_ratio + 0.1 * min(hit_count, 3)) * idf * exact_rare_boost
            entity_scores[eid] += lex_score

    if (
        q_vec_arr is not None
        and index.entity_vectors is not None
        and index.entity_vectors.shape[1] == q_vec_arr.shape[0]
    ):
        sims = index.entity_vectors @ q_vec_arr
        weighted_sims = np.maximum(sims, 0.0) * (
            idf_arr if (idf_arr is not None and idf_arr.shape[0] == sims.shape[0]) else 1.0
        )
        k_cand = min(max(seed_top_m * 4, 20), len(weighted_sims))
        top_indices = np.argpartition(weighted_sims, -k_cand)[-k_cand:]
        ranked_vec_indices = sorted(
            (int(i) for i in top_indices if float(weighted_sims[int(i)]) > 0.0),
            key=lambda i: float(weighted_sims[i]),
            reverse=True,
        )
        df1_chunk_counts: Dict[str, int] = defaultdict(int)
        for idx in ranked_vec_indices:
            eid = index.entity_ids[idx]
            w_sim = float(weighted_sims[idx])
            node = index.graph.nodes[eid]
            etype = str(node.get("type") or "")
            has_lex = ent_hits.get(eid, 0) > 0
            if etype in {"COURSE", "CERTIFICATE"} and not has_lex:
                w_sim *= 0.25
            chunks_set = node.get("chunks", ())
            if not has_lex and len(chunks_set) == 1:
                only_cid = next(iter(chunks_set))
                if df1_chunk_counts[only_cid] >= 2:
                    continue
                df1_chunk_counts[only_cid] += 1
            entity_scores[eid] += w_sim

    if (
        q_vec_arr is not None
        and index.triple_vectors is not None
        and index.triple_vectors.shape[1] == q_vec_arr.shape[0]
    ):
        t_sims = index.triple_vectors @ q_vec_arr
        k_tr = min(max(seed_top_m * 2, 12), len(t_sims))
        top_t_indices = np.argpartition(t_sims, -k_tr)[-k_tr:]
        for idx in top_t_indices:
            sim = float(t_sims[idx])
            if sim > 0.0:
                t_idx = int(idx)
                tr = index.triple_records[t_idx]
                rel_w = _RELATION_WEIGHTS.get(tr["relation"], 1.0)
                h_idf = entity_idf.get(tr["head_id"], 1.0)
                t_idf = entity_idf.get(tr["tail_id"], 1.0)
                tr_score = sim * rel_w
                activated_triple_indices[t_idx] = tr_score
                entity_scores[tr["head_id"]] += 0.5 * tr_score * h_idf
                entity_scores[tr["tail_id"]] += 0.5 * tr_score * t_idf
                base_tr_bonus = 0.35 * tr_score * (0.5 + 0.25 * (h_idf + t_idf))
                for cid in tr["chunk_ids"]:
                    triple_chunk_bonus[cid] += base_tr_bonus
                for scid in tr.get("secondary_chunk_ids", ()):
                    triple_chunk_bonus[scid] += 0.25 * base_tr_bonus

    if not entity_scores and not triple_chunk_bonus:
        return []

    total_chunks = len(index.chunks)
    max_hub_df = max(250, int(0.12 * total_chunks)) if total_chunks >= 50 else total_chunks + 1

    top_seeds = heapq.nlargest(seed_top_m, entity_scores.items(), key=lambda item: item[1])
    seed_set: Set[str] = {eid for eid, _ in top_seeds}
    activated_entities: Dict[str, float] = dict(top_seeds)
    frontier: Dict[str, float] = dict(top_seeds)

    for hop in range(1, max(0, max_hops) + 1):
        if not frontier:
            break
        decay = hop_decay ** hop
        next_frontier: Dict[str, float] = {}
        for u, u_score in frontier.items():
            if not index.graph.has_node(u) or len(index.graph.nodes[u].get("chunks", ())) > max_hub_df:
                continue
            neighbors = set(index.graph.successors(u)) | set(index.graph.predecessors(u))
            if not neighbors:
                continue
            damping = 1.0 / math.sqrt(len(neighbors))
            for v in neighbors:
                if len(index.graph.nodes[v].get("chunks", ())) > max_hub_df:
                    continue
                rel_w = index._max_edge_relation_weight(u, v)
                v_idf = entity_idf.get(v, 1.0)
                prop = u_score * decay * damping * rel_w * v_idf
                prev = activated_entities.get(v, 0.0)
                if prop > prev:
                    activated_entities[v] = prop
                    next_frontier[v] = max(next_frontier.get(v, 0.0), prop)
        if len(next_frontier) > seed_top_m * 3:
            top_next = heapq.nlargest(seed_top_m * 3, next_frontier.items(), key=lambda item: item[1])
            frontier = dict(top_next)
        else:
            frontier = next_frontier

    chunk_scores: Dict[str, float] = defaultdict(float)
    chunk_matched_entities: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def _add_entity_chunks(eid: str, contrib: float, limit: Optional[int] = None) -> None:
        nd = index.graph.nodes[eid]
        cids = nd.get("chunks", ())
        if limit is not None and len(cids) > limit:
            cids = sorted(cids)[:limit]
        for cid in cids:
            chunk_scores[cid] += contrib
            chunk_matched_entities[cid].append((eid, contrib))
        sec_cids = nd.get("secondary_chunks", ())
        if sec_cids:
            sec_contrib = 0.25 * contrib
            for scid in sec_cids:
                chunk_scores[scid] += sec_contrib

    for eid, e_score in activated_entities.items():
        if not index.graph.has_node(eid):
            continue
        node_chunks = index.graph.nodes[eid].get("chunks", ())
        if len(node_chunks) > max_hub_df:
            continue
        _add_entity_chunks(eid, e_score)

    for cid, bonus in triple_chunk_bonus.items():
        chunk_scores[cid] += bonus

    if not chunk_scores and activated_entities:
        best_eid, best_score = max(activated_entities.items(), key=lambda item: item[1])
        if index.graph.has_node(best_eid):
            _add_entity_chunks(best_eid, best_score, limit=max(1, top_k * 4))

    if not chunk_scores:
        return []

    doc_summary_scores: Dict[str, float] = {}
    doc_id_summaries = getattr(index, "doc_id_to_summary", None) or {}
    if doc_id_summaries:
        theme_terms = [
            canonicalize_entity_name(t).casefold()
            for t in (high_level_themes or [])
            if t and not is_forbidden_entity(t, allow_school_names=False)
        ]
        seed_names_low = {
            index.graph.nodes[eid]["name"].casefold()
            for eid in seed_set
            if index.graph.has_node(eid)
        }
        q_low = clean_query.casefold()
        for did, s_info in doc_id_summaries.items():
            s_title = str(s_info.get("title") or "").casefold()
            s_body = str(s_info.get("summary") or "").casefold()
            s_keys = {canonicalize_entity_name(str(k)).casefold() for k in (s_info.get("key_entities") or [])}
            s_score = 0.0
            for th in theme_terms:
                if th in s_title:
                    s_score += 0.45
                elif th in s_body:
                    s_score += 0.25
                if th in s_keys:
                    s_score += 0.30
            overlap_seeds = len(s_keys & seed_names_low)
            if overlap_seeds:
                s_score += min(0.60, 0.25 * overlap_seeds)
            if s_title and len(s_title) >= 10 and s_title in q_low:
                s_score += 0.40
            if s_score >= 0.25:
                doc_summary_scores[did] = min(1.0, s_score)

    if filters:
        filtered_items = [
            (cid, score)
            for cid, score in chunk_scores.items()
            if cid in index.chunk_id_to_idx and index._matches_filters(index.chunks[index.chunk_id_to_idx[cid]], filters)
        ]
    else:
        filtered_items = [(cid, score) for cid, score in chunk_scores.items() if cid in index.chunk_id_to_idx]

    if not filtered_items:
        return []

    top_base_chunks = heapq.nlargest(max(top_k * 4, 200), filtered_items, key=lambda item: item[1])
    pre_smooth_scores: Dict[str, float] = {}
    for cid, base_score in top_base_chunks:
        co_scores: List[float] = []
        for t_idx in index.chunk_to_triples.get(cid, ()):
            tr = index.triple_records[t_idx]
            h_score = activated_entities.get(tr["head_id"], 0.0)
            t_score = activated_entities.get(tr["tail_id"], 0.0)
            if h_score > 0.0 and t_score > 0.0:
                rel_w = _RELATION_WEIGHTS.get(tr["relation"], 1.0)
                both_seed_mult = 1.35 if (tr["head_id"] in seed_set and tr["tail_id"] in seed_set) else 1.0
                co_scores.append(0.45 * math.sqrt(h_score * t_score) * rel_w * both_seed_mult)
        co_bonus = sum(sorted(co_scores, reverse=True)[:3])
        matched_cnt = len({eid for eid, _ in chunk_matched_entities.get(cid, ())})
        bridge_bonus = 0.25 * min(3, max(0, matched_cnt - 1))

        summary_bonus = 0.0
        if doc_summary_scores:
            ch_obj = index.chunks[index.chunk_id_to_idx[cid]]
            ch_did = str(ch_obj.document_id or (ch_obj.metadata or {}).get("document_id") or "")
            if ch_did in doc_summary_scores:
                summary_bonus = 0.18 * doc_summary_scores[ch_did]

        pre_smooth_scores[cid] = base_score + co_bonus + bridge_bonus + summary_bonus

    if not pre_smooth_scores:
        return []

    doc_neighbors = getattr(index, "doc_chunk_neighbors", {})
    smoothed_scores: Dict[str, float] = dict(pre_smooth_scores)
    if adjacent_alpha > 0.0 and doc_neighbors:
        smooth_pool = heapq.nlargest(max(top_k, seed_top_m * 3), pre_smooth_scores.items(), key=lambda item: item[1])
        for cid, base_val in smooth_pool:
            delta = adjacent_alpha * base_val
            for nbr_cid in doc_neighbors.get(cid, ()):
                smoothed_scores[nbr_cid] = smoothed_scores.get(nbr_cid, 0.0) + delta

    target_faculties = detect_query_faculties(query)
    final_scores: List[Tuple[str, float]] = []
    for cid, score in smoothed_scores.items():
        idx = index.chunk_id_to_idx.get(cid)
        if idx is None:
            continue
        chunk = index.chunks[idx]
        if not index._matches_filters(chunk, filters):
            continue
        adj_score = score
        if target_faculties:
            ch_meta = chunk.metadata or {}
            ch_fac = str(ch_meta.get("faculty") or ch_meta.get("site") or "").strip().lower()
            if ch_fac in target_faculties:
                adj_score *= 1.15
            elif ch_fac in KNOWN_FACULTY_KEYS:
                adj_score *= 0.35
        final_scores.append((cid, adj_score))

    if not final_scores:
        return []

    top_chunks = heapq.nlargest(top_k, final_scores, key=lambda item: item[1])
    max_score = top_chunks[0][1] if top_chunks[0][1] > 0 else 1.0

    results: List[RetrievedContext] = []
    summary_injected_docs: Set[str] = set()
    for rank, (cid, raw_score) in enumerate(top_chunks, start=1):
        chunk = index.chunks[index.chunk_id_to_idx[cid]]
        norm_score = round(min(1.0, max(0.01, raw_score / max_score)), 4)

        matched_ents_sorted = sorted(chunk_matched_entities.get(cid, []), key=lambda x: x[1], reverse=True)
        seen_enames: Set[str] = set()
        ent_names: List[str] = []
        for eid, _ in matched_ents_sorted:
            if index.graph.has_node(eid):
                ename = index.graph.nodes[eid]["name"]
                if ename not in seen_enames:
                    seen_enames.add(ename)
                    ent_names.append(ename)
                    if len(ent_names) >= 6:
                        break

        triple_candidates: List[Tuple[float, str]] = []
        seen_t_texts: Set[str] = set()
        for t_idx in index.chunk_to_triples.get(cid, ()):
            tr = index.triple_records[t_idx]
            h_act = activated_entities.get(tr["head_id"], 0.0)
            t_act = activated_entities.get(tr["tail_id"], 0.0)
            co_mult = 2.5 if (h_act > 0.0 and t_act > 0.0) else 1.0
            t_score = co_mult * (h_act + t_act + activated_triple_indices.get(t_idx, 0.0))
            if t_score > 0.0 and tr["text"] not in seen_t_texts:
                seen_t_texts.add(tr["text"])
                triple_candidates.append((t_score, tr["text"]))
        triple_candidates.sort(key=lambda x: x[0], reverse=True)
        top_triples = [t_txt for _, t_txt in triple_candidates[:5]]

        ch_did = str(chunk.document_id or (chunk.metadata or {}).get("document_id") or "")
        if ch_did and ch_did in doc_summary_scores and ch_did not in summary_injected_docs:
            s_info = doc_id_summaries.get(ch_did) or {}
            s_scope = str(s_info.get("summary") or "").strip()
            if s_scope and len(top_triples) < 5:
                top_triples.insert(0, f"Phạm vi văn bản: {s_scope[:140]}")
                summary_injected_docs.add(ch_did)

        meta = dict(chunk.metadata) if chunk.metadata else {}
        meta["search_source"] = "GraphRAG"
        meta["search_type"] = "graph"
        meta["graph_score"] = round(raw_score, 4)
        meta["graph_entities"] = ent_names
        meta["graph_triples"] = top_triples

        results.append(
            RetrievedContext(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                similarity_score=norm_score,
                rank=rank,
                metadata=meta,
            )
        )

    return results


def ground_lightrag_subgraph(
    index: Any,
    query: str,
    seed_entities: List[Tuple[str, float]],
    activated_nodes: Dict[str, float],
    activated_edges: List[Dict[str, Any]],
    high_level_blocks: Optional[List[Dict[str, Any]]] = None,
    query_vector: Optional[List[float]] = None,
    top_k: int = 60,
    filters: Optional[Dict[str, Any]] = None,
    supplement_weight: float = 0.65,
    adjacent_alpha: float = 0.18,
    extracted_entities: Optional[List[str]] = None,
    high_level_themes: Optional[List[str]] = None,
    seed_top_m: int = 10,
    max_hops: int = 2,
    hop_decay: float = 0.65,
) -> List[RetrievedContext]:
    """Neo kết quả suy luận 2 tầng của LightRAGRetriever về các DataChunk nguyên văn."""
    if top_k <= 0 or not index.chunks:
        return []

    hub_cap = index.max_hub_df()

    seed_eids: Set[str] = set()
    for s_name, _ in (seed_entities or ()):
        seid = index.resolve_lightrag_node_id(s_name)
        if seid:
            seed_eids.add(seid)

    lr_chunk_scores: Dict[str, float] = defaultdict(float)
    chunk_matched_entities: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    chunk_matched_seeds: Dict[str, Set[str]] = defaultdict(set)
    chunk_edge_scores: Dict[str, List[float]] = defaultdict(list)
    chunk_lr_triples: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

    for raw_name, act_score in (activated_nodes or {}).items():
        if act_score <= 0.0:
            continue
        eid = index.resolve_lightrag_node_id(raw_name)
        if not eid or not index.graph.has_node(eid):
            continue
        nd = index.graph.nodes[eid]
        prim_cids = nd.get("chunks", ())
        if len(prim_cids) > hub_cap:
            continue
        contrib = float(act_score)
        disp_name = str(nd.get("name") or raw_name)
        is_seed = eid in seed_eids
        for cid in prim_cids:
            lr_chunk_scores[cid] += contrib
            chunk_matched_entities[cid].append((disp_name, contrib))
            if is_seed:
                chunk_matched_seeds[cid].add(eid)
        for scid in nd.get("secondary_chunks", ()):
            lr_chunk_scores[scid] += 0.25 * contrib

    for edge in (activated_edges or ()):
        src_raw = str(edge.get("source") or "").strip()
        tgt_raw = str(edge.get("target") or "").strip()
        if not src_raw or not tgt_raw:
            continue
        rel_raw = str(edge.get("relation_type") or edge.get("predicate") or "LIÊN_QUAN_ĐẾN")
        norm_rel = _normalize_relation(rel_raw)
        rel_w = _RELATION_WEIGHTS.get(norm_rel, 1.0)

        hid = index.resolve_lightrag_node_id(src_raw)
        tid = index.resolve_lightrag_node_id(tgt_raw)
        if not hid and not tid:
            continue

        ev_list = edge.get("evidence_texts") or []
        if isinstance(ev_list, str):
            ev_list = [ev_list]
        ev = str(
            edge.get("evidence")
            or (ev_list[0] if ev_list else "")
            or edge.get("description")
            or ""
        ).strip()[:140]

        h_disp = index.graph.nodes[hid]["name"] if (hid and index.graph.has_node(hid)) else src_raw
        t_disp = index.graph.nodes[tid]["name"] if (tid and index.graph.has_node(tid)) else tgt_raw
        t_text = f"{h_disp} -[{norm_rel}]-> {t_disp}" + (f" (Căn cứ: {ev})" if ev else "")

        edge_cids: Set[str] = set()
        if hid and tid:
            if index.graph.has_edge(hid, tid):
                edge_cids.update(index.graph[hid][tid].get("chunks", ()))
            if index.graph.has_edge(tid, hid):
                edge_cids.update(index.graph[tid][hid].get("chunks", ()))
            if not edge_cids:
                h_c = index.graph.nodes[hid].get("chunks", set()) if index.graph.has_node(hid) else set()
                t_c = index.graph.nodes[tid].get("chunks", set()) if index.graph.has_node(tid) else set()
                if len(h_c) <= hub_cap and len(t_c) <= hub_cap:
                    edge_cids = set(h_c) & set(t_c)

        if not edge_cids:
            continue

        both_seeds = 1.35 if (hid in seed_eids and tid in seed_eids) else 1.0
        base_edge_score = float(edge.get("score", 0.80)) * rel_w * both_seeds
        for cid in edge_cids:
            chunk_edge_scores[cid].append(0.45 * base_edge_score)
            chunk_lr_triples[cid].append((base_edge_score * 2.0, t_text))

    for cid, e_list in chunk_edge_scores.items():
        lr_chunk_scores[cid] += sum(sorted(e_list, reverse=True)[:3])

    doc_summary_map: Dict[str, Tuple[float, str]] = {}
    if high_level_blocks:
        title_map = index._get_title_to_doc_ids()
        for hl in high_level_blocks:
            hl_title_low = str(hl.get("title") or "").strip().casefold()
            hl_doc_low = str(hl.get("doc_id") or "").strip().casefold()
            hl_score = float(hl.get("score") or 0.0)
            hl_sum = str(hl.get("summary") or "").strip()
            matched_dids = set(title_map.get(hl_title_low, ())) | set(title_map.get(hl_doc_low, ()))
            for did in matched_dids:
                if hl_score > doc_summary_map.get(did, (0.0, ""))[0]:
                    doc_summary_map[did] = (hl_score, hl_sum)

    for cid in list(lr_chunk_scores.keys()):
        distinct_ents = len({ename for ename, _ in chunk_matched_entities.get(cid, ())})
        if distinct_ents >= 2:
            lr_chunk_scores[cid] += 0.22 * min(3, distinct_ents - 1)
        matched_seed_cnt = len(chunk_matched_seeds.get(cid, ()))
        if matched_seed_cnt >= 2:
            lr_chunk_scores[cid] *= 1.0 + 0.55 * min(3, matched_seed_cnt - 1)
        idx = index.chunk_id_to_idx.get(cid)
        if idx is not None and doc_summary_map:
            ch = index.chunks[idx]
            did = str(ch.document_id or (ch.metadata or {}).get("document_id") or "")
            if did in doc_summary_map:
                lr_chunk_scores[cid] += 0.20 * doc_summary_map[did][0]

    doc_neighbors = getattr(index, "doc_chunk_neighbors", {})
    if lr_chunk_scores and adjacent_alpha > 0.0 and doc_neighbors:
        top_lr_pool = heapq.nlargest(max(top_k, 30), lr_chunk_scores.items(), key=lambda x: x[1])
        for cid, val in top_lr_pool:
            delta = adjacent_alpha * val
            for nbr_cid in doc_neighbors.get(cid, ()):
                lr_chunk_scores[nbr_cid] = lr_chunk_scores.get(nbr_cid, 0.0) + delta

    supp_hits: List[RetrievedContext] = []
    if not lr_chunk_scores or float(supplement_weight) > 0.0:
        supp_top_k = top_k if not lr_chunk_scores else min(top_k, 40)
        supp_hits = index.search(
            query=query,
            query_vector=query_vector,
            top_k=supp_top_k,
            seed_top_m=seed_top_m,
            max_hops=max_hops,
            hop_decay=hop_decay,
            adjacent_alpha=adjacent_alpha,
            filters=filters,
            extracted_entities=extracted_entities,
            high_level_themes=high_level_themes,
            skip_thesaurus_scan=bool(seed_entities),
        )

    if not lr_chunk_scores:
        for h in supp_hits:
            h.metadata["graph_engine"] = "regex_supplement_fallback"
        return supp_hits

    valid_lr_items = [
        (cid, sc)
        for cid, sc in lr_chunk_scores.items()
        if cid in index.chunk_id_to_idx and index._matches_filters(index.chunks[index.chunk_id_to_idx[cid]], filters)
    ]
    if not valid_lr_items:
        for h in supp_hits:
            h.metadata["graph_engine"] = "regex_supplement_fallback"
        return supp_hits

    max_lr = max(sc for _, sc in valid_lr_items) or 1.0
    combined_scores: Dict[str, float] = {cid: (sc / max_lr) for cid, sc in valid_lr_items}
    supp_meta_map: Dict[str, Dict[str, Any]] = {}

    has_hard_code = index.has_hard_code_entities(query)
    eff_supp_w = min(0.85, max(0.0, float(supplement_weight)) + (0.15 if has_hard_code else 0.0))

    for sh in supp_hits:
        scid = sh.chunk_id
        supp_meta_map[scid] = sh.metadata or {}
        s_val = eff_supp_w * float(sh.similarity_score)
        if scid in combined_scores:
            lr_val = combined_scores[scid]
            combined_scores[scid] = max(lr_val, s_val) + 0.20 * min(lr_val, s_val)
        else:
            combined_scores[scid] = s_val

    target_faculties = detect_query_faculties(query)
    if target_faculties:
        for cid in list(combined_scores.keys()):
            idx = index.chunk_id_to_idx.get(cid)
            if idx is None:
                continue
            ch_meta = index.chunks[idx].metadata or {}
            ch_fac = str(ch_meta.get("faculty") or ch_meta.get("site") or "").strip().lower()
            if ch_fac in target_faculties:
                combined_scores[cid] *= 1.15
            elif ch_fac in KNOWN_FACULTY_KEYS:
                combined_scores[cid] *= 0.35

    ranked_cids = heapq.nlargest(top_k, combined_scores.items(), key=lambda x: x[1])
    top_max = ranked_cids[0][1] if ranked_cids and ranked_cids[0][1] > 0 else 1.0

    results: List[RetrievedContext] = []
    summary_injected_docs: Set[str] = set()
    for rank, (cid, raw_comb) in enumerate(ranked_cids, start=1):
        chunk = index.chunks[index.chunk_id_to_idx[cid]]
        norm_score = round(min(1.0, max(0.01, raw_comb / top_max)), 4)
        supp_meta = supp_meta_map.get(cid, {})

        ent_names: List[str] = []
        seen_ents: Set[str] = set()
        for ename, _ in sorted(chunk_matched_entities.get(cid, []), key=lambda x: x[1], reverse=True):
            if ename not in seen_ents:
                seen_ents.add(ename)
                ent_names.append(ename)
                if len(ent_names) >= 6:
                    break
        for s_ename in (supp_meta.get("graph_entities") or []):
            if len(ent_names) >= 6:
                break
            if s_ename not in seen_ents:
                seen_ents.add(s_ename)
                ent_names.append(s_ename)

        triples_out: List[str] = []
        seen_triples: Set[str] = set()
        for _, t_txt in sorted(chunk_lr_triples.get(cid, []), key=lambda x: x[0], reverse=True):
            if t_txt not in seen_triples:
                seen_triples.add(t_txt)
                triples_out.append(t_txt)
                if len(triples_out) >= 5:
                    break
        for s_tr in (supp_meta.get("graph_triples") or []):
            if len(triples_out) >= 5:
                break
            if s_tr not in seen_triples and not str(s_tr).startswith("Phạm vi văn bản:"):
                seen_triples.add(s_tr)
                triples_out.append(s_tr)

        if not triples_out:
            for t_idx in index.chunk_to_triples.get(cid, ())[:3]:
                tr_rec = index.triple_records[t_idx]
                if tr_rec["text"] not in seen_triples:
                    seen_triples.add(tr_rec["text"])
                    triples_out.append(tr_rec["text"])

        ch_did = str(chunk.document_id or (chunk.metadata or {}).get("document_id") or "")
        if ch_did and ch_did in doc_summary_map and ch_did not in summary_injected_docs:
            s_scope = doc_summary_map[ch_did][1]
            if s_scope and len(triples_out) < 5:
                triples_out.insert(0, f"Phạm vi văn bản: {s_scope[:140]}")
                summary_injected_docs.add(ch_did)

        is_lr_primary = cid in lr_chunk_scores
        meta = dict(chunk.metadata) if chunk.metadata else {}
        meta["search_source"] = "GraphRAG"
        meta["search_type"] = "graph"
        meta["graph_engine"] = "lightrag_primary" if is_lr_primary else "regex_supplement"
        meta["lightrag_grounded"] = is_lr_primary
        meta["graph_score"] = round(raw_comb, 4)
        meta["graph_entities"] = ent_names
        meta["graph_triples"] = triples_out

        results.append(
            RetrievedContext(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                similarity_score=norm_score,
                rank=rank,
                metadata=meta,
            )
        )

    return results

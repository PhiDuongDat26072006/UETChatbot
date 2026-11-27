"""
src/graph/lightrag_matching.py - Entity matching (thesaurus n-grams, graph-node n-grams,
vector resolution), weighted spreading activation, and high-level summary scoring
for LightRAGRetriever.
"""

from __future__ import annotations

import heapq
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.graph.query_entity_extractor import is_forbidden_entity
from src.graph.thesaurus_utils import detect_query_faculties
from src.utils.helpers import get_logger

logger = get_logger("graph.lightrag_retriever")

_WORD_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


def scan_thesaurus_ngrams(retriever: Any, query_text: str) -> List[Tuple[str, float]]:
    """
    Quét n-grams nguyên từ (từ dài xuống ngắn, không chồng lấp) qua Academic Thesaurus.
    Loại bỏ hoàn toàn lỗi bắt nhầm chuỗi con (như bắt 'Thông tin' trong 'Công nghệ thông tin').
    """
    tokens = _WORD_TOKEN_RE.findall(query_text.lower())
    if not tokens:
        return []

    covered_positions: Set[int] = set()
    found: Dict[str, float] = {}

    for n in range(min(6, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            span = set(range(i, i + n))
            if span & covered_positions:
                continue
            phrase = " ".join(tokens[i : i + n])
            if is_forbidden_entity(phrase):
                continue
            canonical = retriever.thesaurus.get_canonical(phrase)
            if canonical and not is_forbidden_entity(canonical):
                covered_positions.update(span)
                score = 1.0 if phrase in retriever.thesaurus.get_aliases(canonical) else 0.95
                found[canonical] = max(found.get(canonical, 0.0), score)

    return sorted(found.items(), key=lambda x: x[1], reverse=True)


def scan_graph_node_ngrams(retriever: Any, query_text: str) -> List[Tuple[str, float]]:
    """
    Quét n-grams đa từ (3..10 từ, từ dài xuống ngắn) khớp chính xác tên node nguyên gốc trong graph.json.
    """
    if not retriever._lower_to_canonical:
        return []
    tokens = _WORD_TOKEN_RE.findall(query_text.lower())
    if len(tokens) < 3:
        return []

    covered_positions: Set[int] = set()
    found: Dict[str, float] = {}
    max_n = min(10, len(tokens))

    for n in range(max_n, 2, -1):
        for i in range(len(tokens) - n + 1):
            span = set(range(i, i + n))
            overlap_cnt = len(span & covered_positions)
            if overlap_cnt > 1 or (n - overlap_cnt) < 2:
                continue
            phrase = " ".join(tokens[i : i + n])
            if len(phrase) < 10 or is_forbidden_entity(phrase):
                continue
            node_name = retriever._lower_to_canonical.get(phrase)
            if not node_name or is_forbidden_entity(node_name) or retriever._is_hub_node(node_name):
                continue
            covered_positions.update(span)
            score = 0.96 if n >= 4 else 0.92
            found[node_name] = max(found.get(node_name, 0.0), score)

    return sorted(found.items(), key=lambda x: x[1], reverse=True)


def resolve_term_in_graph(retriever: Any, term: str) -> Optional[str]:
    """Khớp chính xác tên chuẩn hóa hoặc alias sạch vào tập node thực tế của graph.json."""
    low = term.strip().lower()
    if not low or is_forbidden_entity(low):
        return None
    if low in retriever._lower_to_canonical:
        return retriever._lower_to_canonical[low]
    if low in retriever._alias_to_graph_entity:
        return retriever._alias_to_graph_entity[low]
    return None


def search_entity_vectors(
    retriever: Any,
    terms: List[str],
    threshold: Optional[float] = None,
    precomputed_query_text: Optional[str] = None,
    precomputed_query_vector: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Tìm kiếm ngữ nghĩa các thực thể:
      - Tái sử dụng query_vector đã nhúng sẵn từ UETRetriever nếu term trùng câu hỏi gốc.
      - Ưu tiên nhánh ma trận NumPy trong RAM (<2ms, nạp từ entity_chroma).
      - Fallback sang truy vấn trực tiếp _entity_collection (hỗ trợ mock collection trong unit test).
    """
    if not terms:
        return []
    if retriever._entity_vec_matrix is None and retriever._entity_collection is None:
        return []

    min_sim = retriever.entity_similarity_threshold if threshold is None else float(threshold)
    try:
        vectors: List[List[float]] = []
        if (
            precomputed_query_vector is not None
            and precomputed_query_text
            and len(terms) == 1
            and terms[0].strip() == precomputed_query_text.strip()
        ):
            vectors = [list(precomputed_query_vector)]
        elif retriever.embedding_model is not None:
            if precomputed_query_vector is not None and precomputed_query_text:
                to_embed = [t for t in terms if t.strip() != precomputed_query_text.strip()]
                embedded_map: Dict[str, List[float]] = {}
                if to_embed:
                    for t_str, emb_v in zip(to_embed, retriever.embedding_model.embed_texts(to_embed)):
                        embedded_map[t_str] = emb_v
                for t in terms:
                    if t.strip() == precomputed_query_text.strip():
                        vectors.append(list(precomputed_query_vector))
                    elif t in embedded_map:
                        vectors.append(embedded_map[t])
            else:
                vectors = retriever.embedding_model.embed_texts(terms)
        if not vectors:
            return []

        # Nhánh 1: Ma trận NumPy RAM siêu tốc
        if (
            retriever._entity_vec_matrix is not None
            and len(retriever._entity_vec_names) == retriever._entity_vec_matrix.shape[0]
        ):
            q_arr = np.asarray(vectors, dtype=np.float32)
            if q_arr.ndim == 2 and q_arr.shape[1] == retriever._entity_vec_matrix.shape[1]:
                norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
                norms = np.where(norms == 0.0, 1.0, norms)
                q_norm = q_arr / norms
                sims_mat = q_norm @ retriever._entity_vec_matrix.T
                k_pick = min(max(1, retriever.entity_vector_top_k), sims_mat.shape[1])

                matched_ram: Dict[str, float] = {}
                for row in sims_mat:
                    top_idx = np.argpartition(row, -k_pick)[-k_pick:]
                    for idx in top_idx:
                        sim = float(row[int(idx)])
                        if sim >= min_sim:
                            node_name = retriever._entity_vec_names[int(idx)]
                            score = round(min(0.94, sim), 4)
                            if score > matched_ram.get(node_name, 0.0):
                                matched_ram[node_name] = score
                return sorted(matched_ram.items(), key=lambda x: x[1], reverse=True)

        # Nhánh 2: Truy vấn trực tiếp ChromaDB collection (hoặc mock DummyEntityCollection)
        if retriever._entity_collection is None:
            return []

        res = retriever._entity_collection.query(
            query_embeddings=vectors,
            n_results=max(1, retriever.entity_vector_top_k),
            include=["metadatas", "distances"],
        )
        metadatas_list = res.get("metadatas") or []
        distances_list = res.get("distances") or []

        matched: Dict[str, float] = {}
        for metas, dists in zip(metadatas_list, distances_list):
            for meta, dist in zip(metas or [], dists or []):
                if not isinstance(meta, dict):
                    continue
                cand_name = str(meta.get("canonical_name") or "").strip()
                if not cand_name or is_forbidden_entity(cand_name):
                    continue
                resolved = retriever._resolve_term_in_graph(cand_name)
                if not resolved:
                    continue
                sim = 1.0 - float(dist)
                if sim >= min_sim:
                    score = round(min(0.94, sim), 4)
                    if score > matched.get(resolved, 0.0):
                        matched[resolved] = score

        return sorted(matched.items(), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.warning(f"[LightRAGRetriever] Lỗi truy vấn vector thực thể: {e}")
        return []


def extract_and_match_entities(
    retriever: Any,
    query_text: str,
    pre_extracted_entities: Optional[List[str]] = None,
    pre_extracted_themes: Optional[List[str]] = None,
    allow_llm: Optional[bool] = None,
    query_vector: Optional[List[float]] = None,
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Phễu 3 bước chuẩn LightRAG (PA1 + Graceful Fallback PA2):
      1. Trích xuất thực thể & chủ đề bằng LLM (hoặc nhận kết quả PA1 đã chạy song song) + quét Academic Thesaurus.
      2. Lọc thực thể rác (FORBIDDEN_ENTITIES) & chuẩn hóa tên về Canonical Name.
      3. Ánh xạ vào graph.json: ưu tiên Exact Match (O(1)), phần còn lại Vector Search trên RAM NumPy / entity_chroma.
    Trả về: (seed_entities, high_level_themes).
    """
    clean_query = (query_text or "").strip()
    if not clean_query:
        return [], []

    matched: Dict[str, float] = {}
    unmatched_for_vector: List[str] = []
    high_level_themes: List[str] = []
    used_llm = False

    # Bước 1A: Dùng kết quả PA1 truyền sẵn hoặc gọi QueryEntityExtractor
    raw_low_level: List[str] = []
    if pre_extracted_entities is not None or pre_extracted_themes is not None:
        raw_low_level = list(pre_extracted_entities or [])
        high_level_themes = list(pre_extracted_themes or [])
        used_llm = bool(raw_low_level or high_level_themes)
    else:
        should_call_llm = retriever.use_llm_query_extraction if allow_llm is None else bool(allow_llm)
        if should_call_llm and retriever.query_extractor is not None:
            llm_res = retriever.query_extractor.extract(clean_query)
            used_llm = bool(llm_res.get("used_llm", False))
            raw_low_level = list(llm_res.get("low_level_entities") or [])
            high_level_themes = list(llm_res.get("high_level_themes") or [])

    for idx, raw_ent in enumerate(raw_low_level):
        if is_forbidden_entity(raw_ent):
            continue
        canonical = retriever.thesaurus.normalize(raw_ent)
        if is_forbidden_entity(canonical):
            continue

        base_conf = round(max(0.88, 1.0 - idx * 0.02), 4)
        graph_node = retriever._resolve_term_in_graph(canonical) or retriever._resolve_term_in_graph(raw_ent)
        if graph_node:
            matched[graph_node] = max(matched.get(graph_node, 0.0), base_conf)
        elif canonical not in unmatched_for_vector:
            unmatched_for_vector.append(canonical)

    theme_vector_fallbacks: List[str] = []
    for raw_theme in high_level_themes:
        norm_theme = retriever.thesaurus.normalize(raw_theme)
        if not is_forbidden_entity(norm_theme):
            graph_node = retriever._resolve_term_in_graph(norm_theme)
            if graph_node:
                matched[graph_node] = max(matched.get(graph_node, 0.0), 0.85)
            elif norm_theme not in theme_vector_fallbacks:
                theme_vector_fallbacks.append(norm_theme)

    # Bước 1B: Quét n-gram nguyên cụm qua Academic Thesaurus
    for canonical_th, th_score in retriever._scan_thesaurus_ngrams(clean_query):
        graph_node = retriever._resolve_term_in_graph(canonical_th)
        if graph_node:
            matched[graph_node] = max(matched.get(graph_node, 0.0), th_score)
        elif canonical_th not in unmatched_for_vector:
            unmatched_for_vector.append(canonical_th)

    # Bước 1C: Quét n-gram đa từ khớp trực tiếp tên node trong graph.json
    for node_name, gn_score in retriever._scan_graph_node_ngrams(clean_query):
        matched[node_name] = max(matched.get(node_name, 0.0), gn_score)

    # Bước 2: Semantic Entity Search trên ma trận RAM NumPy / entity_chroma
    vector_lookup_terms = list(unmatched_for_vector)
    if not matched and not vector_lookup_terms:
        vector_lookup_terms.extend(theme_vector_fallbacks[:2])
    if (not used_llm or len(matched) == 0) and clean_query not in vector_lookup_terms:
        vector_lookup_terms.append(clean_query)

    if vector_lookup_terms:
        vec_matches = retriever._search_entity_vectors(
            vector_lookup_terms,
            precomputed_query_text=clean_query,
            precomputed_query_vector=query_vector,
        )
        for node_name, vec_score in vec_matches:
            matched[node_name] = max(matched.get(node_name, 0.0), vec_score)

    sorted_seeds = sorted(matched.items(), key=lambda x: x[1], reverse=True)
    return sorted_seeds[: retriever.max_entities_per_query], high_level_themes


def extract_low_level_subgraph_detailed(
    retriever: Any,
    seed_entities: List[Tuple[str, float]],
    depth: int = 1,
    relation_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]]]:
    """
    Trích xuất đồ thị con cục bộ bằng Lan truyền Kích hoạt có Trọng số (Weighted Spreading Activation):
      - Tích hợp Entity IDF, Hub Gating, trọng số quan hệ pháp quy
        và điểm thưởng đồng kích hoạt (Co-Activation Bonus).
    Trả về: (blocks, activated_nodes, scored_edges).
    """
    rel_weights = relation_weights or {}
    valid_seeds = [(s, float(sc)) for s, sc in seed_entities if s in retriever.entities]
    if not valid_seeds:
        return [], {}, []

    seed_set: Set[str] = {s for s, _ in valid_seeds}
    activated_nodes: Dict[str, float] = {}
    frontier: Dict[str, float] = {}

    for s_name, s_score in valid_seeds:
        weighted_seed = round(s_score * max(0.55, retriever._effective_node_idf(s_name)), 4)
        activated_nodes[s_name] = weighted_seed
        frontier[s_name] = weighted_seed

    max_hops = max(0, int(depth))
    hop_decay_base = 0.68
    raw_edges_seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        decay = hop_decay_base ** hop
        next_frontier: Dict[str, float] = {}

        for u, u_score in frontier.items():
            if hop > 1 and retriever._is_hub_node(u):
                continue
            out_rels = retriever.adjacency.get(u, [])
            in_rels = retriever.reverse_adjacency.get(u, [])
            total_deg = len(out_rels) + len(in_rels)
            if total_deg == 0 or (u not in seed_set and retriever._is_hub_node(u)):
                continue

            damping = 1.0 / math.sqrt(max(1, min(total_deg, 64)))

            for rel in out_rels:
                tgt = rel.get("target")
                if not tgt or tgt not in retriever.entities:
                    continue
                rtype = str(rel.get("relation_type") or rel.get("predicate") or "liên_quan_đến").lower()
                rel_w = rel_weights.get(rtype, 0.95)
                raw_edges_seen.setdefault((u, rtype, tgt), rel)
                if retriever._is_hub_node(tgt) and tgt not in seed_set:
                    continue
                prop = u_score * decay * damping * rel_w * retriever._effective_node_idf(tgt)
                if prop > activated_nodes.get(tgt, 0.0):
                    activated_nodes[tgt] = round(prop, 4)
                    next_frontier[tgt] = max(next_frontier.get(tgt, 0.0), prop)

            for rel in in_rels:
                src = rel.get("source")
                if not src or src not in retriever.entities:
                    continue
                rtype = str(rel.get("relation_type") or rel.get("predicate") or "liên_quan_đến").lower()
                rel_w = rel_weights.get(rtype, 0.95)
                raw_edges_seen.setdefault((src, rtype, u), rel)
                if retriever._is_hub_node(src) and src not in seed_set:
                    continue
                prop = u_score * decay * damping * rel_w * retriever._effective_node_idf(src)
                if prop > activated_nodes.get(src, 0.0):
                    activated_nodes[src] = round(prop, 4)
                    next_frontier[src] = max(next_frontier.get(src, 0.0), prop)

        if len(next_frontier) > retriever.max_entities_per_query * 3:
            top_f = heapq.nlargest(
                retriever.max_entities_per_query * 3, next_frontier.items(), key=lambda x: x[1]
            )
            frontier = dict(top_f)
        else:
            frontier = next_frontier

    # Chấm điểm từng cạnh dựa trên Co-Activation và trọng số quan hệ pháp quy
    scored_edges: List[Dict[str, Any]] = []
    node_co_bonus: Dict[str, float] = {}
    for (src, rtype, tgt), rel in raw_edges_seen.items():
        s_act = activated_nodes.get(src, 0.0)
        t_act = activated_nodes.get(tgt, 0.0)
        if s_act <= 0.0 and t_act <= 0.0:
            continue
        rel_w = rel_weights.get(rtype, 0.95)
        both_seed_mult = (
            1.35 if (src in seed_set and tgt in seed_set)
            else (1.15 if (s_act > 0 and t_act > 0) else 1.0)
        )
        e_score = round(0.5 * (s_act + t_act) * rel_w * both_seed_mult, 4)
        enriched_edge = dict(rel)
        enriched_edge["score"] = e_score
        scored_edges.append(enriched_edge)

        if src in seed_set and tgt in seed_set:
            node_co_bonus[src] = node_co_bonus.get(src, 0.0) + 0.12 * rel_w
            node_co_bonus[tgt] = node_co_bonus.get(tgt, 0.0) + 0.12 * rel_w

    scored_edges.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    # Chọn danh sách thực thể tiêu biểu để dựng Low-level blocks
    ordered_candidates: List[Tuple[str, float]] = []
    seen_block_nodes: Set[str] = set()
    for seed, orig_score in valid_seeds:
        final_s = min(0.99, round(orig_score + node_co_bonus.get(seed, 0.0), 4))
        ordered_candidates.append((seed, final_s))
        seen_block_nodes.add(seed)

    if len(ordered_candidates) < retriever.max_entities_per_query:
        extra_slots = retriever.max_entities_per_query - len(ordered_candidates)
        bridge_nodes = [
            (n, sc) for n, sc in activated_nodes.items() if n not in seen_block_nodes and sc >= 0.25
        ]
        bridge_nodes.sort(key=lambda x: x[1], reverse=True)
        for b_node, b_sc in bridge_nodes[: min(4, extra_slots)]:
            ordered_candidates.append((b_node, round(min(0.86, b_sc), 4)))
            seen_block_nodes.add(b_node)

    blocks: List[Dict[str, Any]] = []
    for node_name, rel_score in ordered_candidates:
        node_info = retriever.entities.get(node_name)
        if not node_info:
            continue

        direct_outgoing = [r for r in scored_edges if r.get("source") == node_name]
        direct_incoming = [r for r in scored_edges if r.get("target") == node_name]

        ent_cat = node_info.get("entity_type") or node_info.get("category", "GENERAL")
        descriptions = node_info.get("descriptions", [])
        if isinstance(descriptions, list):
            ent_desc = " | ".join(descriptions) if descriptions else node_info.get("description", "")
        else:
            ent_desc = str(descriptions)

        blocks.append({
            "entity": node_name,
            "category": ent_cat,
            "description": ent_desc,
            "outgoing_relations": direct_outgoing,
            "incoming_relations": direct_incoming,
            "relevance_score": rel_score,
        })

    blocks.sort(key=lambda b: float(b.get("relevance_score", 0.0)), reverse=True)
    return blocks, activated_nodes, scored_edges


def extract_high_level_summaries(
    retriever: Any,
    query_text: str,
    seed_entities: List[Tuple[str, float]],
    high_level_themes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Trích xuất tóm tắt chủ đề tài liệu toàn cục (High-level Retrieval).
    Kết hợp cả high_level_themes do LLM bóc tách và thực thể hạt giống.
    """
    if not retriever.document_summaries:
        return []

    query_lower = query_text.lower()
    target_faculties = detect_query_faculties(query_text)
    seed_names = {s.lower() for s, _ in seed_entities}
    themes_lower = [
        retriever.thesaurus.normalize(t).lower()
        for t in (high_level_themes or [])
        if not is_forbidden_entity(t)
    ]

    matched_summaries: List[Tuple[str, Dict[str, Any], float]] = []

    for doc_id, summary_info in retriever.document_summaries.items():
        doc_title = summary_info.get("title", "")
        doc_summary = summary_info.get("summary", "")
        title_low = doc_title.lower()
        summary_low = doc_summary.lower()
        key_entities = {e.lower() for e in summary_info.get("key_entities", [])}

        # Lọc/giảm mạnh tóm tắt thuộc Khoa/Viện khác khi câu hỏi nêu đích danh Khoa/Viện cụ thể
        faculty_mult = 1.0
        if target_faculties:
            summary_blob = f"{doc_title} {doc_summary} {' '.join(summary_info.get('key_entities', []))}"
            summary_faculties = detect_query_faculties(summary_blob)
            if summary_faculties:
                if summary_faculties & target_faculties:
                    faculty_mult = 1.25
                else:
                    faculty_mult = 0.15

        score = 0.0
        if title_low and title_low in query_lower:
            score += 0.5

        for theme in themes_lower:
            if theme in title_low:
                score += 0.45
            elif theme in summary_low:
                score += 0.25
            if theme in key_entities:
                score += 0.30

        overlap = key_entities.intersection(seed_names)
        score += len(overlap) * 0.30

        for word in _WORD_TOKEN_RE.findall(query_lower):
            if len(word) > 3 and not is_forbidden_entity(word) and word in summary_low:
                score += 0.04

        score *= faculty_mult
        if score > 0.15:
            matched_summaries.append((doc_id, summary_info, min(score, 0.96)))

    matched_summaries.sort(key=lambda x: x[2], reverse=True)
    return [
        {
            "doc_id": doc_id,
            "title": s.get("title", ""),
            "summary": s.get("summary", ""),
            "document_type": s.get("document_type", ""),
            "key_entities": s.get("key_entities", []),
            "score": score,
        }
        for doc_id, s, score in matched_summaries[:3]
    ]

"""
src/graph/lightrag_formatter.py - Edge formatting, synthesis overview generation,
and dual-level context formatting for LightRAGRetriever.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.base import RetrievedContext


def format_edge_line(rel: Dict[str, Any], max_ev_chars: int = 110) -> str:
    """Format one graph relation edge as an indented bullet line."""
    rel_type = rel.get("relation_type") or rel.get("predicate", "liên_quan")
    src = rel.get("source", "")
    tgt = rel.get("target", "")
    desc = rel.get("description") or (
        rel.get("evidence_texts", [""])[0]
        if rel.get("evidence_texts")
        else rel.get("evidence", "")
    )
    desc_clean = re.sub(r"\s+", " ", str(desc or "").strip())[:max_ev_chars]
    desc_part = f" (Căn cứ: {desc_clean})" if desc_clean else ""
    return f"  └─ [{src}] ──({rel_type})──> [{tgt}]{desc_part}"


def build_synthesis_overview(
    retriever: Any,
    high_level_blocks: List[Dict[str, Any]],
    low_level_blocks: List[Dict[str, Any]],
    max_chars: Optional[int] = None,
) -> Optional[str]:
    """
    Tổng hợp Bản đồ Liên kết Quy chế 2 tầng (High-level + Low-level) gọn nhẹ (~150-220 từ)
    để đính kèm trực tiếp vào vùng ngữ cảnh của LLM cho các câu hỏi quan hệ/tổng hợp.
    """
    limit = max(250, int(max_chars if max_chars is not None else retriever.synthesis_max_chars))
    lines: List[str] = []

    for hl in (high_level_blocks or [])[:2]:
        title = str(hl.get("title") or "").strip()
        summary = re.sub(r"\s+", " ", str(hl.get("summary") or "").strip())[:160]
        if title and summary:
            lines.append(f"• Phạm vi văn bản [{title}]: {summary}")

    seen_edges: Set[str] = set()
    added_entities = 0
    for ll in (low_level_blocks or []):
        if added_entities >= 3:
            break
        ent_name = str(ll.get("entity") or "").strip()
        ent_desc = re.sub(r"\s+", " ", str(ll.get("description") or "").strip())[:160]
        rels = list(ll.get("outgoing_relations") or [])[:2] + list(ll.get("incoming_relations") or [])[:2]
        if not ent_desc and not rels:
            continue

        if ent_desc:
            lines.append(f"• Thực thể [{ent_name}]: {ent_desc}")
        else:
            lines.append(f"• Thực thể [{ent_name}]:")

        for r in rels[:3]:
            edge_line = retriever._format_edge_line(r, max_ev_chars=95)
            if edge_line not in seen_edges:
                seen_edges.add(edge_line)
                lines.append(edge_line)
        added_entities += 1

    if not lines:
        return None

    assembled: List[str] = []
    curr_len = 0
    for ln in lines:
        if curr_len + len(ln) + 1 > limit and assembled:
            break
        assembled.append(ln)
        curr_len += len(ln) + 1

    return "\n".join(assembled).strip() or None


def format_dual_level_contexts(
    retriever: Any,
    high_level_blocks: List[Dict[str, Any]],
    low_level_blocks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[RetrievedContext]:
    """Đóng gói các khối High-level và Low-level thành List[RetrievedContext] độc lập."""
    contexts: List[RetrievedContext] = []
    rank_counter = 1

    for hl in high_level_blocks:
        entities_str = ", ".join(hl["key_entities"][:6]) if hl["key_entities"] else "N/A"
        text_block = (
            f"【TỔNG QUAN QUY CHẾ / VĂN BẢN ĐÀO TẠO UET】\n"
            f"• Văn bản: {hl['title']}\n"
            f"• Phân loại: {hl['document_type']}\n"
            f"• Các thực thể trọng tâm: {entities_str}\n"
            f"• Nội dung bao quát: {hl['summary']}"
        )
        contexts.append(
            RetrievedContext(
                chunk_id=f"lightrag_high_{hl['doc_id']}",
                text=text_block,
                similarity_score=round(hl["score"], 4),
                rank=rank_counter,
                metadata={
                    "source": "knowledge_graph_high_level",
                    "doc_id": hl["doc_id"],
                    "document_type": hl["document_type"],
                    "retrieval_tier": "high_level",
                },
            )
        )
        rank_counter += 1

    for ll in low_level_blocks:
        ent_name = ll["entity"]
        ent_cat = ll["category"]
        ent_desc = ll["description"]

        lines = [
            f"【TRI THỨC LIÊN KẾT HỌC VỤ UET: {ent_name.upper()}】",
            f"• Danh mục: {ent_cat}",
            f"• Định nghĩa / Quy định: {ent_desc}",
        ]

        relations_text = []
        for out_rel in ll["outgoing_relations"][:5]:
            relations_text.append(retriever._format_edge_line(out_rel, max_ev_chars=140))
        for in_rel in ll["incoming_relations"][:5]:
            relations_text.append(retriever._format_edge_line(in_rel, max_ev_chars=140))

        if relations_text:
            lines.append("• Các mối liên kết quy chế liên quan:")
            lines.extend(relations_text)

        formatted_text = "\n".join(lines)
        contexts.append(
            RetrievedContext(
                chunk_id=f"lightrag_low_{ent_name}",
                text=formatted_text,
                similarity_score=round(ll["relevance_score"], 4),
                rank=rank_counter,
                metadata={
                    "source": "knowledge_graph_low_level",
                    "entity": ent_name,
                    "category": ent_cat,
                    "retrieval_tier": "low_level",
                },
            )
        )
        rank_counter += 1

    contexts.sort(key=lambda x: x.similarity_score, reverse=True)
    final_contexts = contexts[:top_k]
    for i, ctx in enumerate(final_contexts, 1):
        ctx.rank = i
    return final_contexts

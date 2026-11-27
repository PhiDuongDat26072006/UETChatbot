"""Chunk expansion (adjacent window & tabular siblings) and contiguous stitching."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.base import DataChunk, RetrievedContext
from src.utils.helpers import get_logger

logger = get_logger("retrieval")

_LIST_QUERY_RE = re.compile(
    r"(?:danh\s+sách|danh\s+mục|liệt\s+kê|toàn\s+bộ|tất\s+cả|những\s+ngành|các\s+ngành|"
    r"những\s+môn|các\s+môn|những\s+chương\s+trình|các\s+chương\s+trình|bao\s+gồm\s+những|"
    r"gồm\s+những|gồm\s+các|có\s+những|có\s+các|những\s+loại|các\s+loại|"
    r"bảng\s+điểm\s+chuẩn|bảng\s+học\s+phí|bảng\s+quy\s+đổi)",
    re.IGNORECASE,
)
_TABULAR_ROW_RE = re.compile(r"(?:\|\s*\*\*|^\s*-\s*\*\*[^*]+\*\*:|\|\s*---+\s*\|)", re.MULTILINE)
_META_LINE_RE = re.compile(r"^\s*(\[META:[^\]]+\])\s*$", re.MULTILINE)
_FOOTNOTE_URL_RE = re.compile(r"\[\[\d+\]\]\(https?://[^)]+\)")


def strip_chunk_header(text: str) -> str:
    """Lược bỏ dòng breadcrumb [Title] và [META: ...] lặp lại khi gộp chunk liền kề."""
    lines = (text or "").splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            idx += 1
            continue
        break
    body = "\n".join(lines[idx:]).strip()
    return body if body else (text or "").strip()


def is_list_query(query_text: str) -> bool:
    """Nhận diện câu hỏi mang ý định liệt kê danh sách / bảng đầy đủ."""
    return bool(_LIST_QUERY_RE.search(query_text or ""))


def extract_meta_header(text: str) -> str:
    """Trích xuất dòng [META: ...] (nếu có) để nhóm các chunk thuộc cùng một bảng."""
    match = _META_LINE_RE.search(text or "")
    return match.group(1).strip() if match else ""


def is_tabular_chunk(text: str) -> bool:
    """Nhận diện chunk chứa dòng bảng biểu / danh mục có cấu trúc."""
    return bool(_TABULAR_ROW_RE.search(strip_chunk_header(text)))


def compact_table_row(text: str) -> str:
    """Bỏ header lặp [Title]/[META: ...] và rút gọn link chú thích dài lặp lại trong từng dòng bảng."""
    body = strip_chunk_header(text)
    body = _FOOTNOTE_URL_RE.sub("", body)
    return re.sub(r"[ \t]{2,}", " ", body).strip()


def extract_header_lines(text: str) -> str:
    """Giữ lại các dòng tiêu đề [Title] và [META: ...] ở đầu khối bảng gộp."""
    headers: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            headers.append(stripped)
        else:
            break
    return "\n".join(headers)


def build_adjacent_lookup(chunks: List[DataChunk]) -> Dict[Tuple[str, int], DataChunk]:
    """Xây dựng bảng tra cứu (document_id, chunk_index) -> DataChunk."""
    lookup: Dict[Tuple[str, int], DataChunk] = {}
    for ch in chunks:
        meta = ch.metadata if isinstance(getattr(ch, "metadata", None), dict) else {}
        doc_id = getattr(ch, "document_id", None) or meta.get("document_id") or meta.get("doc_id")
        cidx = getattr(ch, "chunk_index", None)
        if cidx is None:
            cidx = meta.get("chunk_index")
        if doc_id and cidx is not None:
            try:
                lookup[(str(doc_id), int(cidx))] = ch
            except (TypeError, ValueError):
                continue
    return lookup


def build_doc_chunks_lookup(
    adjacent_lookup: Dict[Tuple[str, int], DataChunk],
) -> Dict[str, List[Tuple[int, DataChunk]]]:
    """Xây dựng bảng tra cứu document_id -> danh sách (chunk_index, DataChunk) đã sắp xếp."""
    grouped: Dict[str, List[Tuple[int, DataChunk]]] = {}
    for (doc_id, cidx), ch in adjacent_lookup.items():
        grouped.setdefault(doc_id, []).append((cidx, ch))
    for doc_id in grouped:
        grouped[doc_id].sort(key=lambda item: item[0])
    return grouped


def _find_tabular_run_around_seed(
    doc_chunks: List[Tuple[int, DataChunk]],
    center_idx: int,
    meta_hdr: str,
    max_chunks: int,
) -> List[DataChunk]:
    """Scan backward and forward around ``center_idx`` for contiguous rows sharing ``meta_hdr``."""
    pos = next((idx for idx, (ci, _) in enumerate(doc_chunks) if ci == center_idx), None)
    if pos is None:
        return []

    left = pos
    while left > 0:
        prev_ci, prev_ch = doc_chunks[left - 1]
        curr_ci, _ = doc_chunks[left]
        if (
            curr_ci - prev_ci <= 1
            and is_tabular_chunk(prev_ch.text)
            and extract_meta_header(prev_ch.text) == meta_hdr
        ):
            left -= 1
        else:
            break

    right = pos
    while right + 1 < len(doc_chunks):
        curr_ci, _ = doc_chunks[right]
        next_ci, next_ch = doc_chunks[right + 1]
        if (
            next_ci - curr_ci <= 1
            and is_tabular_chunk(next_ch.text)
            and extract_meta_header(next_ch.text) == meta_hdr
        ):
            right += 1
        else:
            break

    run_items = [ch for _, ch in doc_chunks[left : right + 1]]
    if len(run_items) < 2:
        return []

    if len(run_items) > max_chunks:
        rel_pos = pos - left
        half = max_chunks // 2
        start_i = max(0, min(rel_pos - half, len(run_items) - max_chunks))
        return run_items[start_i : start_i + max_chunks]
    return run_items


def _find_first_compact_table_run(
    doc_chunks: List[Tuple[int, DataChunk]],
    max_chunks: int,
) -> List[DataChunk]:
    """Find the first compact contiguous table run (2..max_chunks) inside a document."""
    best_run: List[DataChunk] = []
    cur_run: List[DataChunk] = []
    cur_meta: Optional[str] = None
    prev_ci: Optional[int] = None

    for ci, ch in doc_chunks:
        if is_tabular_chunk(ch.text):
            mh = extract_meta_header(ch.text)
            if cur_run and prev_ci is not None and ci - prev_ci <= 1 and mh == cur_meta:
                cur_run.append(ch)
            else:
                if 2 <= len(cur_run) <= max_chunks:
                    best_run = cur_run
                    break
                cur_run = [ch]
                cur_meta = mh
            prev_ci = ci
        else:
            if 2 <= len(cur_run) <= max_chunks:
                best_run = cur_run
                break
            cur_run = []
            cur_meta = None
            prev_ci = None

    if not best_run and 2 <= len(cur_run) <= max_chunks:
        best_run = cur_run
    return best_run


def expand_tabular_siblings(
    query_text: str,
    final_contexts: List[RetrievedContext],
    doc_lookup: Dict[str, List[Tuple[int, DataChunk]]],
    *,
    enabled: bool,
    max_chunks: int,
    max_chars: int,
) -> List[RetrievedContext]:
    """
    Tự động gom các chunk anh em liền kề cùng bảng khi câu hỏi mang ý định liệt kê danh sách đầy đủ.
    Giữ nguyên chunk_id, rank và độ dài final_contexts để không làm sai lệch IR eval.
    """
    if not enabled or not final_contexts or not is_list_query(query_text) or not doc_lookup:
        return final_contexts

    eff_max_chunks = max(2, int(max_chunks))
    eff_max_chars = max(1500, int(max_chars))
    max_tables = 2

    docs_with_tabular_seed = {
        str((ctx.metadata or {}).get("document_id") or (ctx.metadata or {}).get("doc_id") or "")
        for ctx in final_contexts
        if is_tabular_chunk(ctx.text)
    } - {""}

    expanded_tables: set = set()
    expanded_results = list(final_contexts)
    tables_expanded = 0

    for i, ctx in enumerate(expanded_results):
        if tables_expanded >= max_tables:
            break

        meta = dict(ctx.metadata) if ctx.metadata else {}
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or "")
        cidx_raw = meta.get("chunk_index")
        if not doc_id or doc_id not in doc_lookup:
            continue

        doc_chunks = doc_lookup[doc_id]
        sibling_run: List[DataChunk] = []
        table_key: Optional[Tuple[str, str]] = None
        prepend_intro = False

        if is_tabular_chunk(ctx.text) and cidx_raw is not None:
            try:
                center_idx = int(cidx_raw)
            except (TypeError, ValueError):
                continue

            meta_hdr = extract_meta_header(ctx.text)
            table_key = (doc_id, meta_hdr)
            if table_key in expanded_tables:
                continue
            sibling_run = _find_tabular_run_around_seed(doc_chunks, center_idx, meta_hdr, eff_max_chunks)

        elif doc_id not in docs_with_tabular_seed:
            best_run = _find_first_compact_table_run(doc_chunks, eff_max_chunks)
            if not best_run:
                continue
            meta_hdr = extract_meta_header(best_run[0].text)
            table_key = (doc_id, meta_hdr)
            if table_key in expanded_tables:
                continue
            sibling_run = best_run
            prepend_intro = True

        if not sibling_run or table_key is None:
            continue

        parts: List[str] = []
        if prepend_intro:
            parts.append(ctx.text.strip())
            hdr = extract_meta_header(sibling_run[0].text)
            if hdr:
                parts.append(hdr)
        else:
            hdr_block = extract_header_lines(sibling_run[0].text)
            if hdr_block:
                parts.append(hdr_block)

        total_len = sum(len(p) for p in parts)
        merged_ids: List[str] = [ctx.chunk_id] if prepend_intro else []
        seen_bodies: set = set()

        for ch in sibling_run:
            row_body = compact_table_row(ch.text)
            if not row_body or row_body in seen_bodies:
                continue
            if total_len + len(row_body) + 1 > eff_max_chars and len(merged_ids) >= 2:
                break
            parts.append(row_body)
            seen_bodies.add(row_body)
            if ch.chunk_id not in merged_ids:
                merged_ids.append(ch.chunk_id)
            total_len += len(row_body) + 1

        if len(merged_ids) >= 2:
            meta["tabular_expanded"] = True
            meta["expanded_chunk_ids"] = merged_ids
            meta["expanded_chunk_count"] = len(merged_ids)
            expanded_results[i] = RetrievedContext(
                chunk_id=ctx.chunk_id,
                text="\n".join(parts),
                similarity_score=ctx.similarity_score,
                rank=ctx.rank,
                metadata=meta,
            )
            expanded_tables.add(table_key)
            tables_expanded += 1
            logger.info(
                f"[TabularExpansion] Đã gộp {len(merged_ids)} chunks cùng bảng "
                f"(doc={doc_id[:8]}..., {total_len} ký tự) cho câu hỏi danh sách."
            )

    return expanded_results


def expand_adjacent_chunks(
    final_contexts: List[RetrievedContext],
    adjacent_lookup: Dict[Tuple[str, int], DataChunk],
    *,
    enabled: bool,
    expansion_top_k: int,
    window_size: int,
) -> List[RetrievedContext]:
    """Mở rộng ngữ cảnh liền kề (±window_size) cho Top-K chunks điểm cao nhất."""
    if not enabled or not final_contexts or not adjacent_lookup:
        return final_contexts

    window = max(1, int(window_size))
    expand_limit = min(max(0, int(expansion_top_k)), len(final_contexts))
    present_ids = {c.chunk_id for c in final_contexts}
    expanded_results = list(final_contexts)

    for i in range(expand_limit):
        ctx = expanded_results[i]
        meta = dict(ctx.metadata) if ctx.metadata else {}
        if meta.get("tabular_expanded"):
            continue
        doc_id = meta.get("document_id") or meta.get("doc_id")
        cidx_raw = meta.get("chunk_index")
        if not doc_id or cidx_raw is None:
            continue
        try:
            center_idx = int(cidx_raw)
        except (TypeError, ValueError):
            continue

        merged_parts: List[str] = []
        merged_ids: List[str] = []

        for offset in range(-window, window + 1):
            nb_idx = center_idx + offset
            if offset == 0:
                merged_parts.append(ctx.text)
                merged_ids.append(ctx.chunk_id)
                continue
            nb = adjacent_lookup.get((str(doc_id), nb_idx))
            if nb is None or nb.chunk_id in present_ids:
                continue
            nb_body = strip_chunk_header(nb.text)
            if nb_body and nb_body not in ctx.text:
                merged_parts.append(nb_body)
                merged_ids.append(nb.chunk_id)
                present_ids.add(nb.chunk_id)

        if len(merged_ids) > 1:
            meta["expanded_chunk_ids"] = merged_ids
            meta["expanded_chunk_count"] = len(merged_ids)
            expanded_results[i] = RetrievedContext(
                chunk_id=ctx.chunk_id,
                text="\n".join(merged_parts),
                similarity_score=ctx.similarity_score,
                rank=ctx.rank,
                metadata=meta,
            )

    return expanded_results


def stitch_contiguous_chunks(contexts: List[RetrievedContext]) -> List[RetrievedContext]:
    """Ghép nối các chunk liền kề thuộc cùng một tài liệu gốc."""
    if not contexts or len(contexts) < 2:
        return contexts

    doc_groups: Dict[Any, List[RetrievedContext]] = defaultdict(list)
    for ctx in contexts:
        doc_key = (
            ctx.metadata.get("document_id")
            or ctx.metadata.get("source_uri")
            or ctx.metadata.get("file_name")
            or ctx.metadata.get("title")
        )
        doc_groups[doc_key].append(ctx)

    stitched_results: List[RetrievedContext] = []
    for doc_key, group in doc_groups.items():
        if not doc_key:
            stitched_results.extend(group)
            continue

        group.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        merged: List[RetrievedContext] = []
        current = group[0]

        for nxt in group[1:]:
            curr_end_idx = current.metadata.get("chunk_index_end", current.metadata.get("chunk_index", 0))
            nxt_idx = nxt.metadata.get("chunk_index", 0)

            if nxt_idx == curr_end_idx + 1:
                new_text = current.text + "\n\n" + nxt.text
                best_score = max(current.similarity_score, nxt.similarity_score)
                best_rank = min(current.rank, nxt.rank)
                merged_meta = dict(current.metadata)
                merged_meta["chunk_index_end"] = nxt.metadata.get("chunk_index_end", nxt_idx)
                merged_meta["stitched"] = True

                current = RetrievedContext(
                    chunk_id=f"{current.chunk_id}+{nxt.chunk_id}",
                    text=new_text,
                    similarity_score=best_score,
                    rank=best_rank,
                    metadata=merged_meta,
                )
            else:
                merged.append(current)
                current = nxt

        merged.append(current)
        stitched_results.extend(merged)

    stitched_results.sort(key=lambda c: c.similarity_score, reverse=True)
    for i, ctx in enumerate(stitched_results, 1):
        ctx.rank = i
    return stitched_results

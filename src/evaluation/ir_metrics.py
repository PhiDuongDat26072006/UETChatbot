"""
src/evaluation/ir_metrics.py - Standard Information Retrieval (IR) evaluation metrics
and order-preserving ID deduplication.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set


def deduplicate_preserve_order(ids: List[str]) -> List[str]:
    """Khử trùng lặp danh sách ID nhưng vẫn bảo toàn thứ hạng xuất hiện đầu tiên."""
    seen = set()
    unique = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Precision@K: Tỉ lệ documents/chunks liên quan trong top-K kết quả trả về."""
    if k <= 0 or not relevant_ids:
        return 0.0
    unique_ids = deduplicate_preserve_order(retrieved_ids)
    top_k = unique_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_count / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Recall@K: Tỉ lệ documents/chunks liên quan đã được tìm thấy trong top-K."""
    if not relevant_ids or k <= 0:
        return 0.0
    unique_ids = deduplicate_preserve_order(retrieved_ids)
    top_k = unique_ids[:k]
    found = len(set(top_k).intersection(relevant_ids))
    return found / len(relevant_ids)


def ndcg_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int) -> float:
    """NDCG@K: Đánh giá chất lượng xếp hạng có tính mức độ liên quan đa mức (graded relevance)."""
    if k <= 0 or not qrels:
        return 0.0

    unique_ids = deduplicate_preserve_order(retrieved_ids)
    dcg = 0.0
    for i, doc_id in enumerate(unique_ids[:k]):
        rel = qrels.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return min(1.0, dcg / idcg)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """MRR (Mean Reciprocal Rank): Nghịch đảo thứ hạng của kết quả liên quan đầu tiên."""
    unique_ids = deduplicate_preserve_order(retrieved_ids)
    for i, doc_id in enumerate(unique_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Average Precision (AP): Trung bình Precision tại mỗi vị trí có kết quả liên quan."""
    if not relevant_ids:
        return 0.0

    unique_ids = deduplicate_preserve_order(retrieved_ids)
    num_relevant_found = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(unique_ids):
        if doc_id in relevant_ids:
            num_relevant_found += 1
            precision_sum += num_relevant_found / (i + 1)

    return precision_sum / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Hit Rate@K: 1.0 nếu có ít nhất 1 kết quả liên quan trong top-K, 0.0 nếu không."""
    unique_ids = deduplicate_preserve_order(retrieved_ids)
    top_k = unique_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0

"""
src/evaluation/metrics.py - Định nghĩa các chỉ số đo lường chất lượng hệ thống RAG (RAG Metrics).
Tuân theo chuẩn RAG Triad / RAGAS:
1. Context Relevance: Ngữ cảnh trích xuất có thực sự liên quan tới câu hỏi không?
2. Faithfulness (Groundedness): Câu trả lời có hoàn toàn dựa vào ngữ cảnh hay bị ảo giác (hallucination)?
3. Answer Relevance: Câu trả lời có giải đáp đúng trọng tâm câu hỏi không?
4. Context Recall / Hit Rate: Ngữ cảnh có bao phủ đầy đủ thông tin của câu trả lời chuẩn (Ground Truth) không?
"""

from __future__ import annotations
import math
from typing import Dict, List, Set


# ============================================================================
# RAG Triad / RAGAS Text Metrics (Generation & Context Relevance)
# ============================================================================

def compute_keyword_overlap(text_a: str, text_b: str) -> float:
    """Tính toán tỷ lệ trùng lặp từ khóa cơ bản giữa 2 văn bản (Jaccard similarity)."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)
    return len(intersection) / len(union)


def compute_context_relevance(query: str, contexts: List[str]) -> float:
    """
    Ước tính độ liên quan của ngữ cảnh với câu hỏi (thang điểm 0.0 - 1.0).
    (Có thể nâng cấp dùng LLM-as-a-judge hoặc embedding cosine similarity).
    """
    if not contexts:
        return 0.0
    scores = [compute_keyword_overlap(query, c) for c in contexts]
    return sum(scores) / len(scores)


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Ước tính tính trung thực (Groundedness / Faithfulness) của câu trả lời so với ngữ cảnh.
    Trả về điểm từ 0.0 đến 1.0.
    """
    if not contexts or not answer:
        return 0.0
    combined_context = " ".join(contexts).lower()
    answer_words = answer.lower().split()
    if not answer_words:
        return 0.0
    matched = sum(1 for w in answer_words if w in combined_context)
    return min(1.0, (matched / len(answer_words)) * 1.2)  # Hệ số chuẩn hóa


# ============================================================================
# Information Retrieval (IR) Metrics (Retrieval Evaluation)
# ============================================================================

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Precision@K: Tỉ lệ documents liên quan trong top-K kết quả trả về.
    
    :param retrieved_ids: Danh sách ID documents trả về (theo thứ tự xếp hạng)
    :param relevant_ids: Tập ID documents liên quan (ground truth)
    :param k: Số lượng top kết quả cần đánh giá
    :return: Precision@K ∈ [0, 1]
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_count / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@K: Tỉ lệ documents liên quan đã được tìm thấy trong top-K.
    
    :param retrieved_ids: Danh sách ID documents trả về
    :param relevant_ids: Tập ID documents liên quan (ground truth)
    :param k: Số lượng top kết quả
    :return: Recall@K ∈ [0, 1]
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return found / len(relevant_ids)


def ndcg_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain):
    Đánh giá chất lượng xếp hạng có tính đến mức độ liên quan đa mức (graded relevance).
    
    :param retrieved_ids: Danh sách ID documents trả về
    :param qrels: Dict {doc_id: relevance_score} (0, 1, hoặc 2)
    :param k: Số lượng top kết quả
    :return: NDCG@K ∈ [0, 1]
    """
    if k <= 0 or not qrels:
        return 0.0
    
    # DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = qrels.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    
    # IDCG (Ideal DCG)
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    MRR (Mean Reciprocal Rank): Nghịch đảo thứ hạng của document liên quan đầu tiên.
    
    :param retrieved_ids: Danh sách ID documents trả về
    :param relevant_ids: Tập ID documents liên quan
    :return: Reciprocal Rank ∈ [0, 1]
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Average Precision (AP): Trung bình Precision tại mỗi vị trí có document liên quan.
    
    :param retrieved_ids: Danh sách ID documents trả về
    :param relevant_ids: Tập ID documents liên quan
    :return: Average Precision ∈ [0, 1]
    """
    if not relevant_ids:
        return 0.0
    
    num_relevant_found = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_relevant_found += 1
            precision_sum += num_relevant_found / (i + 1)
    
    return precision_sum / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Hit Rate@K: 1.0 nếu có ít nhất 1 document liên quan trong top-K, 0.0 nếu không.
    
    :param retrieved_ids: Danh sách ID documents trả về
    :param relevant_ids: Tập ID documents liên quan
    :param k: Số lượng top kết quả
    :return: 0.0 hoặc 1.0
    """
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


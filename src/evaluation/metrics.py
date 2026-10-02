"""
src/evaluation/metrics.py - Định nghĩa các chỉ số đo lường chất lượng hệ thống RAG (RAG Metrics).
Tuân theo chuẩn RAG Triad / RAGAS:
1. Context Relevance: Ngữ cảnh trích xuất có thực sự liên quan tới câu hỏi không?
2. Faithfulness (Groundedness): Câu trả lời có hoàn toàn dựa vào ngữ cảnh hay bị ảo giác (hallucination)?
3. Answer Relevance: Câu trả lời có giải đáp đúng trọng tâm câu hỏi không?
4. Context Recall / Hit Rate: Ngữ cảnh có bao phủ đầy đủ thông tin của câu trả lời chuẩn (Ground Truth) không?
"""

from __future__ import annotations
from typing import List


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

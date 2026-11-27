"""
src/evaluation/metrics.py - Định nghĩa các chỉ số đo lường chất lượng hệ thống RAG (RAG & IR Metrics).

Gồm 2 nhóm chính:
1. Offline Vietnamese RAG Metrics (0-LLM, <20ms/câu):
   - Chuẩn hóa tiếng Việt & tách bỏ cú pháp bảng Markdown / footer trích dẫn.
   - Token Precision / Recall / F1, ROUGE-L.
   - Numeric & Code Grounding Guard (bắt lỗi bịa GPA, năm 2025/2026, mã học phần, số quyết định).
   - OOD Refusal Detector (nhận diện từ chối lịch sự).
   - Semantic Similarity tùy chọn qua local embedder (bge-m3).
2. Information Retrieval (IR) Metrics:
   - Precision@K, Recall@K, NDCG@K, MRR, MAP, HitRate@K.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from src.evaluation.ir_metrics import (
    average_precision,
    deduplicate_preserve_order,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.text_normalization import (
    _CACHED_AGENT_TEMPLATES,
    _CITATION_FOOTER_RE,
    _CLAUSE_NUM_RE,
    _COURSE_OR_MAJOR_CODE_RE,
    _DATE_DMY_SLASH_RE,
    _DATE_DMY_VERBAL_RE,
    _DATE_MY_SLASH_RE,
    _DATE_MY_VERBAL_RE,
    _DECISION_CODE_RE,
    _DECIMAL_PERCENT_FRACTION_RE,
    _DEFAULT_AGENT_NOTE_TEMPLATES,
    _DOC_REF_RE,
    _GEN_ERROR_PHRASES,
    _GREETING_PREFIX_RE,
    _MD_LINK_RE,
    _MD_SYNTAX_RE,
    _MD_TABLE_SEP_RE,
    _MD_TABLE_STT_RE,
    _NUMBERED_LIST_PREFIX_RE,
    _PREFIX_UNIT_NUMBER_RE,
    _REFUSAL_BOILERPLATE_RE,
    _REFUSAL_PHRASES,
    _TOKEN_RE,
    _UNIT_NUMBER_RE,
    _VI_FUNCTION_STOPWORDS,
    _VOICE_AMBIGUOUS_NOTE_RE,
    _YEAR_RE,
    _BARE_URL_RE,
    _entity_in_context_text,
    _get_default_agent_note_templates,
    _normalize_decimal_commas,
    _normalize_number_str,
    _prepare_text_for_numeric_scan,
    compute_numeric_grounding,
    extract_numeric_and_code_entities,
    is_generation_error_response,
    is_refusal_response,
    normalize_vietnamese_for_eval,
    strip_agent_notes,
    tokenize_vietnamese,
)


# ============================================================================
# Token Overlap, ROUGE-L & Local Embedding Semantic Similarity
# ============================================================================

def compute_token_overlap_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """
    Tính Token Precision, Token Recall, Token F1 và Keyword Recall sau khi loại bỏ Markdown/boilerplate.
    """
    pred_tokens = tokenize_vietnamese(prediction, remove_stopwords=True, strip_boilerplate=True)
    ref_tokens = tokenize_vietnamese(reference, remove_stopwords=True, strip_boilerplate=True)

    if not pred_tokens or not ref_tokens:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
            "keyword_recall": 0.0,
        }

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    keyword_recall = len(pred_set & ref_set) / len(ref_set)

    return {
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
        "keyword_recall": round(keyword_recall, 4),
    }


def compute_rouge_l(prediction: str, reference: str, max_tokens: int = 600) -> Dict[str, float]:
    """
    Tính ROUGE-L (Longest Common Subsequence) trên chuỗi token tiếng Việt đã làm sạch.
    Dùng quy hoạch động 2 hàng tối ưu bộ nhớ O(N).
    """
    pred_tokens = tokenize_vietnamese(prediction, remove_stopwords=False, strip_boilerplate=True)[:max_tokens]
    ref_tokens = tokenize_vietnamese(reference, remove_stopwords=False, strip_boilerplate=True)[:max_tokens]

    if not pred_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    m, n = len(pred_tokens), len(ref_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        p_tok = pred_tokens[i - 1]
        for j in range(1, n + 1):
            if p_tok == ref_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev = curr

    lcs = prev[n]
    if lcs == 0:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    prec = lcs / m
    rec = lcs / n
    f1 = 2 * prec * rec / (prec + rec)
    return {
        "rouge_l_precision": round(prec, 4),
        "rouge_l_recall": round(rec, 4),
        "rouge_l_f1": round(f1, 4),
    }


def _cosine_sim_vectors(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Tính cosine similarity giữa 2 vector và chuẩn hóa về [0.0, 1.0]."""
    if np is not None:
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        sim = float(np.dot(a, b) / denom)
    else:  # pragma: no cover
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        na = math.sqrt(sum(x * x for x in vec_a))
        nb = math.sqrt(sum(y * y for y in vec_b))
        sim = dot / (na * nb) if (na > 0 and nb > 0) else 0.0
    return max(0.0, min(1.0, sim))


def compute_batch_cosine_similarities(
    anchor: str,
    candidates: List[str],
    embedder: Any,
) -> List[float]:
    """Nhúng 1 batch (anchor + candidates) và tính cosine similarity tương ứng."""
    if embedder is None or not anchor or not candidates:
        return []
    clean_anchor = normalize_vietnamese_for_eval(anchor, strip_boilerplate=True) or anchor.strip()
    clean_cands = [normalize_vietnamese_for_eval(c, strip_boilerplate=False) or c.strip() for c in candidates]
    vectors = embedder.embed_texts([clean_anchor] + clean_cands)
    if not vectors or len(vectors) < len(clean_cands) + 1:
        return []
    anchor_vec = vectors[0]
    return [_cosine_sim_vectors(anchor_vec, v) for v in vectors[1:]]


def compute_semantic_similarity(text_a: str, text_b: str, embedder: Any) -> float:
    """Tính tương đồng ngữ nghĩa cosine bằng local embedder (trả về [0.0, 1.0])."""
    sims = compute_batch_cosine_similarities(text_a, [text_b], embedder)
    return round(sims[0], 4) if sims else 0.0


# ============================================================================
# RAG Triad / Generation Metrics (Tương thích ngược & Nâng cấp)
# ============================================================================

def compute_keyword_overlap(text_a: str, text_b: str) -> float:
    """
    Tính tương đồng từ khóa sạch (Token F1 sau khi loại bỏ Markdown/boilerplate).
    Thay thế Jaccard `.split()` thô trước đây vốn phạt nặng bảng Markdown.
    """
    return compute_token_overlap_metrics(text_a, text_b)["token_f1"]


def compute_context_relevance(
    query: str,
    contexts: List[str],
    embedder: Optional[Any] = None,
) -> float:
    """
    Ước tính độ liên quan của ngữ cảnh trích xuất với câu hỏi (thang 0.0 - 1.0).
    Dùng Query Term Coverage trên Top-3 ngữ cảnh tốt nhất, kết hợp Cosine Embedding nếu có.
    """
    if not contexts or not query:
        return 0.0

    q_tokens = set(tokenize_vietnamese(query, remove_stopwords=True, strip_boilerplate=False))
    if not q_tokens:
        q_tokens = set(tokenize_vietnamese(query, remove_stopwords=False, strip_boilerplate=False))
    if not q_tokens:
        return 0.0

    lex_scores: List[float] = []
    for ctx in contexts:
        c_tokens = set(tokenize_vietnamese(ctx, remove_stopwords=True, strip_boilerplate=False))
        cov = len(q_tokens & c_tokens) / len(q_tokens) if c_tokens else 0.0
        lex_scores.append(cov)

    lex_sorted = sorted(lex_scores, reverse=True)
    top3_mean_lex = sum(lex_sorted[:3]) / min(3, len(lex_sorted))
    lexical_agg = 0.5 * lex_sorted[0] + 0.5 * top3_mean_lex

    if embedder is not None:
        sem_scores = compute_batch_cosine_similarities(query, contexts, embedder)
        if sem_scores:
            sem_sorted = sorted(sem_scores, reverse=True)
            top3_mean_sem = sum(sem_sorted[:3]) / min(3, len(sem_sorted))
            semantic_agg = 0.5 * sem_sorted[0] + 0.5 * top3_mean_sem
            return round(min(1.0, 0.65 * semantic_agg + 0.35 * lexical_agg), 4)

    return round(min(1.0, lexical_agg), 4)


def compute_faithfulness(
    answer: str,
    contexts: List[str],
    embedder: Optional[Any] = None,
) -> float:
    """
    Ước tính tính trung thực (Groundedness / Faithfulness) của câu trả lời so với ngữ cảnh.
    Kết hợp độ phủ từ vựng sạch + Numeric & Code Grounding Guard (+ Semantic nếu có embedder).
    """
    if not contexts or not answer:
        return 0.0

    answer_tokens = tokenize_vietnamese(answer, remove_stopwords=True, strip_boilerplate=True)
    if not answer_tokens:
        return 0.0

    combined_ctx_tokens: Set[str] = set()
    for c in contexts:
        combined_ctx_tokens.update(tokenize_vietnamese(c, remove_stopwords=True, strip_boilerplate=False))

    matched = sum(1 for tok in answer_tokens if tok in combined_ctx_tokens)
    lexical_ratio = min(1.0, matched / len(answer_tokens))

    num_info = compute_numeric_grounding(answer, contexts)
    if num_info["answer_entities"]:
        base_score = 0.55 * lexical_ratio + 0.45 * float(num_info["numeric_faithfulness"])
    else:
        base_score = lexical_ratio

    if embedder is not None:
        sem_scores = compute_batch_cosine_similarities(answer, contexts, embedder)
        if sem_scores:
            sem_max = max(sem_scores)
            return round(min(1.0, 0.70 * base_score + 0.30 * sem_max), 4)

    return round(min(1.0, base_score), 4)


def compute_answer_relevance(
    query: str,
    answer: str,
    embedder: Optional[Any] = None,
) -> float:
    """
    Đo mức độ câu trả lời đi thẳng vào câu hỏi (Query <-> Answer, KHÔNG so với Ground Truth).
    """
    if not query or not answer:
        return 0.0

    q_tokens = set(tokenize_vietnamese(query, remove_stopwords=True, strip_boilerplate=False))
    a_tokens = set(tokenize_vietnamese(answer, remove_stopwords=True, strip_boilerplate=True))
    lexical_cov = (len(q_tokens & a_tokens) / len(q_tokens)) if q_tokens and a_tokens else 0.0

    if embedder is not None:
        sem = compute_semantic_similarity(query, answer, embedder)
        return round(min(1.0, 0.65 * sem + 0.35 * lexical_cov), 4)

    return round(min(1.0, lexical_cov), 4)


def compute_answer_correctness_offline(
    answer: str,
    ground_truth: Optional[str],
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Đánh giá Answer Correctness ngoại tuyến so với Ground Truth:
    kết hợp Keyword/Token Recall, Token F1, ROUGE-L Recall, Numeric Correctness và Semantic Cosine.
    """
    if not ground_truth:
        return {
            "answer_correctness": None,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
            "keyword_recall": 0.0,
            "rouge_l_recall": 0.0,
            "rouge_l_f1": 0.0,
            "numeric_correctness": None,
            "semantic_answer_correctness": None,
        }

    tok = compute_token_overlap_metrics(answer, ground_truth)
    rouge = compute_rouge_l(answer, ground_truth)
    num_info = compute_numeric_grounding(answer, contexts=[], ground_truth=ground_truth)
    num_corr = num_info["numeric_correctness"]

    sem_corr: Optional[float] = None
    if embedder is not None:
        sem_corr = compute_semantic_similarity(answer, ground_truth, embedder)

    # Trọng số ưu tiên Recall + Keyword Recall vì câu trả lời bảng Markdown thường dài hơn Ground Truth 4-6x
    base_lex = (
        0.45 * tok["keyword_recall"]
        + 0.25 * tok["token_recall"]
        + 0.15 * rouge["rouge_l_recall"]
        + 0.15 * tok["token_f1"]
    )

    if sem_corr is not None and num_corr is not None:
        combined = 0.45 * sem_corr + 0.30 * base_lex + 0.25 * num_corr
    elif sem_corr is not None:
        combined = 0.55 * sem_corr + 0.45 * base_lex
    elif num_corr is not None:
        combined = 0.65 * base_lex + 0.35 * num_corr
    else:
        combined = base_lex

    # Nếu câu in-domain mà chatbot lại từ chối trả lời, chặn trần điểm
    if is_refusal_response(answer):
        combined = min(combined, 0.15)

    return {
        "answer_correctness": round(min(1.0, max(0.0, combined)), 4),
        "token_precision": tok["token_precision"],
        "token_recall": tok["token_recall"],
        "token_f1": tok["token_f1"],
        "keyword_recall": tok["keyword_recall"],
        "rouge_l_recall": rouge["rouge_l_recall"],
        "rouge_l_f1": rouge["rouge_l_f1"],
        "numeric_correctness": num_corr,
        "semantic_answer_correctness": sem_corr,
        "missing_gt_entities": num_info["missing_gt_entities"],
    }

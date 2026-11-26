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


# ============================================================================
# Hằng số & Regex tiền xử lý tiếng Việt cho Evaluation
# ============================================================================

# Stopwords chức năng tiếng Việt — TUYỆT ĐỐI KHÔNG đưa từ phủ định/biên
# (không, chưa, chỉ, trừ, ngoài, cấm, tối, thiểu, đa, ít, nhất, vượt, quá, trở, lên, xuống)
_VI_FUNCTION_STOPWORDS: Set[str] = {
    "và", "là", "của", "có", "các", "những", "được", "trong", "cho", "về",
    "với", "tại", "theo", "từ", "đến", "khi", "nếu", "thì", "mà", "để",
    "này", "đó", "như", "sau", "đây", "trên", "dưới", "hay", "hoặc", "bởi",
    "vì", "nên", "do", "đã", "đang", "sẽ", "cũng", "còn", "lại", "rất",
    "em", "bạn", "anh", "chị", "nhé", "nha", "ạ", "vui", "lòng", "xem",
    "bảng", "thông", "tin", "dữ", "liệu", "tổng", "hợp", "sau", "nêu",
    "hiện", "nay", "một", "số", "việc", "cụ", "thể", "thêm", "cần",
}

_CITATION_FOOTER_RE = re.compile(
    r"(?im)^\s*(?:📌\s*)?căn\s*cứ\s*(?:quy\s*chế|tài\s*liệu)\s*:.*$"
)
_GREETING_PREFIX_RE = re.compile(
    r"^(?:xin\s+)?(?:chào\s+(?:em|bạn|các\s+em|các\s+bạn)|hello|hi)\s*[,!.:;\-–—]*\s*",
    re.IGNORECASE,
)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|/)[^)]+\)")
_BARE_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MD_TABLE_SEP_RE = re.compile(
    r"(?m)^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*(?::?-{2,}:?\s*\|?)?\s*$"
)
_MD_TABLE_STT_RE = re.compile(r"(?m)^\s*\|\s*(?:\d+|[ivx]+(?:\)|\.))\s*\|", re.IGNORECASE)
_NUMBERED_LIST_PREFIX_RE = re.compile(r"(?m)^\s*(?:\d+|[ivx]+)[\.\)]\s+", re.IGNORECASE)
_DOC_REF_RE = re.compile(
    r"\b(?:tài\s+liệu|bảng)\s+\d+(?:\s*(?:,|và|&|-)\s*\d+)*\b",
    re.IGNORECASE,
)
_MD_SYNTAX_RE = re.compile(r"[`*_>#~|•📌✅⚠️🔍📊📌📌]+")
_TOKEN_RE = re.compile(
    r"\d+(?:[./]\w+(?:-\w+)*)+%?|\d+(?:\.\d+)?%?|\w+",
    re.UNICODE,
)

# Regex trích xuất thực thể số học & mã định danh UET
_DECISION_CODE_RE = re.compile(
    r"\b\d{2,5}/[a-zđ]{1,10}(?:-[a-zđ0-9]{1,16})+\b",
    re.IGNORECASE,
)
_COURSE_OR_MAJOR_CODE_RE = re.compile(
    r"\b(?:int|elt|ema|epn|pes|mat|phy|flf|cn|k)\d{1,4}[a-z]?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_DECIMAL_PERCENT_FRACTION_RE = re.compile(
    r"\b\d+\.\d+%?|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b|(?<!/)\b\d+\s*/\s*\d+\b(?!/)|\b\d+%"
)
_DATE_DMY_VERBAL_RE = re.compile(
    r"\bngày\s+0*(\d{1,2})\s+tháng\s+0*(\d{1,2})\s+năm\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_DATE_DMY_SLASH_RE = re.compile(
    r"(?<!/)\b0*(\d{1,2})\s*/\s*0*(\d{1,2})\s*/\s*((?:19|20)\d{2})\b(?!/|[a-zđ])",
    re.IGNORECASE,
)
_DATE_MY_VERBAL_RE = re.compile(
    r"\btháng\s+0*(\d{1,2})\s*(?:năm|/|-)\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_DATE_MY_SLASH_RE = re.compile(
    r"(?<!/)\b0*([1-9]|1[0-2])\s*/\s*((?:19|20)\d{2})\b(?!/|[a-zđ-])",
    re.IGNORECASE,
)
_CLAUSE_NUM_RE = re.compile(
    r"\b(điều|khoản|chương|mục|bậc)\s+(\d+)\b",
    re.IGNORECASE,
)
_UNIT_NUMBER_RE = re.compile(
    r"\b(\d+)\s*(tín\s*chỉ|tc|điểm|học\s*kỳ|kỳ|năm|tháng|ngày|tuần|triệu|đồng|vnđ|ngành|môn|học\s*phần)\b",
    re.IGNORECASE,
)
_PREFIX_UNIT_NUMBER_RE = re.compile(
    r"\b(ielts|toeic|toefl|vstep|aptis|jlpt|hsk|topik|gpa|điểm\s*chuẩn|điểm\s*trúng\s*tuyển)\s*(?:từ|đạt|>=|<=|>|<|:|=)?\s*(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_REFUSAL_BOILERPLATE_RE = re.compile(
    r"(?:phòng\s+đào\s+tạo\s*\(?\s*phòng\s*105-e3\s*\)?|105-e3)",
    re.IGNORECASE,
)

_REFUSAL_PHRASES = (
    "chưa có thông tin cụ thể",
    "không có thông tin cụ thể",
    "chưa lưu trữ thông tin",
    "không lưu trữ thông tin",
    "không có thông tin về",
    "chưa có thông tin về",
    "chưa có cơ sở",
    "ngoài phạm vi",
    "không thuộc phạm vi",
    "không nằm trong",
    "không tìm thấy thông tin",
    "chưa tìm thấy thông tin",
    "không có dữ liệu",
    "chưa có dữ liệu",
    "không thể trả lời",
)

_DEFAULT_AGENT_NOTE_TEMPLATES: Dict[str, str] = {
    "answer_incomplete_note": (
        "\n\n> **Lưu ý:** Câu trả lời trên có thể chưa đầy đủ hoàn toàn mọi ý trong câu hỏi của bạn{missing_suffix}. "
        "Bạn có thể liên hệ Phòng Đào tạo (Phòng 105-E3, SĐT: 024.3754.7865) hoặc Cổng thông tin đào tạo UET để xác nhận thêm thông tin chi tiết nhé."
    ),
    "ambiguous_followup_note": (
        "\n\n> **Gợi ý:** Nếu bạn muốn tra cứu chính xác theo trường hợp của mình, "
        "bạn có thể cho mình biết thêm **ngành học, khóa tuyển sinh hoặc hệ đào tạo (Chuẩn/CLC)** nhé."
    ),
}

_VOICE_AMBIGUOUS_NOTE_RE = re.compile(
    r"\s*Nếu bạn muốn tra cứu chính xác hơn theo trường hợp của mình,\s*"
    r"bạn hãy cho mình biết thêm ngành học,\s*khóa tuyển sinh hoặc hệ đào tạo cụ thể nhé\.\s*",
    re.IGNORECASE,
)

_CACHED_AGENT_TEMPLATES: Optional[Dict[str, str]] = None


def _get_default_agent_note_templates() -> Dict[str, str]:
    """Đọc mẫu ghi chú từ config.yaml (cache 1 lần), fallback về template mặc định."""
    global _CACHED_AGENT_TEMPLATES
    if _CACHED_AGENT_TEMPLATES is not None:
        return _CACHED_AGENT_TEMPLATES
    merged = dict(_DEFAULT_AGENT_NOTE_TEMPLATES)
    try:
        from src.utils.helpers import load_yaml_config

        cfg = load_yaml_config()
        tpl_cfg = cfg.get("templates") if isinstance(cfg, dict) else None
        if isinstance(tpl_cfg, dict):
            for key in ("answer_incomplete_note", "ambiguous_followup_note"):
                if tpl_cfg.get(key):
                    merged[key] = str(tpl_cfg[key])
    except Exception:
        pass
    _CACHED_AGENT_TEMPLATES = merged
    return merged


def strip_agent_notes(answer: str, templates: Optional[Dict[str, str]] = None) -> str:
    """
    Loại bỏ các khối ghi chú template (`answer_incomplete_note`, `ambiguous_followup_note`,
    và câu gợi ý voice mode) do Agent nối vào cuối câu trả lời, tránh làm sai lệch
    token precision / faithfulness khi chấm offline.
    """
    if not answer:
        return ""
    text = unicodedata.normalize("NFC", str(answer))
    active_templates = templates if templates is not None else _get_default_agent_note_templates()

    for key in ("ambiguous_followup_note", "answer_incomplete_note"):
        tpl = active_templates.get(key)
        if not tpl:
            continue
        tpl_norm = unicodedata.normalize("NFC", str(tpl)).strip()
        if not tpl_norm:
            continue
        pattern = re.escape(tpl_norm).replace(re.escape("{missing_suffix}"), ".*?")
        text = re.sub(r"\s*" + pattern, "", text, flags=re.DOTALL)

    text = _VOICE_AMBIGUOUS_NOTE_RE.sub("", text)
    return text.rstrip()


# ============================================================================
# Tiền xử lý & Chuẩn hóa Tiếng Việt
# ============================================================================

def _normalize_decimal_commas(text: str) -> str:
    """Chuyển dấu phẩy thập phân giữa 2 chữ số (25,5) thành dấu chấm (25.5)."""
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _normalize_number_str(token: str) -> str:
    """Chuẩn hóa số thập phân bỏ 0 thừa ở cuối (3.60 -> 3.6, giữ tối thiểu 1 chữ số thập phân 3.0)."""
    token = token.strip().lower().replace(" ", "")
    suffix = "%" if token.endswith("%") else ""
    core = token[:-1] if suffix else token
    if re.fullmatch(r"\d+\.\d+", core):
        int_part, frac_part = core.split(".", 1)
        frac_trimmed = frac_part.rstrip("0") or "0"
        return f"{int_part}.{frac_trimmed}{suffix}"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/(?:19|20)\d{2}", core):
        day_p, month_p, year_p = core.split("/")
        return f"{int(day_p)}/{int(month_p)}/{year_p}{suffix}"
    return f"{core}{suffix}"


def normalize_vietnamese_for_eval(text: str, strip_boilerplate: bool = True) -> str:
    """
    Làm sạch văn bản tiếng Việt trước khi tính metric offline:
    - Chuẩn hóa Unicode NFC.
    - Xóa link Markdown, URL trần, dòng kẻ phân cách bảng `| :--- |`, ký hiệu `**`, `###`, `>`.
    - Khi `strip_boilerplate=True`, loại bỏ dòng footer `📌 Căn cứ quy chế:...` và lời chào mở đầu.
    """
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFC", str(text))
    if strip_boilerplate:
        cleaned = _CITATION_FOOTER_RE.sub(" ", cleaned)
        cleaned = _GREETING_PREFIX_RE.sub("", cleaned)

    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _BARE_URL_RE.sub(" ", cleaned)
    cleaned = _MD_TABLE_SEP_RE.sub(" ", cleaned)
    cleaned = _MD_SYNTAX_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("—", " ").replace("–", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def tokenize_vietnamese(
    text: str,
    remove_stopwords: bool = True,
    strip_boilerplate: bool = True,
) -> List[str]:
    """
    Tách từ tiếng Việt sạch dùng cho Offline Metrics (bảo toàn mã học phần, số thập phân, từ phủ định).
    """
    cleaned = normalize_vietnamese_for_eval(text, strip_boilerplate=strip_boilerplate)
    if not cleaned:
        return []
    cleaned = _normalize_decimal_commas(cleaned)
    raw_tokens = _TOKEN_RE.findall(cleaned)
    tokens: List[str] = []
    for tok in raw_tokens:
        t = _normalize_number_str(tok) if re.fullmatch(r"\d+\.\d+%?", tok) else tok.lower()
        if len(t) == 1 and not t.isdigit():
            continue
        tokens.append(t)

    if not remove_stopwords:
        return tokens

    filtered = [t for t in tokens if t not in _VI_FUNCTION_STOPWORDS]
    return filtered if filtered else tokens


def is_refusal_response(answer: str) -> bool:
    """
    Phát hiện xem câu trả lời có phải là câu từ chối (do OOD hoặc thiếu ngữ cảnh) hay không.
    Tránh bắt nhầm cụm từ như 'Không nêu rõ' trong ô bảng của câu trả lời đầy đủ.
    """
    if not answer or not answer.strip():
        return True

    raw = unicodedata.normalize("NFC", str(answer)).strip()
    without_footer = _CITATION_FOOTER_RE.sub("", raw).strip()
    without_greeting = _GREETING_PREFIX_RE.sub("", without_footer).strip()
    lowered_full = re.sub(r"\s+", " ", without_greeting).lower()

    # Đếm số dòng dữ liệu bảng Markdown (loại header/separator bằng ước lượng >= 3 dòng có |)
    table_lines = [ln for ln in without_footer.splitlines() if ln.strip().startswith("|")]
    has_rich_table = len(table_lines) >= 4

    # Chỉ quét phần mở đầu (350 ký tự) nếu câu trả lời dài hoặc có bảng Markdown lớn
    scan_text = lowered_full[:350] if (has_rich_table or len(lowered_full) > 450) else lowered_full

    if any(phrase in scan_text for phrase in _REFUSAL_PHRASES):
        return True
    if "xin lỗi" in scan_text and ("phòng đào tạo" in scan_text or "chưa có" in scan_text or "không có" in scan_text):
        return True
    return False


_GEN_ERROR_PHRASES = (
    "mô hình ai tạm thời từ chối dịch vụ",
    "lỗi truy vấn:",
    "không thể tạo câu trả lời do lỗi kết nối",
    "đang gián đoạn dịch vụ",
)


def is_generation_error_response(answer: str, model_name: Optional[str] = None) -> bool:
    """
    Nhận diện câu trả lời thực chất là thông báo lỗi API / rate-limit 429 từ fallback,
    tránh chấm nhầm thành HALLUCINATION hoặc OOD_FAILURE.
    """
    if (model_name or "").strip().lower() == "multi-provider-fallback":
        return True
    low = unicodedata.normalize("NFC", str(answer or "")).strip().lower()
    if not low:
        return True
    return any(phrase in low for phrase in _GEN_ERROR_PHRASES)


# ============================================================================
# Numeric & Code Grounding Guard (đặc trị dữ liệu quy chế / tuyển sinh UET)
# ============================================================================

def _prepare_text_for_numeric_scan(text: str, is_answer: bool = False) -> str:
    """Loại bỏ số thứ tự bảng, số đánh mục đầu dòng và tham chiếu 'Tài liệu N' trước khi trích xuất số."""
    cleaned = unicodedata.normalize("NFC", str(text or ""))
    if is_answer:
        cleaned = _CITATION_FOOTER_RE.sub(" ", cleaned)
        cleaned = _DOC_REF_RE.sub(" ", cleaned)
        if is_refusal_response(cleaned):
            cleaned = _REFUSAL_BOILERPLATE_RE.sub(" ", cleaned)
    cleaned = _MD_TABLE_STT_RE.sub("| ", cleaned)
    cleaned = _NUMBERED_LIST_PREFIX_RE.sub("", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _BARE_URL_RE.sub(" ", cleaned)
    cleaned = _normalize_decimal_commas(cleaned)
    cleaned = _DATE_DMY_VERBAL_RE.sub(r"ngày \1/\2/\3", cleaned)
    cleaned = _DATE_DMY_SLASH_RE.sub(r"\1/\2/\3", cleaned)
    cleaned = _DATE_MY_VERBAL_RE.sub(r"tháng \1/\2", cleaned)
    cleaned = _DATE_MY_SLASH_RE.sub(r"\1/\2", cleaned)
    return cleaned.lower()


def extract_numeric_and_code_entities(text: str, is_answer: bool = False) -> Set[str]:
    """
    Trích xuất các thực thể số học & mã định danh quan trọng:
    - Số quyết định (5115/qđ-đhqghn), mã học phần/ngành/khóa (int1008, cn8, k67).
    - Năm (2025, 2026), số thập phân/phần trăm/phân số (3.60 -> 3.6, 2/3, 50%).
    - Điều/Khoản/Chương/Mục + số và số lượng kèm đơn vị học vụ (15 tín chỉ -> '15').
    """
    prepared = _prepare_text_for_numeric_scan(text, is_answer=is_answer)
    if not prepared:
        return set()

    entities: Set[str] = set()

    for m in _DECISION_CODE_RE.finditer(prepared):
        entities.add(m.group(0).lower())

    for m in _COURSE_OR_MAJOR_CODE_RE.finditer(prepared):
        entities.add(m.group(0).lower())

    for m in _YEAR_RE.finditer(prepared):
        entities.add(m.group(0))

    for m in _DECIMAL_PERCENT_FRACTION_RE.finditer(prepared):
        norm = _normalize_number_str(m.group(0))
        entities.add(norm)

    for m in _CLAUSE_NUM_RE.finditer(prepared):
        entities.add(f"{m.group(1).lower()} {m.group(2)}")

    for m in _UNIT_NUMBER_RE.finditer(prepared):
        entities.add(m.group(1))

    for m in _PREFIX_UNIT_NUMBER_RE.finditer(prepared):
        entities.add(_normalize_number_str(m.group(2)))

    return entities


def _entity_in_context_text(entity: str, context_entities: Set[str], prepared_context: str) -> bool:
    """Kiểm tra thực thể có xuất hiện trong ngữ cảnh đã làm sạch hay không."""
    if entity in context_entities:
        return True
    # Nếu là số thuần hoặc số thập phân đã chuẩn hóa, kiểm tra token số trong context
    if re.fullmatch(r"\d+", entity):
        return bool(re.search(rf"\b{re.escape(entity)}\b", prepared_context))
    if re.fullmatch(r"\d+\.\d+", entity):
        # Cho phép khớp 3.6 với 3.60 trong ngữ cảnh
        int_p, frac_p = entity.split(".", 1)
        pattern = rf"\b{re.escape(int_p)}\.{re.escape(frac_p)}0*\b"
        if re.search(pattern, prepared_context):
            return True
        if frac_p == "0" and re.search(rf"\b{re.escape(int_p)}\b", prepared_context):
            return True
    if "/" in entity:
        compact = entity.replace(" ", "")
        return compact in prepared_context.replace(" ", "")
    return entity in prepared_context


def compute_numeric_grounding(
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kiểm tra tính trung thực và độ chính xác của toàn bộ con số / mã định danh trong câu trả lời.
    """
    answer_entities = extract_numeric_and_code_entities(answer, is_answer=True)
    prepared_context = " ".join(_prepare_text_for_numeric_scan(c, is_answer=False) for c in (contexts or []))
    context_entities = extract_numeric_and_code_entities(prepared_context, is_answer=False)

    ungrounded = sorted(
        ent for ent in answer_entities
        if not _entity_in_context_text(ent, context_entities, prepared_context)
    )
    if answer_entities:
        numeric_faithfulness = (len(answer_entities) - len(ungrounded)) / len(answer_entities)
    else:
        numeric_faithfulness = 1.0

    numeric_correctness: Optional[float] = None
    missing_gt_entities: List[str] = []
    gt_entities: Set[str] = set()

    if ground_truth:
        prepared_gt = _prepare_text_for_numeric_scan(ground_truth, is_answer=False)
        gt_entities = extract_numeric_and_code_entities(prepared_gt, is_answer=False)
        if gt_entities:
            prepared_ans = _prepare_text_for_numeric_scan(answer, is_answer=True)
            matched_gt = 0
            for ent in sorted(gt_entities):
                if _entity_in_context_text(ent, answer_entities, prepared_ans):
                    matched_gt += 1
                else:
                    missing_gt_entities.append(ent)
            numeric_correctness = matched_gt / len(gt_entities)

    return {
        "numeric_faithfulness": round(numeric_faithfulness, 4),
        "numeric_correctness": round(numeric_correctness, 4) if numeric_correctness is not None else None,
        "answer_entities": sorted(answer_entities),
        "ungrounded_entities": ungrounded,
        "gt_entities": sorted(gt_entities),
        "missing_gt_entities": missing_gt_entities,
    }


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


# ============================================================================
# Information Retrieval (IR) Metrics (Retrieval Evaluation)
# ============================================================================

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

"""
src/evaluation/text_normalization.py - Vietnamese text preprocessing,
agent note stripping, refusal/error detection, and numeric/code grounding guards
for offline RAG evaluation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

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

_GEN_ERROR_PHRASES = (
    "mô hình ai tạm thời từ chối dịch vụ",
    "lỗi truy vấn:",
    "không thể tạo câu trả lời do lỗi kết nối",
    "đang gián đoạn dịch vụ",
)


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

    table_lines = [ln for ln in without_footer.splitlines() if ln.strip().startswith("|")]
    has_rich_table = len(table_lines) >= 4

    scan_text = lowered_full[:350] if (has_rich_table or len(lowered_full) > 450) else lowered_full

    if any(phrase in scan_text for phrase in _REFUSAL_PHRASES):
        return True
    if "xin lỗi" in scan_text and ("phòng đào tạo" in scan_text or "chưa có" in scan_text or "không có" in scan_text):
        return True
    return False


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
    if re.fullmatch(r"\d+", entity):
        return bool(re.search(rf"\b{re.escape(entity)}\b", prepared_context))
    if re.fullmatch(r"\d+\.\d+", entity):
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

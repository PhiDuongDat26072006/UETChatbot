"""Hybrid Rule-Based + LLM Graph Extractor for UET chunks."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.base import DataChunk
from src.graph.entity_utils import (
    FACULTY_LABELS,
    _ARTICLE_RE,
    _BREADCRUMB_ARTICLE_COLON_RE,
    _CERT_RE,
    _CONCEPT_PATTERNS,
    _COURSE_CODE_RE,
    _DOC_CODE_RE,
    _EXTRACTION_PROMPT_TEMPLATE,
    _NAMED_REG_RE,
    _ORG_RE,
    _PERSON_RE,
    _UNIT_RE,
    _canonicalize_triple,
    _classify_sentence_relation,
    _normalize_relation,
    _trim_vn_phrase,
    canonicalize_entity_name,
    clean_entity_name,
    normalize_entity_id,
)
from src.graph.thesaurus_utils import is_forbidden_entity
from src.utils.helpers import BASE_DIR, get_logger

logger = get_logger("graph.bridge")

_VALID_EXTRACTION_MODES = ("hybrid", "regex_only", "llm_only")


class HybridGraphExtractor:
    """
    Bộ trích xuất Đồ thị Tri thức lai (Rule-based Academic Regex + OpenRouter Tencent Hy4 LLM).
    Hỗ trợ 3 chế độ độc lập: extraction_mode = "hybrid" | "regex_only" | "llm_only".
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        enable_llm_extraction: bool = False,
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = "tencent/hy4-flash",
        llm_domains: Optional[List[str]] = None,
        max_llm_chunks: Optional[int] = None,
        cache_path: Optional[Path] = None,
        extraction_mode: str = "hybrid",
    ):
        mode = str(extraction_mode or "hybrid").strip().lower()
        if mode not in _VALID_EXTRACTION_MODES:
            raise ValueError(f"Unsupported extraction_mode '{extraction_mode}'. Choose from {_VALID_EXTRACTION_MODES}.")
        self.extraction_mode = mode
        self.llm_client = llm_client
        self.enable_llm_extraction = enable_llm_extraction
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_domains = [d.lower() for d in llm_domains] if llm_domains else []
        self.max_llm_chunks = int(max_llm_chunks) if max_llm_chunks is not None else None
        self.llm_calls_count: int = 0
        self.cache_path = Path(cache_path) if cache_path else (BASE_DIR / "database" / "graph_llm_cache.json")
        self._llm_cache: Dict[str, Any] = self._load_cache()
        self._cache_dirty = False

    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"Không thể đọc graph LLM cache '{self.cache_path}': {exc}")
            return {}

    def flush_cache(self) -> None:
        if not self._cache_dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as fp:
                json.dump(self._llm_cache, fp, ensure_ascii=False, indent=2)
            self._cache_dirty = False
        except Exception as exc:
            logger.warning(f"Không thể lưu graph LLM cache '{self.cache_path}': {exc}")

    @staticmethod
    def _extract_entities_from_text(text: str) -> List[Dict[str, str]]:
        """Trích xuất thực thể học vụ UET bằng regex từ một đoạn hoặc câu."""
        found: Dict[str, Dict[str, str]] = {}

        def _add(name: str, etype: str) -> None:
            cleaned = canonicalize_entity_name(name, etype)
            if len(cleaned) < 2 or is_forbidden_entity(cleaned, allow_school_names=True):
                return
            eid = normalize_entity_id(cleaned)
            if not eid:
                return
            if eid not in found:
                found[eid] = {"id": eid, "name": cleaned, "type": etype}

        for m in _DOC_CODE_RE.finditer(text):
            _add(m.group(1), "REGULATION")
        for m in _NAMED_REG_RE.finditer(text):
            _add(_trim_vn_phrase(m.group(1), max_words=10), "REGULATION")
        for m in _ARTICLE_RE.finditer(text):
            _add(m.group(1), "ARTICLE")
        for m in _COURSE_CODE_RE.finditer(text):
            _add(m.group(1).upper(), "COURSE")
        for m in _UNIT_RE.finditer(text):
            unit = _trim_vn_phrase(m.group(1), max_words=8)
            if len(unit.split()) >= 2:
                etype = "DEPARTMENT" if unit.casefold().startswith("bộ môn") else "FACULTY"
                _add(unit, etype)
        for m in _PERSON_RE.finditer(text):
            _add(m.group(1), "PERSON")
        for m in _CERT_RE.finditer(text):
            _add(m.group(1).upper() if "vnu" in m.group(1).lower() or m.group(1).isupper() else m.group(1), "CERTIFICATE")
        for m in _ORG_RE.finditer(text):
            _add(m.group(1), "FACULTY")
        for pattern, etype in _CONCEPT_PATTERNS:
            for m in pattern.finditer(text):
                _add(m.group(0), etype)

        return list(found.values())

    @staticmethod
    def _extract_breadcrumb_segments(chunk: DataChunk) -> List[str]:
        meta = chunk.metadata or {}
        raw_bc = meta.get("breadcrumb")
        if not raw_bc and chunk.text.lstrip().startswith("["):
            first_line = chunk.text.lstrip().splitlines()[0]
            if first_line.startswith("[") and "]" in first_line and ">" in first_line:
                raw_bc = first_line[1:first_line.index("]")]
        if not raw_bc:
            return []
        segments: List[str] = []
        for seg in str(raw_bc).split(">"):
            cleaned = clean_entity_name(seg)
            if not cleaned or re.fullmatch(r"Trang\s+\d+", cleaned, flags=re.IGNORECASE):
                continue
            m_art = _BREADCRUMB_ARTICLE_COLON_RE.match(cleaned)
            if m_art:
                cleaned = clean_entity_name(m_art.group(1))
            if cleaned:
                segments.append(cleaned)
        return segments

    def extract_rules(self, chunk: DataChunk) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Trích xuất thực thể và bộ ba cấu trúc từ metadata + nội dung chunk."""
        meta = chunk.metadata or {}
        text = unicodedata.normalize("NFC", chunk.text or "")
        entities_map: Dict[str, Dict[str, str]] = {}
        triples: List[Dict[str, str]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        def _register_entity(name: str, etype: str) -> Optional[Dict[str, str]]:
            cleaned = canonicalize_entity_name(name, etype)
            if len(cleaned) < 2 or is_forbidden_entity(cleaned, allow_school_names=True):
                return None
            eid = normalize_entity_id(cleaned)
            if not eid:
                return None
            if eid not in entities_map:
                entities_map[eid] = {"id": eid, "name": cleaned, "type": etype}
            return entities_map[eid]

        def _add_triple(h_name: str, h_type: str, rel: str, t_name: str, t_type: str) -> None:
            h_ent = _register_entity(h_name, h_type)
            t_ent = _register_entity(t_name, t_type)
            if not h_ent or not t_ent or h_ent["id"] == t_ent["id"]:
                return
            norm_rel = _normalize_relation(rel)
            sig = (h_ent["id"], norm_rel, t_ent["id"])
            if sig in seen_triples:
                return
            seen_triples.add(sig)
            triples.append({
                "head": h_ent["name"],
                "head_type": h_ent["type"],
                "relation": norm_rel,
                "tail": t_ent["name"],
                "tail_type": t_ent["type"],
            })

        doc_title = clean_entity_name(str(meta.get("title") or ""))
        bc_segments = self._extract_breadcrumb_segments(chunk)
        if not doc_title and bc_segments:
            doc_title = bc_segments[0]

        doc_code_match = _DOC_CODE_RE.search(text)
        doc_anchor_name = doc_code_match.group(1) if (doc_code_match and chunk.chunk_index == 0) else (doc_title or "")
        doc_anchor_type = "REGULATION"

        if doc_anchor_name:
            _register_entity(doc_anchor_name, doc_anchor_type)

        for idx in range(len(bc_segments) - 1):
            _add_triple(bc_segments[idx], "REGULATION", "BAO_GỒM", bc_segments[idx + 1], "ARTICLE")

        faculty_key = str(meta.get("faculty") or meta.get("site") or "").strip().lower()
        if faculty_key and faculty_key in FACULTY_LABELS:
            fac_label = FACULTY_LABELS[faculty_key]
            _register_entity(fac_label, "FACULTY")
            if doc_anchor_name:
                _add_triple(doc_anchor_name, doc_anchor_type, "THUỘC_ĐƠN_VỊ", fac_label, "FACULTY")

        chunk_entities = self._extract_entities_from_text(text)
        for ent in chunk_entities:
            _register_entity(ent["name"], ent["type"])

        anchor = bc_segments[-1] if bc_segments else doc_anchor_name
        anchor_type = "ARTICLE" if len(bc_segments) > 1 else doc_anchor_type

        sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]
        for sent in sentences:
            sent_entities = self._extract_entities_from_text(sent)
            if not sent_entities:
                continue
            sent_lower = sent.casefold()
            rel = _classify_sentence_relation(sent_lower)

            if "căn cứ" in sent_lower and doc_anchor_name:
                for ent in sent_entities:
                    if ent["type"] == "REGULATION":
                        _add_triple(doc_anchor_name, doc_anchor_type, "CĂN_CỨ", ent["name"], "REGULATION")
            elif any(k in sent_lower for k in ("sửa đổi", "bổ sung")) and doc_anchor_name:
                for ent in sent_entities:
                    if ent["type"] in ("REGULATION", "ARTICLE"):
                        _add_triple(doc_anchor_name, doc_anchor_type, "SỬA_ĐỔI_BỔ_SUNG", ent["name"], ent["type"])

            limited = sent_entities[:5]
            for i in range(len(limited)):
                for j in range(i + 1, len(limited)):
                    _add_triple(
                        limited[i]["name"],
                        limited[i]["type"],
                        rel,
                        limited[j]["name"],
                        limited[j]["type"],
                    )

        if anchor:
            for ent in chunk_entities[:10]:
                if normalize_entity_id(ent["name"]) != normalize_entity_id(anchor):
                    _add_triple(anchor, anchor_type, "QUY_ĐỊNH_VỀ", ent["name"], ent["type"])

        return list(entities_map.values()), triples

    def _is_llm_eligible(self, chunk: DataChunk) -> bool:
        if not self.enable_llm_extraction or self.llm_client is None:
            return False
        if self.max_llm_chunks is not None and self.llm_calls_count >= self.max_llm_chunks:
            return False
        if not self.llm_domains:
            return True
        meta = chunk.metadata or {}
        haystack = " ".join(str(meta.get(k) or "") for k in ("faculty", "site", "domain", "file_name")).casefold()
        return any(dom in haystack for dom in self.llm_domains)

    @staticmethod
    def _parse_llm_triples(raw_response: str) -> List[Dict[str, str]]:
        if not raw_response:
            return []
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_response).strip()
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []

        valid: List[Dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            canon = _canonicalize_triple(item)
            if canon is not None:
                valid.append(canon)
        return valid

    def extract_llm(self, chunk: DataChunk) -> List[Dict[str, str]]:
        """Trích xuất triples bổ sung qua OpenRouter (Tencent Hy4) có Disk Cache theo chunk."""
        content_hash = hashlib.sha256((chunk.text or "").encode("utf-8", errors="replace")).hexdigest()
        cached_entry = self._llm_cache.get(chunk.chunk_id)
        if isinstance(cached_entry, dict) and cached_entry.get("hash") == content_hash:
            res: List[Dict[str, str]] = []
            for raw_tr in (cached_entry.get("triples") or []):
                canon = _canonicalize_triple(raw_tr)
                if canon is not None:
                    res.append(canon)
            return res

        if not self._is_llm_eligible(chunk):
            return []

        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(text=(chunk.text or "")[:2500])
        self.llm_calls_count += 1
        try:
            raw_resp = self.llm_client.complete_text(
                prompt,
                preferred_provider=self.llm_provider,
                model=self.llm_model,
            )
            triples = self._parse_llm_triples(raw_resp)
            self._llm_cache[chunk.chunk_id] = {"hash": content_hash, "triples": triples}
            self._cache_dirty = True
            return triples
        except Exception as exc:
            logger.warning(f"Lỗi gọi LLM trích xuất Graph cho chunk {chunk.chunk_id}: {exc}")
            return []

    @staticmethod
    def _merge_triples(
        base_entities: List[Dict[str, str]],
        base_triples: List[Dict[str, str]],
        extra_triples: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        entities_map = {e["id"]: e for e in base_entities}
        triples = list(base_triples)
        seen = {(normalize_entity_id(t["head"]), t["relation"], normalize_entity_id(t["tail"])) for t in triples}
        for lt in extra_triples:
            hid, tid = normalize_entity_id(lt["head"]), normalize_entity_id(lt["tail"])
            if not hid or not tid or hid == tid:
                continue
            entities_map.setdefault(hid, {"id": hid, "name": lt["head"], "type": lt["head_type"]})
            entities_map.setdefault(tid, {"id": tid, "name": lt["tail"], "type": lt["tail_type"]})
            sig = (hid, lt["relation"], tid)
            if sig not in seen:
                seen.add(sig)
                triples.append(lt)
        return list(entities_map.values()), triples

    def extract(self, chunk: DataChunk) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Trích xuất thực thể & triples theo extraction_mode (hybrid / regex_only / llm_only)."""
        if self.extraction_mode == "regex_only":
            return self.extract_rules(chunk)
        llm_triples = self.extract_llm(chunk)
        if self.extraction_mode == "llm_only":
            return self._merge_triples([], [], llm_triples)
        entities, triples = self.extract_rules(chunk)
        return self._merge_triples(entities, triples, llm_triples) if llm_triples else (entities, triples)

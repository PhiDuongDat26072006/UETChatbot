"""Topic heuristics and site-level stats aggregation for curation."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit
from .corpus.gate import policy_paths
from .corpus.policy import classify, load_policy, normalize as normalize_text
from .crawler.policy import resource_type

_TOPIC_MAP = {
    "quy_che": "Quy chế",
    "chuong_trinh_dao_tao": "Chương trình đào tạo",
    "hoc_phi": "Học phí",
    "hoc_bong": "Học bổng",
    "ngoai_ngu": "Ngoại ngữ",
    "thu_tuc_hoc_tap": "Thủ tục học tập",
    "nhap_hoc_theo_dot": "Thông báo cũ",
    "thong_bao_theo_dot": "Thông báo cũ",
    "thong_bao_tac_nghiep": "Thông báo cũ",
    "lich_su_kien": "Thông báo cũ",
    "danh_sach_ca_nhan": "Thông báo cũ",
    "tin_su_kien": "Thông báo cũ",
}
_HEURISTIC_TOPICS = (
    ("Quy chế", 90, re.compile(r"\b(quy che|quy dinh|noi quy|danh gia drl|ren luyen)\b")),
    (
        "Chương trình đào tạo",
        88,
        re.compile(
            r"\b(chuong trinh dao tao|chuan dau ra|khung chuong trinh|ke hoach giang day|mon hoc|tin chi|de cuong)\b"
        ),
    ),
    ("Học phí", 85, re.compile(r"\b(hoc phi|dinh muc|mien giam)\b")),
    ("Học bổng", 82, re.compile(r"\b(hoc bong|khen thuong|sinh vien 5 tot|sv5t)\b")),
    (
        "Tuyển sinh",
        80,
        re.compile(
            r"\b(tuyen sinh|diem chuan|xet tuyen|diem trung tuyen|chi tieu|de an tuyen sinh)\b"
        ),
    ),
    (
        "Thủ tục học tập",
        70,
        re.compile(
            r"\b(huong dan|thu tuc|bieu mau|quy trinh|dang ky hoc|tot nghiep|nhap hoc|ho tro tai khoan)\b"
        ),
    ),
    (
        "Khoa / Giảng viên",
        65,
        re.compile(
            r"\b(gioi thieu|giang vien|can bo|bo mon|phong thi nghiem|ban chu nhiem|su menh|tam nhin)\b"
        ),
    ),
)
_OLD_NOTICE_RE = re.compile(
    r"\b(thong bao|lich thi|ket qua|ke hoach|hoi thao|tuyen dung|su kien)\b"
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def load_corpus_policies(profiles: dict) -> dict[str, dict]:
    try:
        paths = policy_paths()
    except Exception:
        return {}
    loaded = {}
    for site, path in paths.items():
        if site in profiles and path.is_file():
            try:
                loaded[site] = load_policy(path)
            except Exception:
                continue
    return loaded


def topic_info(row: dict, corpus_policy: dict | None = None) -> tuple[str, int]:
    meta = row.get("metadata", {})
    uri = row.get("uri") or ""
    if corpus_policy is not None:
        res = classify(row, corpus_policy)
        cat = res.get("category", "")
        hint = _TOPIC_MAP.get(cat, "")
        if not hint and res.get("decision") == "exclude" and cat != "outside_scope":
            hint = "Thông báo cũ"
        return hint, int(res.get("priority", 0))
    title = meta.get("title") or meta.get("link_text") or ""
    path = unquote(urlsplit(uri).path)
    haystack = normalize_text(f"{title} {path}")
    for label, prio, pattern in _HEURISTIC_TOPICS:
        if pattern.search(haystack):
            return label, prio
    years = [int(y) for y in _YEAR_RE.findall(haystack)]
    if _OLD_NOTICE_RE.search(haystack) or (years and max(years) <= 2024):
        return "Thông báo cũ", 15
    kind = meta.get("resource_type") or resource_type(uri)
    return ("", 30 if kind == "pdf" else 20)


def compute_site_state(storage, site: str, profile: dict) -> tuple[dict, dict]:
    sources = storage.records("sources", site)
    processed = storage.records("processed", site)
    states = storage.records("state", site)
    latest_raw = {}
    for raw_rec in storage.records("raw", site).values():
        sid = raw_rec["metadata"]["source_id"]
        if sid not in latest_raw or raw_rec["created_at"] > latest_raw[sid]["created_at"]:
            latest_raw[sid] = raw_rec

    counts = {
        "site": site,
        "hosts": list(profile.get("scope", {}).get("hosts", [])),
        "total": 0,
        "pending": 0,
        "accepted_total": 0,
        "accepted_ingested": 0,
        "accepted_queued": 0,
        "listing": 0,
        "rejected_total": 0,
        "rejected_manual": 0,
        "rejected_auto": 0,
        "failed": 0,
    }
    item_status = {}
    for sid, rec in sources.items():
        meta = rec.get("metadata", {})
        dec = meta.get("decision")
        if dec == "alias":
            continue
        counts["total"] += 1
        raw_rec = latest_raw.get(sid)
        proc_rec = processed.get(sid)
        raw_state = states.get(f"raw:{sid}", {})
        proc_state = states.get(f"processed:{sid}", {})
        fail_error = (
            raw_state.get("error") if raw_state.get("status") == "failed" else None
        ) or (
            proc_state.get("error") if proc_state.get("status") == "failed" else None
        )
        is_failed = fail_error is not None and dec != "rejected"
        if is_failed:
            counts["failed"] += 1
        raw_exists = raw_rec is not None
        ingested = bool(
            dec == "accepted"
            and proc_rec is not None
            and not is_failed
            and (raw_rec is None or proc_rec.get("raw_data_id") == raw_rec.get("id"))
        )
        if dec == "accepted":
            counts["accepted_total"] += 1
            if ingested:
                counts["accepted_ingested"] += 1
            else:
                counts["accepted_queued"] += 1
        elif dec == "listing":
            counts["listing"] += 1
        elif dec == "rejected":
            counts["rejected_total"] += 1
            if meta.get("decision_reason") in {"manual_curation", "manual_override"}:
                counts["rejected_manual"] += 1
            else:
                counts["rejected_auto"] += 1
        else:
            counts["pending"] += 1
        item_status[sid] = {
            "raw_exists": raw_exists,
            "ingested": ingested,
            "is_failed": is_failed,
            "fail_error": fail_error or "",
            "char_count": len(proc_rec.get("content") or "") if proc_rec else 0,
            "raw_created_at": raw_rec.get("created_at") if raw_rec else None,
        }
    return counts, item_status

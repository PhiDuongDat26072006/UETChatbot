"""Hydrate SPA / client-rendered pages (StudentHub API, Tuổi trẻ Công nghệ SV5T bundle) into clean HTML."""

from __future__ import annotations

from html import escape
import json
import re
from urllib.parse import urljoin, urlsplit


def _render_studenthub_guide_html(data: dict) -> str:
    title = escape(str(data.get("title") or "Hướng dẫn StudentHub").strip())
    parts = [
        "<!doctype html>",
        '<html lang="vi"><head><meta charset="utf-8">',
        f"<title>{title}</title></head><body>",
        '<article class="entry-content">',
        f'<h1 class="entry-title">{title}</h1>',
    ]
    updated = data.get("updatedAt") or data.get("createdAt")
    if updated:
        parts.append(f'<time datetime="{escape(str(updated))}"></time>')
    desc = str(data.get("description") or "").strip()
    if desc:
        parts.append(f"<p>{escape(desc)}</p>")
    for sec in data.get("sections") or []:
        sec_title = str(sec.get("title") or "").strip()
        sec_content = str(sec.get("content") or "").strip()
        # Strip empty <h2></h2> or <p></p> placeholders from the CMS editor
        sec_content = re.sub(r"<h2[^>]*>\s*</h2>", "", sec_content, flags=re.I)
        if sec_title and not re.search(
            rf"<h[1-3][^>]*>\s*{re.escape(sec_title)}\s*</h[1-3]>", sec_content, re.I
        ):
            parts.append(f"<h2>{escape(sec_title)}</h2>")
        if sec_content:
            parts.append(sec_content)
    parts.append("</article></body></html>")
    return "\n".join(parts)


def _render_sv5t_html(original_html: str, levels: list[dict]) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(original_html, "html.parser")
    page_title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Sinh Viên 5 Tốt | Tuổi trẻ Công nghệ UET"
    )
    parts = [
        "<!doctype html>",
        '<html lang="vi"><head><meta charset="utf-8">',
        f"<title>{escape(page_title)}</title></head><body>",
        '<article class="entry-content">',
        f'<h1 class="entry-title">{escape(page_title)}</h1>',
        "<p>Phong trào Sinh viên 5 tốt UET: Hành trình rèn luyện toàn diện để mỗi sinh viên UET chủ động hoàn thiện bản thân qua 5 tiêu chuẩn: Đạo đức tốt, Học tập tốt, Thể lực tốt, Tình nguyện tốt và Hội nhập tốt.</p>",
    ]
    for level in levels:
        lvl_name = escape(str(level.get("name") or level.get("id") or ""))
        source_doc = escape(str(level.get("source") or ""))
        eff_note = escape(str(level.get("effective_note") or ""))
        parts.append(f"<h2>Tiêu chuẩn xét danh hiệu Sinh viên 5 tốt - {lvl_name}</h2>")
        if source_doc or eff_note:
            meta_line = " | ".join(x for x in [f"Văn bản căn cứ: {source_doc}" if source_doc else "", eff_note] if x)
            parts.append(f"<p><strong>{meta_line}</strong></p>")
        summaries = level.get("summary") or []
        if summaries:
            parts.append("<ul>")
            for s in summaries:
                parts.append(f"<li>{escape(str(s))}</li>")
            parts.append("</ul>")
        for req in level.get("requirements") or []:
            req_name = escape(str(req.get("name") or ""))
            req_summary = escape(str(req.get("summary") or ""))
            parts.append(f"<h3>{req.get('order', '')}. {req_name} ({lvl_name})</h3>")
            if req_summary:
                parts.append(f"<p>{req_summary}</p>")
            req_rule = escape(str(req.get("required_rule") or "Tiêu chuẩn bắt buộc:"))
            req_items = req.get("required") or []
            if req_items:
                parts.append(f"<p><strong>{req_rule}</strong></p><ul>")
                for item in req_items:
                    parts.append(f"<li>{escape(str(item))}</li>")
                parts.append("</ul>")
            add_rule = escape(str(req.get("additional_rule") or "Tiêu chuẩn đạt thêm:"))
            add_items = req.get("additional") or []
            if add_items:
                parts.append(f"<p><strong>{add_rule}</strong></p><ul>")
                for item in add_items:
                    parts.append(f"<li>{escape(str(item))}</li>")
                parts.append("</ul>")
            ev_items = req.get("evidence_suggestions") or []
            if ev_items:
                parts.append("<p><strong>Minh chứng gợi ý:</strong></p><ul>")
                for item in ev_items:
                    parts.append(f"<li>{escape(str(item))}</li>")
                parts.append("</ul>")
    parts.append("</article>")
    nav_links = re.findall(r'<a\s+[^>]*href="[^"]+"[^>]*>.*?</a>', original_html, flags=re.I | re.S)
    if nav_links:
        parts.append('<div id="content"><aside>' + "".join(nav_links) + "</aside></div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def maybe_hydrate_spa(url: str, body: bytes, encoding: str, session, timeout: float) -> tuple[bytes, str]:
    """If url belongs to a known SPA endpoint, fetch its structured API/bundle data and return full HTML bytes."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = parts.path.rstrip("/")

    # 1. StudentHub guide SPA: /huong-dan/<slug>
    if host == "studenthub.uet.edu.vn" and path.startswith("/huong-dan/"):
        slug = path.split("/huong-dan/", 1)[1].strip("/")
        if slug:
            api_url = f"https://studenthub.uet.edu.vn/api/student/guide/slug/{slug}"
            resp = session.get(api_url, timeout=timeout, verify=False)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("sections"):
                rendered = _render_studenthub_guide_html(data)
                return rendered.encode("utf-8"), "utf-8"

    # 2. Tuổi trẻ Công nghệ SV5T Next.js tabs: /sv5t
    if host == "tuoitrecongnghe.uet.vnu.edu.vn" and path == "/sv5t":
        html_str = body.decode(encoding or "utf-8", errors="replace")
        m_chunk = re.search(r'src="(/_next/static/chunks/pages/sv5t-[^"]+\.js)"', html_str)
        if m_chunk:
            chunk_url = urljoin(url, m_chunk.group(1))
            resp = session.get(chunk_url, timeout=timeout, verify=False)
            if resp.ok:
                m_json = re.search(r"JSON\.parse\('(\{.*?\})'\)", resp.text)
                if m_json:
                    decoded = re.sub(
                        r"\\x([0-9a-fA-F]{2})",
                        lambda x: chr(int(x.group(1), 16)),
                        m_json.group(1),
                    )
                    payload = json.loads(decoded)
                    if isinstance(payload, dict) and payload.get("yr"):
                        rendered = _render_sv5t_html(html_str, payload["yr"])
                        return rendered.encode("utf-8"), "utf-8"

    return body, encoding


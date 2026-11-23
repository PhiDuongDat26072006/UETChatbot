"""Native .docx extractor preserving outline indentation, headings, links, and tables."""

from __future__ import annotations

import io
from pathlib import Path


def _render_inline(elem, rels: dict[str, str], qn) -> str:
    tag = elem.tag.rsplit("}", 1)[-1]
    if tag == "r":
        parts = []
        for child in elem:
            ctag = child.tag.rsplit("}", 1)[-1]
            if ctag == "t" and child.text:
                parts.append(child.text)
            elif ctag == "tab":
                parts.append(" ")
            elif ctag in {"br", "cr"}:
                parts.append("\n")
        return "".join(parts)
    if tag == "hyperlink":
        label = " ".join("".join(_render_inline(c, rels, qn) for c in elem).split()).strip()
        target = rels.get(elem.get(qn("r:id")) or "", "")
        if target and label:
            return f"[{label}]({target})"
        return target or label
    return "".join(_render_inline(c, rels, qn) for c in elem)


def extract_docx(source: str | Path | bytes) -> dict:
    """Extract clean Markdown text, outline hierarchy, tables, and links from a .docx file."""
    from docx import Document
    from docx.oxml.ns import qn

    is_bytes = isinstance(source, (bytes, bytearray))
    doc = Document(io.BytesIO(bytes(source))) if is_bytes else Document(str(Path(source).resolve()))
    rels = {
        r_id: str(rel.target_ref)
        for r_id, rel in doc.part.rels.items()
        if "hyperlink" in rel.reltype and getattr(rel, "target_ref", None)
    }

    p_map = {p._p: p for p in doc.paragraphs}
    t_map = {t._tbl: t for t in doc.tables}
    blocks: list[str] = []
    para_count = table_count = link_count = 0
    title = ""

    for child in doc.element.body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p" and child in p_map:
            para = p_map[child]
            raw_text = "".join(_render_inline(sub, rels, qn) for sub in child if not sub.tag.endswith("}pPr"))
            text = " ".join(raw_text.split()).strip()
            if not text:
                continue
            para_count += 1
            style_name = (para.style.name if para.style else "") or ""
            style_low = style_name.strip().lower()
            p_pr = child.find(qn("w:pPr"))
            num_pr = p_pr.find(qn("w:numPr")) if p_pr is not None else None

            if style_low.startswith("heading"):
                suffix = style_low.replace("heading", "").strip()
                level = int(suffix) if suffix.isdigit() and 1 <= int(suffix) <= 6 else 2
                blocks.append(f"{'#' * level} {text}")
                if not title:
                    title = text
            elif num_pr is not None:
                ilvl_el = num_pr.find(qn("w:ilvl"))
                raw_lvl = ilvl_el.get(qn("w:val")) if ilvl_el is not None else "0"
                lvl = int(raw_lvl) if str(raw_lvl).isdigit() else 0
                blocks.append(f"{'  ' * min(lvl, 6)}- {text}")
                if not title:
                    title = text
            else:
                blocks.append(text)
                if not title:
                    title = text
        elif tag == "tbl" and child in t_map:
            tbl = t_map[child]
            rows = []
            for row in tbl.rows:
                cells = [" ".join((cell.text or "").split()).replace("|", "\\|") for cell in row.cells]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            table_count += 1
            width = max(len(r) for r in rows)
            norm_rows = [r + [""] * (width - len(r)) for r in rows]
            md_lines = [
                "| " + " | ".join(norm_rows[0]) + " |",
                "| " + " | ".join(["---"] * width) + " |",
            ]
            md_lines.extend("| " + " | ".join(r) + " |" for r in norm_rows[1:])
            blocks.append("\n".join(md_lines))

    link_count = len(rels)
    if not is_bytes and not title:
        title = Path(source).stem.replace("-", " ").replace("_", " ").strip()
    return {
        "title": title,
        "content": "\n\n".join(blocks).strip(),
        "metadata": {
            "resource_type": "office",
            "extraction_method": "docx_native",
            "paragraph_count": para_count,
            "table_count": table_count,
            "link_count": link_count,
        },
    }

"""Chuyển đổi thẻ HTML <table> sang định dạng Markdown chuẩn."""
from __future__ import annotations

from bs4 import BeautifulSoup

from ..utils import clean_whitespace


def table_to_markdown(table_tag: BeautifulSoup) -> str:
    """Chuyển đổi thẻ <table> HTML thành bảng Markdown chuẩn (| Col 1 | Col 2 |)."""
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(["th", "td"]):
            cell_text = clean_whitespace(cell.get_text(separator=" ", strip=True))
            cell_text = cell_text.replace("\n", " ").replace("|", "\\|")
            cells.append(cell_text)
        if any(c for c in cells):
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    if max_cols == 0:
        return ""

    padded_rows = [r + [""] * (max_cols - len(r)) for r in rows]
    header = padded_rows[0]
    separator = ["---"] * max_cols

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in padded_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n\n" + "\n".join(lines) + "\n\n"

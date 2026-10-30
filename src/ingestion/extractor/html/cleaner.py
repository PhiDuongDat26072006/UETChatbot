"""Convert selected DOM into Markdown without flattening lists and tables."""

import re
from copy import deepcopy
from urllib.parse import urljoin, urlsplit
from bs4 import NavigableString


def table_markdown(table):
    rows = []
    complex_table = any(
        c.get("rowspan", "1") != "1" or c.get("colspan", "1") != "1"
        for c in table.find_all(["th", "td"])
    )
    if complex_table:
        # Retain HTML span semantics instead of inventing flattened cell positions.
        return str(table)
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = [
            c.get_text(" ", strip=True).replace("|", r"\|")
            for c in row.find_all(["th", "td"], recursive=False)
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [r + [""] * (width - len(r)) for r in rows]
    first = table.find("tr")
    if not first or not first.find("th"):
        rows.insert(0, [f"Cột {i + 1}" for i in range(width)])
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def render(node, url, depth=0):
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    name = node.name
    if name == "table":
        return "\n\n" + table_markdown(node) + "\n\n"
    if name in {"ul", "ol"}:
        lines = []
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            body = "".join(render(c, url, depth + 1) for c in li.children).strip()
            lines.append("  " * depth + (f"{i}. " if name == "ol" else "- ") + body)
        return "\n" + "\n".join(lines) + "\n"
    text = "".join(render(c, url, depth) for c in node.children)
    if name and re.fullmatch("h[1-6]", name):
        return "\n\n" + "#" * int(name[1]) + " " + text.strip() + "\n\n"
    if name in {"p", "div", "section", "article", "blockquote"}:
        return "\n\n" + text.strip() + "\n\n"
    if name == "br":
        return "\n"
    if name == "a":
        target = urljoin(url, node.get("href", ""))
        if text.strip() and urlsplit(target).scheme in {"http", "https", "mailto"}:
            return f"[{text.strip()}]({target})"
    if name in {"b", "strong"} and text.strip():
        return "**" + text.strip() + "**"
    if name == "img":
        return node.get("alt", "")
    return text


def clean_content(node, url, remove_selectors):
    node = deepcopy(node)
    for selector in [
        "script",
        "style",
        "noscript",
        "form",
        "button",
        *remove_selectors,
    ]:
        for unwanted in list(node.select(selector)):
            if unwanted.parent is not None:
                unwanted.decompose()
    markdown = render(node, url)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip(), node

"""Convert selected DOM into Markdown without flattening lists and tables."""

import re
from copy import deepcopy
from urllib.parse import urljoin, urlsplit
from bs4 import NavigableString, Comment


from .tables import table_markdown


def render(node, url, depth=0):
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node)).replace("<", "&lt;").replace(">", "&gt;")
    name = node.name
    if name in {"script", "style", "noscript", "iframe", "object", "embed", "svg", "form", "button"}:
        return ""
    if name == "table":
        return "\n\n" + table_markdown(node, url, lambda cell: "".join(render(c, url, depth) for c in cell.children)) + "\n\n"
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
        href = node.get("href", "")
        if not isinstance(href, str):
            return text
        try:
            target = urljoin(url, href)
            parts = urlsplit(target)
            if parts.scheme in {"http", "https"}:
                if not parts.hostname:
                    return text
                _ = parts.port  # Validate malformed/out-of-range ports, too.
        except ValueError:
            return text  # Keep anchor text even when its destination is broken.
        target = target.replace("(", "%28").replace(")", "%29").replace("<", "%3C").replace(">", "%3E").replace(" ", "%20")
        if text.strip() and parts.scheme in {"http", "https", "mailto"}:
            return f"[{text.strip()}]({target})"
    if name in {"b", "strong"} and text.strip():
        return "**" + text.strip() + "**"
    if name == "img":
        return node.get("alt", "").replace("<", "&lt;").replace(">", "&gt;")
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

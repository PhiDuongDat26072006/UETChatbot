"""Conservative quality checks; a pass is not a semantic correctness proof."""

import re

ERROR_PHRASES = (
    "error establishing a database connection",
    "access denied",
    "verify you are human",
    "just a moment...",
    "404 not found",
)


def validate_html(content, profile, node=None):
    text = node.get_text(" ", strip=True) if node is not None else content
    reasons = []
    if re.search(r"</?(?:table|tr|td|th|div|span|script|style|iframe|p|a)\b[^>]*>", content, re.I):
        reasons.append("residual_html")
    if node is not None and (node.get("data-extraction-issue") or node.select("[data-extraction-issue]")):
        reasons.append("unresolved_table_structure")
    if not text.strip():
        reasons.append("empty_content")
    if any(p in text.lower() for p in ERROR_PHRASES):
        reasons.append("error_page")
    minimum = profile.get("quality", {}).get("min_chars", 100)
    if len(text) < minimum:
        reasons.append("thin_content")
    density = 0.0
    if node is not None:
        density = sum(len(a.get_text(" ", strip=True)) for a in node.select("a")) / max(
            1, len(text)
        )
        if density > profile.get("quality", {}).get("max_link_density", 0.65):
            reasons.append("link_heavy")
        if len(node.select("article")) > 1:
            reasons.append("multiple_articles")
    if text.count("�") / max(len(text), 1) > 0.01:
        reasons.append("encoding_damage")
    return {
        "status": "accepted" if not reasons else "review",
        "reasons": reasons,
        "metrics": {"text_length": len(text), "link_density": round(density, 3)},
    }


def validate_refinement(native, candidate):
    reasons = []
    if len(candidate.strip()) < 20:
        reasons.append("empty_or_short_response")
    if candidate.count("�") > 2:
        reasons.append("encoding_damage")
    native_numbers = set(re.findall(r"\b\d+(?:[.,/]\d+)*\b", native))
    candidate_numbers = set(re.findall(r"\b\d+(?:[.,/]\d+)*\b", candidate))
    if native_numbers - candidate_numbers:
        reasons.append("native_numbers_missing")
    if len(native) > 200 and len(candidate) < len(native) * 0.35:
        reasons.append("large_content_loss")
    return {
        "status": "needs_review" if reasons else "validated",
        "reasons": reasons,
        "semantic_accuracy_verified": False,
    }

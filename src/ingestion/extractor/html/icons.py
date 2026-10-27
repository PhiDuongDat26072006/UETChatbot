"""Remove decorative emoji from visible HTML text without changing URL attributes."""

import re
from bs4 import Comment, NavigableString

# Deliberately scoped pictograms; do not remove all Unicode symbols (math, units).
_BASE = (
    r"[\U0001F000-\U0001FAFF"
    r"\u231A\u231B\u23E9-\u23F3\u23F8-\u23FA"
    r"\u2600-\u2604\u260E\u2611\u2614\u2615\u2618\u261D"
    r"\u2620\u2622\u2623\u2626\u262A\u262E\u262F"
    r"\u2638-\u263A\u2648-\u2653\u265F\u2660\u2663\u2665\u2666\u2668"
    r"\u267B\u267E\u267F\u2692-\u2697\u2699\u269B\u269C"
    r"\u26A0\u26A1\u26A7\u26AA\u26AB\u26B0\u26B1\u26BD\u26BE"
    r"\u26C4\u26C5\u26C8\u26CE\u26CF\u26D1\u26D3\u26D4"
    r"\u26E9\u26EA\u26F0-\u26F5\u26F7-\u26FA\u26FD"
    r"\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D"
    r"\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E"
    r"\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27B0\u27BF"
    r"\u2B50\u2B55]"
)
_UNIT = _BASE + r"[\uFE0E\uFE0F\U0001F3FB-\U0001F3FF]*"
_EMOJI = re.compile(_UNIT + r"(?:\u200D" + _UNIT + r")*[\U000E0020-\U000E007F]*")
_KEYCAP = re.compile(r"([0-9#*])\uFE0F?\u20E3")


def clean_icons(text):
    """Keep keycap labels (1️⃣ -> 1); separate words formerly divided by emoji."""
    return _EMOJI.sub(" ", _KEYCAP.sub(r"\1", text))


def clean_html_icons(soup):
    """Mutate text nodes, and remove explicitly marked emoji images only."""
    changed = False
    for node in list(soup.find_all(string=True)):
        if isinstance(node, Comment) or node.parent.name in {
            "script",
            "style",
            "code",
            "pre",
        }:
            continue
        cleaned = clean_icons(str(node))
        if cleaned != str(node):
            node.replace_with(NavigableString(cleaned))
            changed = True
    for img in list(soup.find_all("img")):
        classes = set(img.get("class", []))
        if classes & {"emoji", "wp-smiley", "emojione"}:
            img.decompose()
            changed = True
        elif img.has_attr("alt"):
            cleaned = clean_icons(img["alt"])
            if cleaned != img["alt"]:
                img["alt"] = cleaned
                changed = True
    return changed

"""Page-level heuristics; no OCR or API calls."""


def assess_page(text, image_coverage, table_count, blocks, table_error=None):
    reasons = []
    if not text.strip():
        status = "OCR_REQUIRED" if image_coverage > 0.2 else "MANUAL_REVIEW"
        reasons.append("empty_text")
    elif text.count("�") / max(1, len(text)) > 0.01:
        status = "OCR_REQUIRED"
        reasons.append("encoding_damage")
    elif len(text.strip()) < 80 and image_coverage > 0.5:
        status = "OCR_REQUIRED"
        reasons.append("image_dominant_short_text")
    elif table_error:
        status = "LAYOUT_REVIEW"
        reasons.append("table_extraction_failed")
    else:
        status = "NATIVE_OK"
        # Overlapping vertical spans of horizontally separated text blocks suggest columns.
        for i, a in enumerate(blocks):
            for b in blocks[i + 1 :]:
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                if overlap > 20 and (a[2] + 15 < b[0] or b[2] + 15 < a[0]):
                    status = "LAYOUT_REVIEW"
                    reasons.append("possible_multiple_columns")
                    break
            if reasons:
                break
    return {
        "status": status,
        "reasons": reasons,
        "text_length": len(text),
        "image_coverage": round(image_coverage, 3),
        "table_count": table_count,
        "table_error": table_error,
    }

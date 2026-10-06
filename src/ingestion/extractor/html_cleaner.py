"""Các tiện ích làm sạch HTML, nhận diện bảng biểu, email, môn học và liên kết đám mây."""
from __future__ import annotations

import re
from bs4 import BeautifulSoup

from ..config import CLOUD_STORAGE_DOMAINS
from ..filters import clean_url, is_cloud_storage_url
from ..utils import clean_whitespace

# Các selector phổ biến chứa nội dung bài viết WordPress trên các site UET
WP_CONTENT_SELECTORS = [
    "article",
    "div.content_single",
    "div.content_pages",
    "div.content_detail",
    "div.blog-content",
    "div.entry-content",
    "div.post-content",
    "div.content-area",
    "div.td-post-content",
    "div.single-post-content",
    "section#pages",
    "main#main",
    "main",
    "div.main-content",
    "div.site-content",
    "div#content",
    "div.page-content",
]

# Các tag rác cần loại bỏ triệt để trong fallback
NOISE_TAGS = [
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "noscript",
    "svg",
    "form",
    "iframe",
    "button",
]

# Các class và id liên quan đến nhân sự, giảng viên, CTĐT được bảo vệ tuyệt đối
PROTECTED_CLASSES = {
    "team", "member", "profile", "faculty", "staff", "lecturer",
    "giang-vien", "can-bo", "curriculum", "program", "course", "subject",
    "mon-hoc", "dao-tao", "blog-content", "content_single", "content_pages",
    "content_detail",
}

# Regex nhận diện email cán bộ/giảng viên VNU và mã môn học UET
EMAIL_PATTERN = re.compile(r"[\w\.-]+@(?:[a-zA-Z0-9-]+\.)*vnu\.edu\.vn", re.IGNORECASE)
COURSE_CODE_PATTERN = re.compile(
    r"\b(?:INT|MAT|PHY|AIT|PES|ENG|BSA|EMA|CHE|BIO|ENV|GEO|MNS|ELT|TEL|EPN)\s*\d{4}\b",
    re.IGNORECASE,
)

# Danh sách từ khóa anchor text chung chung
GENERIC_LINK_TEXTS = {
    "tại đây", "tai day", "đây", "day",
    "tại link", "tai link", "link", "đường link", "duong link",
    "xem tại đây", "xem tai day", "xem chi tiết", "xem chi tiet",
    "tải về", "tai ve", "tải tại đây", "tai tai day", "tải file", "tai file",
    "tải xuống", "tai xuong",
    "download", "click here", "here", "chi tiết", "chi tiet",
}


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


def is_thin_content(
    content: str,
    title: str | None = None,
    min_length: int = 100,
    category: str | None = None,
) -> bool:
    """Kiểm tra nội dung có phải là stub rỗng hoặc thin content không có giá trị thông tin tra cứu."""
    if not content or not isinstance(content, str):
        return True

    clean = content.strip()
    if len(clean) < 30:
        return True

    emails = EMAIL_PATTERN.findall(clean)
    course_codes = COURSE_CODE_PATTERN.findall(clean)
    has_markdown_table = "|" in clean and "---" in clean

    non_generic_emails = [e for e in emails if not e.lower().startswith("fit@")]
    if len(non_generic_emails) >= 1 or len(course_codes) >= 1 or has_markdown_table:
        if len(clean) >= 40:
            return False

    if len(clean) < min_length:
        return True

    if title:
        norm_title = re.escape(title.strip())
        substantive = re.sub(norm_title, "", clean, flags=re.IGNORECASE)
        boilerplate_patterns = [
            r"©\s*VNU-UET[^\n]*",
            r"All rights reserved[^\n]*",
            r"Bản quyền thuộc về[^\n]*",
            r"Khoa Công nghệ Thông tin[^\n]*",
            r"Trường Đại học Công nghệ[^\n]*",
            r"Đại học Quốc gia Hà Nội[^\n]*",
            r"fit@vnu\.edu\.vn",
            r"\(024\)\s*3\s*754\s*7064",
        ]
        for bp in boilerplate_patterns:
            substantive = re.sub(bp, "", substantive, flags=re.IGNORECASE)

        substantive = substantive.strip()
        words = substantive.split()
        if len(substantive) < 50 or len(words) < 10:
            return True

    all_words = clean.split()
    if len(all_words) < 12:
        return True

    if not non_generic_emails and not course_codes and not has_markdown_table:
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if len(lines) >= 8:
            avg_line_len = sum(len(line) for line in lines) / len(lines)
            if avg_line_len < 30:
                menu_keywords = {
                    "trang chủ", "giới thiệu", "quá trình phát triển", "tầm nhìn và sứ mệnh",
                    "tầm nhìn", "sứ mệnh", "ban chủ nhiệm khoa", "ban chủ nhiệm",
                    "các bộ môn và phòng thí nghiệm", "giảng viên", "liên hệ",
                    "đào tạo", "đại học", "sau đại học", "nghiên cứu",
                    "các hướng nghiên cứu", "các đề tài dự án", "các sản phẩm công nghệ",
                    "chuyên san cntt-tt", "hợp tác", "đối tác", "đối tác khối công nghệ",
                    "đối tác khối hàn lâm", "tin tức", "người học", "thành tích nổi bật",
                    "thông tin việc làm", "các câu hỏi thường gặp", "cựu sinh viên",
                    "ngôn ngữ", "sitemap", "menu",
                }
                menu_matches = sum(
                    1 for line in lines
                    if line.lower().strip() in menu_keywords
                )
                if (menu_matches / len(lines)) > 0.4:
                    return True

    return False


def extract_cloud_links(
    html_text_or_soup: str | BeautifulSoup,
    allowed_domains: tuple[str, ...] = CLOUD_STORAGE_DOMAINS,
) -> list[dict[str, str]]:
    """Trích xuất và làm sạch danh sách liên kết tới các dịch vụ đám mây từ HTML."""
    if isinstance(html_text_or_soup, str):
        if not html_text_or_soup.strip():
            return []
        soup = BeautifulSoup(html_text_or_soup, "lxml")
    elif isinstance(html_text_or_soup, BeautifulSoup):
        soup = html_text_or_soup
    else:
        return []

    cloud_links: dict[str, dict[str, str]] = {}

    for a_tag in soup.find_all("a", href=True):
        if a_tag.find_parent(["nav", "header", "footer", "aside"]):
            continue

        raw_href = a_tag["href"].strip()
        if not raw_href or not is_cloud_storage_url(raw_href, allowed_domains):
            continue

        cleaned_url = clean_url(raw_href)
        if not cleaned_url:
            continue

        anchor_text = clean_whitespace(a_tag.get_text(separator=" ", strip=True))

        is_generic = (
            not anchor_text
            or anchor_text.lower() in GENERIC_LINK_TEXTS
            or anchor_text.startswith("http://")
            or anchor_text.startswith("https://")
        )

        if is_generic:
            context_text = ""
            if a_tag.previous_sibling:
                prev_str = (
                    a_tag.previous_sibling.get_text(strip=True)
                    if hasattr(a_tag.previous_sibling, "get_text")
                    else str(a_tag.previous_sibling).strip()
                )
                if prev_str:
                    context_text = prev_str

            if not context_text and a_tag.parent and a_tag.parent.name in ("p", "li", "div", "td", "span", "h1", "h2", "h3", "h4", "h5", "h6"):
                parent_full = a_tag.parent.get_text(separator=" ", strip=True)
                if anchor_text and anchor_text in parent_full:
                    context_text = parent_full.split(anchor_text)[0].strip()
                elif not anchor_text:
                    context_text = parent_full.strip()

            if context_text:
                cleaned_ctx = re.sub(r"^[\s\-\*•\d+\.]+\s*", "", context_text).strip()
                cleaned_ctx = re.sub(r"[:：\s-]+$", "", cleaned_ctx).strip()
                if 3 <= len(cleaned_ctx) <= 100:
                    anchor_text = cleaned_ctx

        if not anchor_text or anchor_text.lower() in GENERIC_LINK_TEXTS:
            if "forms.gle" in cleaned_url or "/forms" in cleaned_url:
                anchor_text = "Biểu mẫu đăng ký"
            else:
                anchor_text = "Tài liệu đính kèm"
        else:
            anchor_text = re.sub(r"[:：\s-]+$", "", anchor_text).strip()

        if cleaned_url not in cloud_links:
            cloud_links[cleaned_url] = {"text": anchor_text, "url": cleaned_url}
        else:
            existing_text = cloud_links[cleaned_url]["text"]
            default_labels = ("Tài liệu đính kèm", "Biểu mẫu đăng ký")
            if existing_text in default_labels and anchor_text not in default_labels:
                cloud_links[cleaned_url]["text"] = anchor_text

    return list(cloud_links.values())


def format_cloud_links_markdown(cloud_links: list[dict[str, str]]) -> str:
    """Định dạng danh sách cloud links thành khối Markdown đính kèm cuối bài viết."""
    if not cloud_links:
        return ""
    lines = ["[Tài liệu & Biểu mẫu liên kết]:"]
    for item in cloud_links:
        text = item.get("text", "").strip() or "Tài liệu liên kết"
        clean_item_text = re.sub(r"[:：\s-]+$", "", text)
        lines.append(f"- {clean_item_text}: {item['url']}")
    return "\n\n" + "\n".join(lines)

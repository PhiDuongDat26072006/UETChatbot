"""Unit tests cho module html_extractor — bóc tách HTML và bảo tồn liên kết tài liệu đám mây."""
import pytest
from bs4 import BeautifulSoup

from src.ingestion.extractor.html_extractor import (
    extract_cloud_links,
    extract_html_content,
    format_cloud_links_markdown,
)
from src.ingestion.filters import is_cloud_storage_url


class TestExtractCloudLinks:
    """Kiểm tra hàm extract_cloud_links trích xuất tài liệu đám mây từ HTML."""

    def test_extract_google_drive_explicit_text(self):
        html = """
        <div class="entry-content">
            <p>Khoa CNTT thông báo danh sách đề tài khóa luận tốt nghiệp năm học 2023-2024.</p>
            <p><a href="https://drive.google.com/drive/folders/1w1r0Hk8eZ_N_7-0eS-9P0F">Định hướng nghiên cứu KLTN</a></p>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["text"] == "Định hướng nghiên cứu KLTN"
        assert links[0]["url"] == "https://drive.google.com/drive/folders/1w1r0Hk8eZ_N_7-0eS-9P0F"

    def test_extract_google_forms_generic_text_resolved(self):
        html = """
        <div class="entry-content">
            <p>Sinh viên tham gia seminar vui lòng điền thông tin.</p>
            <p>Form đăng ký đề tài: <a href="https://forms.gle/abCD1234efGH">tại đây</a></p>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["text"] == "Form đăng ký đề tài"
        assert links[0]["url"] == "https://forms.gle/abCD1234efGH"

    def test_extract_google_docs_parent_context(self):
        html = """
        <div class="post-content">
            <ul>
                <li>Đề cương chi tiết môn học: <a href="https://docs.google.com/document/d/1XyZ/edit?usp=sharing">tại link</a></li>
            </ul>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["text"] == "Đề cương chi tiết môn học"
        assert "docs.google.com" in links[0]["url"]
        assert "usp=sharing" in links[0]["url"]

    def test_extract_onedrive_and_sharepoint(self):
        html = """
        <div>
            <p><a href="https://1drv.ms/u/s!Alkjshdf832">Tài liệu tham khảo môn học</a></p>
            <p><a href="https://uetvnu-my.sharepoint.com/:b:/g/personal/tailieu.pdf">Biểu mẫu xét tốt nghiệp</a></p>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 2
        urls = [item["url"] for item in links]
        assert "https://1drv.ms/u/s!Alkjshdf832" in urls
        assert "https://uetvnu-my.sharepoint.com/:b:/g/personal/tailieu.pdf" in urls

    def test_extract_dropbox(self):
        html = """
        <p><a href="https://www.dropbox.com/s/xyz/slides.pdf?dl=0">Slide bài giảng tuần 1</a></p>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["text"] == "Slide bài giảng tuần 1"
        assert "dropbox.com" in links[0]["url"]

    def test_cleans_tracking_query_params(self):
        html = """
        <p><a href="https://forms.gle/abCD1234efGH?utm_source=facebook&utm_medium=post&fbclid=IwAR123">Đăng ký tham gia</a></p>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["url"] == "https://forms.gle/abCD1234efGH"
        assert "utm_source" not in links[0]["url"]
        assert "fbclid" not in links[0]["url"]

    def test_rejects_non_whitelisted_links(self):
        html = """
        <div>
            <p><a href="https://facebook.com/uet.vnu.edu.vn">Fanpage chính thức</a></p>
            <p><a href="https://google.com/search?q=uet">Tìm kiếm Google</a></p>
            <p><a href="https://youtube.com/watch?v=123">Video giới thiệu</a></p>
            <p><a href="https://drive.google.com/file/d/123/view">File quy chế</a></p>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["url"] == "https://drive.google.com/file/d/123/view"
        assert links[0]["text"] == "File quy chế"

    def test_ignores_links_in_nav_and_footer(self):
        html = """
        <div>
            <nav>
                <a href="https://forms.gle/nav_form">Khảo sát sinh viên</a>
            </nav>
            <article>
                <p>Nội dung thông báo hướng dẫn tốt nghiệp.</p>
                <p><a href="https://drive.google.com/drive/folders/valid_folder">Thư mục đề tài</a></p>
            </article>
            <footer>
                <a href="https://docs.google.com/document/d/footer_doc">Chính sách bảo mật</a>
            </footer>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["url"] == "https://drive.google.com/drive/folders/valid_folder"
        assert links[0]["text"] == "Thư mục đề tài"

    def test_deduplicates_identical_urls(self):
        html = """
        <div>
            <p><a href="https://forms.gle/sample_form">tại đây</a></p>
            <p>Form đăng ký chính thức: <a href="https://forms.gle/sample_form">Đăng ký tại đây</a></p>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 1
        assert links[0]["url"] == "https://forms.gle/sample_form"

    def test_fallback_labels_when_no_context(self):
        html = """
        <div>
            <p><a href="https://forms.gle/sample_form"></a></p>
            <p><a href="https://drive.google.com/file/d/123"></a></p>
        </div>
        """
        links = extract_cloud_links(html)
        assert len(links) == 2
        labels = {item["url"]: item["text"] for item in links}
        assert labels["https://forms.gle/sample_form"] == "Biểu mẫu đăng ký"
        assert labels["https://drive.google.com/file/d/123"] == "Tài liệu đính kèm"

    def test_empty_or_invalid_html(self):
        assert extract_cloud_links("") == []
        assert extract_cloud_links("   ") == []
        assert extract_cloud_links("<p>Trang web không có liên kết nào</p>") == []

    def test_soup_object_input(self):
        soup = BeautifulSoup('<a href="https://forms.gle/abc">Đăng ký</a>', "lxml")
        links = extract_cloud_links(soup)
        assert len(links) == 1
        assert links[0]["text"] == "Đăng ký"


class TestFormatCloudLinksMarkdown:
    """Kiểm tra hàm format_cloud_links_markdown định dạng khối Markdown."""

    def test_empty_list_returns_empty_string(self):
        assert format_cloud_links_markdown([]) == ""

    def test_markdown_formatting_structure(self):
        cloud_links = [
            {"text": "Định hướng nghiên cứu", "url": "https://drive.google.com/drive/folders/123"},
            {"text": "Form đăng ký đề tài", "url": "https://forms.gle/abc"},
        ]
        result = format_cloud_links_markdown(cloud_links)
        assert result.startswith("\n\n[Tài liệu & Biểu mẫu liên kết]:")
        assert "- Định hướng nghiên cứu: https://drive.google.com/drive/folders/123" in result
        assert "- Form đăng ký đề tài: https://forms.gle/abc" in result

    def test_strips_trailing_colons_from_text(self):
        cloud_links = [
            {"text": "Định hướng nghiên cứu:", "url": "https://drive.google.com/123"},
        ]
        result = format_cloud_links_markdown(cloud_links)
        assert "- Định hướng nghiên cứu: https://drive.google.com/123" in result
        assert "::" not in result


class TestExtractHtmlContentEndToEnd:
    """Kiểm tra extract_html_content tích hợp bảo tồn liên kết tài liệu đám mây."""

    def test_extract_html_content_with_cloud_links(self):
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Thông báo Đăng ký KLTN - FIT UET</title></head>
        <body>
            <div class="entry-content">
                <h1>Thông báo Đăng ký Khóa luận Tốt nghiệp</h1>
                <p>Khoa Công nghệ Thông tin thông báo kế hoạch thực hiện Khóa luận Tốt nghiệp kỳ 2 năm học 2023-2024 dành cho sinh viên khóa K65. Các em sinh viên chú ý theo dõi kỹ các hướng dẫn và biểu mẫu.</p>
                <p>Danh sách định hướng nghiên cứu và giảng viên hướng dẫn: <a href="https://drive.google.com/drive/folders/sample_folder">Xem tại đây</a></p>
                <p>Form đăng ký nguyện vọng: <a href="https://forms.gle/kltn_form">Điền thông tin</a></p>
            </div>
        </body>
        </html>
        """
        res = extract_html_content("https://fit.uet.vnu.edu.vn/thong-bao-kltn/", html_text=sample_html, min_length=50)
        assert res is not None
        assert "Thông báo Đăng ký KLTN" in res["title"]
        assert "[Tài liệu & Biểu mẫu liên kết]:" in res["text"]
        assert "https://drive.google.com/drive/folders/sample_folder" in res["text"]
        assert "https://forms.gle/kltn_form" in res["text"]
        assert len(res["cloud_links"]) == 2

    def test_extract_html_content_without_cloud_links(self):
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Giới thiệu Khoa Điện tử Viễn thông</title></head>
        <body>
            <div class="entry-content">
                <h1>Lịch sử hình thành và phát triển</h1>
                <p>Khoa Điện tử Viễn thông được thành lập với mục tiêu đào tạo nguồn nhân lực chất lượng cao trong các lĩnh vực viễn thông, vi mạch và hệ thống nhúng thông minh hàng đầu tại Việt Nam.</p>
            </div>
        </body>
        </html>
        """
        res = extract_html_content("https://fet.uet.vnu.edu.vn/gioi-thieu/", html_text=sample_html, min_length=50)
        assert res is not None
        assert "[Tài liệu & Biểu mẫu liên kết]:" not in res["text"]
        assert res["cloud_links"] == []

    def test_extract_html_content_too_short_rejected(self):
        short_html = "<p>Quá ngắn</p>"
        res = extract_html_content("https://fit.uet.vnu.edu.vn/short/", html_text=short_html, min_length=100)
        assert res is None

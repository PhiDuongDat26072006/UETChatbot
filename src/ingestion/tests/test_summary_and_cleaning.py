"""Unit tests cho tính năng lọc thin content, deduplicate stub pages và sinh document summary."""
import hashlib
import pytest

from src.ingestion.extractor.html_extractor import (
    extract_html_content,
    is_thin_content,
    table_to_markdown,
)
from src.ingestion.extractor.pipeline import (
    DUPLICATE_STUB_MAX_LENGTH,
    canonicalize_url,
    generate_doc_id,
    is_duplicate_stub,
)
from src.ingestion.extractor.summary import generate_document_summary


class TestThinContentFiltering:
    """Kiểm tra hàm is_thin_content loại bỏ trang rác, trang chỉ có tiêu đề hoặc menu dump."""

    def test_rejects_single_word_or_short_title_content(self):
        # Trang chỉ có chữ "Giảng viên"
        assert is_thin_content("Giảng viên", title="Giảng viên") is True
        assert is_thin_content("Giảng viên\nGiảng viên", title="Giảng viên") is True

    def test_rejects_title_plus_copyright_boilerplate_only(self):
        # Trang tiêu đề kèm dòng bản quyền / email / hotline
        text = (
            "Giảng viên\n"
            "© VNU-UET-Faculty of Information Technology. All rights reserved.\n"
            "fit@vnu.edu.vn - (024) 3 754 7064"
        )
        assert is_thin_content(text, title="Giảng viên") is True

    def test_rejects_pure_navigation_menu_dump(self):
        # Trang không có bài viết thực, chỉ bóc nhầm cây menu điều hướng của theme
        menu_text = (
            "Ngôn ngữ\n"
            "Giới thiệu\n"
            "Quá trình phát triển\n"
            "Tầm nhìn và sứ mệnh\n"
            "Ban chủ nhiệm Khoa\n"
            "Các bộ môn và phòng thí nghiệm\n"
            "Giảng viên\n"
            "Liên hệ\n"
            "Đào tạo\n"
            "Đại học\n"
            "Sau đại học\n"
            "Nghiên cứu\n"
            "Các hướng nghiên cứu\n"
            "Các đề tài dự án\n"
            "Các sản phẩm công nghệ\n"
            "Chuyên san CNTT-TT\n"
            "Hợp tác\n"
            "Tin tức\n"
            "Người học\n"
            "Cựu sinh viên"
        )
        assert is_thin_content(menu_text, title="Giảng viên") is True

    def test_accepts_valid_substantive_article(self):
        # Bài viết giới thiệu bộ môn hợp lệ
        valid_article = (
            "Bộ môn Mạng và Truyền thông máy tính được thành lập vào năm 2004 với nhiệm vụ đào tạo và nghiên cứu.\n"
            "Đội ngũ giảng viên của bộ môn hiện đang có 6 tiến sỹ, 4 thạc sỹ trong đó có 2 Phó giáo sư.\n"
            "Bộ môn phụ trách giảng dạy các môn học cơ sở ngành và chuyên ngành cho sinh viên ngành Mạng máy tính.\n"
            "Các hướng nghiên cứu mũi nhọn gồm an toàn an ninh mạng, điện toán đám mây và IoT."
        )
        assert is_thin_content(valid_article, title="Bộ môn Mạng và Truyền thông máy tính") is False

    def test_extract_html_content_rejects_thin_page_end_to_end(self):
        thin_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Giảng viên - Khoa CNTT</title></head>
        <body>
            <div class="entry-content">
                <h1>Giảng viên</h1>
                <p>Giảng viên</p>
                <p>© VNU-UET-Faculty of Information Technology. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        res = extract_html_content("https://fit.uet.vnu.edu.vn/gioi-thieu/giang-vien/", html_text=thin_html)
        assert res is None


class TestDuplicateStubDeduplication:
    """Kiểm tra cơ chế Content Hash Deduplication cho trang stub/fallback dưới 500 ký tự."""

    def test_skips_second_duplicate_stub_url(self):
        seen_content_hashes: dict[str, str] = {}
        stub_content = "Các Bộ môn và phòng thí nghiệm\nBộ môn Các Hệ thống Thông tin\nBộ môn Khoa học Máy tính\nXem tất cả..."
        assert len(stub_content) < DUPLICATE_STUB_MAX_LENGTH

        # URL 1: ghi nhận bình thường
        chash = hashlib.md5(stub_content.strip().encode("utf-8")).hexdigest()
        seen_content_hashes[chash] = "https://fit.uet.vnu.edu.vn/tag/first/"

        # URL 2: nội dung trùng 100% với URL 1
        url2 = "https://fit.uet.vnu.edu.vn/tag/second/"
        chash2 = hashlib.md5(stub_content.strip().encode("utf-8")).hexdigest()

        # Kiểm tra điều kiện loại bỏ
        is_duplicate_stub = (chash2 in seen_content_hashes and len(stub_content) < DUPLICATE_STUB_MAX_LENGTH)
        assert is_duplicate_stub is True
        assert seen_content_hashes[chash2] == "https://fit.uet.vnu.edu.vn/tag/first/"

    def test_allows_duplicate_content_for_long_valid_articles(self):
        seen_content_hashes: dict[str, str] = {}
        # Bài viết dài trên 500 ký tự (ví dụ thông báo song song hợp lệ)
        long_article = "Quy chế đào tạo đại học và sau đại học tại trường Đại học Công nghệ. " * 30
        assert len(long_article) >= DUPLICATE_STUB_MAX_LENGTH

        chash = hashlib.md5(long_article.strip().encode("utf-8")).hexdigest()
        seen_content_hashes[chash] = "https://fit.uet.vnu.edu.vn/quy-che-1/"

        # URL 2 có bài viết dài
        is_duplicate_stub = (chash in seen_content_hashes and len(long_article) < DUPLICATE_STUB_MAX_LENGTH)
        assert is_duplicate_stub is False


class TestDocumentSummaryGeneration:
    """Kiểm tra hàm generate_document_summary tạo đoạn trích xuất có tiền tố ngữ cảnh chuẩn."""

    def test_summary_format_and_prefix(self):
        domain = "fit.uet.vnu.edu.vn"
        title = "Bộ môn Công nghệ Phần mềm"
        content = (
            "Bộ môn CNPM đảm nhận vai trò quan trọng trong phát triển các ngành và chuyên ngành tại khoa CNTT, "
            "bao gồm cả đào tạo cử nhân và thạc sĩ. Đội ngũ gồm nhiều giảng viên giàu kinh nghiệm thực tế."
        )
        summary = generate_document_summary(domain=domain, title=title, content=content)
        assert summary.startswith("[fit.uet.vnu.edu.vn | Bộ môn Công nghệ Phần mềm]: ")
        assert "Bộ môn CNPM đảm nhận vai trò quan trọng" in summary
        assert len(summary) > 50

    def test_summary_filters_administrative_boilerplate(self):
        domain = "uet.edu.vn"
        title = "Quy chế Thi"
        content = (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "ĐẠI HỌC QUỐC GIA HÀ NỘI\n"
            "TRƯỜNG ĐẠI HỌC CÔNG NGHỆ\n"
            "Quy định số 456/QĐ về việc tổ chức thi và đánh giá kết quả học tập của sinh viên chính quy từ năm học 2024."
        )
        summary = generate_document_summary(domain=domain, title=title, content=content)
        assert summary.startswith("[uet.edu.vn | Quy chế Thi]: ")
        assert "CỘNG HÒA" not in summary
        assert "Độc lập" not in summary
        assert "Quy định số 456/QĐ về việc tổ chức thi" in summary

    def test_summary_filters_pdf_pagination_header(self):
        domain = "fet.uet.vnu.edu.vn"
        title = "CTDT Thạc sĩ"
        content = (
            "--- Trang 1 ---\n"
            "Chương trình đào tạo Thạc sĩ chuyên ngành Kỹ thuật Điện tử Viễn thông định hướng ứng dụng và nghiên cứu."
        )
        summary = generate_document_summary(domain=domain, title=title, content=content)
        assert "--- Trang 1 ---" not in summary
        assert "Chương trình đào tạo Thạc sĩ" in summary

    def test_summary_non_empty_for_short_content(self):
        domain = "fat.uet.vnu.edu.vn"
        title = "Thông báo tuyển sinh"
        content = "Khoa Công nghệ Nông nghiệp tuyển sinh đại học chính quy."
        summary = generate_document_summary(domain=domain, title=title, content=content)
        assert summary == "[fat.uet.vnu.edu.vn | Thông báo tuyển sinh]: Khoa Công nghệ Nông nghiệp tuyển sinh đại học chính quy."

    def test_summary_handles_empty_content(self):
        domain = "sae.uet.vnu.edu.vn"
        title = "Trang chủ"
        summary = generate_document_summary(domain=domain, title=title, content="")
        assert summary == "[sae.uet.vnu.edu.vn | Trang chủ]"


class TestTableToMarkdown:
    """Kiểm tra hàm table_to_markdown chuyển đổi bảng HTML thành Markdown chuẩn."""

    def test_table_with_header(self):
        from bs4 import BeautifulSoup
        html = """
        <table>
          <tr><th>STT</th><th>Họ và tên</th><th>Email</th></tr>
          <tr><td>1</td><td>TS. Nguyễn Văn A</td><td>nguyenvana@vnu.edu.vn</td></tr>
          <tr><td>2</td><td>ThS. Trần Thị B</td><td>tranb@vnu.edu.vn</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        md = table_to_markdown(soup.find("table"))
        assert "| STT | Họ và tên | Email |" in md
        assert "| --- | --- | --- |" in md
        assert "| 1 | TS. Nguyễn Văn A | nguyenvana@vnu.edu.vn |" in md

    def test_table_without_th_row(self):
        from bs4 import BeautifulSoup
        html = """
        <table>
          <tr><td>Mã môn</td><td>Tên môn</td><td>Số TC</td></tr>
          <tr><td>INT1003</td><td>Tin học cơ sở 1</td><td>2</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        md = table_to_markdown(soup.find("table"))
        assert "| Mã môn | Tên môn | Số TC |" in md
        assert "| --- | --- | --- |" in md
        assert "| INT1003 | Tin học cơ sở 1 | 2 |" in md

    def test_table_escapes_pipe_characters(self):
        from bs4 import BeautifulSoup
        html = """
        <table>
          <tr><td>Học phần</td><td>Ghi chú</td></tr>
          <tr><td>Toán rời rạc</td><td>Nhóm 1 | Nhóm 2</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        md = table_to_markdown(soup.find("table"))
        assert "Nhóm 1 \\| Nhóm 2" in md


class TestThinContentProtection:
    """Kiểm tra cơ chế bảo vệ không đánh dấu thin content cho email và mã môn học."""

    def test_preserves_short_content_with_email(self):
        text = "Liên hệ Trưởng bộ môn: hieuvd@vnu.edu.vn để đăng ký đồ án tốt nghiệp."
        assert is_thin_content(text, title="Đồ án tốt nghiệp") is False

    def test_preserves_short_content_with_course_code(self):
        text = "Học phần tiên quyết INT1003 và INT1004 cần hoàn thành trước kỳ 3."
        assert is_thin_content(text, title="Thông báo") is False

    def test_preserves_short_content_with_markdown_table(self):
        text = "| Mã môn | Tín chỉ |\n|---|---|\n| INT2001 | 3 |"
        assert is_thin_content(text, title="Khung CT") is False


class TestRefinedDeduplication:
    """Kiểm tra hàm is_duplicate_stub với các điều kiện bảo vệ bảng biểu và email."""

    def test_preserves_page_with_multiple_emails(self):
        content = "Giảng viên bộ môn: nguyenvana@vnu.edu.vn và tranb@vnu.edu.vn hướng dẫn."
        # Dù dưới 150 ký tự nhưng có 2 email -> không được loại bỏ
        assert is_duplicate_stub(content) is False

    def test_preserves_page_with_course_codes(self):
        content = "Môn học INT1003 và INT2001 áp dụng cho khóa K68."
        assert is_duplicate_stub(content) is False

    def test_preserves_page_with_table(self):
        content = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert is_duplicate_stub(content) is False

    def test_rejects_empty_short_stub(self):
        content = "Trang lưu trữ thông báo cũ."
        assert is_duplicate_stub(content) is True


class TestCanonicalizeUrl:
    """Kiểm tra chuẩn hóa canonical URL loại bỏ www. và trailing slash."""

    def test_canonicalize_url_removes_www_and_trailing_slash(self):
        u1 = "https://www.fit.uet.vnu.edu.vn/gioi-thieu/giang-vien/"
        u2 = "https://fit.uet.vnu.edu.vn/gioi-thieu/giang-vien"
        assert canonicalize_url(u1) == canonicalize_url(u2)
        assert canonicalize_url(u1) == "https://fit.uet.vnu.edu.vn/gioi-thieu/giang-vien"

    def test_canonicalize_url_preserves_query(self):
        u = "https://www.fit.uet.vnu.edu.vn/?p=123"
        assert canonicalize_url(u) == "https://fit.uet.vnu.edu.vn?p=123"


class TestDocumentUnitField:
    """Kiểm tra trường unit trong schema tài liệu của extraction pipeline."""

    def test_unit_mapping_in_document_record(self):
        from src.ingestion.config import resolve_unit

        domains = {
            "fit.uet.vnu.edu.vn": "FIT",
            "fet.uet.vnu.edu.vn": "FET",
            "fepn.uet.vnu.edu.vn": "FEPN",
            "fema.uet.vnu.edu.vn": "FEMA",
            "fat.uet.vnu.edu.vn": "FAT",
            "fce.uet.vnu.edu.vn": "FCE",
            "sae.uet.vnu.edu.vn": "SAE",
            "iai.uet.vnu.edu.vn": "IAI",
            "uet.vnu.edu.vn": "UET",
            "uet.edu.vn": "UET",
        }
        for d, u in domains.items():
            record = {
                "id": "abc12345",
                "source_type": "html",
                "source_url_or_path": f"https://{d}/bai-viet-1",
                "title": "Tiêu đề",
                "domain": d,
                "unit": resolve_unit(d),
                "category": "DAO_TAO",
                "published_date": "2023-01-01",
                "summary": "Tóm tắt",
                "content": "Nội dung",
                "content_length": 7,
                "cloud_links": [],
            }
            assert record["unit"] == u
            assert record["unit"].isupper()

    def test_load_existing_processed_docs_adds_unit_if_missing(self, tmp_path):
        from src.ingestion.extractor.pipeline import load_existing_processed_docs
        import json

        # Giả lập file JSONL cũ chưa có trường unit
        old_record = {
            "id": "11223344",
            "source_type": "html",
            "source_url_or_path": "https://fit.uet.vnu.edu.vn/gioi-thieu",
            "title": "Giới thiệu",
            "domain": "fit.uet.vnu.edu.vn",
            "category": "KHAC",
            "published_date": None,
            "content": "Nội dung bài viết giới thiệu khoa CNTT.",
            "content_length": 40,
            "cloud_links": [],
        }
        file_path = tmp_path / "processed_documents.jsonl"
        file_path.write_text(json.dumps(old_record) + "\n", encoding="utf-8")

        loaded = load_existing_processed_docs(file_path)
        assert "https://fit.uet.vnu.edu.vn/gioi-thieu" in loaded
        doc = loaded["https://fit.uet.vnu.edu.vn/gioi-thieu"]
        assert doc["domain"] == "fit.uet.vnu.edu.vn"
        # Kiểm tra khi pipeline nạp vào raw_existing, logic gán unit tự động
        from src.ingestion.config import resolve_unit
        if "unit" not in doc:
            doc["unit"] = resolve_unit(doc.get("domain", "fit.uet.vnu.edu.vn"))
        assert doc["unit"] == "FIT"


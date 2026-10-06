"""Unit tests cho module date_extractor — trích xuất và chuẩn hóa thời gian công bố."""
import pytest
from bs4 import BeautifulSoup

from src.ingestion.extractor.date_extractor import (
    extract_date_from_filename,
    extract_date_from_text,
    extract_date_from_url,
    extract_doc_date,
    extract_html_date,
    is_evergreen_page,
    is_recent_document,
    normalize_date_string,
)
from src.ingestion.extractor.html_extractor import extract_html_content
from src.ingestion.extractor.doc_extractor import extract_document_text


class TestNormalizeDateString:
    """Kiểm tra hàm chuẩn hóa chuỗi ngày tháng."""

    def test_iso_datetime_with_timezone(self):
        assert normalize_date_string("2023-04-18T08:30:00+07:00") == "2023-04-18"

    def test_iso_date_only(self):
        assert normalize_date_string("2023-04-18") == "2023-04-18"

    def test_vietnamese_slash_format(self):
        assert normalize_date_string("18/04/2023") == "2023-04-18"
        assert normalize_date_string("5/4/2023") == "2023-04-05"

    def test_vietnamese_dash_format(self):
        assert normalize_date_string("18-04-2023") == "2023-04-18"

    def test_year_month(self):
        assert normalize_date_string("2023-04") == "2023-04"
        assert normalize_date_string("2023-4") == "2023-04"

    def test_year_only(self):
        assert normalize_date_string("2023") == "2023"

    def test_invalid_strings(self):
        assert normalize_date_string("") is None
        assert normalize_date_string(None) is None
        assert normalize_date_string("not_a_date") is None
        assert normalize_date_string("32/13/2023") is None
        assert normalize_date_string("1800") is None  # Ngoài khoảng 1990-2035


class TestExtractDateFromUrl:
    """Kiểm tra trích xuất ngày từ đường dẫn URL."""

    def test_url_with_full_date(self):
        url = "https://fet.uet.vnu.edu.vn/2023/04/18/dao-tao-sau-dai-hoc/"
        assert extract_date_from_url(url) == "2023-04-18"

    def test_url_with_year_and_month(self):
        url = "https://fet.uet.vnu.edu.vn/2023/04/thong-bao-tuyen-sinh/"
        assert extract_date_from_url(url) == "2023-04"

    def test_url_with_year_only(self):
        url = "https://uet.edu.vn/2022/hoi-thao-quoc-te/"
        assert extract_date_from_url(url) == "2022"

    def test_url_without_date(self):
        assert extract_date_from_url("https://fit.uet.vnu.edu.vn/gioi-thieu/cac-bo-mon/") is None
        assert extract_date_from_url("https://uet.edu.vn/4593-2/") is None


class TestExtractDateFromText:
    """Kiểm tra trích xuất ngày từ đoạn văn bản (500 ký tự đầu)."""

    def test_standard_vietnamese_administrative_date(self):
        text = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nHà Nội, ngày 18 tháng 4 năm 2023\nTHÔNG BÁO"
        assert extract_date_from_text(text) == "2023-04-18"

    def test_single_digit_day_and_month(self):
        text = "Hà Nội, ngày 5 tháng 9 năm 2024. Thông báo tựu trường..."
        assert extract_date_from_text(text) == "2024-09-05"

    def test_slash_date_in_text(self):
        text = "Thông báo số 299/ĐT ngày 30/09/2022 về việc hướng dẫn sinh viên..."
        assert extract_date_from_text(text) == "2022-09-30"

    def test_month_and_year_in_text(self):
        text = "Kế hoạch tốt nghiệp tháng 6 năm 2024 dành cho sinh viên khóa K65..."
        assert extract_date_from_text(text) == "2024-06"

    def test_academic_year_in_text(self):
        text = "Thông báo công tác sinh viên năm học 2023-2024..."
        assert extract_date_from_text(text) == "2023"

    def test_no_date_in_text(self):
        text = "Chương trình đào tạo ngành Khoa học Máy tính bao gồm các học phần cơ bản và nâng cao."
        assert extract_date_from_text(text) is None


class TestExtractDateFromFilename:
    """Kiểm tra trích xuất ngày từ tên tệp tin."""

    def test_academic_year_in_filename(self):
        assert extract_date_from_filename("Danh-sach-de-tai-2022-2023.docx") == "2022"
        assert extract_date_from_filename("Ke-hoach_2023-2024_signed.pdf") == "2023"

    def test_full_date_in_filename(self):
        assert extract_date_from_filename("2026.4.10_725_Signed.pdf") == "2026-04-10"
        assert extract_date_from_filename("15-05-2024_Thong-bao.pdf") == "2024-05-15"

    def test_month_year_in_filename(self):
        assert extract_date_from_filename("BM_CHTTT_12.2025-1.pdf") == "2025-12"
        assert extract_date_from_filename("GioiThieu_BoMon_CNPM_2024.09.pdf") == "2024-09"

    def test_embedded_year_in_filename(self):
        assert extract_date_from_filename("Gioi-thieu-huong-nghien-cuu-Lab-HMI2024.pdf") == "2024"
        assert extract_date_from_filename("Procon_2019_kyogi_vi.docx") == "2019"

    def test_filename_without_date(self):
        assert extract_date_from_filename("Don-xin-nghi-hoc.pdf") is None
        assert extract_date_from_filename("Mau-Bien-ban-Test.doc") is None


class TestExtractHtmlDate:
    """Kiểm tra hàm extract_html_date kết hợp nhiều nguồn (Trafilatura, meta, URL, text)."""

    def test_extract_from_article_published_time_meta(self):
        html = """
        <html>
        <head>
            <meta property="article:published_time" content="2023-04-18T08:30:00+07:00" />
        </head>
        <body><p>Nội dung bài viết thông báo tuyển sinh...</p></body>
        </html>
        """
        assert extract_html_date(html, url="https://fet.uet.vnu.edu.vn/post/") == "2023-04-18"

    def test_extract_from_pubdate_meta(self):
        html = """
        <html>
        <head>
            <meta name="pubdate" content="2022-11-20" />
        </head>
        <body><p>Nội dung bài viết kỷ niệm ngày nhà giáo...</p></body>
        </html>
        """
        assert extract_html_date(html) == "2022-11-20"

    def test_extract_from_time_tag(self):
        html = """
        <html>
        <body>
            <span class="posted-on">
                <time class="entry-date published" datetime="2024-01-15T10:00:00+07:00">15 Tháng Một, 2024</time>
            </span>
            <p>Thông báo lịch thi học kỳ 1 năm học 2023-2024...</p>
        </body>
        </html>
        """
        assert extract_html_date(html) == "2024-01-15"

    def test_fallback_to_url_when_no_html_meta(self):
        html = "<html><body><p>Trang không có meta date</p></body></html>"
        url = "https://fet.uet.vnu.edu.vn/2021/03/19/thong-bao-khoa-hoc/"
        assert extract_html_date(html, url=url) == "2021-03-19"

    def test_fallback_to_text_when_no_meta_and_no_url_date(self):
        html = "<html><body><p>Hà Nội, ngày 12 tháng 5 năm 2023. Ban chủ nhiệm thông báo...</p></body></html>"
        url = "https://fit.uet.vnu.edu.vn/thong-bao-hoc-bong/"
        text = "Hà Nội, ngày 12 tháng 5 năm 2023. Ban chủ nhiệm thông báo..."
        assert extract_html_date(html, url=url, text=text) == "2023-05-12"

    def test_returns_none_for_static_page(self):
        html = "<html><head><title>Giới thiệu Bộ môn</title></head><body><p>Bộ môn thành lập với sứ mệnh...</p></body></html>"
        url = "https://fit.uet.vnu.edu.vn/bo-mon-cong-nghe-phan-mem/"
        assert extract_html_date(html, url=url, text="Bộ môn thành lập...") is None


class TestExtractDocDate:
    """Kiểm tra trích xuất ngày từ file tài liệu (nội dung text và tên file)."""

    def test_extract_from_document_text(self):
        text = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\nHà Nội, ngày 25 tháng 08 năm 2023\nQUYẾT ĐỊNH"
        assert extract_doc_date("Thong_bao.pdf", text=text) == "2023-08-25"

    def test_fallback_to_filename_when_text_has_no_date(self):
        text = "Danh sách chi tiết kết quả đánh giá điểm rèn luyện của sinh viên."
        assert extract_doc_date("Danh_sach_2022-2023.xlsx", text=text) == "2022"

    def test_returns_none_when_neither_has_date(self):
        text = "Mẫu đơn xin tạm hoãn thi kết thúc học phần."
        assert extract_doc_date("Mau-don-xin-hoan-thi.docx", text=text) is None


class TestIsRecentDocument:
    """Kiểm tra hàm lọc bài viết không quá cũ (từ min_year trở lại đây)."""

    def test_recent_date_accepted(self):
        assert is_recent_document("2023-04-18", min_year=2020) is True
        assert is_recent_document("2020-01-01", min_year=2020) is True
        assert is_recent_document("2024", min_year=2020) is True

    def test_old_date_rejected(self):
        assert is_recent_document("2019-12-31", category="KHAC", min_year=2020) is False
        assert is_recent_document("2018", category="TIN_TUC_SU_KIEN", min_year=2020) is False
        assert is_recent_document("2016-05", category="KHAC", min_year=2020) is False

    def test_none_date_accepted_as_evergreen(self):
        # Các trang tĩnh (giới thiệu bộ môn, CTĐT) không có ngày cần được giữ lại
        assert is_recent_document(None, min_year=2020) is True


class TestExtractHtmlContentPublishedDateEndToEnd:
    """Kiểm tra extract_html_content tích hợp trường published_date."""

    def test_html_content_includes_published_date(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kế hoạch KLTN năm 2023 - Khoa CNTT</title>
            <meta property="article:published_time" content="2023-04-18T08:30:00+07:00" />
        </head>
        <body>
            <div class="entry-content">
                <h1>Kế hoạch Khóa luận tốt nghiệp</h1>
                <p>Khoa Công nghệ Thông tin thông báo kế hoạch thực hiện Khóa luận Tốt nghiệp kỳ 2 năm học 2022-2023 dành cho sinh viên khóa K64.</p>
                <p>Danh sách đề tài xem tại link: <a href="https://drive.google.com/drive/folders/sample">Xem chi tiết</a></p>
            </div>
        </body>
        </html>
        """
        res = extract_html_content("https://fit.uet.vnu.edu.vn/kltn-2023/", html_text=html, min_length=50)
        assert res is not None
        assert "published_date" in res
        assert res["published_date"] == "2023-04-18"
        assert res["cloud_links"] != []

    def test_html_content_without_date_has_none(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Giới thiệu Bộ môn Khoa học Máy tính</title></head>
        <body>
            <div class="entry-content">
                <h1>Giới thiệu Bộ môn</h1>
                <p>Bộ môn Khoa học Máy tính chịu trách nhiệm đào tạo các học phần nền tảng về thuật toán, cấu trúc dữ liệu và trí tuệ nhân tạo.</p>
            </div>
        </body>
        </html>
        """
        res = extract_html_content("https://fit.uet.vnu.edu.vn/bo-mon-khmt/", html_text=html, min_length=50)
        assert res is not None
        assert "published_date" in res
        assert res["published_date"] is None


class TestEvergreenContentPreservation:
    """Kiểm tra cơ chế Evergreen: ngày seed CMS, content recency override, miễn trừ lọc."""

    SEED_HTML = (
        '<html><head><meta property="article:published_time" '
        'content="2015-08-09T10:00:00+07:00" /></head>'
        "<body><p>{body}</p></body></html>"
    )

    def test_is_evergreen_page(self):
        assert is_evergreen_page("https://fit.uet.vnu.edu.vn/bo-mon-cong-nghe-phan-mem")
        assert is_evergreen_page("https://fit.uet.vnu.edu.vn/gioi-thieu/cac-bo-mon")
        assert is_evergreen_page("https://fit.uet.vnu.edu.vn/x", category="DAO_TAO")
        assert not is_evergreen_page("https://fit.uet.vnu.edu.vn/thong-bao-abc/", category="KHAC")

    def test_seed_date_on_department_url_becomes_none(self):
        html = self.SEED_HTML.format(body="Bộ môn được thành lập với nhiệm vụ đào tạo.")
        url = "https://fit.uet.vnu.edu.vn/bo-mon-mang-va-truyen-thong-may-tinh"
        assert extract_html_date(html, url=url, text="Bộ môn được thành lập") is None

    def test_seed_date_on_evergreen_category_becomes_none(self):
        html = self.SEED_HTML.format(body="Chương trình đào tạo đại học.")
        url = "https://fit.uet.vnu.edu.vn/dao-tao-dai-hoc"
        assert extract_html_date(html, url=url, text="Chương trình đào tạo", category="DAO_TAO") is None

    def test_content_recency_override_when_meta_stale(self):
        html = self.SEED_HTML.format(body="Thông báo lịch thi năm học 2023-2024")
        url = "https://fit.uet.vnu.edu.vn/thong-bao-lich-thi/"
        text = "Thông báo lịch thi học kỳ I năm học 2023-2024 dành cho sinh viên."
        assert extract_html_date(html, url=url, text=text, category="TIN_TUC_SU_KIEN") == "2023"

    def test_stale_meta_without_new_year_in_text_is_kept(self):
        html = self.SEED_HTML.format(body="Tin cũ")
        url = "https://fit.uet.vnu.edu.vn/tin-cu/"
        assert extract_html_date(html, url=url, text="Tin cũ không có năm", category="TIN_TUC_SU_KIEN") is None

    def test_seed_date_dropped_for_any_category(self):
        html = self.SEED_HTML.format(body="Trang liên hệ")
        for cat, url in [("KHAC", "https://fit.uet.vnu.edu.vn/contact-us/"),
                         ("TUYEN_SINH", "https://fit.uet.vnu.edu.vn/change-of-major/")]:
            assert extract_html_date(html, url=url, text="Trang liên hệ", category=cat) is None


class TestIsRecentDocumentCategory:
    def test_evergreen_category_kept_even_if_old(self):
        assert is_recent_document("2015-08-09", category="CAN_BO_GIANG_VIEN") is True
        assert is_recent_document("2016", category="DAO_TAO") is True

    def test_evergreen_category_kept_when_none(self):
        assert is_recent_document(None, category="CAN_BO_GIANG_VIEN") is True

    def test_old_news_filtered(self):
        assert is_recent_document("2018-05-01", category="TIN_TUC_SU_KIEN") is False
        assert is_recent_document("2018", category="KHAC") is False

    def test_other_categories_with_old_date_are_kept(self):
        for cat in ("QUY_CHE_BIEU_MAU", "TUYEN_SINH", "NGHIEN_CUU_HOP_TAC"):
            assert is_recent_document("2018-03-01", category=cat) is True

    def test_recent_news_kept(self):
        assert is_recent_document("2024-05-01", category="TIN_TUC_SU_KIEN") is True

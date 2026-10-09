"""Unit tests cho module filters — bộ lọc và chuẩn hóa URL."""
import pytest
from src.ingestion.filters import (
    clean_url,
    normalize_endpoint_url,
    is_document_file,
    is_media_or_asset_file,
    is_valid_html_endpoint,
    is_allowed_file_domain,
    is_cloud_storage_url,
    get_domain_folder_name,
)


class TestCleanUrl:
    """Kiểm tra hàm clean_url."""

    def test_removes_fragment(self):
        assert clean_url("https://example.com/page#section") == "https://example.com/page"

    def test_removes_tracking_params(self):
        result = clean_url("https://example.com/page?utm_source=fb&p=123")
        assert "utm_source" not in result
        assert "p=123" in result

    def test_removes_ver_param(self):
        result = clean_url("https://example.com/page?ver=1.2.3")
        assert "ver" not in result

    def test_removes_fbclid(self):
        result = clean_url("https://example.com/page?fbclid=abc123")
        assert "fbclid" not in result

    def test_normalizes_double_slashes(self):
        result = clean_url("https://example.com//path//page")
        assert "//" not in result.split("://")[1]

    def test_preserves_wp_post_param(self):
        result = clean_url("https://fit.uet.vnu.edu.vn/?p=4593")
        assert "p=4593" in result

    def test_empty_query_no_question_mark(self):
        result = clean_url("https://example.com/page?ver=1.0")
        assert result == "https://example.com/page"


class TestNormalizeEndpointUrl:
    """Kiểm tra hàm normalize_endpoint_url."""

    def test_basic_normalization(self):
        result = normalize_endpoint_url("https://fit.uet.vnu.edu.vn/dao-tao/")
        assert result == "https://fit.uet.vnu.edu.vn/dao-tao/"

    def test_adds_root_path(self):
        result = normalize_endpoint_url("https://fit.uet.vnu.edu.vn")
        assert result == "https://fit.uet.vnu.edu.vn/"


class TestIsDocumentFile:
    """Kiểm tra nhận diện file tài liệu."""

    def test_pdf(self):
        assert is_document_file("https://example.com/doc.pdf") is True

    def test_docx(self):
        assert is_document_file("https://example.com/doc.docx") is True

    def test_xlsx(self):
        assert is_document_file("https://example.com/data.xlsx") is True

    def test_html_page(self):
        assert is_document_file("https://example.com/page.html") is False

    def test_image(self):
        assert is_document_file("https://example.com/img.png") is False

    def test_no_extension(self):
        assert is_document_file("https://example.com/dao-tao") is False


class TestIsMediaOrAssetFile:
    """Kiểm tra nhận diện file media/asset."""

    def test_png(self):
        assert is_media_or_asset_file("https://example.com/img.png") is True

    def test_js(self):
        assert is_media_or_asset_file("https://example.com/app.js") is True

    def test_css(self):
        assert is_media_or_asset_file("https://example.com/style.css") is True

    def test_pdf_not_media(self):
        assert is_media_or_asset_file("https://example.com/doc.pdf") is False


class TestIsValidHtmlEndpoint:
    """Kiểm tra bộ lọc endpoint HTML hợp lệ."""

    def test_valid_endpoint(self):
        assert is_valid_html_endpoint(
            "https://fit.uet.vnu.edu.vn/dao-tao/",
            "fit.uet.vnu.edu.vn"
        ) is True

    def test_different_domain_rejected(self):
        assert is_valid_html_endpoint(
            "https://google.com/page",
            "fit.uet.vnu.edu.vn"
        ) is False

    def test_wp_json_rejected(self):
        assert is_valid_html_endpoint(
            "https://fit.uet.vnu.edu.vn/wp-json/wp/v2/posts",
            "fit.uet.vnu.edu.vn"
        ) is False

    def test_feed_rejected(self):
        assert is_valid_html_endpoint(
            "https://fit.uet.vnu.edu.vn/feed/",
            "fit.uet.vnu.edu.vn"
        ) is False

    def test_login_rejected(self):
        assert is_valid_html_endpoint(
            "https://fit.uet.vnu.edu.vn/login/",
            "fit.uet.vnu.edu.vn"
        ) is False

    def test_document_file_rejected(self):
        assert is_valid_html_endpoint(
            "https://fit.uet.vnu.edu.vn/doc.pdf",
            "fit.uet.vnu.edu.vn"
        ) is False

    def test_image_file_rejected(self):
        assert is_valid_html_endpoint(
            "https://fit.uet.vnu.edu.vn/logo.png",
            "fit.uet.vnu.edu.vn"
        ) is False

    def test_base64_cache_rejected(self):
        assert is_valid_html_endpoint(
            "https://fit.uet.vnu.edu.vn/Ly9mb250cy5nb29nbGVhcGlz",
            "fit.uet.vnu.edu.vn"
        ) is False

    def test_uet_vnu_and_uet_edu_interchangeable(self):
        assert is_valid_html_endpoint(
            "https://uet.edu.vn/page/",
            "uet.vnu.edu.vn"
        ) is True

    def test_subpath_filter(self):
        assert is_valid_html_endpoint(
            "https://uet.vnu.edu.vn/vien-tri-tue-nhan-tao/bai-viet/",
            "uet.vnu.edu.vn",
            subpath="vien-tri-tue-nhan-tao"
        ) is True

    def test_subpath_rejected_without_match(self):
        assert is_valid_html_endpoint(
            "https://uet.vnu.edu.vn/other-page/",
            "uet.vnu.edu.vn",
            subpath="vien-tri-tue-nhan-tao"
        ) is False


class TestIsAllowedFileDomain:
    """Kiểm tra nhận diện domain file cho phép."""

    def test_same_domain(self):
        assert is_allowed_file_domain(
            "https://fit.uet.vnu.edu.vn/doc.pdf",
            "fit.uet.vnu.edu.vn"
        ) is True

    def test_uet_domain_always_allowed(self):
        assert is_allowed_file_domain(
            "https://uet.vnu.edu.vn/doc.pdf",
            "fit.uet.vnu.edu.vn"
        ) is True

    def test_cdn_allowed(self):
        assert is_allowed_file_domain(
            "https://cdn.cdneverest.net/doc.pdf",
            "fit.uet.vnu.edu.vn"
        ) is True

    def test_foreign_domain_rejected(self):
        assert is_allowed_file_domain(
            "https://google.com/doc.pdf",
            "fit.uet.vnu.edu.vn"
        ) is False


class TestGetDomainFolderName:
    """Kiểm tra hàm phân loại domain → thư mục."""

    def test_iai_special_case(self):
        assert get_domain_folder_name("https://uet.vnu.edu.vn/vien-tri-tue-nhan-tao/") == "iai.uet.vnu.edu.vn"

    def test_regular_domain(self):
        assert get_domain_folder_name("https://fit.uet.vnu.edu.vn") == "fit.uet.vnu.edu.vn"

    def test_removes_www(self):
        assert get_domain_folder_name("https://www.fit.uet.vnu.edu.vn") == "fit.uet.vnu.edu.vn"


class TestIsCloudStorageUrl:
    """Kiểm tra hàm is_cloud_storage_url nhận diện dịch vụ lưu trữ đám mây."""

    def test_google_drive(self):
        assert is_cloud_storage_url("https://drive.google.com/drive/folders/1w1r0Hk8eZ_N_7-0eS-9P0F") is True
        assert is_cloud_storage_url("https://drive.google.com/file/d/1A2B3C/view?usp=sharing") is True

    def test_google_docs(self):
        assert is_cloud_storage_url("https://docs.google.com/document/d/1XyZ/edit") is True
        assert is_cloud_storage_url("https://docs.google.com/spreadsheets/d/1XyZ/edit") is True

    def test_google_forms(self):
        assert is_cloud_storage_url("https://forms.gle/abCD1234efGH") is True
        assert is_cloud_storage_url("https://docs.google.com/forms/d/e/1FAIpQLSc/viewform") is True

    def test_onedrive_and_sharepoint(self):
        assert is_cloud_storage_url("https://1drv.ms/u/s!Alkjshdf832") is True
        assert is_cloud_storage_url("https://onedrive.live.com/?id=root") is True
        assert is_cloud_storage_url("https://uetvnu-my.sharepoint.com/:b:/g/personal/doc.pdf") is True

    def test_dropbox(self):
        assert is_cloud_storage_url("https://dropbox.com/s/xyz/slides.pdf") is True
        assert is_cloud_storage_url("https://www.dropbox.com/scl/fi/xyz?dl=0") is True

    def test_rejects_non_whitelisted_domains(self):
        assert is_cloud_storage_url("https://www.facebook.com/uet.vnu.edu.vn") is False
        assert is_cloud_storage_url("https://www.google.com/search?q=uet") is False
        assert is_cloud_storage_url("https://fit.uet.vnu.edu.vn/tin-tuc/") is False
        assert is_cloud_storage_url("https://youtube.com/watch?v=123") is False
        assert is_cloud_storage_url("https://shopee.vn/") is False

    def test_invalid_and_empty_urls(self):
        assert is_cloud_storage_url("") is False
        assert is_cloud_storage_url("javascript:void(0)") is False
        assert is_cloud_storage_url("mailto:support@uet.vnu.edu.vn") is False
        assert is_cloud_storage_url("ftp://drive.google.com/test") is False


class TestResolveUnit:
    """Kiểm tra hàm resolve_unit chuẩn hóa domain sang mã đơn vị (Unit Code)."""

    def test_all_standard_domains(self):
        from src.ingestion.config import FACULTY_TARGETS, faculty_storage_name, resolve_unit

        for target in FACULTY_TARGETS:
            assert resolve_unit(faculty_storage_name(target["faculty_id"])) == target["faculty_id"]

    def test_case_insensitivity_and_whitespace(self):
        from src.ingestion.config import resolve_unit

        assert resolve_unit("FIT.UET.VNU.EDU.VN") == "FIT"
        assert resolve_unit("  fet.uet.vnu.edu.vn  ") == "FET"
        assert resolve_unit("IAI.UET.VNU.EDU.VN") == "IAI"

    def test_handles_www_prefix(self):
        from src.ingestion.config import resolve_unit

        assert resolve_unit("www.fit.uet.vnu.edu.vn") == "FIT"
        assert resolve_unit("www.uet.vnu.edu.vn") == "UET"
        assert resolve_unit("www.fema.uet.vnu.edu.vn") == "FEMA"

    def test_unknown_or_empty_domain_defaults_to_uet(self):
        from src.ingestion.config import resolve_unit

        assert resolve_unit("unknown.vnu.edu.vn") == "UET"
        assert resolve_unit("example.com") == "UET"
        assert resolve_unit("") == "UET"
        assert resolve_unit(None) == "UET"


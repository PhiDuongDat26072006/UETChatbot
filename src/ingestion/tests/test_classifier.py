"""Unit tests cho module classifier — phân loại URL theo danh mục."""
import pytest
from src.ingestion.classifier import (
    Category,
    classify_endpoint,
    normalize_text,
    clean_page_title,
)


class TestNormalizeText:
    """Kiểm tra hàm chuẩn hóa văn bản cho regex tiếng Việt."""

    def test_basic(self):
        result = normalize_text("Đào tạo")
        assert "đào tạo" in result
        assert "dao tao" in result

    def test_empty(self):
        assert normalize_text("") == ""

    def test_d_stroke(self):
        result = normalize_text("Đường dẫn")
        assert "duong dan" in result


class TestCleanPageTitle:
    """Kiểm tra hàm làm sạch tiêu đề trang."""

    def test_removes_uet_suffix(self):
        result = clean_page_title("Đào tạo - Trường Đại học Công nghệ")
        assert "Trường" not in result
        assert "Đào tạo" in result

    def test_removes_dhqghn_suffix(self):
        result = clean_page_title("Tuyển sinh - ĐHQGHN")
        assert "ĐHQGHN" not in result
        assert "Tuyển sinh" in result

    def test_html_entity_decoded(self):
        result = clean_page_title("B&agrave;i vi&#7871;t")
        assert "&agrave;" not in result

    def test_empty(self):
        assert clean_page_title("") == ""


class TestClassifyEndpoint:
    """Kiểm tra hàm phân loại endpoint theo 7 danh mục."""

    def test_dao_tao_by_slug(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/dao-tao/") == Category.DAO_TAO

    def test_tuyen_sinh_by_slug(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/tuyen-sinh/") == Category.TUYEN_SINH

    def test_quy_che_by_slug(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/quy-che/") == Category.QUY_CHE_BIEU_MAU

    def test_nghien_cuu_by_slug(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/nghien-cuu/") == Category.NGHIEN_CUU_HOP_TAC

    def test_can_bo_by_slug(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/can-bo/") == Category.CAN_BO_GIANG_VIEN

    def test_tin_tuc_by_slug(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/tin-tuc/") == Category.TIN_TUC_SU_KIEN

    def test_homepage_is_khac(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/") == Category.KHAC

    def test_root_is_khac(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn") == Category.KHAC

    def test_title_based_classification(self):
        # Khi slug không có ngữ nghĩa (?p=4593), title phải chứa từ khóa khớp
        # 'tuyen sinh' (accented) -> sẽ khớp pattern TUYEN_SINH qua normalize_text
        result = classify_endpoint(
            "https://fit.uet.vnu.edu.vn/tuyen-sinh-dai-hoc/",
            title="Thông báo tuyển sinh đại học năm 2024"
        )
        assert result == Category.TUYEN_SINH

    def test_tuyen_sinh_priority_over_dao_tao(self):
        """TUYEN_SINH phải ưu tiên hơn DAO_TAO khi cùng xuất hiện."""
        result = classify_endpoint(
            "https://fit.uet.vnu.edu.vn/tuyen-sinh-dao-tao/"
        )
        assert result == Category.TUYEN_SINH

    def test_giang_vien_by_slug(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/giang-vien/") == Category.CAN_BO_GIANG_VIEN

    def test_hoc_bong_classified_as_nghien_cuu(self):
        result = classify_endpoint("https://fit.uet.vnu.edu.vn/hoc-bong/")
        assert result == Category.NGHIEN_CUU_HOP_TAC

    def test_unknown_slug_is_khac(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/about/") == Category.KHAC

    def test_seminar_is_nghien_cuu(self):
        assert classify_endpoint("https://fit.uet.vnu.edu.vn/seminar-khoa-hoc/") == Category.NGHIEN_CUU_HOP_TAC


class TestConfig:
    """Kiểm tra cấu hình hệ thống."""

    def test_domain_folders_has_uet(self):
        from src.ingestion.config import DOMAIN_FOLDERS
        folders = [d["folder"] for d in DOMAIN_FOLDERS]
        assert "uet.edu.vn" in folders

    def test_domain_folders_has_all_faculties(self):
        from src.ingestion.config import DOMAIN_FOLDERS, FACULTY_TARGETS
        folders = [d["folder"] for d in DOMAIN_FOLDERS]
        for t in FACULTY_TARGETS:
            assert t["domain"] in folders

    def test_create_http_session_returns_session(self):
        import requests
        from src.ingestion.config import create_http_session
        session = create_http_session(pool_size=5)
        assert isinstance(session, requests.Session)
        assert "User-Agent" in session.headers

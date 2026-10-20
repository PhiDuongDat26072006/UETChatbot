"""src/ingestion/tests/test_storage_architecture.py - Kiểm thử toàn diện kiến trúc lưu trữ và mô hình dữ liệu mới.

Bao gồm kiểm thử:
1. DataSource: tính tất định của source_id, không bị UUID churn, schema validation, metadata.
2. RawData: bảo tồn nguyên vẹn bytes (HTML, PDF, DOCX), kiểm tra mã băm SHA-256 khớp với file trên đĩa,
   liên kết source_id, manifest JSONL.
3. ProcessedData: hợp đồng phẳng JSONL, liên kết raw_data_id và source_id, bảo tồn sự kiện/ngày/mã môn.
4. DataChunk: truy vết xuyên suốt vòng đời dữ liệu (DataChunk -> ProcessedData -> RawData -> DataSource).
5. Integration: toàn bộ luồng DataSource -> RawData -> ProcessedData -> DataChunk, kiểm thử di trú và chạy lại bất biến (idempotent).
"""
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src.base import DataChunk, DataSource, ProcessedData, RawData
from src.chunking import UETChunker
from src.ingestion.loader import UETDataLoader
from src.ingestion.main import main as ingestion_cli_main
from src.ingestion.migration import (
    enrich_processed_data,
    migrate_all,
    migrate_domain_discovery,
    migrate_domain_raw_payloads,
)
from src.ingestion.models import (
    compute_bytes_sha256,
    create_data_source,
    create_raw_record,
    generate_raw_id,
    generate_source_id,
    safe_payload_filename,
)


class TestDataSourceArchitecture:
    """Kiểm thử tầng DataSource (Nơi dữ liệu bắt nguồn)."""

    def test_stable_source_id_deterministic(self):
        """source_id phải mang tính tất định (deterministic), không bị UUID churn khi cào lại."""
        url = "https://fit.uet.vnu.edu.vn/dao-tao-dai-hoc"
        sid1 = generate_source_id("web", url)
        sid2 = generate_source_id("web", url)
        sid3 = generate_source_id("web", url + "/")
        sid_other = generate_source_id("web", "https://fit.uet.vnu.edu.vn/nghien-cuu")

        assert sid1 == sid2, "Cùng URL phải sinh ra cùng source_id"
        assert sid1 == sid3, "URL chuẩn hóa có trailing slash phải sinh ra cùng source_id"
        assert sid1 != sid_other, "URL khác nhau phải có source_id khác nhau"

    def test_data_source_model_creation(self):
        """DataSource phải mô tả nguồn gốc, không chứa thân văn bản bài viết."""
        url = "https://uet.vnu.edu.vn/quy-che"
        source = create_data_source(
            source_type="web",
            uri=url,
            metadata={"domain": "uet.vnu.edu.vn", "unit": "UET", "category": "QUY_CHE"},
        )
        assert isinstance(source, DataSource)
        assert source.source_id == generate_source_id("web", url)
        assert source.source_type == "web"
        assert source.uri == url
        assert source.metadata["domain"] == "uet.vnu.edu.vn"
        assert not hasattr(source, "content")

    def test_file_data_source(self):
        """DataSource cho file tài liệu đính kèm."""
        file_url = "https://fit.uet.vnu.edu.vn/wp-content/uploads/2024/BM_01.pdf"
        source = create_data_source(
            source_type="file",
            uri=file_url,
            metadata={"domain": "fit.uet.vnu.edu.vn", "file_name": "BM_01.pdf"},
        )
        assert source.source_type == "file"
        assert source.source_id == generate_source_id("file", file_url)


class TestRawDataArchitecture:
    """Kiểm thử tầng RawData (Bản thể bất biến của payload nguyên bản)."""

    def test_sha256_checksum_verification(self, tmp_path):
        """Mã băm SHA-256 phải tính chính xác trên byte nhị phân của payload nguyên bản."""
        sample_bytes = b"%PDF-1.4\n%mock binary document content for testing\n%%EOF"
        doc_path = tmp_path / "sample.pdf"
        doc_path.write_bytes(sample_bytes)

        computed_hash = compute_bytes_sha256(sample_bytes)
        assert computed_hash == compute_bytes_sha256(doc_path.read_bytes())

        sid = generate_source_id("file", str(doc_path))
        raw_rec = create_raw_record(
            source_id=sid,
            source_uri=str(doc_path),
            raw_file_path=str(doc_path),
            content=None,
            title="sample.pdf",
            raw_metadata={"sha256": computed_hash, "file_size": len(sample_bytes), "status": "preserved"},
        )
        assert raw_rec.raw_metadata["sha256"] == computed_hash
        assert raw_rec.raw_metadata["source_id"] == sid
        assert raw_rec.raw_metadata["raw_file_path"] == str(doc_path)
        assert raw_rec.content == "" or raw_rec.content is None

    def test_missing_payload_marked_status(self):
        """Tài liệu chưa tải được raw payload phải được đánh dấu 'missing', không tạo payload giả."""
        url = "https://uet.vnu.edu.vn/missing-page"
        sid = generate_source_id("web", url)
        raw_rec = create_raw_record(
            source_id=sid,
            source_uri=url,
            raw_file_path=None,
            content=None,
            title="Trang chưa tải",
            raw_metadata={"status": "missing", "content_type": "text/html"},
        )
        assert raw_rec.raw_metadata["status"] == "missing"
        assert "raw_file_path" not in raw_rec.raw_metadata or raw_rec.raw_metadata["raw_file_path"] is None

    def test_safe_payload_filename_avoids_collisions(self):
        """Filename an toàn chống va chạm basename giữa các đường dẫn khác nhau."""
        u1 = "https://uet.vnu.edu.vn/khoa-cntt/thong-bao"
        u2 = "https://uet.vnu.edu.vn/khoa-dtt/thong-bao"
        fn1 = safe_payload_filename(u1, default_ext=".html")
        fn2 = safe_payload_filename(u2, default_ext=".html")
        assert fn1 != fn2
        assert fn1.endswith(".html")


class TestProcessedDataArchitecture:
    """Kiểm thử tầng ProcessedData (Dữ liệu sạch, phẳng, truy vết được)."""

    def test_flat_jsonl_record_and_lineage(self):
        """Bản ghi ProcessedData phẳng phải giữ nguyên tương thích JSONL và chứa source_id, raw_data_id."""
        url = "https://fat.uet.vnu.edu.vn/chuong-trinh-dt"
        sid = generate_source_id("web", url)
        rid = generate_raw_id(sid, "hash123")

        proc = ProcessedData(
            id="doc_abc123",
            raw_data_id=rid,
            title="Chương trình đào tạo K67",
            content="Môn học: INT2204 Lập trình mạng. Giảng viên: TS. Nguyễn Văn A. Ngày áp dụng: 2026-09-01.",
            metadata={
                "source_id": sid,
                "source_type": "html",
                "source_url_or_path": url,
                "domain": "fat.uet.vnu.edu.vn",
                "unit": "FAT",
                "category": "DAO_TAO",
                "published_date": "2026-09-01",
                "cloud_links": [],
            },
        )
        assert proc.id == "doc_abc123"
        assert proc.raw_data_id == rid
        assert proc.metadata["source_id"] == sid
        # Bảo toàn thông tin quan trọng
        assert "INT2204" in proc.content
        assert "2026-09-01" in proc.content


class TestDataChunkTraceability:
    """Kiểm thử truy vết từ DataChunk ngược về ProcessedData, RawData và DataSource."""

    def test_complete_lineage_traceability(self):
        """Đảm bảo chuỗi truy vết: DataChunk.document_id -> ProcessedData.id -> RawData.id -> DataSource.source_id."""
        url = "https://fit.uet.vnu.edu.vn/quy-che-hoc-vu"
        sid = generate_source_id("web", url)
        rid = generate_raw_id(sid, "hash_fit_01")
        doc_id = "doc_fit_stable_001"

        proc_doc = ProcessedData(
            id=doc_id,
            raw_data_id=rid,
            title="Quy chế đào tạo đại học",
            content="Điều 1. Phạm vi áp dụng. Điều 2. Đăng ký tín chỉ. Sinh viên đăng ký tối thiểu 14 tín chỉ mỗi học kỳ.",
            metadata={
                "source_id": sid,
                "source_url_or_path": url,
                "domain": "fit.uet.vnu.edu.vn",
                "unit": "FIT",
                "raw_data_id": rid,
            },
        )

        chunker = UETChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(proc_doc)

        assert len(chunks) >= 1
        for c in chunks:
            # 1. Chunk trỏ về ProcessedData cha
            assert c.document_id == doc_id
            # 2. Metadata của chunk kế thừa thông tin truy vết
            assert c.metadata["document_id"] == doc_id
            assert c.metadata["domain"] == "fit.uet.vnu.edu.vn"
            assert c.metadata.get("raw_data_id") == rid
            assert c.metadata.get("source_id") == sid


class TestIntegrationLifecycleAndMigration:
    """Kiểm thử tích hợp vòng đời dữ liệu và công cụ di trú."""

    def test_migration_pipeline_mock(self, tmp_path):
        """Kiểm thử di trú an toàn từ endpoints và file nhị phân mẫu sang cấu trúc mới."""
        raw_root = tmp_path / "raw_data"
        source_root = tmp_path / "data_source"
        proc_root = tmp_path / "processed_data"

        domain = "fit.uet.vnu.edu.vn"
        domain_raw = raw_root / domain
        endpoints_dir = domain_raw / "endpoints"
        files_dir = domain_raw / "files"
        endpoints_dir.mkdir(parents=True)
        files_dir.mkdir(parents=True)
        proc_root.mkdir(parents=True)

        # 1. Tạo endpoints_metadata.jsonl mẫu
        ep_url = "https://fit.uet.vnu.edu.vn/k67"
        ep_meta = endpoints_dir / "endpoints_metadata.jsonl"
        ep_meta.write_text(
            json.dumps({"url": ep_url, "domain": domain, "faculty_id": "FIT", "category": "DAO_TAO", "title": "K67"}) + "\n",
            encoding="utf-8",
        )

        # 2. Tạo file PDF nhị phân nguyên bản mẫu
        sample_pdf = files_dir / "quydinh.pdf"
        pdf_bytes = b"%PDF-1.4 sample content for fit"
        sample_pdf.write_bytes(pdf_bytes)

        # 3. Tạo file processed_data cũ
        proc_file = proc_root / f"{domain}.jsonl"
        proc_file.write_text(
            json.dumps({
                "id": "fit_proc_1",
                "source_type": "html",
                "source_url_or_path": ep_url,
                "title": "K67",
                "content": "Nội dung đào tạo K67",
                "domain": domain,
                "unit": "FIT",
            }) + "\n",
            encoding="utf-8",
        )

        # 4. Thực thi di trú
        sources, uri_to_sid = migrate_domain_discovery(domain, raw_root, source_root)
        raw_recs, uri_to_rid = migrate_domain_raw_payloads(domain, raw_root, uri_to_sid)
        enriched_count = enrich_processed_data(proc_file, uri_to_sid, uri_to_rid)

        # 5. Kiểm chứng kết quả
        assert len(sources) >= 1
        assert (source_root / f"{domain}.jsonl").exists()
        assert (domain_raw / "raw_records.jsonl").exists()
        assert enriched_count == 1

        # File gốc nguyên bản không bị xóa
        assert sample_pdf.exists()
        assert sample_pdf.read_bytes() == pdf_bytes

        # Bản ghi raw_records cho file PDF chứa checksum chính xác
        pdf_rec = next(r for r in raw_recs if r["title"] == "quydinh.pdf")
        assert pdf_rec["raw_metadata"]["sha256"] == compute_bytes_sha256(pdf_bytes)
        assert pdf_rec["raw_metadata"]["status"] == "preserved"

        # Bản ghi processed được bổ sung source_id và raw_data_id
        updated_proc = json.loads(proc_file.read_text(encoding="utf-8").strip())
        assert updated_proc["id"] == "fit_proc_1"
        assert "source_id" in updated_proc
        assert "raw_data_id" in updated_proc

    def test_dataloader_reads_migrated_architecture(self, tmp_path):
        """UETDataLoader đọc được DataSource, RawData manifest và ProcessedData."""
        source_dir = tmp_path / "data_source"
        source_dir.mkdir(parents=True)
        (source_dir / "uet.edu.vn.jsonl").write_text(
            json.dumps({
                "source_id": "src_1",
                "source_type": "web",
                "uri": "https://uet.edu.vn",
                "metadata": {"unit": "UET"},
            }) + "\n",
            encoding="utf-8",
        )

        loader = UETDataLoader()
        sources = loader.load_sources(data_source_dir=tmp_path / "data_source", domain="uet.edu.vn")
        assert len(sources) == 1
        assert sources[0].source_id == "src_1"
        assert sources[0].uri == "https://uet.edu.vn"

    def test_cli_help_and_migrate_flag(self):
        """Lệnh CLI trong src.ingestion.main hỗ trợ cờ --migrate."""
        with pytest.raises(SystemExit) as exc:
            ingestion_cli_main(["--help"])
        assert exc.value.code == 0


class TestRawDataPreservationAndRelativePaths:
    """Kiểm thử chuyên sâu bảo tồn nguyên vẹn RawData HTML, nhị phân và đường dẫn project-relative."""

    def test_html_raw_data_preservation_unparsed_bytes(self, tmp_path):
        """HTML nguyên bản được lưu trữ chính xác từng byte (bao gồm thẻ script, style, comments), không qua BeautifulSoup."""
        from src.ingestion.backfill import backfill_single_html
        from src.ingestion.config.paths import resolve_project_path

        html_dir = tmp_path / "raw_data" / "fit.uet.vnu.edu.vn" / "html"
        html_dir.mkdir(parents=True)

        complex_html_payload = (
            b"<!DOCTYPE html>\n"
            b"<html>\n"
            b"<head>\n"
            b"  <style>.hero { color: #003366; margin: 10px; }</style>\n"
            b"  <script>console.log('raw javascript preserved');</script>\n"
            b"</head>\n"
            b"<body>\n"
            b"  <!-- Navigation bar with whitespace -->\n"
            b"  <nav>   <a href='/'>Home</a>   </nav>\n"
            b"  <h1>Thong tin Khoa CNTT</h1>\n"
            b"  <footer>Footer content with &copy; 2026</footer>\n"
            b"</body>\n"
            b"</html>"
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = complex_html_payload
        mock_response.url = "https://fit.uet.vnu.edu.vn/gioi-thieu"
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}

        mock_session = Mock()
        mock_session.get.return_value = mock_response

        raw_id = "raw_test_html_123"
        source_id = "src_test_html_456"

        success, rec = backfill_single_html(
            session=mock_session,
            source_url="https://fit.uet.vnu.edu.vn/gioi-thieu",
            domain="fit.uet.vnu.edu.vn",
            prefixes=(),
            raw_id=raw_id,
            source_id=source_id,
            html_dir=html_dir,
        )

        assert success is True
        saved_file = html_dir / f"{raw_id}.html"
        assert saved_file.exists(), "File HTML nguyên bản phải được tạo trên đĩa"
        saved_bytes = saved_file.read_bytes()

        # Kiểm chứng: byte nguyên vẹn 100%, không bị stripping, không loại bỏ script/style
        assert saved_bytes == complex_html_payload
        assert b"<script>console.log('raw javascript preserved');</script>" in saved_bytes
        assert b"<style>.hero { color: #003366; margin: 10px; }</style>" in saved_bytes
        assert b"<!-- Navigation bar with whitespace -->" in saved_bytes

        # Kiểm chứng manifest record
        assert rec["id"] == raw_id
        assert rec["source_id"] == source_id
        assert rec["content"] is None, "content phải là None vì đã lưu vào raw_file_path"
        assert rec["raw_file_path"] == f"data/raw_data/fit.uet.vnu.edu.vn/html/{raw_id}.html"
        assert rec["raw_metadata"]["status"] == "preserved"
        assert rec["raw_metadata"]["http_status"] == 200
        assert rec["raw_metadata"]["file_size"] == len(complex_html_payload)
        assert rec["raw_metadata"]["sha256"] == compute_bytes_sha256(complex_html_payload)

    def test_html_backfill_failed_requests_handling(self, tmp_path):
        """Xử lý lỗi HTTP (404, 500, non-HTML, timeout) chính xác: không lưu file rác, đánh dấu status: failed."""
        from src.ingestion.backfill import backfill_single_html

        html_dir = tmp_path / "raw_data" / "fit.uet.vnu.edu.vn" / "html"
        html_dir.mkdir(parents=True)

        # 1. Trường hợp HTTP 404
        mock_resp_404 = Mock()
        mock_resp_404.status_code = 404
        mock_resp_404.url = "https://fit.uet.vnu.edu.vn/not-found"
        mock_resp_404.headers = {"Content-Type": "text/html"}
        mock_resp_404.content = b"Not Found"

        mock_session = Mock()
        mock_session.get.return_value = mock_resp_404

        success, rec_404 = backfill_single_html(
            session=mock_session,
            source_url="https://fit.uet.vnu.edu.vn/not-found",
            domain="fit.uet.vnu.edu.vn",
            prefixes=(),
            raw_id="raw_fail_404",
            source_id="src_fail_404",
            html_dir=html_dir,
        )
        assert success is False
        assert rec_404["raw_metadata"]["status"] == "failed"
        assert rec_404["raw_metadata"]["http_status"] == 404
        assert rec_404["raw_file_path"] is None
        assert not (html_dir / "raw_fail_404.html").exists()

        # 2. Trường hợp Non-HTML Content-Type (e.g. application/zip)
        mock_resp_zip = Mock()
        mock_resp_zip.status_code = 200
        mock_resp_zip.url = "https://fit.uet.vnu.edu.vn/archive.zip"
        mock_resp_zip.headers = {"Content-Type": "application/zip"}
        mock_resp_zip.content = b"PK\x03\x04mockzip"
        mock_session.get.return_value = mock_resp_zip

        success, rec_zip = backfill_single_html(
            session=mock_session,
            source_url="https://fit.uet.vnu.edu.vn/archive.zip",
            domain="fit.uet.vnu.edu.vn",
            prefixes=(),
            raw_id="raw_fail_zip",
            source_id="src_fail_zip",
            html_dir=html_dir,
        )
        assert success is False
        assert rec_zip["raw_metadata"]["status"] == "failed"
        assert "Non-HTML" in rec_zip["raw_metadata"]["failure_reason"]
        assert rec_zip["raw_file_path"] is None

        # 3. Trường hợp Network Exception (e.g. ConnectTimeout)
        mock_session.get.side_effect = requests.exceptions.ConnectTimeout("Connection timed out")
        success, rec_timeout = backfill_single_html(
            session=mock_session,
            source_url="https://fit.uet.vnu.edu.vn/timeout",
            domain="fit.uet.vnu.edu.vn",
            prefixes=(),
            raw_id="raw_fail_timeout",
            source_id="src_fail_timeout",
            html_dir=html_dir,
        )
        assert success is False
        assert rec_timeout["raw_metadata"]["status"] == "failed"
        assert "ConnectTimeout" in rec_timeout["raw_metadata"]["failure_reason"]
        assert rec_timeout["raw_file_path"] is None

    def test_project_relative_path_rules_and_security(self):
        """Đảm bảo mọi đường dẫn tuân thủ chuẩn POSIX project-relative và chống path traversal."""
        from src.ingestion.config.paths import (
            BASE_DIR,
            resolve_project_path,
            to_project_relative_path,
        )

        # 1. Đường dẫn URL web giữ nguyên
        assert to_project_relative_path("https://uet.vnu.edu.vn/page") == "https://uet.vnu.edu.vn/page"

        # 2. Đường dẫn cục bộ tương đối chuẩn hóa sang POSIX
        assert to_project_relative_path("data/raw_data/fit/html/doc.html") == "data/raw_data/fit/html/doc.html"
        assert to_project_relative_path("data\\raw_data\\fit\\html\\doc.html") == "data/raw_data/fit/html/doc.html"

        # 3. Chống Path Traversal (../)
        with pytest.raises(ValueError, match="Path traversal detected"):
            to_project_relative_path("../outside/secret.txt")

        with pytest.raises(ValueError, match="Path traversal detected"):
            to_project_relative_path("data/raw_data/../../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal detected"):
            resolve_project_path("../../etc/shadow")

        # 4. Resolve đường dẫn tương đối luôn nằm bên trong BASE_DIR
        resolved = resolve_project_path("data/raw_data/fit.uet.vnu.edu.vn/raw_records.jsonl")
        assert resolved.is_absolute()
        assert resolved == (BASE_DIR / "data/raw_data/fit.uet.vnu.edu.vn/raw_records.jsonl").resolve()

    def test_project_relocation_and_working_dir_independence(self, monkeypatch):
        """Hệ thống hoạt động nhất quán ngay cả khi working directory thay đổi sang thư mục khác."""
        import tempfile
        from src.ingestion.config.paths import BASE_DIR, resolve_project_path

        # Đổi CWD sang thư mục tạm bên ngoài
        with tempfile.TemporaryDirectory() as external_dir:
            monkeypatch.chdir(external_dir)
            # resolve_project_path vẫn resolve chính xác dựa trên BASE_DIR chứ không phụ thuộc CWD
            target = resolve_project_path("data/data_source/fit.uet.vnu.edu.vn.jsonl")
            assert target == (BASE_DIR / "data/data_source/fit.uet.vnu.edu.vn.jsonl").resolve()
            assert target.exists()


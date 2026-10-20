"""
tests/test_ingestion_stages.py - Kiểm thử bộ điều phối dữ liệu đa chặng & chống trùng lặp.
"""

import tempfile
import unittest
from pathlib import Path
from src.ingestion.manifest import DataManifestTracker
from src.pipeline import UETIngestionPipeline


class TestDataManifestTracker(unittest.TestCase):
    """Kiểm thử hoạt động của sổ cái trạng thái chống trùng lặp."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manifest_file = Path(self.temp_dir.name) / ".ingest_manifest.json"
        self.tracker = DataManifestTracker(manifest_path=self.manifest_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_hash_deterministic(self):
        """Mã băm SHA-256 phải tất định và thay đổi khi nội dung đổi."""
        h1 = DataManifestTracker.compute_hash("Nội dung quy chế A", "Tiêu đề A")
        h2 = DataManifestTracker.compute_hash("Nội dung quy chế A", "Tiêu đề A")
        h3 = DataManifestTracker.compute_hash("Nội dung quy chế B", "Tiêu đề A")

        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)

    def test_mark_and_is_processed(self):
        """Kiểm tra đánh dấu và truy vấn trạng thái xử lý."""
        doc_id = "doc_test_123"
        h = DataManifestTracker.compute_hash("Nội dung", "Tiêu đề")

        # Chưa xử lý
        self.assertFalse(self.tracker.is_processed(doc_id, "processed"))
        self.assertFalse(self.tracker.is_processed(doc_id, "chunked"))

        # Đánh dấu processed
        self.tracker.mark_processed(doc_id, "processed", content_hash=h, title="Tiêu đề")
        self.tracker.save()

        # Đọc lại từ file
        new_tracker = DataManifestTracker(manifest_path=self.manifest_file)
        self.assertTrue(new_tracker.is_processed(doc_id, "processed", current_hash=h))
        self.assertFalse(new_tracker.is_processed(doc_id, "chunked"))

        # Nếu hash thay đổi -> Phải báo False (cần xử lý lại)
        self.assertFalse(new_tracker.is_processed(doc_id, "processed", current_hash="hash_khac_999"))

    def test_get_stats_and_clear(self):
        """Kiểm tra báo cáo thống kê và hàm xóa cache."""
        self.tracker.mark_processed("d1", "processed", "h1")
        self.tracker.mark_processed("d1", "chunked", "h1")
        self.tracker.mark_processed("d2", "processed", "h2")

        stats = self.tracker.get_stats()
        self.assertEqual(stats["total_tracked_documents"], 2)
        self.assertEqual(stats["stages_completed"]["processed"], 2)
        self.assertEqual(stats["stages_completed"]["chunked"], 1)

        self.tracker.clear()
        stats_after = self.tracker.get_stats()
        self.assertEqual(stats_after["total_tracked_documents"], 0)


class TestUETIngestionPipelineStages(unittest.TestCase):
    """Kiểm thử tính năng điều phối chặng của UETIngestionPipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manifest_file = Path(self.temp_dir.name) / ".ingest_manifest.json"
        self.tracker = DataManifestTracker(manifest_path=self.manifest_file)
        self.pipeline = UETIngestionPipeline(manifest=self.tracker)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_invalid_stages_raise_error(self):
        """Giai đoạn không tồn tại hoặc đi lùi phải báo lỗi ValueError."""
        with self.assertRaises(ValueError):
            self.pipeline.run_stages(from_stage="invalid_stage", to_stage="chunked")

        with self.assertRaises(ValueError):
            self.pipeline.run_stages(from_stage="chunked", to_stage="processed")

        with self.assertRaises(ValueError):
            self.pipeline.run_stages(from_stage="vectordb", to_stage="raw")

    def test_get_stage_stats(self):
        """Kiểm tra thống kê các giai đoạn trả về đúng cấu trúc."""
        stats = self.pipeline.get_stage_stats()
        self.assertIn("stages", stats)
        self.assertIn("raw", stats["stages"])
        self.assertIn("processed", stats["stages"])
        self.assertIn("chunked", stats["stages"])
        self.assertIn("vectordb", stats["stages"])
        self.assertIn("manifest_tracking", stats)


if __name__ == "__main__":
    unittest.main()

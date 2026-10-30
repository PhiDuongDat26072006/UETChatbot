"""
tests/test_chunking.py - Bộ kiểm thử đơn vị cho thuật toán Text Chunking (UETChunker).
Kiểm tra:
1. Nạp và đồng bộ tham số từ config.yaml
2. Fallback xử lý độ gối đầu (Overlap preservation khi câu cuối dài hơn chunk_overlap)
3. Context Injection (Gắn tiêu đề vào nội dung chunk)
4. Tích hợp nạp cấu hình trong UETIngestionPipeline
"""

import unittest
from src.base import ProcessedData
from src.chunking import UETChunker
from src.pipeline import UETIngestionPipeline


class TestUETChunker(unittest.TestCase):
    def test_default_init_loads_from_config(self):
        """1. Kiểm tra UETChunker tự động đọc tham số từ config.yaml."""
        chunker = UETChunker()
        self.assertEqual(chunker.chunk_size, 800)
        self.assertEqual(chunker.chunk_overlap, 150)
        # Separators phải chứa các ký tự Markdown từ config.yaml
        self.assertIn("### ", chunker.separators)
        self.assertIn("## ", chunker.separators)
        self.assertIn("# ", chunker.separators)
        self.assertIn("\n\n", chunker.separators)

    def test_custom_parameters_override(self):
        """2. Tham số truyền trực tiếp phải ghi đè cấu hình config.yaml."""
        chunker = UETChunker(chunk_size=300, chunk_overlap=40, separators=["\n", " "])
        self.assertEqual(chunker.chunk_size, 300)
        self.assertEqual(chunker.chunk_overlap, 40)
        self.assertEqual(chunker.separators, ["\n", " "])

    def test_overlap_fallback_when_last_split_longer_than_overlap(self):
        """
        3. Kiểm tra khắc phục lỗi mất Overlap:
        Khi câu cuối cùng của chunk dài hơn chunk_overlap (ví dụ 120 ký tự > overlap 80 ký tự),
        thuật toán phải fallback cắt lấy phần đuôi (80 ký tự) thay vì để mất 100% độ gối đầu.
        """
        chunk_size = 200
        chunk_overlap = 80
        chunker = UETChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Tạo 3 câu, mỗi câu dài 110 ký tự (> chunk_overlap = 80)
        sentence1 = "Câu thứ nhất rất dài nhằm vượt ngưỡng gối đầu overlap của thuật toán phân đoạn văn bản một hai ba bốn năm. "
        sentence2 = "Câu thứ hai tiếp tục rất dài để ép đóng chunk đầu tiên và kiểm tra cơ chế fallback gối đầu sáu bảy tám chín mười. "
        sentence3 = "Câu thứ ba để hoàn tất phần nội dung kiểm thử cuối cùng cho quy trình chunking của hệ thống mười một mười hai. "
        content = sentence1 + sentence2 + sentence3

        doc = ProcessedData(
            id="doc_test_overlap",
            title="Quy định kiểm thử Overlap",
            content=content,
        )

        chunks = chunker.chunk(doc)
        self.assertGreater(len(chunks), 1, "Văn bản phải được chia thành từ 2 chunk trở lên")

        # Kiểm tra nội dung giữa 2 chunk liên tiếp (bỏ qua header [doc.title]):
        chunk0_body = chunks[0].text.replace(f"[{doc.title}]\n", "")
        chunk1_body = chunks[1].text.replace(f"[{doc.title}]\n", "")

        # Kiểm tra nội dung gối đầu giữa 2 chunk:
        # Phần đầu của chunk 1 (với độ dài chunk_overlap) phải xuất hiện trọn vẹn trong chunk 0
        overlap_content = chunk1_body[:chunk_overlap].strip()
        self.assertIn(
            overlap_content,
            chunk0_body,
            f"Phần gối đầu của Chunk 1 ('{overlap_content}') phải xuất hiện trong Chunk 0",
        )
        self.assertGreaterEqual(
            len(overlap_content),
            chunk_overlap - 5,
            "Độ dài phần gối đầu phải xấp xỉ chunk_overlap",
        )

    def test_context_injection(self):
        """4. Kiểm tra Context Injection: tiêu đề tài liệu được prepend vào mọi chunk."""
        chunker = UETChunker(chunk_size=150, chunk_overlap=30)
        doc = ProcessedData(
            id="doc_ctx",
            title="Quy chế đào tạo Đại học Chính quy UET",
            content="Khoản 1. Sinh viên phải hoàn thành tất cả học phần theo đúng lộ trình đào tạo chuẩn.\n\n"
                    "Khoản 2. Sinh viên không được nợ quá 24 tín chỉ tính từ đầu khóa học.\n\n"
                    "Khoản 3. Điều kiện tốt nghiệp yêu cầu chuẩn đầu ra tiếng Anh bậc 4 VSTEP.",
            metadata={"domain": "dao_tao", "year": 2024},
        )

        chunks = chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)

        expected_header = f"[{doc.title}]\n"
        for i, c in enumerate(chunks):
            # Nội dung text được prepend tiêu đề [doc.title]
            self.assertTrue(
                c.text.startswith(expected_header),
                f"Chunk {i} không có Context Injection header mong muốn!",
            )
            # Metadata vẫn bảo toàn tiêu đề và thông tin gốc
            self.assertEqual(c.metadata.get("title"), doc.title)
            self.assertEqual(c.metadata.get("domain"), "dao_tao")
            self.assertEqual(c.metadata.get("year"), 2024)
            self.assertEqual(c.chunk_index, i)

    def test_pipeline_config_integration(self):
        """5. Kiểm tra UETIngestionPipeline nạp trực tiếp tham số chunking từ config.yaml."""
        pipeline = UETIngestionPipeline()
        self.assertIsNotNone(pipeline.chunker)
        self.assertEqual(pipeline.chunker.chunk_size, 800)
        self.assertEqual(pipeline.chunker.chunk_overlap, 150)
        self.assertIn("### ", pipeline.chunker.separators)


if __name__ == "__main__":
    unittest.main()

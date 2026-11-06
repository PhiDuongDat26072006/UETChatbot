"""
tests/test_chunking.py - Bộ kiểm thử đơn vị toàn diện cho thuật toán Text Chunking (UETChunker).
Kiểm tra:
1. Nạp và đồng bộ tham số từ config.yaml
2. Fallback xử lý độ gối đầu (Overlap preservation khi câu cuối dài hơn chunk_overlap)
3. Context Injection (Gắn tiêu đề vào nội dung chunk)
4. Tích hợp nạp cấu hình trong UETIngestionPipeline
5. Nhận diện cấu trúc quy chế đào tạo (Chương / Điều) và Context Breadcrumbs
6. Bảo vệ ranh giới từ (Word Boundary Protection) khi gối đầu Overlap
7. Nhận diện và làm giàu dữ liệu bảng Markdown (Tabular Chunks)
8. Lọc chunk rác và gộp chunk mồ côi bằng min_chunk_size
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
        self.assertEqual(chunker.min_chunk_size, 100)
        # Separators phải chứa các ký tự Markdown và regex từ config.yaml
        self.assertIn("### ", chunker.separators)
        self.assertIn("## ", chunker.separators)
        self.assertIn("# ", chunker.separators)
        self.assertIn("\n\n", chunker.separators)
        self.assertTrue(any("Điều" in s for s in chunker.separators))
        self.assertTrue(any("Chương" in s for s in chunker.separators))

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

        chunk0_body = chunks[0].text.replace(f"[{doc.title}]\n", "")
        chunk1_body = chunks[1].text.replace(f"[{doc.title}]\n", "")

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
            self.assertTrue(
                c.text.startswith(expected_header),
                f"Chunk {i} không có Context Injection header mong muốn!",
            )
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

    def test_academic_regulation_breadcrumbs(self):
        """6. Kiểm tra nhận diện cấu trúc Chương / Điều và tiêm Context Breadcrumbs."""
        chunker = UETChunker(chunk_size=400, chunk_overlap=50)
        content = (
            "Chương I: QUY ĐỊNH CHUNG\n\n"
            "Điều 1. Phạm vi điều chỉnh\n"
            "Quy chế này quy định về đào tạo đại học chính quy tại Trường Đại học Công nghệ.\n\n"
            "Điều 2. Đối tượng áp dụng\n"
            "Áp dụng đối với sinh viên tất cả các khóa đào tạo chính quy.\n\n"
            "Chương II: ĐĂNG KÝ HỌC PHẦN\n\n"
            "Điều 14. Đăng ký học tập\n"
            "Sinh viên phải đăng ký học phần trước khi bắt đầu học kỳ mới qua cổng Portal."
        )
        doc = ProcessedData(
            id="doc_breadcrumbs",
            title="Quy chế UET",
            content=content,
        )

        chunks = chunker.chunk(doc)
        self.assertGreaterEqual(len(chunks), 3)

        # Kiểm tra breadcrumb ở chunk Điều 1
        dieu1_chunks = [c for c in chunks if "Điều 1." in c.text]
        self.assertTrue(len(dieu1_chunks) > 0)
        self.assertIn("Quy chế UET > Chương I: QUY ĐỊNH CHUNG > Điều 1. Phạm vi điều chỉnh", dieu1_chunks[0].text)
        self.assertEqual(dieu1_chunks[0].metadata.get("heading"), "Điều 1. Phạm vi điều chỉnh")

        # Kiểm tra breadcrumb ở chunk Điều 14 thuộc Chương II
        dieu14_chunks = [c for c in chunks if "Điều 14." in c.text]
        self.assertTrue(len(dieu14_chunks) > 0)
        self.assertIn("Chương II: ĐĂNG KÝ HỌC PHẦN > Điều 14. Đăng ký học tập", dieu14_chunks[0].text)

    def test_word_boundary_overlap_preservation(self):
        """7. Kiểm tra cơ chế Overlap không cắt gãy giữa từ tiếng Việt."""
        chunk_size = 180
        chunk_overlap = 60
        chunker = UETChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        sentence1 = "Sinh viên được xét cấp học bổng khuyến khích học tập nếu đạt kết quả học tập từ loại khá trở lên. "
        sentence2 = "Điểm rèn luyện học kỳ phải đạt từ loại tốt trở lên và không bị kỷ luật trong học kỳ xét học bổng. "
        content = sentence1 + sentence2

        doc = ProcessedData(id="doc_wb", title="Học bổng UET", content=content)
        chunks = chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)

        # Lấy nội dung thân của chunk thứ 2
        chunk1_body = chunks[1].text.split("]\n", 1)[1] if "]\n" in chunks[1].text else chunks[1].text
        # Từ đầu tiên của chunk 1 không được là mảnh vỡ cắt cụt (ví dụ: 'hích', 'yến', 'ược', 'ập')
        first_word = chunk1_body.split()[0]
        self.assertFalse(first_word in ["ược", "ên", "ập", "hích", "yến"], f"Từ '{first_word}' bị cắt cụt do mất ranh giới từ!")

    def test_markdown_table_chunking(self):
        """8. Kiểm tra nhận diện bảng Markdown và làm giàu ngữ cảnh theo hàng (Tabular Chunks)."""
        chunker = UETChunker(chunk_size=800, chunk_overlap=100)
        content = (
            "### Bảng điểm chuẩn trúng tuyển\n\n"
            "| Mã ngành | Tên ngành | Điểm chuẩn |\n"
            "| --- | --- | --- |\n"
            "| CN1 | Công nghệ thông tin | 27.85 |\n"
            "| CN2 | Khoa học máy tính | 27.50 |\n\n"
            "Sinh viên theo dõi thông tin trên cổng tuyển sinh."
        )
        doc = ProcessedData(id="doc_table", title="Thông báo tuyển sinh", content=content)
        chunks = chunker.chunk(doc)

        table_chunks = [c for c in chunks if c.metadata.get("data_type") == "tabular"]
        self.assertTrue(len(table_chunks) > 0, "Phải nhận diện và tạo ít nhất 1 chunk dạng bảng (tabular)")
        self.assertIn("Mã ngành: CN1", table_chunks[0].text)
        self.assertIn("Tên ngành: Công nghệ thông tin", table_chunks[0].text)
        self.assertIn("Điểm chuẩn: 27.85", table_chunks[0].text)
        self.assertIn("[META: Bảng điểm chuẩn trúng tuyển]", table_chunks[0].text)

    def test_min_chunk_size_merging_and_noise_filtering(self):
        """9. Kiểm tra gộp chunk mồ côi (< min_chunk_size) và lọc bỏ rác."""
        chunker = UETChunker(chunk_size=400, min_chunk_size=80)
        # Tạo đoạn văn dài ~600 ký tự (> chunk_size 400) để kích hoạt chia tách
        long_para = "Khoản 1. Sinh viên có trách nhiệm bảo vệ tài sản của nhà trường, chấp hành nghiêm túc quy định nội quy phòng học, giảng đường, thư viện và khu vực công cộng trong khuôn viên trường. " * 3
        short_para = "Khoản 2. Sinh viên không được xả rác bừa bãi."
        noise = "\n\n$/$\n\n---"

        content = long_para + "\n\n" + short_para + noise
        doc = ProcessedData(id="doc_orphan", title="Nội quy", content=content)

        chunks = chunker.chunk(doc)
        for c in chunks:
            self.assertNotIn("$/$", c.text)
            self.assertNotIn("---", c.text)
        self.assertTrue(any("Khoản 2. Sinh viên không được xả rác" in c.text for c in chunks))


if __name__ == "__main__":
    unittest.main()

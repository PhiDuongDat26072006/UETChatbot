# Module Indexing (Sparse Lexical & Chunk Validation)

Module `src/indexing` chịu trách nhiệm nạp, kiểm tra tính toàn vẹn `DataChunk` và xây dựng **Sparse Lexical Index (BM25Plus + Inverted Index)** từ `data/chunked_data/*.jsonl`, lưu tại `database/bm25_index.pkl`.

> **Lưu ý kiến trúc:** Toàn bộ logic **Knowledge Graph / LightRAG** (trích xuất đồ thị 2-Pass, từ điển học vụ `AcademicThesaurus`, cầu nối `LightRAGChunkBridge` / `graph_index.pkl`) đã được tách sang phân hệ chuyên biệt [`src/graph/`](../graph/doc.md).

---

## 1. Cấu Trúc Thư Mục

```text
src/indexing/
├── __init__.py     # Lazy exports: UETIndexer, load_chunks_from_dir, BM25Index
├── __main__.py     # Entry point chạy trực tiếp: python -m src.indexing [args]
├── chunks.py       # Trình nạp, chuẩn hóa và kiểm tra tính toàn vẹn DataChunk từ JSONL
├── sparse.py       # Lớp BM25Index (BM25Plus, Inverted Index, Tokenizer tiếng Việt)
├── indexer.py      # Điều phối viên UETIndexer đối chiếu ChromaDB và dựng BM25Index
└── README.md       # Tài liệu kỹ thuật này
```

---

## 2. Chỉ Mục Từ Khóa BM25 (`sparse.py` & `indexer.py`)

Trước khi build BM25, `UETIndexer` đối chiếu toàn bộ `chunk_id` và nội dung `text` (sau chuẩn hóa Unicode NFC) của đầu vào với kho vector hiện có trong `UETVectorStore` (`database/vector_db/`). Thứ tự chunks không ảnh hưởng kết quả.

- **Khớp 100%:** Xây dựng `BM25Index` (BM25Plus kết hợp tách từ tiếng Việt và giữ nguyên mã học phần / số hiệu văn bản), kiểm tra lại với dense rồi ghi nguyên tử (atomic replacement) ra `database/bm25_index.pkl`.
- **Không khớp:** Báo lỗi (`ValueError`) và giữ nguyên artifact `bm25_index.pkl` cũ để bảo toàn tính nhất quán giữa Sparse Index và Dense Vector DB.

---

## 3. Lệnh Vận Hành

```bash
# Xây dựng / tái tạo BM25 Index từ data/chunked_data:
python main.py --ingest --from chunked --to indexed_data

# Làm mới sạch và đồng bộ lại BM25 Index:
python main.py --ingest --from chunked --to indexed_data --clear

# Kiểm tra trạng thái đồng bộ giữa Vector DB và BM25 Index qua module CLI:
python -m src.indexing --status
```

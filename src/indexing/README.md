# Indexing

Module indexing xây dựng **BM25Plus và inverted index** từ dữ liệu trong `data/chunked_data/*.jsonl`.

Trước khi build, `UETIndexer` đối chiếu toàn bộ `chunk_id` và `text` của chunks với dữ liệu trong `UETVectorStore`. Thứ tự chunks không ảnh hưởng kết quả.

- **Khớp:** build BM25, kiểm tra lại với dense rồi lưu vào `database/bm25_index.pkl`.
- **Không khớp:** báo lỗi, giữ nguyên artifact BM25 cũ.
- Không tạo embedding hoặc thay đổi dữ liệu trong VectorStore.

## Trình tự thực hiện

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant I as UETIndexer
    participant V as UETVectorStore
    participant B as BM25Index

    P->>P: Load chunks từ JSONL
    P->>I: index_chunks(chunks)
    I->>V: get_chunks()
    V-->>I: Chunks đã lưu trong dense
    I->>I: So sánh chunk_id và text

    alt Dữ liệu không khớp
        I-->>P: Báo lỗi, dừng xử lý
    else Dữ liệu khớp
        I->>B: BM25Index(chunks)
        B->>B: Tokenize tiếng Việt
        B->>B: Build BM25Plus và inverted index
        I->>V: Đọc lại và đối chiếu corpus
        Note over I,V: Nếu lệch, dừng trước khi lưu
        I->>B: save(bm25_index.pkl)
        I-->>P: Trả kết quả
        P->>P: Cập nhật manifest
    end
```

## Sử dụng

Chạy dense trước, sau đó chạy sparse trên cùng bộ chunks:

```bash
python main.py --ingest --from chunked --to vectordb
python main.py --ingest --from chunked --to indexed_data
```

| File | Vai trò |
|---|---|
| `chunks.py` | Nạp và chuẩn hóa chunks từ JSONL |
| `indexer.py` | Đối chiếu với dense, điều phối build và lưu BM25 |
| `sparse.py` | Tokenization, BM25Plus, inverted index, lưu/load và tìm kiếm |

Retrieval nạp artifact đã xây để tìm kiếm, không build lại index. Kiểm tra đồng bộ chỉ đối chiếu **ID và text**, không đánh giá chất lượng embedding.
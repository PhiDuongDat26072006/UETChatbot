# Cơ sở dữ liệu vector

Tài liệu mô tả `ChromaVectorStore` trong `src/vectordb/vector_store.py`. Lớp
này triển khai `BaseVectorStore`, lưu chunk cùng embedding trong ChromaDB bền
vững trên đĩa và trả về các đoạn gần với câu hỏi nhất.

## Khởi tạo và lưu trữ

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `persist_dir` | `database/vector_db/` (có kiểm tra thư mục `vector_db/` cũ) | Thư mục ChromaDB lưu dữ liệu. |
| `collection_name` | `uet_knowledge_base` | Tên collection được lấy hoặc tạo khi khởi tạo. |
| `embedding_model` | `None` | Mô hình tùy chọn để tự tạo embedding nếu `store()` không được truyền vector. |

Collection được cấu hình dùng khoảng cách cosine (`hnsw:space: cosine`). Trong
pipeline, vector store thường được khởi tạo cùng `UETEmbedder`, và các vector
đã nhúng được truyền vào `store()`.

```python
from pathlib import Path

from src.embeddings import UETEmbedder
from src.vectordb import UETVectorStore

embedder = UETEmbedder()
store = UETVectorStore(
    persist_dir=Path("database") / "vector_db",
    collection_name="uet_knowledge_base",
    embedding_model=embedder,
)
```

`UETVectorStore` là alias của `ChromaVectorStore`.

## Lưu chunk

```python
from src.base import DataChunk

chunks = [
    DataChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Điều kiện đăng ký học phần...",
        chunk_index=0,
        metadata={"title": "Quy chế đào tạo", "unit": "uet"},
    )
]

vectors = embedder.embed_chunks(chunks)
saved_count = store.store(chunks, vectors)
```

`store(chunks, vectors=None)` trả về số chunk đã lưu. Dữ liệu được ghi theo
batch tối đa 500 bản ghi; thao tác là `upsert`, tức bản ghi có cùng `chunk_id`
sẽ được thêm mới hoặc cập nhật. Khi có vector truyền vào với số lượng khớp
`chunks`, chúng được lưu trực tiếp. Nếu không, lớp dùng `embedding_model` đã
truyền khi khởi tạo; nếu không có mô hình đó, collection ChromaDB có thể dùng
embedding function mặc định của chính nó khi nhận documents.

Metadata được chuẩn hóa trước khi gửi tới ChromaDB: các giá trị `str`, `int`,
`float`, `bool` được giữ nguyên; `None` đổi thành chuỗi rỗng; kiểu khác được
chuyển thành chuỗi. `document_id` và `chunk_index` của `DataChunk` được đưa vào
metadata lưu trữ.

## Tìm kiếm

```python
from src.base import UserQuery

query_vector = embedder.embed_query(
    UserQuery(query_text="Điều kiện đăng ký học phần là gì?")
)
results = store.search(query_vector, top_k=5)

for result in results:
    print(result.rank, result.similarity_score, result.text)
```

`search(query_vector, top_k=5, filters=None)` tìm theo embedding của câu hỏi
(hoặc theo `query_text` nếu vector không có), và trả về danh sách
`RetrievedContext` theo thứ hạng của ChromaDB. Số kết quả yêu cầu không vượt quá
số bản ghi hiện có. `filters` được chuyển thành điều kiện `where` của ChromaDB,
nên khóa và cấu trúc điều kiện phải đúng định dạng metadata của ChromaDB.

Điểm trả về được tính từ cosine distance bằng `1 - distance` rồi giới hạn trong
khoảng `[0, 1]`; đây là điểm được chuyển đổi để tiện sử dụng, không phải xác
suất. Mỗi kết quả gồm `chunk_id`, văn bản, `similarity_score`, `rank` và
metadata. Nếu collection rỗng, phương thức trả về `[]`. Lỗi truy vấn được ghi
log và cũng trả về danh sách rỗng, vì vậy hãy kiểm tra log khi kết quả trống
bất thường.

## Các thao tác quản lý khác

- `count()` trả về tổng số bản ghi trong collection.
- `get_chunks(batch_size=5000)` đọc các bản ghi thành `List[DataChunk]` theo
  từng trang.
- `delete(chunk_ids)` xóa các ID được chỉ định, chia yêu cầu thành batch tối đa
  500 ID.
- `clear()` xóa toàn bộ dữ liệu trong collection. Đây là thao tác phá hủy dữ
  liệu; chỉ gọi khi chủ đích làm sạch hoặc lập chỉ mục lại.

## Lưu ý vận hành

- Dùng cùng mô hình embedding và cùng số chiều cho dữ liệu được lưu lẫn truy
  vấn. Nếu thay mô hình, hãy lập kế hoạch tái tạo embedding và kiểm tra
  collection tương ứng.
- `chunk_id` là khóa của thao tác upsert. Nếu mỗi lần tạo chunk sinh UUID mới,
  chạy lại ingest không tự thay thế chunk cũ trừ khi pipeline có cơ chế dọn
  hoặc đồng bộ dữ liệu.
- Dữ liệu được lưu bền vững dưới `persist_dir`; không xóa hoặc đổi thư mục nếu
  muốn giữ collection hiện tại.
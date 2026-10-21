# Tạo vector embedding

Tài liệu mô tả `UETEmbedder` trong `src/embeddings/embedder.py`. Thành phần này
chuyển văn bản của chunk và câu hỏi thành các vector số thực để lưu trữ, tìm
kiếm tương đồng trong hệ thống RAG.

## Mô hình và cấu hình hiện tại

`UETEmbedder` sử dụng `chromadb.utils.embedding_functions.DefaultEmbeddingFunction`
của ChromaDB và khai báo kích thước vector là `384`. Mặc định, tên mô hình được
ghi nhận là `all-MiniLM-L6-v2`. Tham số `model_name` tùy chỉnh tên được báo cáo;
trong phần triển khai hiện tại, nó không thay thế embedding function mặc định
của ChromaDB.

Trong `config.yaml`, phần `embedding` đang đặt `provider: chroma_default` và
`dimension: 384`. Lớp `UETEmbedder` hiện cố định kích thước `384`; các khóa
cấu hình này không được lớp đọc trực tiếp. Nếu đổi nhà cung cấp hoặc kích thước
vector, cần bảo đảm mô hình nhúng tài liệu và câu hỏi tương thích, đồng thời
kiểm tra cách khởi tạo trong pipeline trước khi lập chỉ mục lại.

## API chính

| Thuộc tính/phương thức | Kết quả |
| --- | --- |
| `dimension` | Số chiều vector, hiện là `384`. |
| `model_name` | Tên mô hình được khai báo cho instance. |
| `embed_texts(texts)` | Nhúng danh sách chuỗi thành danh sách vector; danh sách rỗng trả về `[]`. |
| `embed_chunks(chunks)` | Tạo `EmbeddedVector` tương ứng với mỗi `DataChunk`, giữ `chunk_id` và metadata. |
| `embed_query(query)` | Tạo `EmbeddedQueryVector` cho `UserQuery`, giữ ID và nội dung câu hỏi. |

Văn bản được xử lý theo từng lô tối đa 64 chuỗi. Thứ tự kết quả được giữ theo
thứ tự đầu vào, do đó mỗi vector tương ứng với phần tử cùng vị trí trong danh
sách ban đầu.

## Luồng sử dụng

```text
DataChunk.text ── embed_chunks() ──> EmbeddedVector ──> Vector DB
UserQuery.query_text ── embed_query() ──> EmbeddedQueryVector ──> tìm kiếm
```

Ví dụ:

```python
from src.base import DataChunk, UserQuery
from src.embeddings import UETEmbedder

embedder = UETEmbedder()

chunks = [
    DataChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Điều kiện đăng ký học phần...",
        chunk_index=0,
        metadata={"title": "Quy chế đào tạo"},
    )
]
vectors = embedder.embed_chunks(chunks)

query_vector = embedder.embed_query(
    UserQuery(query_text="Điều kiện đăng ký học phần là gì?")
)
```

Mỗi phần tử trong `vectors` có kiểu `EmbeddedVector`, gồm `chunk_id`, `vector`,
`dimension` và metadata. `query_vector` có kiểu `EmbeddedQueryVector`, chứa
`query_id`, `query_text`, `vector` và `dimension`.

## Xử lý lỗi

Nếu không khởi tạo được embedding function của ChromaDB hoặc có lỗi khi nhúng,
lớp ghi log và tạo vector dự phòng 384 chiều dựa trên MD5 của văn bản. Vector
dự phòng này giúp giữ đúng hình dạng dữ liệu nhưng **không biểu diễn ngữ nghĩa**
và không phù hợp để đánh giá chất lượng truy xuất. Khi thấy cảnh báo/lỗi
fallback trong log, cần kiểm tra môi trường và mô hình; không nên coi kết quả
tìm kiếm trên các vector đó là kết quả semantic search bình thường.

Không trộn vector từ các mô hình hoặc cấu hình khác nhau trong cùng collection.
Đặc biệt, thay đổi số chiều thường yêu cầu tạo collection tương thích và nhúng
lại cả tài liệu lẫn câu hỏi bằng cùng một mô hình.
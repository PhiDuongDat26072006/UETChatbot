# Phân đoạn văn bản (Chunking)

Tài liệu mô tả bộ phân đoạn trong `src/chunking`, chịu trách nhiệm chuyển
`ProcessedData` thành các `DataChunk` để đưa qua bước embedding và lập chỉ mục.
Lớp triển khai chính là `UETChunker` trong `chunker.py`.

## Vai trò trong pipeline

```text
ProcessedData -> UETChunker -> List[DataChunk] -> Embedding -> Vector DB
```

Mỗi chunk giữ nội dung văn bản cùng thông tin định danh và metadata của tài liệu
nguồn. Việc chia nhỏ giúp truy xuất những đoạn liên quan thay vì phải tìm kiếm
trên toàn bộ tài liệu dài.

## Cấu hình

`UETChunker` đọc cấu hình `chunking` trong `config.yaml`; có thể ghi đè từng
tham số khi khởi tạo:

| Tham số | Mặc định trong `config.yaml` | Ý nghĩa |
| --- | ---: | --- |
| `chunk_size` | `800` | Độ dài tối đa mục tiêu của một chunk, tính theo ký tự Python (`len`), không phải số token. |
| `chunk_overlap` | `150` | Số ký tự mục tiêu được giữ lại giữa hai chunk liên tiếp. |
| `separators` | `\n\n`, `\n`, `### `, `## `, `# `, `. `, ` ` | Các ranh giới được thử theo thứ tự, từ lớn đến nhỏ. Bộ phân đoạn thêm chuỗi rỗng làm phương án cuối nếu cấu hình chưa có. |

Ví dụ cấu hình:

```yaml
chunking:
  chunk_size: 800
  chunk_overlap: 150
  separators:
    - "\n\n"
    - "\n"
    - "### "
    - "## "
    - "# "
    - ". "
    - " "
```

Có thể truyền tham số trực tiếp; giá trị truyền vào được ưu tiên hơn cấu hình:

```python
from src.chunking import UETChunker

chunker = UETChunker(chunk_size=600, chunk_overlap=100)
```

`chunk_overlap` nên nhỏ hơn `chunk_size` để tiến trình chia đoạn tiến triển
hợp lý. Hãy giữ cùng cấu hình khi tái lập chỉ mục để kích thước và ranh giới
chunk nhất quán.

## Cách chia văn bản

1. Lấy `doc.content` và loại bỏ khoảng trắng ở hai đầu. Nếu nội dung rỗng,
   phương thức `chunk()` ghi log cảnh báo và trả về danh sách rỗng.
2. Nếu nội dung không dài hơn `chunk_size`, tạo một chunk duy nhất.
3. Với tài liệu dài hơn, `_split_text()` tìm separator đầu tiên có trong văn
   bản, tách theo separator đó và đệ quy với các separator nhỏ hơn cho những
   mảnh vẫn quá dài. Separator rỗng là mức cuối để tách thành ký tự; nếu một
   mảnh vẫn dài hơn giới hạn mà không còn separator nhỏ hơn, nó được cắt theo
   `chunk_size` với bước tối đa là `chunk_size - chunk_overlap` (tối thiểu một
   ký tự).
4. `_merge_splits()` ghép các mảnh liền kề trong giới hạn kích thước, đồng thời
   giữ lại các mảnh cuối của chunk trước để tạo phần chồng lặp. Khi một mảnh
   riêng lẻ dài hơn phần overlap, bộ phân đoạn lấy phần đuôi của mảnh đó.
5. Gắn tiêu đề tài liệu ở đầu mỗi chunk dưới dạng `[tiêu đề]\n` để giữ ngữ cảnh.

`chunk_size` là giới hạn cho phần nội dung được chia; tiêu đề được thêm sau đó,
nên độ dài cuối cùng của `DataChunk.text` có thể lớn hơn `chunk_size`. Việc
chia dựa trên số ký tự, do đó không đảm bảo mỗi chunk có cùng số token.

## Nhận diện bảng Markdown

Chunker duyệt từng dòng trong nội dung để tìm bảng Markdown, nên bảng vẫn được nhận ra khi có tiêu đề hoặc đoạn văn đứng trước. Dòng tiêu đề bảng phải bắt đầu bằng dấu `|`, có ít nhất hai ô không rỗng. Phía sau cần có ít nhất một dòng bắt đầu bằng `|` với cùng số ô. Dòng phân cách Markdown như `| --- | --- |` được bỏ qua nếu có.

Chunker tiếp tục quét phần văn bản còn lại sau mỗi bảng, vì một tài liệu có thể chứa nhiều bảng xen giữa các đoạn văn. Mỗi đoạn văn được chia theo cách thông thường. Trước khi chunking, mỗi hàng bảng được phẳng hóa thành các cặp `tên cột: giá trị`; tên bảng được chèn vào đầu mỗi chunk dưới dạng `[META: ...]`. Sau đó, cửa sổ trượt gom các hàng đã làm giàu và lặp một số hàng cuối ở chunk kế tiếp để tạo overlap. Hàng dài hơn ngân sách được giữ nguyên trong chunk riêng. Các chunk được ghép theo đúng thứ tự ban đầu. `chunk_size` và `chunk_overlap` tính bằng ký tự, không phải token. Chunk bảng có metadata `data_type: tabular`; nội dung tên bảng và tên cột đã nằm trực tiếp trong `text` để được embedding.

## Dữ liệu đầu ra

`chunk(doc)` trả về `List[DataChunk]`. Mỗi chunk có:

- `chunk_id`: ID do mô hình dữ liệu tạo, mặc định là UUID.
- `document_id`: ID của `ProcessedData` nguồn.
- `text`: nội dung chunk, có thể kèm tiêu đề ở đầu.
- `chunk_index`: vị trí chunk trong tài liệu, bắt đầu từ `0`.
- `metadata`: metadata ban đầu của tài liệu, được bổ sung hoặc ghi đè bởi:
  `title`, `document_id`, `chunk_index`, `total_chunks`, `char_count`.

`char_count` là độ dài phần segment trước khi thêm tiền tố tiêu đề. Các khóa
chuẩn nêu trên được ghi đè nếu metadata đầu vào có cùng tên.

## Sử dụng

```python
from src.base import ProcessedData
from src.chunking import UETChunker

document = ProcessedData(
    id="regulation-2025",
    title="Quy chế đào tạo",
    content="Nội dung văn bản đã được làm sạch...",
    metadata={"unit": "uet", "source": "https://example.edu.vn/regulation"},
)

chunker = UETChunker()
chunks = chunker.chunk(document)

for chunk in chunks:
    print(chunk.chunk_index, chunk.text)
```

Để xử lý một danh sách, gọi `chunk_batch(docs)`. Ngoài việc trả về toàn bộ
chunk, phương thức này còn ghi từng dòng JSONL vào
`data/chunked_data/<khoa>_chunks.jsonl`. Tên nhóm được xác định từ metadata
`unit`/khoa; nếu không nhận diện được thì dùng `domain` hoặc `other`. File được
mở ở chế độ nối thêm (`append`), vì vậy gọi lặp lại sẽ tiếp tục ghi vào file
đang có, không tự xóa dữ liệu cũ. Quy trình lập chỉ mục chính dùng các chunk
được trả về để chuyển sang các bước tiếp theo.

## Lưu ý khi vận hành

- Đưa vào bộ phân đoạn `ProcessedData` đã được làm sạch; nội dung còn HTML hoặc
  lỗi trích xuất sẽ được chia nguyên trạng.
- Điều chỉnh separator theo cấu trúc tài liệu. Separator xuất hiện sớm hơn
  trong danh sách được ưu tiên; chuỗi rỗng là phương án chia đến từng ký tự.
- Các chunk có overlap lặp lại một phần nội dung giữa những đoạn liên tiếp;
  đây là chủ đích nhằm giữ ngữ cảnh, không phải bản ghi trùng ngoài ý muốn.

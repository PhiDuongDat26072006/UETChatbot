# Phân đoạn văn bản (Text Chunking)

Tài liệu mô tả bộ phân đoạn văn bản trong `src/chunking`, chịu trách nhiệm chuyển đổi `ProcessedData` thành danh sách các `DataChunk` thông minh (Structure-Aware Chunks) để phục vụ quá trình nhúng vector (`BAAI/bge-m3`) và lập chỉ mục tìm kiếm kết hợp (Hybrid Search: Vector DB + BM25).

Lớp triển khai chính là `UETChunker` trong [`src/chunking/chunker.py`](chunker.py).

---

## 1. Vai trò trong Pipeline

```text
ProcessedData ──> UETChunker ──> List[DataChunk] ──> Embedding ──> Vector DB & BM25
```

Mỗi chunk lưu giữ nội dung văn bản hoàn chỉnh kèm metadata chi tiết (nguồn, thứ tự, tiêu đề phân cấp, đường dẫn breadcrumb). Bộ phân đoạn được thiết kế đặc thù cho tài liệu học vụ và quy chế đào tạo của Trường Đại học Công nghệ (UET) - ĐHQGHN, bảo tồn tối đa ngữ cảnh của từng điều khoản pháp lý và bảng biểu.

---

## 2. Cấu hình tham số

`UETChunker` tự động nạp cấu hình khối `chunking` từ [`config.yaml`](../../config.yaml) và cho phép ghi đè linh hoạt khi khởi tạo:

| Tham số | Mặc định trong `config.yaml` | Ý nghĩa kỹ thuật |
| --- | :---: | --- |
| `chunk_size` | `800` | Độ dài tối đa mục tiêu của một chunk, tính theo ký tự Python (`len(str)`). |
| `chunk_overlap` | `150` | Độ dài mục tiêu của phần gối đầu giữa 2 chunk liên tiếp để bảo toàn mạch ngữ nghĩa. |
| `min_chunk_size` | `100` | Ngưỡng ký tự tối thiểu của một chunk. Các đoạn mồ côi nhỏ hơn ngưỡng này sẽ được gộp vào chunk liền trước nếu cùng Điều/Heading. |
| `separators` | Danh sách phân cấp | Thứ tự ưu tiên cắt văn bản từ mức cấu trúc lớn (Heading, Chương, Điều) đến đoạn (`\n\n`), dòng (`\n`), câu (`. `), từ (` `). |

### Cấu hình chuẩn trong `config.yaml`:

```yaml
chunking:
  chunk_size: 800
  chunk_overlap: 150
  min_chunk_size: 100
  separators:
    - 'regex:(?<=\n)(?=(?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+)'
    - 'regex:(?<=\n)(?=(?:Mục|MỤC)\s+\d+)'
    - 'regex:(?<=\n)(?=(?:Điều|ĐIỀU)\s+\d+[\.:])'
    - "### "
    - "## "
    - "# "
    - 'regex:(?<=\n)(?=(?:Khoản\s+\d+[\.:]|\d+\.\s))'
    - "\n\n"
    - "\n"
    - ". "
    - "; "
    - " "
    - ""
```

### Khởi tạo tùy biến trong code:

```python
from src.chunking import UETChunker

# Ghi đè cấu hình runtime
chunker = UETChunker(
    chunk_size=600,
    chunk_overlap=120,
    min_chunk_size=80,
)
```

---

## 3. Các cơ chế thuật toán cốt lõi

### 3.1. Phân cấp văn bản & Context Breadcrumb Injection
* **Nhận diện cấu trúc quy chế:** Tự động phát hiện các ranh giới pháp lý và phân cấp Markdown:
  * **Cấp 1:** Markdown `# ` hoặc `Chương [IVXLCDM\d]+` (Chương I, Chương II...)
  * **Cấp 2:** Markdown `## ` hoặc `Mục \d+` (Mục 1, Mục 2...)
  * **Cấp 3:** Markdown `### ` hoặc `Điều \d+[\.:]` (Điều 1., Điều 14:...)
* **Tiêm đường dẫn phân cấp (Breadcrumbs):** Mỗi chunk được tiêm chuỗi định danh cấu trúc ở đầu:
  ```text
  [Quy chế đào tạo UET > Chương II: TỔ CHỨC ĐÀO TẠO > Điều 14. Đăng ký học tập]
  1. Đầu mỗi học kỳ, sinh viên phải đăng ký học phần qua cổng Portal...
  ```
* **Quản lý ngân sách độ dài (Length Budget):** Thuật toán tự động trừ độ dài chuỗi breadcrumb khỏi `chunk_size` để đảm bảo tổng độ dài thực tế của `DataChunk.text` không bị vượt ngưỡng.

### 3.2. Cắt đệ quy bảo vệ tiền tố (Regex Lookahead Splitter)
* Khi chia văn bản theo tiêu đề Markdown hoặc các điều khoản, thuật toán sử dụng Regex lookahead (`(?=...)`).
* Nhờ đó, tiêu đề (`### `, `Điều 14.`) luôn được giữ nguyên vẹn ở **đầu** của đoạn văn bản tiếp theo, không bị cắt dời vô lý về cuối đoạn trước.

### 3.3. Bảo vệ ranh giới từ khi gối đầu (Word Boundary Protection)
* Khi một câu văn đơn lẻ dài hơn `chunk_overlap`, cơ chế fallback cần cắt lấy phần đuôi của đoạn trước.
* Thuật toán kiểm tra và **lùi về khoảng trắng gần nhất** (`rfind(' ')`) thay vì cắt `slice` cứng theo ký tự.
* **Ngăn chặn 100%** lỗi cắt đứt giữa từ tiếng Việt (ví dụ: tránh cắt *"khuyến khích"* thành *"hích"*, *"được"* thành *"ược"*), bảo vệ tính toàn vẹn của Tokenizer (`BAAI/bge-m3`) và BM25.

### 3.4. Lọc rác & Gộp Chunk mồ côi (`min_chunk_size`)
* **Lọc bỏ dòng vô nghĩa:** Loại bỏ các dòng chỉ chứa ký tự phân cách hoặc rác cào web (`$/$`, `---`, `***`, khoảng trắng rỗng).
* **Gộp chunk mồ côi thông minh:**
  * Nếu phần đuôi của một Điều luật sau khi chia bị dư lại một đoạn ngắn ($< \text{min\_chunk\_size}$), đoạn này sẽ được tự động gộp vào chunk liền trước.
  * **Nguyên tắc an toàn:** Chỉ gộp khi 2 chunk có **cùng heading/Điều**, và kích thước sau khi gộp không vượt quá $1.3 \times \text{chunk\_size}$. Nếu là một Điều luật mới (khác heading), chunk vẫn được bảo tồn độc lập để không làm mất tiêu đề.

---

## 4. Xử lý bảng biểu Markdown (Tabular Chunks)

Tài liệu học vụ chứa nhiều bảng biểu (khung chương trình, điều kiện tiên quyết, điểm chuẩn, biểu phí):
1. **Phát hiện bảng:** Tự động nhận diện cấu trúc bảng Markdown (`| Cột 1 | Cột 2 |`), hỗ trợ cả các bảng có ô tiêu đề trống (tự điền `Cột {i}`).
2. **Trích xuất ngữ cảnh bảng:** Tự động dò ngược tiêu đề gần nhất trước bảng để làm nhãn `[META: Tiêu đề bảng]`.
3. **Tuyến tính hóa từng hàng (Row-level Serialization):** Chuyển đổi mỗi dòng bảng thành dạng cặp thuộc tính rõ ràng:
   ```text
   - Mã ngành: CN1 | Tên ngành: Công nghệ thông tin | Điểm chuẩn: 27.85
   ```
4. **Cửa sổ trượt hàng:** Gom các hàng theo ngân sách `chunk_size`, duy trì gối đầu lặp lại một số hàng giữa các chunk bảng liền kề. Chunk bảng được gán `metadata["data_type"] = "tabular"`.

---

## 5. Cấu trúc dữ liệu đầu ra (`DataChunk`)

Phương thức `chunk(doc)` trả về `List[DataChunk]`. Mỗi đối tượng gồm:

* `chunk_id`: Mã định danh UUIDv4 duy nhất của chunk.
* `document_id`: ID của tài liệu `ProcessedData` nguồn.
* `text`: Nội dung văn bản hoàn chỉnh của chunk (đã kèm header breadcrumbs).
* `chunk_index`: Vị trí của chunk trong tài liệu (bắt đầu từ `0`).
* `metadata`: Kế thừa metadata của tài liệu nguồn và bổ sung các trường chuẩn hóa:
  * `title`: Tiêu đề tài liệu gốc.
  * `breadcrumb`: Chuỗi đường dẫn phân cấp đầy đủ (ví dụ: `Quy chế UET > Chương I > Điều 1`).
  * `heading`: Tiêu đề mục trực tiếp chứa chunk (ví dụ: `Điều 1. Phạm vi điều chỉnh`).
  * `char_count`: Tổng số ký tự thực tế của `chunk.text`.
  * `data_type`: Loại dữ liệu (`"text"` hoặc `"tabular"`).
  * `chunk_index`: Vị trí chunk.
  * `total_chunks`: Tổng số chunk của tài liệu sau khi lọc/gộp.

---

## 6. Hướng dẫn sử dụng

### Phân đoạn một tài liệu:

```python
from src.base import ProcessedData
from src.chunking import UETChunker

doc = ProcessedData(
    id="doc_quy_che_2024",
    title="Quy chế đào tạo Đại học UET 2024",
    content="""
Chương I: QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Quy chế này quy định về đào tạo đại học chính quy tại Trường Đại học Công nghệ.

Điều 2. Thời gian và kế hoạch đào tạo
1. Năm học được tổ chức theo học chế tín chỉ.
2. Mỗi năm học gồm hai học kỳ chính và một học kỳ phụ.
    """,
    metadata={"unit": "dao_tao", "year": 2024},
)

chunker = UETChunker()
chunks = chunker.chunk(doc)

for c in chunks:
    print(f"[{c.chunk_index + 1}/{len(chunks)}] Heading: {c.metadata['heading']}")
    print(f"Breadcrumb: {c.metadata['breadcrumb']}")
    print(f"Nội dung:\n{c.text}\n{'-'*40}")
```

### Phân đoạn hàng loạt và lưu trữ theo Khoa (`chunk_batch`):

```python
# Phân đoạn hàng loạt tài liệu ProcessedData
all_chunks = chunker.chunk_batch(processed_docs, append=True)

# Kết quả tự động phân loại theo khoa/nguồn và lưu tại:
# data/chunked_data/{faculty}_chunks.jsonl (fit_chunks.jsonl, uet_chunks.jsonl, v.v.)
```

---

## 7. Kiểm thử đơn vị (Unit Tests)

Bộ kiểm thử tại [`tests/test_chunking.py`](../../tests/test_chunking.py) kiểm tra tự động toàn diện:
* Đọc và ghi đè tham số cấu hình `config.yaml`.
* Bảo toàn độ gối đầu overlap và cơ chế không cắt đứt từ ngữ tiếng Việt.
* Nhận diện cấu trúc phân cấp Chương / Điều và tiêm Context Breadcrumbs.
* Bóc tách bảng biểu Markdown và tuần tự hóa theo hàng.
* Lọc bỏ dòng rác (`$/$`, `---`) và gộp chunk mồ côi (`min_chunk_size`).

Chạy kiểm thử:
```bash
python -m unittest tests/test_chunking.py
```

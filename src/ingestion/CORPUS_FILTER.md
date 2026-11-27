# Corpus filtering during ingestion

Bản tích hợp dựa trên ingestion_current.zip. Chép các file src/ trong patch vào
đúng dự án. RawData.metadata phải tồn tại như hợp đồng hiện tại; patch không sửa
src/base.py, chunker, embeddings, indexing hoặc root pipeline.

## Áp dụng ngay trên dữ liệu đã có

```bash
python3 -m src.ingestion.main filter --sites UET --export-processed
```

Không HTTP, không extraction, không embedding. Đọc data/data_source/UET.jsonl và
data/processed_data/UET.jsonl, xuất:

- data/corpus/UET/UET-ranked.jsonl và UET-ranked.tsv: tất cả nguồn đã xếp hạng.
- data/corpus/UET/UET-include-links.txt, UET-review-links.txt, UET-exclude-links.txt.
- data/corpus/UET/processed/UET.jsonl: payload ProcessedData gốc của nguồn include
  VÀ crawl accepted, có content không rỗng; bổ sung metadata.corpus.
- data/corpus/UET/summary.json: số exported/empty/missing_processed/not_exported.

Cho chunker đọc thư mục data/corpus/UET/processed thay vì toàn bộ processed_data.
Patch không đổi input mặc định của chunker hoặc tự xóa vector/BM25 cũ; index đã
build từ corpus cũ vẫn cần rebuild theo quy trình của nhóm.

Thiếu file processed: lệnh báo lỗi, không tạo dữ liệu giả. Thiếu một document:
report missing_processed. Chạy sau khi ingestion kết thúc; writer.lock đang tồn
tại thì filter từ chối. Không chạy nhiều filter cùng output-dir. Mỗi file output
được replace nguyên tử, cả bộ report không phải transaction chung. Export không
kiểm chứng raw revision hiện tại hay hiệu lực văn bản; không dedup khác source ID.
Chạy lại --export-processed sau mỗi lần chỉnh policy để thay toàn bộ tập chọn cũ.
Chỉ chạy filter không có --export-processed sẽ không cập nhật file export cũ.

## Lấy thêm dữ liệu và xử lý streaming

```bash
python3 -m src.ingestion.main ingest --from web --to processed --sites UET --corpus-mode balanced
python3 -m src.ingestion.main ingest --from web --to processed --sites UET --corpus-mode strict
python3 -m src.ingestion.main ingest --from data_source --to processed --sites UET --corpus-mode strict
python3 -m src.ingestion.main ingest --from raw --to processed --sites UET --corpus-mode strict
```

| Mode | Quyết định lưu raw/gửi extractor |
|---|---|
| balanced (mặc định mới) | include + review; bỏ exclude |
| strict | chỉ include; site chưa cấu hình policy báo lỗi |
| off | bỏ qua corpus gate, giữ crawl policy hiện có |

balanced cho khoa chưa đăng ký policy giữ hành vi cũ. Report corpus_enabled cho
biết site có policy hay không. CLI extractor độc lập không dùng corpus gate;
để lọc raw theo site hãy dùng `main ingest --from raw --to processed`.

1. Trước HTTP: prefetch.exclude_prefixes và exclude_patterns chặn URL rõ ràng.
   Prefix đối chiếu path gốc, pattern đối chiếu path bỏ dấu/chuyển dấu gạch thành
   khoảng trắng. Không chặn tất cả hoc-phi hoặc tất cả năm cũ.
2. URL chưa rõ và trang danh mục vẫn được tải để đọc tiêu đề/khám phá link.
   Listing không lưu raw. Prefix chặn rõ (tag/author/en...) có thể chặn listing.
3. Sau HTML: cùng policy URL/title quyết định giữ raw, ghi metadata.corpus trong
   data_source. Trang bị loại ở bước này vẫn khám phá link con.
4. Raw được commit trước khi submit extractor, dùng process pool và batch writer
   hiện có. Corpus gate áp dụng cả raw backlog và trước khi publish kết quả.
5. Không xóa raw/processed chỉ vì corpus exclude. Khi tái sử dụng processed cũ,
   export là nơi áp dụng lại policy hiện tại; không tin nhãn corpus cũ trong file.
   Cơ chế xóa theo crawl rejection/alias có sẵn vẫn giữ nguyên.

Cần GET mới biết title của slug số: không thể hứa bỏ hết request rác. Chặn trước
HTTP cũng mất link chỉ xuất hiện trên trang bị chặn; thêm seed/override chính xác
cho tài liệu quan trọng. Giữ ưu tiên seed đào tạo và ngân sách crawl để bù rủi ro.
Không dùng API WordPress. Không gọi LLM phân loại.

## Thứ tự và checkpoint

Exact include overrides được thêm thành seed ưu tiên (vẫn phải nằm trong crawl
scope và được crawl policy cho phép). Các seed khác giữ thứ tự cấu hình. BFS giữ
độ sâu; trong cùng depth, URL/anchor/title có priority cao được xếp trước khi submit.
Concurrency không bảo đảm thứ tự hoàn tất. Anchor chỉ dùng xếp lịch, không cấp
quyền include hay tự loại tài liệu. Đây không phải priority toàn cục vượt BFS.

Policy content hash và mode nằm trong khóa checkpoint crawl. Khi đổi policy/mode,
crawler duyệt lại từ seeds để không bỏ sót link từng bị chặn; không cần xóa data.
Khi policy/mode không đổi, checkpoint complete vẫn được tôn trọng. Muốn cập nhật
web mới dùng --refresh như trước. Muốn chỉ lọc dữ liệu cũ: filter/export, không HTTP.

## Mở rộng cho khoa

Registry: src/ingestion/config/corpus/sites.yaml

```yaml
version: 1
sites:
  UET: ../uet_corpus.yaml
  FIT: fit.yaml
```

Copy config/corpus/fit.example.yaml thành fit.yaml, chỉnh host/hosts, rule,
overrides và prefetch theo dữ liệu khoa rồi đăng ký. Template chưa bật tự động.
Hosts của policy phải là tập con scope.hosts trong site profile khi chạy ingest.
Không cần đổi Python. Không sao chép toàn bộ bộ lọc UET sang khoa khác.

```bash
python3 -m src.ingestion.main filter --sites FIT
python3 -m src.ingestion.main filter --sites UET FIT --export-processed
python3 -m src.ingestion.main ingest --from web --to processed --sites UET FIT --corpus-mode balanced
```

--corpus-config nhận registry khác cho cả filter/ingest. --policy của filter nhận
policy trực tiếp cho một site; --sources cũng chỉ dùng với một site. --output-dir
cho nhiều site sẽ tạo thư mục con tương ứng.

## Override và giới hạn

Corpus overrides dùng include/review/exclude và priority 0..100. Crawl overrides
trong config/overrides.yaml vẫn dùng accepted/listing/rejected/deferred. Hai loại
khác mục đích: corpus include không vượt qua scope/rejection của crawler. Crawl
accepted cho phép request nhưng không tự ép corpus include. Sửa cả hai khi cần.
Exact corpus exclude chặn trước HTTP (ngoại trừ explicit crawl accepted/listing
cho phép request để khám phá; vẫn không lưu corpus bị exclude).

Các suffix -2/-43 chưa thể xác định trùng/cũ bằng URL: giữ để đối chiếu nội dung,
khóa học, ngày hiệu lực. Mức thu/hướng dẫn theo khóa có thể review. Quality của
extractor vẫn là advisory, không tự xóa payload thành công. include là heuristic
chủ đề, không chứng nhận văn bản hiện hành.

## File chịu trách nhiệm

- corpus/gate.py: registry, policy snapshot, prefetch, eligibility, priority.
- corpus/policy.py: phân loại URL/title; hỗ trợ nhiều host của một site.
- crawler/engine.py: prefetch, ưu tiên trong BFS, gate trước raw, checkpoint.
- runner.py: gate data_source -> raw, tái đánh giá nguồn trước mỗi lần chạy.
- streaming.py: gate backlog raw và kết quả extractor, vẫn dùng pool/batch cũ.
- corpus/runner.py: filter nhiều site, reports, điều phối export.
- corpus/export.py: join source_id với processed, lọc tập include, thống kê thiếu.
- main.py: các cờ mới; sửa handlers chưa khởi tạo trong nhánh ingest.

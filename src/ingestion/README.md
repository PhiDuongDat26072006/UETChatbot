# UET ingestion — HTML-first, streaming extraction

Chạy từ thư mục gốc dự án. Dùng DataSource, RawData, ProcessedData trong
src/base.py; RawData cần trường `metadata` (thay cho `raw_metadata`). Cài dependencies
trong `src/ingestion/requirements.txt`. Bản vá này áp dụng sau bản network speed patch;
ZIP chỉ chứa các file thay đổi, không có dữ liệu hoặc tests/.

## Lệnh chạy

```bash
python3 -m src.ingestion.main ingest --from web --to data_source --sites UET FIT
python3 -m src.ingestion.main ingest --from data_source --to raw --sites UET FIT
python3 -m src.ingestion.main ingest --from web --to raw --sites UET FIT
python3 -m src.ingestion.main ingest --from web --to processed --sites UET FIT
python3 -m src.ingestion.main ingest --from data_source --to processed --sites UET FIT
python3 -m src.ingestion.main ingest --from raw --to processed --sites UET FIT
python3 -m src.ingestion.main status --sites UET FIT
```

`web/data_source → processed`: các site chạy song song; khi một batch raw đã commit,
coordinator gửi đường dẫn/hash sang process pool chung. Extractor chạy khi network
vẫn tiếp tục, không đợi hết site hay toàn chặng raw. Web dùng lại HTTP response để
lưu raw, không tải bài viết lần hai. Raw tồn đọng được đưa vào pool ngay đầu lượt chạy.

## Cấu hình

CLI ghi đè `config/runtime.yaml`. Khởi động lại để nhận thay đổi YAML.

| Cấu hình | Mặc định | Ý nghĩa |
|---|---:|---|
| download_workers | 40 | Tổng request đồng thời |
| site_workers | 4 | Số site coordinator đồng thời |
| per_host | 10 | Quota request theo nhóm host của site |
| stream_extract_workers | null | null = download_workers; số process extract dùng chung |
| extract_workers | 8 | Thread cho lệnh raw → processed chạy riêng |
| flush_every | 20 | Batch commit metadata mạng và kết quả extract |
| max_pages / max_depth | 10000 / 10 | Ngân sách trang mỗi site / độ sâu BFS |
| timeout / retries / delay | 30 / 3 / 0.25 | Timeout HTTP, retry, delay mỗi worker |

Với mặc định: tối đa 40 request và pool 40 process extract cho toàn lượt chạy,
không phải 40 process/site. Nếu chỉ còn UET, per_host vẫn giới hạn 10 request;
extractor có thể sử dụng toàn bộ pool nếu đủ raw. Không tự tăng quota UET.

```bash
# Ghi kết quả mỗi 20 mục hoàn tất; pool extract tự lấy download_workers.
python3 -m src.ingestion.main ingest --from web --to processed \
  --sites UET FIT --download-workers 40 --per-host 10 --flush-every 20

# Ghi đè riêng số process nếu cần.
python3 -m src.ingestion.main ingest --from web --to processed \
  --sites UET --stream-extract-workers 40
```

## Ghi theo batch và resume

- Worker chỉ đọc raw đã lưu, kiểm hash và trả kết quả; coordinator của site là writer.
- Thu kết quả theo thứ tự hoàn thành. Đủ `flush_every` kết quả thì ghi snapshot JSONL
  và state; batch cuối chưa đủ cũng được flush. Checkpoint mạng có thể flush sớm hơn.
- Khi HTTP đang chờ, coordinator vẫn polling kết quả extract để ghi batch.
- Raw → processed chạy riêng cũng dùng rolling window và batch commit; không giữ
  tất cả kết quả đến cuối lượt chạy. Mỗi snapshot được thay thế atomic, không append
  trùng record. Review payload riêng vẫn ghi từng file.
- Số future pending mỗi site bị chặn ở kích thước pool; khi đầy, producer chờ kết quả.
- Kết quả của raw cũ bị bỏ nếu source đã có raw mới hoặc không còn accepted.
- Lỗi extract được ghi failed; không đánh dấu completed. Process pool không ghi file.
- Sau crash, raw chưa có kết quả hợp lệ được xử lý lại; state completed được ghi sau
  processed. Đây là replay idempotent, không phải transaction chung nhiều file.
- `--limit` vẫn là ngân sách toàn lượt theo từng chặng; các site chạy tuần tự khi bật.
- `--refresh` tải lại; `--reprocess` extract lại; `--retry-failed` áp dụng riêng cho
  data_source → raw hoặc raw → processed. Checkpoint web đã complete thì resume bỏ qua.

## File và giao tiếp

| File | Trách nhiệm |
|---|---|
| main.py, config/settings.py, config/runtime.yaml | CLI, kiểm tra cấu hình, mặc định |
| network_runner.py | Pool HTTP chung, site threads, process pool chung và report |
| runner.py | Điều phối chặng, lưu nguồn/raw, batch cho extract độc lập |
| crawler/engine.py | BFS, checkpoint; callback commit raw và polling extract |
| streaming.py | Backlog, pending futures, chống kết quả cũ, batch publish |
| extractor/worker.py | Hàm process worker: đọc raw → process_raw → payload + quality |
| extractor/pipeline.py | Thuật toán extract/quality hiện có, không đổi |
| utils.py | Rolling window; polling khi chờ HTTP |
| storage.py | JSONL snapshots, atomic write, batch và writer lock hiện có |

Dữ liệu giữ nguyên: `data/data_source/<SITE>.jsonl`,
`data/raw_data/<SITE>/records.jsonl`, `data/processed_data/<SITE>.jsonl`.
Metadata nằm trong `metadata: {}`; giữ nguyên quy tắc ID. Không tạo FIT-accepted.jsonl.
Selectors/policy trong `config/sites/` không thay đổi. Không sửa indexing, embeddings,
retrieval, root pipeline hoặc base. Delay/retry hiện có giữ nguyên; chưa thêm jitter.

Extractor CLI riêng vẫn hoạt động:

```bash
python3 -m src.ingestion.extractor --html page.html --url https://www.fit.uet.vnu.edu.vn/example/ --site FIT
python3 -m src.ingestion.extractor --pdf document.pdf
```

Streaming ở đây chỉ xử lý HTML, không tự gọi OCR API. Xem VERIFICATION.md để biết
phạm vi đã kiểm tra; chưa benchmark trên máy người dùng hoặc crawl thật toàn bộ UET.

## HTML icon cleanup — 2026-10-27

Bộ lọc extractor/html/icons.py xử lý emoji chữ trước cả selector và trafilatura,
loại ảnh có class emoji/wp-smiley/emojione; giữ URL, ảnh nội dung, toán học/đơn vị.
Keycap 1️⃣ thành 1, emoji giữa hai từ thành khoảng trắng. Không sửa raw.
Không nhận diện nguồn gốc AI, không tự xóa SVG/icon font chưa có rule.
Bỏ qua code/pre/script/style; phạm vi là nội dung văn bản HTML thông thường.
PROCESSOR_VERSION=html-v2-icons làm processing_key thay đổi: chạy lại raw→processed
để tái xử lý. Downstream chunk/index không tự cập nhật theo bản vá này.
Kiểm tra ngoài gói PASS: emoji ghép ZWJ/màu da/cờ, keycap, ký hiệu toán học, URL,
ảnh thường và cả hai nhánh selector/fallback. Không thêm dependency hoặc tests/.

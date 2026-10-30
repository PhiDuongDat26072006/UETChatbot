# Streaming + batch extraction verification — 2026-10-27

20 kiểm tra unittest PASS, chạy ngoài gói trên Python 3.12:

- 14 kiểm tra regression network: web→raw tái sử dụng response, source→raw,
  resume/refresh 304, ngân sách limit, max_depth/max_pages, standalone extraction,
  batch metadata, checkpoint khi lỗi commit, rolling refill, quota host aliases/tổng,
  redirect scope, writer lock, lỗi download.
- 6 kiểm tra bổ sung:
  1. ProcessPoolExecutor spawn thật, 2 processes: processed JSONL có batch ít nhất
     5 record khi request /slow vẫn chưa trả về; cuối lượt đủ 13 record. Chạy lại
     không GET, không submit extract lại, skip đủ 13.
  2. Backlog raw→process qua data_source→processed, không GET lại; override pool=1.
  3. Standalone raw→processed: 12 kết quả, flush_every=5, snapshot lần lượt 5/10/12.
  4. Raw bị sửa sai hash: failed, không publish processed cho source lỗi.
  5. Hai site với --limit 3: tổng raw và processed đều 3, không vượt ngân sách.
  6. Future của raw cũ không thể ghi đè phiên bản raw mới.

Ruff check --select F và compileall đạt. Site profiles không thay đổi so với bản
network speed patch. ZIP không chứa tests/, __pycache__, .pyc, dữ liệu, base.py.
Dùng base.py thật trong bản sao tạm với raw_metadata đổi thành metadata.

Giới hạn: các HTTP fixture chạy local, không gọi OCR/API; chưa đo CPU/RAM hoặc
throughput trên máy người dùng với 40 process, chưa chạy crawl đầy đủ UET.
Không kiểm chứng crash ngay giữa mọi cặp file snapshot; replay không phải một
transaction nhiều file. Raw→processed độc lập giữ ThreadPool, streaming mới dùng
ProcessPool. PDF CLI riêng không nằm trong thay đổi batch HTML này.

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

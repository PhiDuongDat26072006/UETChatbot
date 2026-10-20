# UET Chatbot — Data Pipeline

Hệ thống thu thập, phân loại và bóc tách dữ liệu tự động cho Trợ lý ảo (Chatbot RAG) từ các cổng thông tin của Trường Đại học Công nghệ (UET - ĐHQGHN) và 8 Khoa/Viện trực thuộc.

## Tổng quan Pipeline

```
Websites / WordPress APIs
        ↓
[crawler/] → Cào BFS + WP REST API → 3,184 endpoints sạch + files gốc
        ↓
[validator/] → Xác thực Trailing Slash & Soft-404 → endpoints.txt
        ↓
[classifier/] → Phân loại 7 danh mục (Taxonomy) → endpoints_metadata.jsonl
        ↓
[extractor/] → Bóc tách HTML + PDF/DOCX/XLSX/TXT → data/processed_data/<domain>.jsonl
        ↓
[loader.py] → UETDataLoader (BaseDataCrawler) → List[RawData] cho các module RAG phía sau
```

## Cấu trúc thư mục

```
data/                                      # (bị ignore khỏi git, trừ data/doc.md)
├── raw_data/<domain>/
│   ├── endpoints/
│   │   ├── endpoints.txt                  # URLs sạch đã thẩm định
│   │   ├── endpoints_metadata.jsonl       # Metadata phân loại
│   │   └── by_category/                   # URLs chia theo danh mục
│   └── files/                             # Tài liệu tải về (.pdf, .docx...)
└── processed_data/<domain>.jsonl          # Tài liệu sạch sau bóc tách (3,244 docs)

src/ingestion/
├── __init__.py                            # Module 1 public API exports
├── __main__.py                            # CLI entry point (python -m src.ingestion)
├── main.py                                # CLI entry point (crawl/verify/classify/extract)
├── loader.py                              # UETDataLoader: processed_data/*.jsonl -> RawData
├── utils.py                               # Tiện ích dùng chung (clean_whitespace)
├── filters.py                             # Bộ lọc URL, chuẩn hóa, kiểm tra cloud storage
├── config/                                # Cấu hình module hóa (< 100 dòng/file)
│   ├── paths.py                           # Thư mục, đường dẫn đọc từ config.yaml
│   ├── constants.py                       # User-Agent, timeouts, đuôi file tài liệu/media
│   ├── faculties.py                       # Danh sách 8 khoa/viện + UET portal, mapping unit
│   ├── session.py                         # HTTP Session pooling với retry adapter
│   └── __init__.py                        # Re-exports thống nhất
├── crawler/                               # Bộ cào dữ liệu song song
│   ├── downloader.py                      # Tải file streaming an toàn
│   ├── wp_api.py                          # Quét WordPress REST API (posts/pages/media)
│   ├── engine.py                          # Crawler engine BFS đa luồng
│   └── __init__.py
├── validator/                             # Thẩm định và sửa lỗi URLs
│   ├── detector.py                        # Phát hiện soft-404, redirect, trailing slash
│   ├── fixer.py                           # Tự động chuẩn hóa và cập nhật endpoints.txt
│   └── __init__.py
├── classifier/                            # Phân loại URL theo Taxonomy 7 danh mục
│   ├── taxonomy.py                        # Định nghĩa Category enum, từ khóa, regex chuẩn
│   ├── scoring.py                         # Thuật toán tính điểm weighted scoring
│   ├── runner.py                          # Thực thi phân loại toàn bộ endpoints
│   └── __init__.py
├── extractor/                             # Bóc tách văn bản sạch và tóm tắt RAG
│   ├── date_patterns.py                   # Regex ngày tháng, evergreen rules
│   ├── date_extractor.py                  # Trích xuất thời gian công bố từ URL/HTML/file
│   ├── html_cleaner.py                    # Trafilatura, cloud links preservation, markdown table
│   ├── html_extractor.py                  # Trích xuất bài viết HTML hoàn chỉnh
│   ├── doc_extractor.py                   # Trích xuất PDF, DOCX, XLSX, DOC, TXT
│   ├── summary.py                         # Sinh tóm tắt ngữ cảnh [Domain | Title]
│   ├── pipeline_helpers.py                # Canonical URL, băm ID, chống trùng lặp stub
│   ├── pipeline.py                        # Điều phối trích xuất đa luồng ra JSONL
│   └── __init__.py
└── tests/                                 # 170 unit tests (pytest)
```

## Cài đặt và sử dụng (chạy từ thư mục gốc repo)

### 1. Môi trường
```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

### 2. Nạp dữ liệu cho RAG
```bash
python src/ingestion/loader.py
```

### 3. Chạy CLI thu thập / bóc tách
```bash
# Thu thập dữ liệu (Crawler)
python -m src.ingestion --crawl
python -m src.ingestion --crawl --no-download

# Xác thực trailing slash (Validator)
python -m src.ingestion --verify

# Phân loại URL theo Taxonomy (Classifier)
python -m src.ingestion --classify
python -m src.ingestion --classify --domain fit.uet.vnu.edu.vn

# Bóc tách văn bản sạch (Extractor)
python -m src.ingestion --extract
python -m src.ingestion --extract --domain fema.uet.vnu.edu.vn
python -m src.ingestion --extract --force --workers 15
```

### 4. Chạy Tests
```bash
pytest src/ingestion/tests -v
```

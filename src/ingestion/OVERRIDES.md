# Manual URL overrides

Áp dụng sau ingestion_streaming_batch_patch.zip và ingestion_icon_cleanup_patch.zip.
Không sửa config/runtime.yaml, site profiles, base.py hoặc dữ liệu của bạn.

Sửa config/overrides.yaml (URL chính xác, theo canonicalization hiện có):

```yaml
version: 1
sites:
  UET:
    urls:
      "https://uet.vnu.edu.vn/doi-song-sinh-vien/":
        decision: listing
        note: "Trang tổng hợp; chỉ duyệt link"
      "https://uet.vnu.edu.vn/en/author/root/":
        decision: rejected
        note: "Archive tác giả"
```

- accepted: cho phép tải/extract HTML.
- listing: duyệt link, không đưa chính trang vào corpus.
- rejected: không request URL đó, không giữ processed hiện hành.
- deferred: tạm hoãn request/extract, giữ nguồn để xem lại.
- Xóa entry: trở lại policy tự động. Không chỉnh tay DataSource JSONL.

```bash
python3 -m src.ingestion.main ingest --from web --to processed --sites UET --stream-extract-workers 8
python3 -m src.ingestion.main ingest --from data_source --to processed --sites UET --stream-extract-workers 8
python3 -m src.ingestion.main ingest --from raw --to processed --sites UET
# File tùy chọn:
python3 -m src.ingestion.main ingest --from web --to processed --sites UET --overrides config/my-overrides.yaml
```

YAML được đọc và validate một lần trước network, snapshot dùng chung toàn lượt.
Không hot reload: sửa YAML khi đang chạy chỉ có hiệu lực khi chạy lại. Dừng lượt
hiện tại trước khi chạy lượt mới; giữ cơ chế writer.lock hiện có.

Reconcile trước streaming: thêm URL khai báo nhưng chưa được khám phá; áp dụng
quyết định lên nguồn đã có; loại processed khi không còn accepted. Raw lịch sử giữ
nguyên. Chuyển sang accepted/listing được đưa lại vào checkpoint dù trước đó complete.
Checkpoint replay intent ghi trước khi đổi quyết định. Với web, URL vừa duyệt lại
có thể được GET lại để khám phá link; data_source→processed tái sử dụng raw hợp lệ.
raw→processed chỉ xử lý raw đã có, không tự tải URL mới. Các giới hạn CLI vẫn áp dụng.

Metadata bổ sung nằm trong metadata: automatic_decision, override_note;
decision_reason=manual_override. Xóa entry sẽ xóa thông tin override khỏi nguồn.
Không sửa processing_key của tất cả corpus chỉ vì một ghi chú/rule thay đổi.

Streaming vẫn dùng coordinator làm writer; source không còn accepted bị loại trước
submit và trước publish bằng các guard hiện có trong streaming.py. Snapshot cố định
tránh các worker nhìn policy khác nhau. Không cần thay streaming.py.

accepted/listing có thể vượt các rule loại path trong cùng host, nhưng không tự mở
host bên ngoài. Host ngoài profile hoặc đuôi file không phải HTML được yêu cầu dùng
deferred/chỉnh profile thích hợp. Không thêm downloader Google Docs/PDF/Office.
Rule là exact URL sau normalize; nếu redirect sang URL khác, đích phải qua policy
của nó. Muốn duyệt/chặn cả alias và đích thì khai báo cả hai URL tương ứng.

Không tự đồng bộ/xóa chunk, embedding, Chroma hay BM25 đã tạo. Cần chạy quy trình
đồng bộ downstream sau thay đổi corpus. Bản này không sửa lỗi query legacy đã nêu.

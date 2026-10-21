Load DataChunk
      ↓
Validate Input
      ↓
Đọc ChromaDB (Read-only)
      ↓
So sánh chunk_id + text
      ↓
    Khớp?
   /     \
 Không    Có
   ↓       ↓
 Báo lỗi  Build BM25Plus
           ↓
       Kiểm tra lại Dense
           ↓
       Backup BM25 cũ
           ↓
       Atomic Save
           ↓
   database/bm25_index.pkl
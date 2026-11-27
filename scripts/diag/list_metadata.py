"""
scripts/diag/list_metadata.py - Liệt kê phân bố metadata (faculty, title) trong cơ sở dữ liệu Vector DB (Phase 0).
Giúp kiểm chứng thực tế tài liệu điểm chuẩn / tuyển sinh mang metadata gì.
"""

import os
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.helpers import load_yaml_config


def main():
    cfg = load_yaml_config()
    db_dir = BASE_DIR / cfg.get("paths", {}).get("vector_db_dir", "database/vector_db")
    col_name = cfg.get("paths", {}).get("collection_name", "chatbot_collection")

    print(f"=== PHÂN TÍCH METADATA VECTOR DB ===")
    print(f"Vector DB path: {db_dir}")
    print(f"Collection: {col_name}")

    if not db_dir.exists():
        print(f"Thư mục Vector DB không tồn tại: {db_dir}")
        return

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(db_dir))
        col = client.get_collection(col_name)
        count = col.count()
        print(f"Tổng số chunks trong Collection: {count}\n")

        # Lấy metadata mẫu
        data = col.get(limit=min(count, 500), include=["metadatas"])
        metas = data.get("metadatas", [])

        faculty_counts = Counter()
        admission_titles = []

        for m in metas:
            if not m:
                continue
            fac = m.get("faculty", "UNKNOWN")
            faculty_counts[fac] += 1
            title = str(m.get("title", ""))
            if any(kw in title.lower() for kw in ["tuyển sinh", "điểm chuẩn", "trúng tuyển", "tuyensinh"]):
                admission_titles.append((title, fac))

        print("--- Phân bố theo trường 'faculty' ---")
        for fac, cnt in faculty_counts.most_common():
            print(f"  * {fac}: {cnt} chunks")

        print("\n--- Tài liệu Tuyển sinh / Điểm chuẩn tìm thấy trong mẫu ---")
        if admission_titles:
            for title, fac in set(admission_titles)[:15]:
                print(f"  * [{fac}] {title}")
        else:
            print("  (Không tìm thấy tài liệu tuyển sinh rõ ràng trong mẫu kiểm tra)")

    except Exception as e:
        print(f"Lỗi khi đọc Chroma DB: {e}")


if __name__ == "__main__":
    main()

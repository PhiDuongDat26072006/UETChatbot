"""src/indexing/indexer.py - Phụ trách quy trình lập chỉ mục dữ liệu (Indexing Engine).
Nhiệm vụ:
  1. Đọc dữ liệu các đoạn văn bản (DataChunk) từ thư mục `data/chunk_data/` (*_chunks.jsonl).
  2. Tạo vector nhúng ngữ nghĩa (Embedding) qua UETEmbedder (all-MiniLM-L6-v2).
  3. Lập chỉ mục và lưu trữ bền vững vào ChromaDB (vector_db/chroma.sqlite3).
  4. Hỗ trợ batch processing, lọc theo domain, chạy thử mẫu (sample) và xóa làm mới chỉ mục.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Đảm bảo root directory có trong sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import BaseEmbeddingModel, BaseVectorStore, DataChunk
from src.embeddings.embedder import UETEmbedder
from src.vectordb.vector_store import UETVectorStore
from src.utils.helpers import get_logger, Timer

logger = get_logger("indexing")


def find_chunk_files(base_dir: Path | str) -> List[Path]:
    """Tìm kiếm các tệp chunk (*.jsonl) trong thư mục chỉ định."""
    path = Path(base_dir)
    if not path.exists():
        logger.warning(f"Thư mục không tồn tại: {path}")
        return []

    return sorted(path.glob("*.jsonl"))


def load_chunks_from_dir(
    data_dir: Path | str,
    domain: Optional[str] = None,
    max_chunks: Optional[int] = None,
) -> List[DataChunk]:
    """
    Đọc và phân tích các tệp JSONL thành danh sách đối tượng DataChunk chuẩn hóa.

    :param data_dir: Đường dẫn thư mục chứa dữ liệu chunk (ví dụ data/chunk_data)
    :param domain: Bộ lọc domain cụ thể (ví dụ: fit.uet.vnu.edu.vn, uet.edu.vn...)
    :param max_chunks: Giới hạn tối đa số chunk cần nạp (tiện lợi cho việc test nhanh/sample)
    :return: Danh sách các DataChunk
    """
    path = Path(data_dir)
    jsonl_files = find_chunk_files(path)

    if not jsonl_files:
        logger.warning(f"Không tìm thấy file .jsonl nào trong: {path}")
        return []

    chunks: List[DataChunk] = []
    seen_ids: set[str] = set()
    skipped_dup = 0
    skipped_bad = 0

    for jf in jsonl_files:
        # Nếu chỉ định lọc domain, kiểm tra theo tên tệp
        if domain and domain.lower() not in jf.name.lower():
            continue

        file_chunks_count = 0
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_bad += 1
                        continue

                    cid = str(item.get("chunk_id") or f"{jf.stem}_{line_idx}")
                    if cid in seen_ids:
                        skipped_dup += 1
                        continue

                    text = item.get("text", "").strip()
                    if not text:
                        continue

                    metadata = dict(item.get("metadata") or {})
                    # Bổ sung thông tin nguồn tệp nếu chưa có
                    if "file_name" not in metadata:
                        metadata["file_name"] = jf.name

                    chunk = DataChunk(
                        chunk_id=cid,
                        document_id=item.get("document_id"),
                        text=text,
                        chunk_index=int(item.get("chunk_index", 0)),
                        metadata=metadata,
                    )

                    seen_ids.add(cid)
                    chunks.append(chunk)
                    file_chunks_count += 1

                    if max_chunks and len(chunks) >= max_chunks:
                        logger.info(f"Đã đạt giới hạn max_chunks={max_chunks}. Dừng nạp thêm.")
                        return chunks

            logger.info(f"  • Đã đọc tệp [{jf.name}]: {file_chunks_count} chunks hợp lệ.")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {jf}: {e}")

    logger.info(
        f"Tổng cộng nạp được: {len(chunks)} chunks từ {len(jsonl_files)} tệp "
        f"(bỏ qua {skipped_dup} trùng lặp, {skipped_bad} lỗi dòng)."
    )
    return chunks


class UETIndexer:
    """
    Bộ quản lý và thực thi quy trình Lập chỉ mục (Indexing Engine) cho UET Chatbot.
    Điều phối luồng: DataChunk -> UETEmbedder -> ChromaDB Vector Store.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        persist_dir: Optional[Path | str] = None,
        collection_name: str = "uet_knowledge_base",
    ):
        self.persist_dir = Path(persist_dir) if persist_dir else (BASE_DIR / "vector_db")
        self.collection_name = collection_name
        self.embedding_model = embedding_model or UETEmbedder()
        self.vector_store = vector_store or UETVectorStore(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )

    def index_chunks(
        self,
        chunks: List[DataChunk],
        batch_size: int = 128,
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Lập chỉ mục và lưu danh sách chunks vào Vector Store theo từng batch.

        :param chunks: Danh sách DataChunk cần lập chỉ mục
        :param batch_size: Kích thước batch nhúng và lưu vào database (mặc định 128)
        :param clear_existing: Có xóa toàn bộ dữ liệu cũ trong collection không?
        :return: Thống kê kết quả lập chỉ mục
        """
        if not chunks:
            logger.warning("Danh sách chunks rỗng. Không có dữ liệu để lập chỉ mục.")
            return {
                "status": "empty",
                "indexed_chunks": 0,
                "total_in_db": self.vector_store.count(),
                "time_seconds": 0.0,
            }

        if clear_existing:
            logger.info(f"Đang làm sạch toàn bộ dữ liệu cũ trong collection '{self.collection_name}'...")
            self.vector_store.clear()

        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        indexed_count = 0

        logger.info(
            f"Bắt đầu lập chỉ mục {total_chunks} chunks (chia làm {total_batches} batches, "
            f"mỗi batch {batch_size} chunks)..."
        )

        with Timer() as timer:
            for b_idx in range(total_batches):
                start = b_idx * batch_size
                end = min(start + batch_size, total_chunks)
                batch = chunks[start:end]

                b_start_time = time.time()
                # 1. Nhúng vector cho batch
                vectors = self.embedding_model.embed_chunks(batch)

                # 2. Lưu batch vào Vector Store
                saved = self.vector_store.store(chunks=batch, vectors=vectors)
                indexed_count += saved
                b_duration = max(0.001, time.time() - b_start_time)
                speed = len(batch) / b_duration

                progress_pct = (indexed_count / total_chunks) * 100
                logger.info(
                    f"  [Batch {b_idx + 1}/{total_batches}] "
                    f"Đã lập chỉ mục {indexed_count}/{total_chunks} ({progress_pct:.1f}%) "
                    f"| Tốc độ: {speed:.1f} chunks/s"
                )

        elapsed = round(timer.elapsed, 2)
        final_total = self.vector_store.count()

        logger.info(
            f"✔ Hoàn tất lập chỉ mục: {indexed_count} chunks trong {elapsed}s. "
            f"Tổng số vectors hiện có trong DB: {final_total}."
        )

        return {
            "status": "success",
            "indexed_chunks": indexed_count,
            "total_in_db": final_total,
            "time_seconds": elapsed,
            "average_speed": round(indexed_count / max(0.1, elapsed), 1),
        }

    def index_from_dir(
        self,
        data_dir: Path | str = "data/chunk_data",
        batch_size: int = 128,
        domain: Optional[str] = None,
        max_chunks: Optional[int] = None,
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Quy trình trọn gói: Đọc toàn bộ thư mục dữ liệu chunk và lập chỉ mục.
        """
        resolved_path = Path(data_dir)
        if not resolved_path.is_absolute():
            resolved_path = BASE_DIR / resolved_path

        logger.info(f"Đang chuẩn bị nạp dữ liệu từ thư mục: {resolved_path}")
        chunks = load_chunks_from_dir(
            data_dir=resolved_path,
            domain=domain,
            max_chunks=max_chunks,
        )

        return self.index_chunks(
            chunks=chunks,
            batch_size=batch_size,
            clear_existing=clear_existing,
        )

    def get_status(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái hiện tại của cơ sở dữ liệu Vector."""
        count = self.vector_store.count()
        return {
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name,
            "total_vectors": count,
            "is_ready_for_demo": count > 0,
        }


def main():
    """Điểm chạy dòng lệnh (CLI Entry Point) cho module Indexing."""
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="UET Chatbot - Công cụ lập chỉ mục dữ liệu (Indexing Engine)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/chunk_data",
        help="Đường dẫn tới thư mục chứa tệp chunk (mặc định: data/chunk_data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Số lượng chunks xử lý trong mỗi batch (mặc định: 128)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Chỉ lập chỉ mục N chunks đầu tiên để chạy thử nghiệm nhanh (ví dụ: --sample 500)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Chỉ lập chỉ mục các chunk thuộc domain chỉ định (ví dụ: --domain fit.uet.vnu.edu.vn)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Xóa sạch dữ liệu cũ trong Vector Store trước khi lập chỉ mục lại",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Xem thống kê số lượng vectors hiện có trong database",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("🚀 UET CHATBOT - HỆ THỐNG LẬP CHỈ MỤC DỮ LIỆU (INDEXING ENGINE)")
    print("=" * 65)

    indexer = UETIndexer()

    if args.status:
        st = indexer.get_status()
        print(f"  • Thư mục lưu trữ  : {st['persist_dir']}")
        print(f"  • Tên Collection   : {st['collection_name']}")
        print(f"  • Tổng số vectors  : {st['total_vectors']}")
        print(f"  • Trạng thái Demo  : {'SẴN SÀNG DEMO ✅' if st['is_ready_for_demo'] else 'CHƯA CÓ DỮ LIỆU ⚠️'}")
        print("=" * 65 + "\n")
        return

    res = indexer.index_from_dir(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        domain=args.domain,
        max_chunks=args.sample,
        clear_existing=args.clear,
    )

    print("\n" + "=" * 65)
    print("📊 BÁO CÁO KẾT QUẢ LẬP CHỈ MỤC (INDEXING REPORT)")
    print("=" * 65)
    print(f"  • Trạng thái          : {res['status'].upper()}")
    print(f"  • Số chunks đã nạp    : {res.get('indexed_chunks', 0)}")
    print(f"  • Tổng số trong DB    : {res.get('total_in_db', 0)}")
    print(f"  • Thời gian thực thi  : {res.get('time_seconds', 0)}s")
    if "average_speed" in res:
        print(f"  • Tốc độ trung bình   : {res['average_speed']} chunks/s")
    print("=" * 65)
    print("🎉 Bây giờ bạn có thể chạy 'python main.py --web' hoặc 'python main.py --cli' để demo ngay!\n")


if __name__ == "__main__":
    main()

"""src/indexing - Phân hệ Lập chỉ mục dữ liệu (Indexing Module) cho UET Chatbot.

Cung cấp các công cụ đọc DataChunk từ thư mục `data/chunk_data`,
nhúng vector và lưu trữ vào Vector Database (ChromaDB) phục vụ cho RAG Pipeline.
"""

from src.indexing.indexer import UETIndexer, load_chunks_from_dir

__all__ = ["UETIndexer", "load_chunks_from_dir"]

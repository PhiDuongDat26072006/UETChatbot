"""
tests/test_app.py - Kịch bản kiểm thử toàn diện (Unit & Integration Tests) cho dự án.
Cả nhóm có thể chạy file này để kiểm tra tính toàn vẹn của kiến trúc và các module.
"""

import sys
import unittest
from pathlib import Path

# Cấu hình UTF-8 cho Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn gốc vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from base import (
    DataSource,
    RawData,
    ProcessedData,
    DataChunk,
    UserQuery,
    RetrievedContext,
    AugmentedPrompt,
    Response,
)
from src.utils.helpers import load_yaml_config, get_logger, Timer


class TestUETChatbotArchitecture(unittest.TestCase):
    """Kiểm tra các thành phần cốt lõi của kiến trúc."""

    def test_data_models_validation(self):
        """1. Kiểm tra tính hợp lệ của các mô hình dữ liệu Pydantic."""
        raw = RawData(
            source_uri="https://uet.vnu.edu.vn",
            content="Quy chế đào tạo",
            title="Quy chế UET",
        )
        self.assertIsNotNone(raw.id)
        self.assertEqual(raw.title, "Quy chế UET")

        chunk = DataChunk(
            text="Nội dung điều 1...",
            chunk_index=0,
            metadata={"source": "quy_che.pdf"},
        )
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.metadata["source"], "quy_che.pdf")

        query = UserQuery(query_text="Học bổng UET?")
        self.assertEqual(query.query_text, "Học bổng UET?")

    def test_utils_config_and_logger(self):
        """2. Kiểm tra bộ tiện ích helpers (config, logger, timer)."""
        logger = get_logger("test_logger")
        self.assertIsNotNone(logger)

        config = load_yaml_config()
        self.assertIsInstance(config, dict)

        with Timer() as t:
            _ = sum(i for i in range(1000))
        self.assertGreaterEqual(t.elapsed, 0.0)

    def test_modules_importability(self):
        """3. Kiểm tra khả năng import của tất cả các package trong src/."""
        from src.ingestion import UETDataLoader
        from src.chunking import UETChunker
        from src.embeddings import UETEmbedder
        from src.vectordb import UETVectorStore
        from src.retrieval import UETRetriever
        from src.prompts import UETPromptAugmenter
        from src.llm import UETLLMClient
        from src.evaluation import UETEvaluator
        from src.interfaces import create_app, run_cli

        self.assertIsNotNone(UETDataLoader)
        self.assertIsNotNone(UETChunker)
        self.assertIsNotNone(UETEmbedder)
        self.assertIsNotNone(UETVectorStore)
        self.assertIsNotNone(UETRetriever)
        self.assertIsNotNone(UETPromptAugmenter)
        self.assertIsNotNone(UETLLMClient)
        self.assertIsNotNone(UETEvaluator)
        self.assertIsNotNone(create_app)
        self.assertIsNotNone(run_cli)


if __name__ == "__main__":
    unittest.main()

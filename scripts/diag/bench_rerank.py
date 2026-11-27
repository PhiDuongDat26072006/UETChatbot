"""
scripts/diag/bench_rerank.py - Đo thời gian thực thi và bộ nhớ của Reranker (Phase 0).
Khảo sát batch_size, max_length và số ứng viên để tối ưu hóa hiệu năng CPU/GPU laptop.
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import RetrievedContext
from src.retrieval.reranker import JinaONNXReranker


def get_process_memory_mb():
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def make_dummy_candidates(num_candidates: int = 30) -> list:
    sample_texts = [
        "Trường Đại học Công nghệ thông báo điểm trúng tuyển đại học chính quy các ngành năm 2026.",
        "Ngành Công nghệ thông tin có điểm chuẩn 28.15 điểm, tiêu chí phụ môn Toán xét từ 9.0 trở lên.",
        "Ngành Trí tuệ nhân tạo có điểm chuẩn 27.85 điểm, thời gian đào tạo 4 năm với 134 tín chỉ.",
        "Quy chế đào tạo quy định sinh viên có điểm trung bình chung tích lũy từ 3.20 được xét loại giỏi.",
        "Học phí chương trình chất lượng cao năm học 2025-2026 áp dụng theo định mức kinh tế kỹ thuật.",
    ]
    candidates = []
    for i in range(num_candidates):
        txt = sample_texts[i % len(sample_texts)] + f" (Đoạn trích số {i+1})"
        candidates.append(
            RetrievedContext(
                chunk_id=f"chunk_{i+1:03d}",
                text=txt,
                similarity_score=round(0.9 - i * 0.01, 3),
                rank=i + 1,
                metadata={"title": f"Tài liệu quy chế {i+1}"},
            )
        )
    return candidates


def main():
    print("=== BENCHMARK JINA ONNX RERANKER ===")
    start_mem = get_process_memory_mb()
    print(f"RAM ban đầu: {start_mem:.2f} MB")

    query = "So sánh điểm chuẩn ngành trí tuệ nhân tạo và công nghệ thông tin"
    candidates = make_dummy_candidates(30)

    for b_size in [4, 8, 16]:
        for max_len in [384, 512]:
            print(f"\n--- Thử nghiệm: batch_size={b_size}, max_length={max_len} ---")
            try:
                reranker = JinaONNXReranker(max_length=max_len, batch_size=b_size)
                t0 = time.perf_counter()
                results = reranker.rerank(query, candidates, top_n=10)
                elapsed = time.perf_counter() - t0
                curr_mem = get_process_memory_mb()
                print(f"  -> Hoàn thành trong: {elapsed:.3f}s | RAM: {curr_mem:.2f} MB (+{curr_mem - start_mem:.2f} MB)")
                print(f"  -> Top 1: {results[0].chunk_id} (score={results[0].similarity_score})")
            except Exception as e:
                print(f"  -> Thất bại hoặc OOM: {e}")


if __name__ == "__main__":
    main()

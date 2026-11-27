"""
scripts/diag/trace_one.py - Chạy một câu hỏi qua Graph và in bảng phân tích thời gian chi tiết từng Node (Phase 0, 6).
Thay thế cho việc phải đọc log thủ công.
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.graph import UETAgentGraph
from src.pipeline import get_active_pipeline, get_rag_pipeline


def main():
    parser = argparse.ArgumentParser(description="Chạy một câu hỏi qua Graph và in bảng thời gian từng node.")
    parser.add_argument(
        "--query",
        "-q",
        default="So sánh điểm chuẩn ngành trí tuệ nhân tạo và công nghệ thông tin",
        help="Nội dung câu hỏi cần kiểm tra",
    )
    args = parser.parse_args()

    print(f"=== TRACE ONE QUERY TRÊN UET AGENT GRAPH ===")
    print(f"Câu hỏi: '{args.query}'\n")

    pipeline = get_active_pipeline() or get_rag_pipeline()
    graph = pipeline.agent_graph if hasattr(pipeline, "agent_graph") else UETAgentGraph(retriever=pipeline.retriever, llm_client=pipeline.llm_client)

    t0 = time.perf_counter()
    state = graph.run(args.query)
    total_time = round(time.perf_counter() - t0, 3)

    resp = state.get("response")
    trace = state.get("trace", [])
    steps = state.get("execution_steps", [])

    print(f"{'Node / Bước':<25} | {'Quyết định / Kết quả':<30} | {'Thời gian (s)':<12} | {'Nguồn':<8}")
    print("-" * 85)

    for entry in trace:
        node = entry.get("node", "unknown")
        dec = entry.get("decision", "")
        res = str(entry.get("result", entry.get("error", "")))
        lat = entry.get("latency_s", 0.0)
        src = entry.get("source", "local")
        dec_str = f"{dec}: {res}" if dec else res
        print(f"{node:<25} | {dec_str[:28]:<30} | {lat:<12.3f} | {src:<8}")

    print("-" * 85)
    print(f"Tổng thời gian toàn bộ luồng: {total_time:.3f} s")
    print(f"Bộ đếm: retried1={state.get('retried1', 0)}, retried2={state.get('retried2', 0)}, retried3={state.get('retried3', 0)}")
    print(f"Trạng thái cuối: Status={getattr(resp, 'status', 'N/A')}, State={getattr(resp, 'state', 'N/A')}")
    if resp and getattr(resp, "fail_reason", None):
        print(f"Lý do thất bại (fail_reason): {resp.fail_reason}")

    print(f"\n--- Câu trả lời sinh viên nhận được ---")
    print(state.get("answer", "(Không có câu trả lời)"))


if __name__ == "__main__":
    main()

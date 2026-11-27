"""
scripts/run_golden_eval.py - Đánh giá bộ câu hỏi vàng trên kiến trúc JEV Agentic RAG v3.
"""

import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.graph import UETAgentGraph
from src.decision import JevDecisionClient, FakeDecisionClient
from src.utils.helpers import load_yaml_config

_LEGACY_EXPECTED_MAP = {
    "SINGLE_HOP": "QUALIFIED_QUESTION",
    "MULTI_HOP": "QUALIFIED_QUESTION",
    "AMBIGUOUS": "QUALIFIED_QUESTION",
}


def main():
    golden_path = BASE_DIR / "data" / "golden_questions.json"
    if not golden_path.exists():
        print(f"[ERROR] Không tìm thấy tệp {golden_path}")
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("=" * 70)
    print(f"📊 ĐÁNH GIÁ BỘ CÂU HỎI VÀNG ({len(questions)} CÂU) TRÊN KIẾN TRÚC JEV RAG v3")
    print("=" * 70)

    cfg = load_yaml_config()
    api_key = os.getenv("JEV_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    endpoint = os.getenv("JEV_ENDPOINT") or cfg.get("jev", {}).get("endpoint")

    if api_key or (endpoint and "openrouter.ai" not in endpoint):
        print(f"[*] Sử dụng JEV Decision Client thật ({endpoint})")
        client = JevDecisionClient(
            endpoint=endpoint,
            api_key=api_key,
            model=cfg.get("jev", {}).get("model", "strands-decider-12B-gemma4-v1-2610"),
            timeout_s=float(cfg.get("jev", {}).get("timeout_s", 30.0)),
        )
    else:
        print("[*] Không có API key -> Sử dụng FakeDecisionClient kịch bản giả lập")
        client = FakeDecisionClient()

    graph = UETAgentGraph(decision_client=client, config=cfg)

    results = []
    route_counts = {}
    latencies = []
    state_counts = {0: 0, 1: 0}

    for i, item in enumerate(questions, 1):
        qid = item["id"]
        q_text = item["query"]
        raw_expected = item.get("expected_route")
        expected_route = _LEGACY_EXPECTED_MAP.get(raw_expected, raw_expected)

        start_t = time.time()
        try:
            state = graph.run(query=q_text, session_id=f"eval_{qid}")
            latency = round(time.time() - start_t, 3)
            latencies.append(latency)

            resp = state.get("response")
            actual_route = state.get("route").value if state.get("route") else "UNKNOWN"
            state_val = resp.state if resp else 0
            status_val = resp.status if resp else "unknown"

            route_counts[actual_route] = route_counts.get(actual_route, 0) + 1
            state_counts[state_val] = state_counts.get(state_val, 0) + 1

            is_match = (actual_route == expected_route) if expected_route else True
            mark = "✅" if is_match else "⚠️"

            print(
                f"[{i:02d}/{len(questions):02d}] {qid} | Route: {actual_route:<18} "
                f"(Expected: {expected_route or 'N/A':<18}) {mark} | Retry: {state.get('retry', 0)} | "
                f"Latency: {latency:.2f}s | Status: {status_val}"
            )

            results.append({
                "id": qid,
                "query": q_text,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "route_match": is_match,
                "state": state_val,
                "status": status_val,
                "retry": state.get("retry", 0),
                "latency_s": latency,
            })
        except Exception as e:
            print(f"[{i:02d}/{len(questions):02d}] {qid} | LỖI: {e}")

    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0
    route_matches = sum(1 for r in results if r.get("route_match"))

    print("\n" + "=" * 70)
    print("📈 TỔNG KẾT ĐÁNH GIÁ v3:")
    print(f"- Tổng số câu hỏi: {len(results)}")
    if results:
        print(f"- Độ khớp nhãn Router dự kiến: {route_matches}/{len(results)} ({route_matches/len(results)*100:.1f}%)")
    print(f"- Phân phối Route: {route_counts}")
    print(f"- Trạng thái phản hồi (state): {state_counts}")
    print(f"- Độ trễ trung bình: {avg_latency}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

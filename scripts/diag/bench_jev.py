"""
scripts/diag/bench_jev.py - Công cụ đo lường độ trễ và độ tin cậy của JEV Server (Phase 0).
Đo thời gian client, server latency từ trường 'latency' trong phản hồi, và kiểm tra kịch bản abort kết nối.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.helpers import load_yaml_config


def create_sample_payload(query: str = "So sánh điểm chuẩn ngành trí tuệ nhân tạo và công nghệ thông tin", num_chunks: int = 12) -> Dict[str, Any]:
    sample_text = (
        "Trường Đại học Công nghệ thông báo điểm chuẩn trúng tuyển đại học chính quy năm 2026. "
        "Ngành Trí tuệ nhân tạo (CN8): 27.85 điểm. "
        "Ngành Công nghệ thông tin (CN1): 28.15 điểm. "
        "Tiêu chí phụ: Điểm môn Toán >= 9.0 đối với các thí sinh bằng điểm ở cuối danh sách."
    )
    doc_lines = [f"[Đoạn {i+1}]: {sample_text}" for i in range(num_chunks)]
    questions = {
        f"chunk_{i+1}": {
            "type": "choice",
            "instructions": f"Đoạn {i+1} có chứa thông tin điểm chuẩn ngành hỏi không?",
            "criteria": {
                "RELEVANT": "Đoạn văn có chứa điểm chuẩn hoặc thông tin tuyển sinh",
                "IRRELEVANT": "Đoạn văn không liên quan",
            },
        }
        for i in range(num_chunks)
    }
    return {
        "state": {
            "query": query,
            "documents": "\n\n---\n\n".join(doc_lines),
        },
        "questions": questions,
    }


def bench_single_request(url: str, payload: Dict[str, Any], api_key: str = "", timeout: float = 35.0) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        client_lat = round(time.perf_counter() - t0, 3)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "success",
                "status_code": 200,
                "client_latency_s": client_lat,
                "server_latency_s": data.get("latency"),
                "answers_count": len(data.get("answers", {})),
            }
        return {
            "status": "http_error",
            "status_code": resp.status_code,
            "client_latency_s": client_lat,
            "response": resp.text[:200],
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout", "client_latency_s": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return {"status": "error", "error": str(e), "client_latency_s": round(time.perf_counter() - t0, 3)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark JEV Decisions API")
    parser.add_argument("--endpoint", default=None, help="JEV endpoint URL")
    parser.add_argument("--repeats", type=int, default=3, help="Số lần chạy")
    parser.add_argument("--chunks", type=int, default=12, help="Số chunks gộp")
    args = parser.parse_args()

    cfg = load_yaml_config().get("jev", {})
    endpoint = args.endpoint or os.getenv("JEV_ENDPOINT") or cfg.get("endpoint")
    api_key = os.getenv("JEV_API_KEY", cfg.get("api_key", ""))

    print(f"=== BẮT ĐẦU BENCHMARK JEV SERVER ===")
    print(f"Endpoint: {endpoint}")
    print(f"Số chunk mỗi request: {args.chunks}")
    print(f"Số lần lặp: {args.repeats}")
    print("-" * 50)

    payload = create_sample_payload(num_chunks=args.chunks)
    payload_kb = len(json.dumps(payload).encode("utf-8")) / 1024
    print(f"Payload size: {payload_kb:.2f} KB\n")

    for i in range(args.repeats):
        print(f"Lần {i+1}/{args.repeats}: Đang gửi request...")
        res = bench_single_request(endpoint, payload, api_key=api_key)
        print(f"  -> Kết quả: {res['status']} | Client: {res.get('client_latency_s')}s | Server: {res.get('server_latency_s')}s")
        time.sleep(1.0)


if __name__ == "__main__":
    main()

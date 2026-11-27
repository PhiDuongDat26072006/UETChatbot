"""
Benchmark gọi API OpenRouter LLM và so sánh thông số với Baseline Kaggle Gemma-4-26B (GGUF).

Cách chạy:
    python tests/test_openrouter_benchmark.py
    python tests/test_openrouter_benchmark.py --model "meta-llama/llama-3.3-70b-instruct:free"
    python tests/test_openrouter_benchmark.py --no-stream
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

# =========================================================================
# BASELINE: Kết quả thực tế từ Kaggle Gemma-4-26B-A4B-IT (Q4_0 GGUF)
# =========================================================================
GEMMA_BASELINE = {
    "name": "Gemma-4-26B (Kaggle GGUF)",
    "model": "gemma-4-26B_q4_0-it.gguf",
    "prompt_tokens": 21,
    "cached_tokens": 16,
    "completion_tokens": 339,
    "total_tokens": 360,
    "prompt_ms": 88.383,
    "predicted_ms": 6136.901,
    "total_ms": 88.383 + 6136.901,  # ~6225.284 ms
    "prompt_tps": 56.57196519692701,
    "predicted_tps": 55.0766583981068,
    "ms_per_token": 18.156511834319527,
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROMPT = "Hoàng Sa là của nước nào?"


def load_default_model() -> str:
    """Đọc model_name mặc định từ config.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model = cfg.get("llm", {}).get("model_name")
            if model:
                return model
        except Exception:
            pass
    return "baidu/ernie-5.5-flash-preview:free"


def call_openrouter_stream(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int = 120,
) -> dict:
    """Gọi OpenRouter ở chế độ stream để đo chính xác TTFT, Generation time và tok/s."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "UET Chatbot Benchmark",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first_token = None
    content_chunks = []
    usage = {}
    actual_model = model
    finish_reason = None
    chunk_count = 0

    with requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        status_code = resp.status_code
        if status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {status_code}: {resp.text}")

        resp.encoding = "utf-8"
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data_str = raw_line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if chunk.get("model"):
                actual_model = chunk["model"]
            if chunk.get("usage"):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning") or ""
                if (text or reasoning) and t_first_token is None:
                    t_first_token = time.perf_counter()
                if text:
                    content_chunks.append(text)
                    chunk_count += 1
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0
    ttft_ms = ((t_first_token - t_start) * 1000.0) if t_first_token else total_ms
    gen_ms = max(total_ms - ttft_ms, 1e-6)

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or chunk_count
    total_tokens = usage.get("total_tokens") or (
        (prompt_tokens + completion_tokens) if prompt_tokens is not None else None
    )

    gen_tps = (completion_tokens / (gen_ms / 1000.0)) if completion_tokens else 0.0
    e2e_tps = (completion_tokens / (total_ms / 1000.0)) if completion_tokens else 0.0
    ms_per_token = (gen_ms / completion_tokens) if completion_tokens else 0.0

    return {
        "mode": "stream",
        "status_code": status_code,
        "model": actual_model,
        "finish_reason": finish_reason,
        "answer": "".join(content_chunks),
        "usage": usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": ttft_ms,
        "gen_ms": gen_ms,
        "total_ms": total_ms,
        "gen_tps": gen_tps,
        "e2e_tps": e2e_tps,
        "ms_per_token": ms_per_token,
    }


def call_openrouter_non_stream(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int = 120,
) -> dict:
    """Gọi OpenRouter ở chế độ non-stream (giống test_gemma_host.py)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "UET Chatbot Benchmark",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t_start = time.perf_counter()
    resp = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    t_end = time.perf_counter()

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    total_ms = (t_end - t_start) * 1000.0
    usage = data.get("usage") or {}
    choices = data.get("choices") or [{}]
    answer = choices[0].get("message", {}).get("content", "")
    finish_reason = choices[0].get("finish_reason")

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or 0
    total_tokens = usage.get("total_tokens")
    e2e_tps = (completion_tokens / (total_ms / 1000.0)) if completion_tokens else 0.0
    ms_per_token = (total_ms / completion_tokens) if completion_tokens else 0.0

    return {
        "mode": "non-stream",
        "status_code": resp.status_code,
        "model": data.get("model", model),
        "finish_reason": finish_reason,
        "answer": answer,
        "usage": usage,
        "raw_response": data,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": None,
        "gen_ms": None,
        "total_ms": total_ms,
        "gen_tps": None,
        "e2e_tps": e2e_tps,
        "ms_per_token": ms_per_token,
    }


def fmt_val(val, suffix="", decimals=2):
    if val is None:
        return "N/A"
    if isinstance(val, int):
        return f"{val}{suffix}"
    return f"{val:.{decimals}f}{suffix}"


def fmt_diff(or_val, base_val, lower_is_better=True):
    if or_val is None or base_val is None or base_val == 0:
        return "N/A"
    ratio = or_val / base_val
    pct = (ratio - 1.0) * 100.0
    if lower_is_better:
        status = "nhanh hơn" if or_val < base_val else "chậm hơn"
    else:
        status = "cao hơn" if or_val > base_val else "thấp hơn"
    return f"{pct:+.1f}% ({ratio:.2f}x, {status})"


def print_comparison(res: dict):
    b = GEMMA_BASELINE
    print("\n" + "=" * 92)
    print(f"SO SÁNH HIỆU NĂNG: OpenRouter ({res['model']}) vs {b['name']}")
    print("=" * 92)

    rows = [
        (
            "Prompt tokens",
            fmt_val(res["prompt_tokens"]),
            fmt_val(b["prompt_tokens"]) + f" (cache: {b['cached_tokens']})",
            "-",
        ),
        (
            "Completion tokens",
            fmt_val(res["completion_tokens"]),
            fmt_val(b["completion_tokens"]),
            "-",
        ),
        (
            "Total tokens",
            fmt_val(res["total_tokens"]),
            fmt_val(b["total_tokens"]),
            "-",
        ),
        (
            "TTFT / Prompt time",
            fmt_val(res["ttft_ms"], " ms"),
            fmt_val(b["prompt_ms"], " ms"),
            fmt_diff(res["ttft_ms"], b["prompt_ms"], lower_is_better=True),
        ),
        (
            "Generation time",
            fmt_val(res["gen_ms"], " ms"),
            fmt_val(b["predicted_ms"], " ms"),
            fmt_diff(res["gen_ms"], b["predicted_ms"], lower_is_better=True),
        ),
        (
            "Tổng thời gian (Total)",
            fmt_val(res["total_ms"], " ms") + f" ({res['total_ms']/1000:.2f}s)",
            fmt_val(b["total_ms"], " ms") + f" ({b['total_ms']/1000:.2f}s)",
            fmt_diff(res["total_ms"], b["total_ms"], lower_is_better=True),
        ),
        (
            "Tốc độ sinh (Gen tok/s)",
            fmt_val(res["gen_tps"], " tok/s"),
            fmt_val(b["predicted_tps"], " tok/s"),
            fmt_diff(res["gen_tps"], b["predicted_tps"], lower_is_better=False),
        ),
        (
            "Tốc độ E2E (Total tok/s)",
            fmt_val(res["e2e_tps"], " tok/s"),
            fmt_val(b["completion_tokens"] / (b["total_ms"] / 1000.0), " tok/s"),
            fmt_diff(
                res["e2e_tps"],
                b["completion_tokens"] / (b["total_ms"] / 1000.0),
                lower_is_better=False,
            ),
        ),
        (
            "Độ trễ / token (ms/tok)",
            fmt_val(res["ms_per_token"], " ms"),
            fmt_val(b["ms_per_token"], " ms"),
            fmt_diff(res["ms_per_token"], b["ms_per_token"], lower_is_better=True),
        ),
    ]

    header = f"{'Thông số':<26} | {'OpenRouter':<22} | {'Gemma-4-26B (Kaggle)':<22} | {'Chênh lệch'}"
    print(header)
    print("-" * 92)
    for label, v_or, v_base, diff in rows:
        print(f"{label:<26} | {v_or:<22} | {v_base:<22} | {diff}")
    print("=" * 92)
    print(
        "Lưu ý: Gemma-4-26B timings là server-side (chưa tính network tunnel); "
        "OpenRouter timings là client-side (đã bao gồm network + queue)."
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Test tốc độ gọi API OpenRouter và so sánh với Baseline Gemma-4-26B."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=load_default_model(),
        help="Tên model trên OpenRouter (mặc định đọc từ config.yaml).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Câu hỏi test (mặc định giống test_gemma_host.py).",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Tắt streaming (không đo được TTFT riêng, chỉ đo tổng thời gian).",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Lỗi: Không tìm thấy biến môi trường OPENROUTER_API_KEY (trong .env hoặc system env).")
        sys.exit(1)

    print(f"Model      : {args.model}")
    print(f"Prompt     : {args.prompt}")
    print(f"Mode       : {'non-stream' if args.no_stream else 'stream'}")
    print("Đang gọi OpenRouter API...")

    if args.no_stream:
        res = call_openrouter_non_stream(
            api_key=api_key,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        res = call_openrouter_stream(
            api_key=api_key,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    print(f"\nStatus       : {res['status_code']} (finish_reason={res['finish_reason']})")
    print(f"Usage        : {res['usage']}")
    print("\n--- Answer ---")
    print(res["answer"])
    print("--------------")

    print_comparison(res)


if __name__ == "__main__":
    main()

import json
import time
import requests

BASE_URL = "https://assisted-mill-calvin-orders.trycloudflare.com"

with requests.Session() as session:
    # 1. Kiểm tra kết nối server
    health = session.get(f"{BASE_URL}/health", timeout=10)
    print("Health:", health.status_code, health.text)

    # 2. Gọi thử chat completions ở chế độ streaming ("stream": True)
    t0 = time.perf_counter()
    first_token_time = None

    with session.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "gemma-4",
            "messages": [
                {"role": "user", "content": "Hoàng Sa là của nước nào?"}
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
        },
        stream=True,
        timeout=(10, 120),
    ) as resp:
        print("Status:", resp.status_code)
        resp.raise_for_status()
        resp.encoding = "utf-8"

        print("Answer: ", end="", flush=True)
        for raw_line in resp.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                    print(delta, end="", flush=True)
            except json.JSONDecodeError:
                continue

        total_time = time.perf_counter() - t0
        print(f"\n\nTTFT: {first_token_time:.3f}s | Total: {total_time:.3f}s" if first_token_time else f"\n\nTotal: {total_time:.3f}s")


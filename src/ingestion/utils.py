"""Tiện ích dùng chung (shared utilities) cho toàn bộ pipeline UET Chatbot."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor


def clean_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng và dòng trống trong văn bản.

    - Xóa khoảng trắng thừa ở đầu/cuối mỗi dòng.
    - Giới hạn tối đa 1 dòng trống liên tiếp.
    - Loại bỏ dòng trống dư ở đầu và cuối chuỗi.
    """
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            cleaned_lines.append(line)
            blank_count = 0
        elif blank_count < 1:
            cleaned_lines.append("")
            blank_count += 1
    return "\n".join(cleaned_lines).strip()


def bounded_map(function, items, workers):
    """Submit only one worker-sized batch at a time; never retain all HTML bodies."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(items)
        while True:
            batch = []
            for _ in range(workers):
                item = next(iterator, None)
                if item is None:
                    break
                batch.append(item)
            if not batch:
                return
            yield from pool.map(function, batch)


def completion_map(function, items, workers, on_tick=None):
    """Bounded rolling window, yielding (input, output) in completion order."""
    from concurrent.futures import wait, FIRST_COMPLETED

    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < workers:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[pool.submit(function, item)] = item
                if not pending:
                    break
                done, _ = wait(
                    pending,
                    timeout=0.1 if on_tick else None,
                    return_when=FIRST_COMPLETED,
                )
                if on_tick is not None:
                    on_tick()
                for future in done:
                    item = pending.pop(future)
                    yield item, future.result()
        finally:
            for future in pending:
                future.cancel()

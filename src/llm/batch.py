"""Multi-provider worker queue for batch LLM evaluation."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.base import AugmentedPrompt, Response
from src.utils.helpers import Timer, get_logger

logger = get_logger("llm")

WorkerSpec = Tuple[str, Callable[[str], Tuple[str, str, Dict[str, Any]]], float]


def run_batch_queue(
    prompts: List[AugmentedPrompt],
    *,
    get_cached: Callable[[str], Optional[Tuple[str, str, Dict[str, Any]]]],
    save_cached: Callable[[str, str, str, Dict[str, Any]], None],
    workers_cfg: List[WorkerSpec],
    fallback_generate: Callable[[AugmentedPrompt], Response],
) -> List[Response]:
    """Execute batch prompts across concurrent provider workers while preserving order."""
    total = len(prompts)
    if total == 0:
        return []

    logger.info(f"🚀 Bắt đầu Multi-Provider Batch Queue trên {total} câu hỏi...")
    results: Dict[int, Response] = {}
    task_queue: Queue = Queue()

    cache_hits = 0
    for idx, prompt in enumerate(prompts):
        cached = get_cached(prompt.formatted_prompt)
        if cached:
            cached_ans, cached_model, cached_meta = cached
            results[idx] = Response(
                query_id=prompt.user_query.query_id,
                answer=cached_ans,
                sources=prompt.contexts,
                model_name=f"{cached_model} (cached)",
                latency_seconds=0.005,
                metadata={**cached_meta, "cache_hit": True},
            )
            cache_hits += 1
        else:
            task_queue.put((idx, prompt))

    logger.info(f"📊 Disk Cache Hit: {cache_hits}/{total} câu. Số câu cần gọi API qua Queue: {task_queue.qsize()}")

    if task_queue.empty():
        return [results[i] for i in range(total)]

    if not workers_cfg:
        logger.warning("Không có API provider nào sẵn sàng. Chuyển sang fallback tuần tự.")
        for idx, prompt in enumerate(prompts):
            if idx not in results:
                results[idx] = fallback_generate(prompt)
        return [results[i] for i in range(total)]

    stop_event = threading.Event()

    def worker_loop(worker_name: str, call_fn: Callable[[str], Tuple[str, str, Dict[str, Any]]], delay_sec: float) -> None:
        logger.info(f"🟢 Worker [{worker_name}] đã sẵn sàng (Nhịp nghỉ: {delay_sec}s)...")
        while not stop_event.is_set():
            try:
                item = task_queue.get(timeout=1.0)
            except Empty:
                if task_queue.empty():
                    break
                continue

            idx, prompt = item
            try:
                with Timer() as timer:
                    ans, used_model, meta = call_fn(prompt.formatted_prompt)
                save_cached(prompt.formatted_prompt, ans, used_model, meta)
                results[idx] = Response(
                    query_id=prompt.user_query.query_id,
                    answer=ans,
                    sources=prompt.contexts,
                    model_name=used_model,
                    latency_seconds=round(timer.elapsed, 3),
                    metadata=meta,
                )
                logger.info(f"[{worker_name}] ✅ Đã xong câu #{idx + 1}/{total} ({timer.elapsed:.2f}s). Nghỉ {delay_sec}s...")
                task_queue.task_done()
                time.sleep(delay_sec)
            except Exception as exc:
                err_str = str(exc)
                logger.warning(f"[{worker_name}] ⚠️ Gặp lỗi tại câu #{idx + 1}: {err_str[:100]}... Đẩy lại vào hàng đợi.")
                task_queue.put((idx, prompt))
                task_queue.task_done()
                time.sleep(5.0)

    threads: List[threading.Thread] = []
    for w_name, w_fn, w_delay in workers_cfg:
        thread = threading.Thread(target=worker_loop, args=(w_name, w_fn, w_delay), daemon=True)
        threads.append(thread)
        thread.start()

    task_queue.join()
    stop_event.set()

    for thread in threads:
        thread.join(timeout=2.0)

    logger.info(f"🎉 Hoàn thành xử lý toàn bộ {total} câu hỏi bằng Multi-Provider Queue!")
    return [results[i] for i in range(total)]

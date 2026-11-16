/**
 * src/interfaces/frontend/src/hooks/useStreamBuffer.ts
 * Bộ đệm token streaming sử dụng requestAnimationFrame (~16ms/frame, 60fps)
 * để tránh quá tải React re-render khi nhận SSE token tốc độ cao.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseStreamBufferReturn {
  text: string;
  pushChunk: (chunk: string) => void;
  setImmediateText: (nextText: string) => void;
  flush: () => string;
  reset: () => void;
}

const SINGLE_SHOT_REVEAL_THRESHOLD = 240;

export function useStreamBuffer(): UseStreamBufferReturn {
  const [text, setText] = useState<string>("");

  const committedRef = useRef<string>("");
  const pendingRef = useRef<string>("");
  const revealQueueRef = useRef<string>("");
  const rafIdRef = useRef<number | null>(null);

  const cancelScheduledFrame = useCallback(() => {
    if (rafIdRef.current !== null && typeof cancelAnimationFrame === "function") {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
  }, []);

  const stepFrame = useCallback(() => {
    rafIdRef.current = null;

    if (pendingRef.current.length > 0) {
      committedRef.current += pendingRef.current;
      pendingRef.current = "";
      setText(committedRef.current);
    } else if (revealQueueRef.current.length > 0) {
      const remaining = revealQueueRef.current;
      const chunkSize = Math.max(24, Math.ceil(remaining.length / 18));
      let splitAt = Math.min(chunkSize, remaining.length);
      if (splitAt < remaining.length) {
        const nextSpace = remaining.indexOf(" ", splitAt);
        if (nextSpace !== -1 && nextSpace - splitAt <= 14) {
          splitAt = nextSpace + 1;
        }
      }
      committedRef.current += remaining.slice(0, splitAt);
      revealQueueRef.current = remaining.slice(splitAt);
      setText(committedRef.current);
    }

    if (
      (pendingRef.current.length > 0 || revealQueueRef.current.length > 0) &&
      typeof requestAnimationFrame === "function"
    ) {
      rafIdRef.current = requestAnimationFrame(stepFrame);
    }
  }, []);

  const scheduleFrame = useCallback(() => {
    if (rafIdRef.current !== null) return;
    if (typeof requestAnimationFrame === "function") {
      rafIdRef.current = requestAnimationFrame(stepFrame);
    } else {
      stepFrame();
    }
  }, [stepFrame]);

  const pushChunk = useCallback(
    (chunk: string) => {
      if (!chunk) return;

      // Nếu backend fallback (/api/chat) đẩy cả câu trả lời dài trong 1 lần gọi đầu tiên,
      // đưa vào revealQueue để mở dần ở 60fps mà không chặn UI (và flush() sẽ xả hết ngay khi onDone).
      if (
        committedRef.current.length === 0 &&
        pendingRef.current.length === 0 &&
        revealQueueRef.current.length === 0 &&
        chunk.length > SINGLE_SHOT_REVEAL_THRESHOLD
      ) {
        revealQueueRef.current = chunk;
        scheduleFrame();
        return;
      }

      if (revealQueueRef.current.length > 0) {
        pendingRef.current += revealQueueRef.current + chunk;
        revealQueueRef.current = "";
      } else {
        pendingRef.current += chunk;
      }
      scheduleFrame();
    },
    [scheduleFrame]
  );

  const setImmediateText = useCallback(
    (nextText: string) => {
      cancelScheduledFrame();
      pendingRef.current = "";
      revealQueueRef.current = "";
      committedRef.current = nextText;
      setText(nextText);
    },
    [cancelScheduledFrame]
  );

  const flush = useCallback((): string => {
    cancelScheduledFrame();
    const fullText =
      committedRef.current + pendingRef.current + revealQueueRef.current;
    pendingRef.current = "";
    revealQueueRef.current = "";
    committedRef.current = fullText;
    setText(fullText);
    return fullText;
  }, [cancelScheduledFrame]);

  const reset = useCallback(() => {
    cancelScheduledFrame();
    pendingRef.current = "";
    revealQueueRef.current = "";
    committedRef.current = "";
    setText("");
  }, [cancelScheduledFrame]);

  useEffect(() => {
    return () => {
      cancelScheduledFrame();
    };
  }, [cancelScheduledFrame]);

  return {
    text,
    pushChunk,
    setImmediateText,
    flush,
    reset,
  };
}


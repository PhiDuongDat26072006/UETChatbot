/**
 * src/interfaces/frontend/src/components/chat/StreamingMessage.tsx
 * Component tách biệt chỉ dành cho câu trả lời AI đang stream bên dưới danh sách ảo hóa,
 * giúp toàn bộ lịch sử tin nhắn phía trên không bị re-render ở tốc độ 60fps.
 */

import { motion } from "motion/react";
import type { SourceItem } from "../../types/chat";
import { BubbleTail } from "./MessageBubble";
import { StreamdownRenderer } from "./StreamdownRenderer";

export interface StreamingMessageProps {
  text: string;
  sources: SourceItem[];
  modelName?: string;
  onOpenSource: (source: SourceItem) => void;
  onStop?: () => void;
  onSkip?: () => void;
}

export function StreamingMessage({
  text,
  onStop,
  onSkip,
}: StreamingMessageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 420, damping: 30 }}
      style={{ willChange: "transform, opacity" }}
      className="message-bubble-row bot-row flex w-full items-end justify-start gap-2.5"
      aria-live="polite"
    >
      <div className="bot-avatar-col mb-0.5 w-[34px] shrink-0">
        <div
          className="telegram-avatar bot-avatar-mini flex h-[34px] w-[34px] items-center justify-center overflow-hidden rounded-full border border-[rgba(8,117,201,0.18)] bg-white shadow-[0_2px_6px_rgba(8,117,201,0.2)]"
          title="UET AI Assistant"
        >
          <img
            src="/static/uet-logo.png?v=v14_robot_mascot"
            alt="UET ChatBot Logo"
            className="uet-logo-img"
          />
        </div>
      </div>

      <div className="message-bubble bot-bubble bubble-group-first bubble-group-last relative max-w-[92%] md:max-w-[88%]">
        <div className="mb-1.5 flex items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full bg-[var(--primary-light)] px-2 py-0.5 text-[10px] font-semibold text-[var(--primary)] dark:text-[#38BDF8]">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--primary)] dark:bg-[#38BDF8] animate-pulse" />
            đang trả lời
          </span>
        </div>

        <div className="message-text">
          <StreamdownRenderer content={text} isStreaming={true} />
        </div>

        {(onStop || onSkip) && (
          <div className="message-meta-footer mt-2.5 flex items-center justify-end gap-2 border-t border-[var(--border-divider)] pt-2 text-[11px]">
            {onSkip && (
              <button
                type="button"
                onClick={onSkip}
                className="inline-flex items-center rounded-md border border-[var(--border-subtle)] bg-[var(--bg-search)] px-2.5 py-1 text-[11.5px] font-medium text-[var(--text-secondary)] transition-colors hover:border-[var(--primary)] hover:text-[var(--primary)] dark:hover:text-[#38BDF8]"
                title="Hiển thị toàn bộ nội dung hiện có ngay"
              >
                <span>Hoàn tất ngay</span>
              </button>
            )}
            {onStop && (
              <button
                type="button"
                onClick={onStop}
                className="inline-flex items-center rounded-md border border-red-200 bg-red-50 px-2.5 py-1 text-[11.5px] font-medium text-red-600 transition-colors hover:bg-red-100 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-400 dark:hover:bg-red-950/70"
                title="Dừng tạo phản hồi"
              >
                <span>Dừng</span>
              </button>
            )}
          </div>
        )}

        <BubbleTail sender="bot" />
      </div>
    </motion.div>
  );
}


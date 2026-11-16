/**
 * src/interfaces/frontend/src/components/chat/ThinkingIndicator.tsx
 * Bong bóng chờ phản hồi phong cách Telegram với hiệu ứng 3 chấm nảy
 * và chữ shimmer mượt mà trước khi token đầu tiên xuất hiện.
 */

import { motion } from "motion/react";

export interface ThinkingIndicatorProps {
  label?: string;
}

export function ThinkingIndicator({
  label = "Đang tra cứu quy chế & tổng hợp câu trả lời...",
}: ThinkingIndicatorProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -4, scale: 0.98 }}
      transition={{ type: "spring", stiffness: 420, damping: 30 }}
      style={{ willChange: "transform, opacity" }}
      className="message-bubble-row bot-row flex w-full items-end justify-start gap-2.5"
      role="status"
      aria-live="polite"
      aria-label="UET AI đang suy nghĩ"
    >
      <div className="bot-avatar-col mb-0.5 shrink-0">
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

      <div className="message-bubble bot-bubble bubble-group-first bubble-group-last typing-bubble relative max-w-[82%] px-4 py-3">
        <div className="bot-name-tag mb-1.5 text-[12px] font-bold text-[var(--primary)] dark:text-[#38BDF8]">
          UET AI Assistant
        </div>

        <div className="flex items-center gap-3">
          <div className="telegram-typing-dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <span className="shimmer-text text-[13px] font-medium">{label}</span>
        </div>

        <svg
          className="bubble-tail-svg tail-bot"
          viewBox="0 0 11 18"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M11 0V18H0C4.6 15.5 7.4 10.6 8.6 0H11Z"
            fill="currentColor"
          />
        </svg>
      </div>
    </motion.div>
  );
}


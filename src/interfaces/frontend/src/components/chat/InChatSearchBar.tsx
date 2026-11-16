/**
 * src/interfaces/frontend/src/components/chat/InChatSearchBar.tsx
 * Thanh tìm kiếm từ khóa ngay bên trong cuộc trò chuyện (Ctrl+F),
 * hỗ trợ đếm kết quả khớp, điều hướng Trước/Sau (Enter / Shift+Enter) và đóng nhanh (Esc).
 */

import { useEffect, useRef, type KeyboardEvent } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";

export interface InChatSearchBarProps {
  isOpen: boolean;
  query: string;
  currentMatchIndex: number;
  totalMatches: number;
  onQueryChange: (q: string) => void;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
}

export function InChatSearchBar({
  isOpen,
  query,
  currentMatchIndex,
  totalMatches,
  onQueryChange,
  onPrev,
  onNext,
  onClose,
}: InChatSearchBarProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    const raf = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
    return () => cancelAnimationFrame(raf);
  }, [isOpen]);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (e.shiftKey) {
        onPrev();
      } else {
        onNext();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  const displayCount =
    totalMatches > 0 ? `${currentMatchIndex + 1}/${totalMatches}` : "0/0";

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: -12, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -12, scale: 0.98 }}
          transition={{ type: "spring", stiffness: 460, damping: 32 }}
          style={{ willChange: "transform, opacity" }}
          className="in-chat-search-bar pointer-events-none absolute inset-x-0 top-3 z-[35] px-3 pl-14 md:pl-4"
        >
          <div
            className="in-chat-search-inner pointer-events-auto mx-auto flex max-w-[720px] items-center gap-2 rounded-full border border-[var(--border-subtle)] bg-white/72 px-3.5 py-1.5 shadow-[0_8px_24px_rgba(15,23,42,0.12)] backdrop-blur-md dark:bg-[#181D26]/72 dark:shadow-[0_10px_28px_rgba(0,0,0,0.45)]"
            style={{
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
            }}
          >
            <Search className="h-4 w-4 shrink-0 text-[var(--text-muted)]" />

            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Tìm kiếm từ khóa trong tin nhắn..."
              autoComplete="off"
              aria-label="Tìm kiếm trong tin nhắn"
              className="min-w-0 flex-1 bg-transparent py-1 text-[13.5px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)]"
            />

            <span className="search-count-badge whitespace-nowrap rounded-full bg-black/5 px-2 py-0.5 text-[11.5px] font-bold text-[var(--text-muted)] dark:bg-white/10 dark:text-[var(--text-secondary)]">
              {displayCount}
            </span>

            <button
              type="button"
              onClick={onPrev}
              disabled={totalMatches === 0}
              className="btn-search-nav flex h-7 w-7 items-center justify-center rounded-full text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover-chat)] hover:text-[var(--primary)] disabled:opacity-40"
              title="Kết quả trước (Shift+Enter)"
              aria-label="Kết quả trước"
            >
              <ChevronUp className="h-4 w-4" />
            </button>

            <button
              type="button"
              onClick={onNext}
              disabled={totalMatches === 0}
              className="btn-search-nav flex h-7 w-7 items-center justify-center rounded-full text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover-chat)] hover:text-[var(--primary)] disabled:opacity-40"
              title="Kết quả tiếp theo (Enter)"
              aria-label="Kết quả tiếp theo"
            >
              <ChevronDown className="h-4 w-4" />
            </button>

            <button
              type="button"
              onClick={onClose}
              className="btn-close-in-chat flex h-7 w-7 items-center justify-center rounded-full text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-hover-chat)] hover:text-[var(--text-primary)]"
              title="Đóng tìm kiếm (Esc)"
              aria-label="Đóng tìm kiếm"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}


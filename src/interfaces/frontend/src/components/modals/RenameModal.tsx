import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { X } from "lucide-react";

export interface RenameModalProps {
  isOpen: boolean;
  initialTitle: string;
  onSave: (newTitle: string) => void;
  onClose: () => void;
}

const MODAL_SPRING = { type: "spring", stiffness: 420, damping: 32 } as const;

export function RenameModal({
  isOpen,
  initialTitle,
  onSave,
  onClose,
}: RenameModalProps) {
  const [title, setTitle] = useState(initialTitle);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    setTitle(initialTitle);
    const timer = window.setTimeout(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    }, 80);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, initialTitle, onClose]);

  const handleConfirm = () => {
    const trimmed = title.trim();
    if (trimmed) {
      onSave(trimmed);
    }
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          key="rename-modal-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="telegram-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="rename-modal-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            key="rename-modal-card"
            initial={{ opacity: 0, scale: 0.94, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 8 }}
            transition={MODAL_SPRING}
            className="telegram-modal-card modal-compact gpu-layer"
          >
            <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-5 py-4">
              <h3
                id="rename-modal-title"
                className="text-[16px] font-bold text-[var(--text-primary)]"
              >
                Đổi tên cuộc trò chuyện
              </h3>

              <button
                type="button"
                onClick={onClose}
                className="flex h-8 w-8 items-center justify-center rounded-full text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--text-primary)]"
                title="Đóng"
                aria-label="Đóng"
              >
                <X size={16} />
              </button>
            </div>

            <div className="p-5">
              <div className="flex flex-col gap-2">
                <label
                  htmlFor="input-rename-chat"
                  className="text-[13px] font-semibold text-[var(--text-secondary)]"
                >
                  Tiêu đề cuộc trò chuyện
                </label>
                <input
                  ref={inputRef}
                  id="input-rename-chat"
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleConfirm();
                    }
                  }}
                  placeholder="Nhập tên mới..."
                  maxLength={60}
                  autoComplete="off"
                  className="w-full rounded-[10px] border border-[var(--border-subtle)] bg-[var(--bg-search)] px-3.5 py-2.5 text-[14px] text-[var(--text-primary)] transition-all focus:border-[var(--primary)] focus:bg-[var(--bg-sidebar)] focus:ring-2 focus:ring-[rgba(8,117,201,0.2)]"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5 border-t border-[var(--border-subtle)] px-5 py-3.5">
              <button
                type="button"
                onClick={onClose}
                className="rounded-[10px] border border-[var(--border-subtle)] bg-[var(--bg-search)] px-4.5 py-2 text-[13.5px] font-semibold text-[var(--text-secondary)] transition-colors hover:bg-[var(--border-subtle)] hover:text-[var(--text-primary)]"
              >
                Hủy
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                disabled={!title.trim()}
                className="rounded-[10px] bg-[var(--primary)] px-4.5 py-2 text-[13.5px] font-semibold text-white transition-colors hover:bg-[var(--primary-hover)] disabled:cursor-not-allowed disabled:opacity-50"
              >
                Lưu thay đổi
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}


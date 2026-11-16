import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { X } from "lucide-react";

export interface ConfirmDialogState {
  title: string;
  message: string;
  confirmText?: string;
  isDanger?: boolean;
  onConfirm: () => void;
}

export interface ConfirmModalProps {
  dialog: ConfirmDialogState | null;
  onClose: () => void;
}

const MODAL_SPRING = { type: "spring", stiffness: 420, damping: 32 } as const;

export function ConfirmModal({ dialog, onClose }: ConfirmModalProps) {
  const confirmBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!dialog) return;
    const timer = window.setTimeout(() => {
      confirmBtnRef.current?.focus();
    }, 60);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [dialog, onClose]);

  const isDanger = dialog?.isDanger ?? true;

  return (
    <AnimatePresence>
      {dialog && (
        <motion.div
          key="confirm-modal-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="telegram-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="confirm-modal-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            key="confirm-modal-card"
            initial={{ opacity: 0, scale: 0.94, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 8 }}
            transition={MODAL_SPRING}
            className="telegram-modal-card modal-compact gpu-layer"
          >
            <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-5 py-4">
              <h3
                id="confirm-modal-title"
                className="text-[16px] font-bold text-[var(--text-primary)]"
              >
                {dialog.title || "Xác nhận"}
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
              <p className="text-[14.5px] leading-[1.6] text-[var(--text-primary)]">
                {dialog.message}
              </p>
            </div>

            <div className="flex items-center justify-end gap-2.5 border-t border-[var(--border-subtle)] px-5 py-3.5">
              <button
                type="button"
                onClick={onClose}
                className="rounded-[10px] border border-[var(--border-subtle)] bg-[var(--bg-search)] px-4.5 py-2 text-[13.5px] font-semibold text-[var(--text-secondary)] transition-colors hover:bg-[var(--border-subtle)] hover:text-[var(--text-primary)]"
              >
                Hủy bỏ
              </button>
              <button
                ref={confirmBtnRef}
                type="button"
                onClick={() => {
                  dialog.onConfirm();
                  onClose();
                }}
                className={`rounded-[10px] px-4.5 py-2 text-[13.5px] font-semibold text-white transition-colors ${
                  isDanger
                    ? "bg-[#dc2626] hover:bg-[#b91c1c]"
                    : "bg-[var(--primary)] hover:bg-[var(--primary-hover)]"
                }`}
              >
                {dialog.confirmText || "Tiếp tục"}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}


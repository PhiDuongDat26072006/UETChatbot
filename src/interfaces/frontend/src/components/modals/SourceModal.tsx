import { useEffect } from "react";
import { AnimatePresence, motion } from "motion/react";
import { X } from "lucide-react";
import type { SourceItem } from "../../types/chat";

export interface SourceModalProps {
  source: SourceItem | null;
  onClose: () => void;
}

const MODAL_SPRING = { type: "spring", stiffness: 420, damping: 32 } as const;

export function SourceModal({ source, onClose }: SourceModalProps) {
  useEffect(() => {
    if (!source) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [source, onClose]);

  const meta = source?.metadata ?? {};
  const title =
    typeof meta.title === "string" && meta.title.trim()
      ? meta.title
      : "Văn bản quy chế đào tạo UET";
  const scoreText = source?.similarity_score
    ? `${(source.similarity_score * 100).toFixed(1)}%`
    : "Độ tin cậy cao";

  return (
    <AnimatePresence>
      {source && (
        <motion.div
          key="source-modal-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="telegram-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="source-modal-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            key="source-modal-card"
            initial={{ opacity: 0, scale: 0.94, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 8 }}
            transition={MODAL_SPRING}
            className="telegram-modal-card gpu-layer"
          >
            <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-5 py-4">
              <h3
                id="source-modal-title"
                className="min-w-0 truncate pr-3 text-[16px] font-bold text-[var(--text-primary)]"
              >
                {title}
              </h3>

              <button
                type="button"
                onClick={onClose}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--text-primary)]"
                title="Đóng"
                aria-label="Đóng hộp thoại nguồn trích dẫn"
              >
                <X size={16} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5">
              <div className="mb-3.5 flex flex-wrap gap-2">
                <span className="rounded-md bg-[var(--bg-search)] px-2.5 py-1 text-[11.5px] font-medium text-[var(--text-secondary)]">
                  Độ liên quan: {scoreText}
                </span>
                <span className="rounded-md bg-[var(--bg-search)] px-2.5 py-1 text-[11.5px] font-medium text-[var(--text-secondary)]">
                  Mã chunk: {source.chunk_id || "N/A"}
                </span>
                {meta.page !== undefined && meta.page !== null && String(meta.page).trim() !== "" && (
                  <span className="rounded-md bg-[var(--bg-search)] px-2.5 py-1 text-[11.5px] font-medium text-[var(--text-secondary)]">
                    Trang tham chiếu: {String(meta.page)}
                  </span>
                )}
                {typeof meta.source === "string" && meta.source.trim() !== "" && (
                  <span className="rounded-md bg-[var(--bg-search)] px-2.5 py-1 text-[11.5px] font-medium text-[var(--text-secondary)]">
                    Tài liệu: {meta.source}
                  </span>
                )}
              </div>

              <div className="whitespace-pre-wrap rounded-[10px] border border-[var(--border-subtle)] bg-[var(--bg-search)] p-3.5 text-[14px] leading-[1.7] text-[var(--text-primary)]">
                {source.text || "Không có trích đoạn nội dung."}
              </div>
            </div>

            <div className="flex justify-end border-t border-[var(--border-subtle)] px-5 py-3.5">
              <button
                type="button"
                onClick={onClose}
                className="rounded-[10px] bg-[var(--primary)] px-4.5 py-2 text-[13.5px] font-semibold text-white transition-colors hover:bg-[var(--primary-hover)]"
              >
                Đã hiểu
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}


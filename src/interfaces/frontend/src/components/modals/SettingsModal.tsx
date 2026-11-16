import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Moon, RefreshCw, Sun, Trash2, X } from "lucide-react";
import type { HealthStatus, ThemeMode } from "../../types/chat";

export interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthStatus;
  onRefreshHealth?: () => void | Promise<unknown>;
  theme: ThemeMode;
  onSelectTheme: (theme: ThemeMode) => void;
  soundEnabled: boolean;
  onToggleSound: (enabled: boolean) => void;
  onRequestClearAllChats: () => void;
}

const MODAL_SPRING = { type: "spring", stiffness: 420, damping: 32 } as const;

export function SettingsModal({
  isOpen,
  onClose,
  health,
  onRefreshHealth,
  theme,
  onSelectTheme,
  soundEnabled,
  onToggleSound,
  onRequestClearAllChats,
}: SettingsModalProps) {
  const onRefreshHealthRef = useRef(onRefreshHealth);
  onRefreshHealthRef.current = onRefreshHealth;

  useEffect(() => {
    if (!isOpen) return;
    void onRefreshHealthRef.current?.();
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          key="settings-modal-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="telegram-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="settings-modal-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            key="settings-modal-card"
            initial={{ opacity: 0, scale: 0.94, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 8 }}
            transition={MODAL_SPRING}
            className="telegram-modal-card gpu-layer"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-5 py-4">
              <h3
                id="settings-modal-title"
                className="text-[16px] font-bold text-[var(--text-primary)]"
              >
                Cài đặt &amp; Thông tin hệ thống
              </h3>

              <button
                type="button"
                onClick={onClose}
                className="flex h-8 w-8 items-center justify-center rounded-full text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--text-primary)]"
                title="Đóng"
                aria-label="Đóng cài đặt"
              >
                <X size={16} />
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 space-y-5 overflow-y-auto p-5">
              {/* Section 1: Model & Health */}
              <div className="border-b border-[var(--border-divider)] pb-4">
                <div className="mb-2.5 flex items-center justify-between">
                  <h4 className="text-[13.5px] font-bold text-[var(--text-primary)]">
                    Mô hình AI &amp; Trực tuyến
                  </h4>
                  {onRefreshHealth && (
                    <button
                      type="button"
                      onClick={onRefreshHealth}
                      className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11.5px] font-medium text-[var(--primary)] transition-colors hover:bg-[var(--primary-light)]"
                      title="Kiểm tra lại kết nối"
                    >
                      <RefreshCw size={12} />
                      <span>Làm mới</span>
                    </button>
                  )}
                </div>

                <div className="mb-2 flex items-center justify-between gap-3">
                  <span className="text-[13px] text-[var(--text-secondary)]">
                    Mô hình hiện tại:
                  </span>
                  <span className="rounded-full bg-[var(--bg-search)] px-3 py-1 text-[12px] font-semibold text-[var(--text-primary)]">
                    {health.model_name || "Qwen / Gemini RAG"}
                  </span>
                </div>

                <div className="flex items-center justify-between gap-3">
                  <span className="text-[13px] text-[var(--text-secondary)]">
                    Trạng thái RAG Service:
                  </span>
                  <span
                    className={`rounded-full px-3 py-1 text-[12px] font-semibold ${
                      health.online
                        ? "bg-[#dcfce7] text-[#166534] dark:bg-[#064e3b] dark:text-[#6ee7b7]"
                        : "bg-[#fef9c3] text-[#854d0e] dark:bg-[#451a03] dark:text-[#fcd34d]"
                    }`}
                  >
                    {health.online
                      ? "Đang hoạt động (Connected)"
                      : "Chưa kết nối (Offline)"}
                  </span>
                </div>
              </div>

              {/* Section 2: Appearance & Sound */}
              <div className="border-b border-[var(--border-divider)] pb-4">
                <h4 className="mb-3 text-[13.5px] font-bold text-[var(--text-primary)]">
                  Giao diện &amp; Âm thanh
                </h4>

                <div className="mb-3.5 flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[13px] text-[var(--text-secondary)]">
                    Chế độ hiển thị:
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      type="button"
                      onClick={() => onSelectTheme("light")}
                      className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[12.5px] font-medium transition-colors ${
                        theme === "light"
                          ? "border-[var(--primary)] bg-[var(--primary)] text-white"
                          : "border-[var(--border-subtle)] bg-[var(--bg-search)] text-[var(--text-primary)] hover:border-[var(--primary)] hover:text-[var(--primary)]"
                      }`}
                    >
                      <Sun size={14} />
                      <span>Sáng</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => onSelectTheme("dark")}
                      className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[12.5px] font-medium transition-colors ${
                        theme === "dark"
                          ? "border-[var(--primary)] bg-[var(--primary)] text-white"
                          : "border-[var(--border-subtle)] bg-[var(--bg-search)] text-[var(--text-primary)] hover:border-[var(--primary)] hover:text-[var(--primary)]"
                      }`}
                    >
                      <Moon size={14} />
                      <span>Tối</span>
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-[13px] text-[var(--text-secondary)]">
                      Âm thanh thông báo:
                    </div>
                    <div className="text-[11.5px] text-[var(--text-muted)]">
                      Phát âm thanh khi AI hoàn thành câu trả lời
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2.5">
                    <span className="min-w-[24px] text-right text-[12.5px] font-semibold text-[var(--text-primary)]">
                      {soundEnabled ? "Bật" : "Tắt"}
                    </span>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={soundEnabled}
                      aria-label="Âm thanh thông báo khi AI hoàn thành câu trả lời"
                      onClick={() => onToggleSound(!soundEnabled)}
                      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full p-0.5 transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-[var(--primary)] focus-visible:ring-offset-2 ${
                        soundEnabled
                          ? "bg-[var(--primary)]"
                          : "bg-slate-300 dark:bg-slate-600"
                      }`}
                    >
                      <span
                        className={`inline-block h-5 w-5 transform rounded-full bg-white shadow-sm transition-transform duration-200 ${
                          soundEnabled ? "translate-x-5" : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>
                </div>
              </div>

              {/* Section 3: Local Data */}
              <div>
                <h4 className="mb-2.5 text-[13.5px] font-bold text-[var(--text-primary)]">
                  Dữ liệu cục bộ
                </h4>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[13px] text-[var(--text-secondary)]">
                    Lịch sử trò chuyện:
                  </span>
                  <button
                    type="button"
                    onClick={onRequestClearAllChats}
                    className="inline-flex items-center gap-1.5 rounded-md bg-[var(--bg-search)] px-3.5 py-1.5 text-[12.5px] font-semibold text-[#dc2626] transition-colors hover:bg-[#fee2e2] dark:hover:bg-[#451a1a] dark:hover:text-[#f87171]"
                  >
                    <Trash2 size={14} />
                    <span>Xóa tất cả hội thoại</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="flex justify-end border-t border-[var(--border-subtle)] px-5 py-3.5">
              <button
                type="button"
                onClick={onClose}
                className="rounded-[10px] bg-[var(--primary)] px-4.5 py-2 text-[13.5px] font-semibold text-white transition-colors hover:bg-[var(--primary-hover)]"
              >
                Đóng
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

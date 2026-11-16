import { AnimatePresence, motion } from "motion/react";
import { AlertCircle, AlertTriangle, X } from "lucide-react";
import type { ToastItem } from "../../types/chat";

export interface ToastContainerProps {
  toasts: ToastItem[];
  onDismiss: (id: string) => void;
}

export function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
  const getIcon = (type: ToastItem["type"]) => {
    switch (type) {
      case "error":
        return <AlertCircle size={18} className="shrink-0 text-[#dc2626]" />;
      case "warning":
        return <AlertTriangle size={18} className="shrink-0 text-[#d97706]" />;
      case "success":
      case "info":
      default:
        return null;
    }
  };

  return (
    <div
      className="toast-container"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <AnimatePresence initial={false}>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            layout
            initial={{ opacity: 0, x: 24, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 450, damping: 30 }}
            className={`toast-item toast-${toast.type}`}
          >
            {getIcon(toast.type)}
            <span className="flex-1 text-[13.5px] leading-snug">
              {toast.message}
            </span>
            <button
              type="button"
              onClick={() => onDismiss(toast.id)}
              className="ml-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--text-primary)]"
              aria-label="Đóng thông báo"
            >
              <X size={13} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}


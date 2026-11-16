/**
 * src/interfaces/frontend/src/components/chat/ChatHeader.tsx
 * Thanh Header Glassmorphism phía trên cửa sổ chat, hiển thị logo UET,
 * tên hội thoại, trạng thái trực tuyến/đang trả lời, nút tìm kiếm Ctrl+F, theme & menu 3 chấm.
 */

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  Eraser,
  Menu,
  Moon,
  MoreVertical,
  Pencil,
  Pin,
  PinOff,
  Search,
  Settings,
  Sun,
  Trash2,
} from "lucide-react";
import type { ThemeMode } from "../../types/chat";

export interface ChatHeaderProps {
  title: string;
  isPinned: boolean;
  isGenerating: boolean;
  isOnline: boolean;
  modelName?: string;
  isInChatSearchOpen: boolean;
  onOpenMobileSidebar: () => void;
  onToggleInChatSearch: () => void;
  onTogglePin: () => void;
  onRenameChat: () => void;
  onClearMessages: () => void;
  onDeleteChat: () => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
}

export function ChatHeader({
  title,
  isPinned,
  isGenerating,
  isOnline,
  modelName,
  isInChatSearchOpen,
  onOpenMobileSidebar,
  onToggleInChatSearch,
  onTogglePin,
  onRenameChat,
  onClearMessages,
  onDeleteChat,
  theme,
  onToggleTheme,
  onOpenSettings,
}: ChatHeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;

    const handleOutsideClick = (event: MouseEvent) => {
      if (
        menuContainerRef.current &&
        !menuContainerRef.current.contains(event.target as Node)
      ) {
        setMenuOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [menuOpen]);

  return (
    <header
      className="telegram-header glass-surface sticky top-0 z-20 flex h-16 shrink-0 items-center justify-between gap-3 border-b border-[var(--border-subtle)] bg-[var(--bg-chat-header)] px-4 backdrop-blur-md md:px-5"
      style={{
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
      }}
    >
      <div className="header-left-col flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={onOpenMobileSidebar}
          className="btn-mobile-menu flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--primary)] md:hidden"
          title="Mở danh sách hội thoại"
          aria-label="Mở danh sách hội thoại"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="bot-avatar-header flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-full border border-[rgba(8,117,201,0.2)] bg-white shadow-sm">
          <img
            src="/static/uet-logo.png?v=v14_robot_mascot"
            alt="UET ChatBot Logo"
            className="uet-logo-img"
          />
        </div>

        <div className="chat-header-info flex min-w-0 flex-col">
          <div className="flex items-center gap-1.5">
            <h2 className="chat-header-title truncate text-[15px] font-semibold leading-tight text-[var(--text-primary)] max-w-[180px] sm:max-w-[320px] md:max-w-[460px]">
              {title || "UET AI Assistant"}
            </h2>
            {isPinned && (
              <span
                className="inline-flex shrink-0 items-center rounded-full bg-[var(--primary-light)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--primary)] dark:text-[#38BDF8]"
                title="Cuộc trò chuyện đã ghim"
              >
                <Pin className="h-2.5 w-2.5" />
              </span>
            )}
          </div>

          <div className="chat-header-status flex items-center gap-1.5 text-[12px] text-[var(--text-muted)]">
            {isGenerating ? (
              <span className="typing-indicator-text font-semibold text-[var(--primary)] dark:text-[#38BDF8] animate-pulse">
                đang trả lời...
              </span>
            ) : isOnline ? (
              <>
                <span className="online-indicator" />
                <span>trực tuyến</span>
                {modelName && (
                  <span className="hidden sm:inline-flex items-center rounded-full border border-[var(--border-subtle)] bg-[var(--bg-search)] px-2 py-0.5 text-[10.5px] font-medium text-[var(--text-secondary)]">
                    {modelName}
                  </span>
                )}
              </>
            ) : (
              <>
                <span className="offline-indicator" />
                <span>ngoại tuyến</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="header-right-col flex shrink-0 items-center gap-1">
        <button
          type="button"
          onClick={onToggleInChatSearch}
          className={`btn-header-icon flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
            isInChatSearchOpen
              ? "bg-[var(--primary-light)] text-[var(--primary)] dark:text-[#38BDF8]"
              : "text-[var(--text-muted)] hover:bg-[var(--bg-search)] hover:text-[var(--primary)]"
          }`}
          title="Tìm kiếm trong cuộc trò chuyện (Ctrl+F)"
          aria-label="Tìm kiếm trong cuộc trò chuyện"
          aria-pressed={isInChatSearchOpen}
        >
          <Search className="h-[18px] w-[18px]" />
        </button>

        <button
          type="button"
          onClick={onToggleTheme}
          className="btn-header-icon flex h-9 w-9 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--primary)]"
          title={
            theme === "dark"
              ? "Chuyển sang chế độ Sáng"
              : "Chuyển sang chế độ Tối"
          }
          aria-label="Chuyển chế độ Sáng / Tối"
        >
          {theme === "dark" ? (
            <Sun className="h-[18px] w-[18px]" />
          ) : (
            <Moon className="h-[18px] w-[18px]" />
          )}
        </button>

        <button
          type="button"
          onClick={onOpenSettings}
          className="btn-header-icon hidden h-9 w-9 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--primary)] sm:flex"
          title="Cài đặt hệ thống & mô hình"
          aria-label="Cài đặt hệ thống"
        >
          <Settings className="h-[18px] w-[18px]" />
        </button>

        <div className="header-menu-container relative" ref={menuContainerRef}>
          <button
            type="button"
            onClick={() => setMenuOpen((prev) => !prev)}
            className={`btn-header-icon flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
              menuOpen
                ? "bg-[var(--bg-search)] text-[var(--primary)]"
                : "text-[var(--text-muted)] hover:bg-[var(--bg-search)] hover:text-[var(--primary)]"
            }`}
            title="Tùy chọn cuộc trò chuyện"
            aria-label="Tùy chọn cuộc trò chuyện"
            aria-expanded={menuOpen}
            aria-haspopup="menu"
          >
            <MoreVertical className="h-[18px] w-[18px]" />
          </button>

          <AnimatePresence>
            {menuOpen && (
              <motion.div
                initial={{ opacity: 0, y: -6, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.96 }}
                transition={{ type: "spring", stiffness: 460, damping: 30 }}
                style={{ willChange: "transform, opacity" }}
                role="menu"
                className="header-dropdown-menu absolute right-0 top-[calc(100%+6px)] z-50 min-w-[205px] rounded-xl border border-[var(--border-card)] bg-[var(--bg-card)] p-1.5 shadow-[var(--shadow-dropdown)]"
              >
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    onTogglePin();
                  }}
                  className="header-dropdown-item flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-search)]"
                >
                  {isPinned ? (
                    <PinOff className="h-4 w-4 text-[var(--primary)]" />
                  ) : (
                    <Pin className="h-4 w-4 text-[var(--primary)]" />
                  )}
                  <span>{isPinned ? "Bỏ ghim hội thoại" : "Ghim hội thoại"}</span>
                </button>

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    onRenameChat();
                  }}
                  className="header-dropdown-item flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-search)]"
                >
                  <Pencil className="h-4 w-4 text-[var(--text-secondary)]" />
                  <span>Đổi tên hội thoại</span>
                </button>

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    onOpenSettings();
                  }}
                  className="header-dropdown-item flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-search)] sm:hidden"
                >
                  <Settings className="h-4 w-4 text-[var(--text-secondary)]" />
                  <span>Cài đặt hệ thống</span>
                </button>

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    onClearMessages();
                  }}
                  className="header-dropdown-item flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-search)]"
                >
                  <Eraser className="h-4 w-4 text-[var(--text-secondary)]" />
                  <span>Xóa toàn bộ tin nhắn</span>
                </button>

                <div className="my-1 h-px bg-[var(--border-divider)]" />

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    onDeleteChat();
                  }}
                  className="header-dropdown-item text-danger flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] font-medium text-red-600 transition-colors hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/40"
                >
                  <Trash2 className="h-4 w-4" />
                  <span>Xóa cuộc trò chuyện</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
}


import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  GraduationCap,
  LayoutGrid,
  MoreVertical,
  PanelLeftClose,
  PenSquare,
  Pin,
  PinOff,
  Search,
  Settings,
  User,
  X,
} from "lucide-react";
import type { ChatSession, GroupedChats, ThemeMode } from "../../types/chat";
import { MiniRail } from "./MiniRail";

export interface SidebarProps {
  groupedChats: GroupedChats;
  activeChatId: string | null;
  searchQuery: string;
  onSearchQueryChange: (q: string) => void;
  isCollapsed: boolean;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
  onToggleCollapse: () => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  onTogglePin: (chatId: string) => void;
  onOpenSettings: () => void;
  onOpenInChatSearch: () => void;
  onOpenContextMenu: (x: number, y: number, chat: ChatSession) => void;
  onInsertRegulationsPrompt: () => void;
  onShowModulesInfo: () => void;
}

const MOBILE_DRAWER_SPRING = { type: "spring", stiffness: 380, damping: 36 } as const;

export function Sidebar({
  groupedChats,
  activeChatId,
  searchQuery,
  onSearchQueryChange,
  isCollapsed,
  isMobileOpen,
  onCloseMobile,
  onToggleCollapse,
  theme,
  onToggleTheme,
  onSelectChat,
  onNewChat,
  onTogglePin,
  onOpenSettings,
  onOpenInChatSearch,
  onOpenContextMenu,
  onInsertRegulationsPrompt,
  onShowModulesInfo,
}: SidebarProps) {
  const [isMobile, setIsMobile] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth < 768 : false
  );
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (!isSearchOpen) return;
    const raf = requestAnimationFrame(() => {
      searchInputRef.current?.focus();
    });
    return () => cancelAnimationFrame(raf);
  }, [isSearchOpen]);

  const handleToggleSearch = () => {
    if (isSearchOpen || searchQuery.length > 0) {
      onSearchQueryChange("");
      setIsSearchOpen(false);
    } else {
      setIsSearchOpen(true);
    }
  };

  const handleOpenModules = () => {
    onShowModulesInfo();
  };

  const recentChats = [
    ...groupedChats.today,
    ...groupedChats.yesterday,
    ...groupedChats.earlier,
  ];
  const hasAnyChats = groupedChats.pinned.length > 0 || recentChats.length > 0;

  const renderGroupSection = (title: string, chats: ChatSession[]) => {
    if (chats.length === 0) return null;

    return (
      <div key={title}>
        <div className="chat-group-header">
          <span>{title}</span>
        </div>

        {chats.map((chat) => {
          const isActive = chat.id === activeChatId;
          const displayTitle = chat.title || "Cuộc trò chuyện mới";

          return (
            <div
              key={chat.id}
              className={`chat-list-item ${isActive ? "active" : ""} ${
                chat.pinned ? "pinned-chat" : ""
              }`}
              data-chat-id={chat.id}
              role="button"
              tabIndex={0}
              onClick={() => {
                onSelectChat(chat.id);
                if (isMobile) onCloseMobile();
              }}
              onKeyDown={(e) => {
                if (e.target !== e.currentTarget) return;
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectChat(chat.id);
                  if (isMobile) onCloseMobile();
                }
              }}
              onContextMenu={(e) => {
                e.preventDefault();
                onOpenContextMenu(e.clientX, e.clientY, chat);
              }}
            >
              <span className="chat-item-title truncate" title={displayTitle}>
                {displayTitle}
              </span>

              <div className="chat-item-right">
                {chat.pinned && (
                  <span
                    className="chat-item-pin-icon"
                    title="Đã ghim"
                    aria-label="Đã ghim"
                  >
                    <Pin size={13} />
                  </span>
                )}

                <div className="chat-item-actions">
                  <button
                    type="button"
                    className={`btn-chat-item-menu btn-chat-item-pin ${
                      chat.pinned ? "is-pinned text-[var(--primary)] dark:text-[#38BDF8]" : ""
                    }`}
                    title={chat.pinned ? "Bỏ ghim hội thoại" : "Ghim hội thoại"}
                    aria-label={
                      chat.pinned
                        ? `Bỏ ghim ${displayTitle}`
                        : `Ghim ${displayTitle}`
                    }
                    onClick={(e) => {
                      e.stopPropagation();
                      onTogglePin(chat.id);
                    }}
                  >
                    {chat.pinned ? <PinOff size={14} /> : <Pin size={14} />}
                  </button>

                  <button
                    type="button"
                    className="btn-chat-item-menu"
                    title="Tùy chọn"
                    aria-label={`Tùy chọn cho ${displayTitle}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      const rect = e.currentTarget.getBoundingClientRect();
                      onOpenContextMenu(rect.left - 120, rect.bottom + 4, chat);
                    }}
                  >
                    <MoreVertical size={15} />
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const fullSidebarContent = (
    <div className="sidebar-full-content">
      {/* Top Header (Gemini-inspired: Brand left | Search icon + Collapse right) */}
      <div className="sidebar-top-header">
        <div className="uet-brand-box">
          <div
            className="uet-logo-emblem"
            title="Trường Đại học Công nghệ – ĐHQGHN"
          >
            <img
              src="/static/uet-logo.png?v=v14_robot_mascot"
              alt="UET ChatBot Logo"
              className="uet-logo-img"
            />
          </div>

          <h1 className="brand-name">UET AI</h1>
        </div>

        <div className="sidebar-header-actions">
          <button
            type="button"
            className={`btn-sidebar-action ${
              isSearchOpen || searchQuery.length > 0
                ? "bg-[rgba(8,117,201,0.12)] text-[var(--primary)] dark:text-[#38BDF8]"
                : ""
            }`}
            onClick={handleToggleSearch}
            title="Tìm kiếm hội thoại"
            aria-label="Tìm kiếm hội thoại"
            aria-expanded={isSearchOpen || searchQuery.length > 0}
          >
            <Search size={18} strokeWidth={2} />
          </button>

          <button
            type="button"
            className="btn-sidebar-action btn-sidebar-menu"
            onClick={(e) => {
              e.stopPropagation();
              if (isMobile) {
                onCloseMobile();
              } else {
                onToggleCollapse();
              }
            }}
            title={isMobile ? "Đóng thanh bên" : "Thu gọn thanh bên (Ctrl+B)"}
            aria-label={isMobile ? "Đóng thanh bên" : "Thu gọn thanh bên"}
          >
            {isMobile ? (
              <X size={19} strokeWidth={2} />
            ) : (
              <PanelLeftClose size={18} strokeWidth={2} />
            )}
          </button>
        </div>
      </div>

      {/* Collapsible Search Box */}
      {(isSearchOpen || searchQuery.length > 0) && (
        <div className="sidebar-search-box">
          <div className="search-input-wrapper">
            <Search size={15} className="search-icon" />
            <input
              ref={searchInputRef}
              type="text"
              className="search-input"
              placeholder="Tìm kiếm hội thoại..."
              autoComplete="off"
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  onSearchQueryChange("");
                  setIsSearchOpen(false);
                }
              }}
            />
            <button
              type="button"
              className="btn-clear-search"
              onClick={() => {
                onSearchQueryChange("");
                setIsSearchOpen(false);
              }}
              title="Đóng tìm kiếm"
              aria-label="Đóng tìm kiếm"
            >
              <X size={13} />
            </button>
          </div>
        </div>
      )}

      {/* Gemini-style Quick Features */}
      <nav
        className="sidebar-quick-actions"
        aria-label="Tính năng nhanh UET AI"
      >
        <button
          type="button"
          className="sidebar-quick-item"
          onClick={() => {
            onNewChat();
            if (isMobile) onCloseMobile();
          }}
        >
          <PenSquare size={17} strokeWidth={1.9} className="shrink-0" />
          <span>Cuộc trò chuyện mới</span>
        </button>

        <button
          type="button"
          className="sidebar-quick-item"
          onClick={handleOpenModules}
        >
          <LayoutGrid size={17} strokeWidth={1.9} className="shrink-0" />
          <span>Tiện ích UET</span>
        </button>

        <button
          type="button"
          className="sidebar-quick-item"
          onClick={() => {
            onInsertRegulationsPrompt();
            if (isMobile) onCloseMobile();
          }}
        >
          <GraduationCap size={18} strokeWidth={1.9} className="shrink-0" />
          <span>Tra cứu quy chế</span>
        </button>
      </nav>

      {/* Grouped Chat List */}
      <div className="chat-list-wrapper">
        {!hasAnyChats ? (
          <div className="chat-list-empty">
            <div className="text-[13px]">
              Không tìm thấy cuộc trò chuyện nào
            </div>
          </div>
        ) : (
          <>
            {renderGroupSection("ĐÃ GHIM", groupedChats.pinned)}
            {renderGroupSection("GẦN ĐÂY", recentChats)}
          </>
        )}
      </div>

      {/* Footer: Student Profile */}
      <div className="sidebar-student-footer">
        <div className="student-profile-left">
          <div className="student-avatar">
            <User size={20} strokeWidth={2} />
          </div>
          <div className="min-w-0">
            <div className="student-name">Sinh viên UET</div>
          </div>
        </div>

        <button
          type="button"
          className="btn-student-settings"
          onClick={onOpenSettings}
          title="Cài đặt hệ thống & mô hình"
          aria-label="Cài đặt hệ thống"
        >
          <Settings size={17} />
        </button>
      </div>
    </div>
  );

  // Mobile Drawer Layout
  if (isMobile) {
    return (
      <AnimatePresence>
        {isMobileOpen && (
          <>
            <motion.div
              key="mobile-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22 }}
              onClick={onCloseMobile}
              className="fixed inset-0 z-[998] bg-black/50 backdrop-blur-[2px]"
              aria-hidden="true"
            />
            <motion.aside
              key="mobile-drawer"
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={MOBILE_DRAWER_SPRING}
              style={{ minWidth: 0 }}
              className="telegram-sidebar fixed left-0 top-0 z-[1000] h-full w-[86%] max-w-[340px] rounded-l-none shadow-[4px_0_28px_rgba(0,0,0,0.35)] gpu-layer"
              aria-label="Danh sách cuộc trò chuyện"
            >
              {fullSidebarContent}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    );
  }

  // Desktop Collapsible Bento Sidebar <-> MiniRail (giữ nguyên DOM cả hai lớp để chuyển cảnh 60fps trên mọi trình duyệt)
  return (
    <aside
      className={`telegram-sidebar ${isCollapsed ? "sidebar-closed" : ""}`}
      aria-label="Thanh bên hội thoại"
    >
      <div className="sidebar-mini-rail-wrapper" aria-hidden={!isCollapsed}>
        <MiniRail
          theme={theme}
          onExpand={onToggleCollapse}
          onNewChat={onNewChat}
          onOpenInChatSearch={onOpenInChatSearch}
          onOpenModules={handleOpenModules}
          onInsertRegulationsPrompt={onInsertRegulationsPrompt}
          onToggleTheme={onToggleTheme}
          onOpenSettings={onOpenSettings}
        />
      </div>

      <div className="sidebar-full-clip" aria-hidden={isCollapsed}>
        {fullSidebarContent}
      </div>
    </aside>
  );
}


import {
  GraduationCap,
  LayoutGrid,
  Moon,
  PanelLeftOpen,
  PenSquare,
  Search,
  Settings,
  Sun,
  User,
} from "lucide-react";
import type { ThemeMode } from "../../types/chat";

export interface MiniRailProps {
  theme: ThemeMode;
  onExpand: () => void;
  onNewChat: () => void;
  onOpenInChatSearch: () => void;
  onOpenModules: () => void;
  onInsertRegulationsPrompt: () => void;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
}

export function MiniRail({
  theme,
  onExpand,
  onNewChat,
  onOpenInChatSearch,
  onOpenModules,
  onInsertRegulationsPrompt,
  onToggleTheme,
  onOpenSettings,
}: MiniRailProps) {
  return (
    <div className="sidebar-mini-rail" aria-label="Thanh công cụ thu gọn">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onExpand();
        }}
        className="mini-rail-btn btn-rail-menu"
        data-tooltip="Mở rộng thanh bên (Ctrl+B)"
        title="Mở rộng thanh bên (Ctrl+B)"
        aria-label="Mở rộng thanh bên"
      >
        <PanelLeftOpen size={18} strokeWidth={2} />
      </button>

      <div className="mini-rail-divider" />

      <div className="mini-rail-group">
        <button
          type="button"
          onClick={onNewChat}
          className="mini-rail-btn"
          data-tooltip="Tạo cuộc trò chuyện mới"
          title="Tạo cuộc trò chuyện mới"
          aria-label="Tạo cuộc trò chuyện mới"
        >
          <PenSquare size={17} strokeWidth={2} />
        </button>

        <button
          type="button"
          onClick={onOpenInChatSearch}
          className="mini-rail-btn"
          data-tooltip="Tìm kiếm trong cuộc trò chuyện (Ctrl+F)"
          title="Tìm kiếm trong cuộc trò chuyện (Ctrl+F)"
          aria-label="Tìm kiếm trong cuộc trò chuyện"
        >
          <Search size={17} strokeWidth={2} />
        </button>

        <button
          type="button"
          onClick={onOpenModules}
          className="mini-rail-btn"
          data-tooltip="Tính năng & Tiện ích UET"
          title="Tính năng & Tiện ích UET"
          aria-label="Tính năng và Tiện ích UET"
        >
          <LayoutGrid size={17} strokeWidth={2} />
        </button>

        <button
          type="button"
          onClick={onInsertRegulationsPrompt}
          className="mini-rail-btn"
          data-tooltip="Tra cứu Quy chế đào tạo UET"
          title="Tra cứu Quy chế đào tạo UET"
          aria-label="Tra cứu Quy chế đào tạo UET"
        >
          <GraduationCap size={17} strokeWidth={2} />
        </button>
      </div>

      <div className="mini-rail-spacer" />

      <div className="mini-rail-group mini-rail-bottom">
        <button
          type="button"
          onClick={onToggleTheme}
          className="mini-rail-btn"
          data-tooltip={
            theme === "dark" ? "Chuyển sang chế độ Sáng" : "Chuyển sang chế độ Tối"
          }
          title={
            theme === "dark" ? "Chuyển sang chế độ Sáng" : "Chuyển sang chế độ Tối"
          }
          aria-label="Chuyển chế độ Sáng / Tối"
        >
          {theme === "dark" ? (
            <Sun size={17} strokeWidth={2} />
          ) : (
            <Moon size={17} strokeWidth={2} />
          )}
        </button>

        <button
          type="button"
          onClick={onOpenSettings}
          className="mini-rail-btn"
          data-tooltip="Cài đặt hệ thống & mô hình"
          title="Cài đặt hệ thống & mô hình"
          aria-label="Cài đặt hệ thống và mô hình"
        >
          <Settings size={17} strokeWidth={2} />
        </button>

        <button
          type="button"
          onClick={onOpenSettings}
          className="mini-rail-btn mini-rail-avatar-btn"
          data-tooltip="Sinh viên UET"
          title="Sinh viên UET"
          aria-label="Hồ sơ sinh viên UET"
        >
          <div className="mini-rail-avatar">
            <User size={14} strokeWidth={2.2} />
          </div>
        </button>
      </div>
    </div>
  );
}


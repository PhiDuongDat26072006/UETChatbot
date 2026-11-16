import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Eraser, Pencil, Pin, PinOff, Search, Trash2 } from "lucide-react";

export interface ContextMenuState {
  x: number;
  y: number;
  chatId: string;
  pinned: boolean;
}

export interface ContextMenuProps {
  menu: ContextMenuState | null;
  onClose: () => void;
  onTogglePin: (chatId: string) => void;
  onRename: (chatId: string) => void;
  onSearchInChat: (chatId: string) => void;
  onClearMessages: (chatId: string) => void;
  onDelete: (chatId: string) => void;
}

export function ContextMenu({
  menu,
  onClose,
  onTogglePin,
  onRename,
  onSearchInChat,
  onClearMessages,
  onDelete,
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menu) return;

    const handlePointerDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    const handleWindowResizeOrScroll = () => onClose();

    document.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", handleWindowResizeOrScroll);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", handleWindowResizeOrScroll);
    };
  }, [menu, onClose]);

  const computePosition = (x: number, y: number) => {
    const menuWidth = 224;
    const menuHeight = 224;
    const vw = typeof window !== "undefined" ? window.innerWidth : 1024;
    const vh = typeof window !== "undefined" ? window.innerHeight : 768;

    let left = Math.max(8, x);
    let top = Math.max(8, y);
    if (left + menuWidth > vw - 8) left = Math.max(8, vw - menuWidth - 8);
    if (top + menuHeight > vh - 8) top = Math.max(8, vh - menuHeight - 8);
    return { left, top };
  };

  return (
    <AnimatePresence>
      {menu && (
        <motion.div
          ref={menuRef}
          key={`${menu.chatId}-${menu.x}-${menu.y}`}
          initial={{ opacity: 0, scale: 0.92, y: -4 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.92, y: -4 }}
          transition={{ type: "spring", stiffness: 520, damping: 32 }}
          style={{
            ...computePosition(menu.x, menu.y),
            transformOrigin: "top left",
          }}
          className="telegram-context-menu gpu-layer"
          role="menu"
          aria-label="Tùy chọn hội thoại"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            className="context-menu-item"
            role="menuitem"
            onClick={() => {
              onTogglePin(menu.chatId);
              onClose();
            }}
          >
            {menu.pinned ? <PinOff size={14} /> : <Pin size={14} />}
            <span>{menu.pinned ? "Bỏ ghim hội thoại" : "Ghim hội thoại"}</span>
          </button>

          <button
            type="button"
            className="context-menu-item"
            role="menuitem"
            onClick={() => {
              onRename(menu.chatId);
              onClose();
            }}
          >
            <Pencil size={14} />
            <span>Đổi tên hội thoại</span>
          </button>

          <button
            type="button"
            className="context-menu-item"
            role="menuitem"
            onClick={() => {
              onSearchInChat(menu.chatId);
              onClose();
            }}
          >
            <Search size={14} />
            <span>Tìm kiếm trong hội thoại</span>
          </button>

          <div className="context-menu-divider" />

          <button
            type="button"
            className="context-menu-item"
            role="menuitem"
            onClick={() => {
              onClearMessages(menu.chatId);
              onClose();
            }}
          >
            <Eraser size={14} />
            <span>Xóa toàn bộ tin nhắn</span>
          </button>

          <button
            type="button"
            className="context-menu-item text-danger"
            role="menuitem"
            onClick={() => {
              onDelete(menu.chatId);
              onClose();
            }}
          >
            <Trash2 size={14} />
            <span>Xóa hội thoại</span>
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}


/**
 * src/interfaces/frontend/src/services/storage.ts
 * Quản lý trạng thái hội thoại, giao diện và hình nền trên LocalStorage,
 * giữ tương thích 100% các khóa lưu trữ từ src/interfaces/static/js/storage.js.
 */

import type {
  ChatMessage,
  ChatSession,
  CreateMessageInput,
  GroupedChats,
  ThemeMode,
  WallpaperMode,
} from "../types/chat";

export const STORAGE_KEYS = {
  CONVERSATIONS: "uet_ai_telegram_conversations_v3",
  ACTIVE_CHAT: "uet_ai_active_chat_id_v3",
  THEME: "uet_ai_theme_mode",
  WALLPAPER: "uet_ai_chat_wallpaper",
  SIDEBAR_CLOSED: "uet_ai_sidebar_closed",
} as const;

const VALID_WALLPAPERS: readonly WallpaperMode[] = [
  "telegram-blue",
  "plain",
  "dark-night",
];

function hasLocalStorage(): boolean {
  try {
    return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
  } catch {
    return false;
  }
}

function generateId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
}

function normalizeMessage(raw: Partial<ChatMessage>): ChatMessage {
  return {
    id: raw.id || generateId("msg"),
    sender: raw.sender === "user" ? "user" : "bot",
    text: typeof raw.text === "string" ? raw.text : "",
    timestamp: typeof raw.timestamp === "number" ? raw.timestamp : Date.now(),
    sources: Array.isArray(raw.sources) ? raw.sources : [],
    latency: typeof raw.latency === "number" ? raw.latency : 0.0,
    model: typeof raw.model === "string" ? raw.model : "",
    attachments: Array.isArray(raw.attachments) ? raw.attachments : [],
    status: raw.status || "delivered",
    error: Boolean(raw.error),
    ...(raw.errorType ? { errorType: raw.errorType } : {}),
  };
}

function normalizeSession(raw: Partial<ChatSession>): ChatSession {
  const now = Date.now();
  return {
    id: raw.id || generateId("chat"),
    title:
      typeof raw.title === "string" && raw.title.trim()
        ? raw.title
        : "Cuộc trò chuyện mới",
    createdAt: typeof raw.createdAt === "number" ? raw.createdAt : now,
    updatedAt: typeof raw.updatedAt === "number" ? raw.updatedAt : now,
    pinned: Boolean(raw.pinned),
    unreadCount: typeof raw.unreadCount === "number" ? raw.unreadCount : 0,
    messages: Array.isArray(raw.messages)
      ? raw.messages.map((m) => normalizeMessage(m))
      : [],
  };
}

export function loadConversations(): ChatSession[] {
  if (!hasLocalStorage()) return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEYS.CONVERSATIONS);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed
        .filter((item): item is Partial<ChatSession> => Boolean(item && typeof item === "object"))
        .map((item) => normalizeSession(item));
    }
  } catch (e) {
    console.error("Lỗi đọc LocalStorage:", e);
  }
  return [];
}

export function saveConversations(conversations: ChatSession[]): void {
  if (!hasLocalStorage()) return;
  try {
    window.localStorage.setItem(
      STORAGE_KEYS.CONVERSATIONS,
      JSON.stringify(conversations)
    );
  } catch (e) {
    console.error("Lỗi ghi LocalStorage:", e);
  }
}

export function getActiveChatId(): string | null {
  if (!hasLocalStorage()) return null;
  try {
    return window.localStorage.getItem(STORAGE_KEYS.ACTIVE_CHAT) || null;
  } catch {
    return null;
  }
}

export function setActiveChatId(id: string | null): void {
  if (!hasLocalStorage()) return;
  try {
    if (id) {
      window.localStorage.setItem(STORAGE_KEYS.ACTIVE_CHAT, id);
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.ACTIVE_CHAT);
    }
  } catch (e) {
    console.error("Lỗi lưu activeChatId:", e);
  }
}

export function initializeConversations(): {
  conversations: ChatSession[];
  activeChatId: string;
} {
  let conversations = loadConversations();
  let activeChatId = getActiveChatId();

  if (conversations.length === 0) {
    const result = createChat([], "Tư vấn học vụ UET");
    return {
      conversations: result.conversations,
      activeChatId: result.chat.id,
    };
  }

  if (!activeChatId || !conversations.some((c) => c.id === activeChatId)) {
    activeChatId = conversations[0].id;
    setActiveChatId(activeChatId);
  }

  return { conversations, activeChatId };
}

export function sortChats(conversations: ChatSession[]): ChatSession[] {
  return [...conversations].sort((a, b) => {
    if (a.pinned && !b.pinned) return -1;
    if (!a.pinned && b.pinned) return 1;
    return (b.updatedAt || 0) - (a.updatedAt || 0);
  });
}

export function createChat(
  conversations: ChatSession[],
  title = "Cuộc trò chuyện mới"
): { conversations: ChatSession[]; chat: ChatSession } {
  // Tránh tạo trùng hội thoại rỗng liên tiếp giống storage.js
  if (conversations.length > 0) {
    const firstChat = conversations[0];
    if (
      (!firstChat.messages || firstChat.messages.length === 0) &&
      (firstChat.title === "Cuộc trò chuyện mới" ||
        firstChat.title === "Tư vấn học vụ UET")
    ) {
      setActiveChatId(firstChat.id);
      return { conversations, chat: firstChat };
    }
  }

  const now = Date.now();
  const newChat: ChatSession = {
    id: generateId("chat"),
    title: title.trim() || "Cuộc trò chuyện mới",
    createdAt: now,
    updatedAt: now,
    pinned: false,
    unreadCount: 0,
    messages: [],
  };

  const next = [newChat, ...conversations];
  saveConversations(next);
  setActiveChatId(newChat.id);
  return { conversations: next, chat: newChat };
}

export function addMessage(
  conversations: ChatSession[],
  chatId: string,
  messageInput: CreateMessageInput
): { conversations: ChatSession[]; message: ChatMessage | null } {
  const target = conversations.find((c) => c.id === chatId);
  if (!target) {
    return { conversations, message: null };
  }

  const msg = normalizeMessage({
    sender: messageInput.sender || "user",
    text: messageInput.text || "",
    timestamp: messageInput.timestamp || Date.now(),
    sources: messageInput.sources || [],
    latency: messageInput.latency || 0.0,
    model: messageInput.model || "",
    attachments: messageInput.attachments || [],
    status: messageInput.status || "delivered",
    error: messageInput.error || false,
    errorType: messageInput.errorType,
  });

  const next = conversations.map((chat) => {
    if (chat.id !== chatId) return chat;
    const updatedMessages = [...chat.messages, msg];
    let updatedTitle = chat.title;
    if (
      updatedMessages.length === 1 &&
      msg.sender === "user" &&
      (!chat.title ||
        chat.title.startsWith("Cuộc trò chuyện mới") ||
        chat.title === "Tư vấn học vụ UET")
    ) {
      const trimmed = msg.text.trim();
      updatedTitle =
        trimmed.substring(0, 36) + (trimmed.length > 36 ? "..." : "");
    }
    return {
      ...chat,
      title: updatedTitle || chat.title,
      updatedAt: Date.now(),
      messages: updatedMessages,
    };
  });

  saveConversations(next);
  return { conversations: next, message: msg };
}

export function updateMessage(
  conversations: ChatSession[],
  chatId: string,
  messageId: string,
  patch: Partial<ChatMessage>,
  persist = true
): ChatSession[] {
  const next = conversations.map((chat) => {
    if (chat.id !== chatId) return chat;
    return {
      ...chat,
      updatedAt: Date.now(),
      messages: chat.messages.map((m) =>
        m.id === messageId ? { ...m, ...patch } : m
      ),
    };
  });
  if (persist) {
    saveConversations(next);
  }
  return next;
}

export function togglePinChat(
  conversations: ChatSession[],
  id: string
): ChatSession[] {
  const next = conversations.map((chat) =>
    chat.id === id ? { ...chat, pinned: !chat.pinned } : chat
  );
  saveConversations(next);
  return next;
}

export function renameChat(
  conversations: ChatSession[],
  id: string,
  newTitle: string
): ChatSession[] {
  const trimmed = (newTitle || "").trim();
  if (!trimmed) return conversations;
  const next = conversations.map((chat) =>
    chat.id === id
      ? { ...chat, title: trimmed, updatedAt: Date.now() }
      : chat
  );
  saveConversations(next);
  return next;
}

export function deleteChat(
  conversations: ChatSession[],
  id: string,
  currentActiveId?: string | null
): { conversations: ChatSession[]; activeChatId: string } {
  const filtered = conversations.filter((c) => c.id !== id);
  if (filtered.length === 0) {
    const freshResult = createChat([], "Cuộc trò chuyện mới");
    return {
      conversations: freshResult.conversations,
      activeChatId: freshResult.chat.id,
    };
  }

  let nextActiveId = currentActiveId ?? getActiveChatId();
  if (!nextActiveId || nextActiveId === id || !filtered.some((c) => c.id === nextActiveId)) {
    nextActiveId = filtered[0].id;
    setActiveChatId(nextActiveId);
  }

  saveConversations(filtered);
  return { conversations: filtered, activeChatId: nextActiveId };
}

export function clearChatMessages(
  conversations: ChatSession[],
  id: string
): ChatSession[] {
  const next = conversations.map((chat) =>
    chat.id === id
      ? { ...chat, messages: [], updatedAt: Date.now() }
      : chat
  );
  saveConversations(next);
  return next;
}

export function clearAllChats(): {
  conversations: ChatSession[];
  activeChatId: string;
} {
  if (hasLocalStorage()) {
    try {
      window.localStorage.removeItem(STORAGE_KEYS.CONVERSATIONS);
      window.localStorage.removeItem(STORAGE_KEYS.ACTIVE_CHAT);
    } catch (e) {
      console.error("Lỗi xóa LocalStorage:", e);
    }
  }
  const fresh = createChat([], "Cuộc trò chuyện mới");
  return {
    conversations: fresh.conversations,
    activeChatId: fresh.chat.id,
  };
}

export function groupChats(
  conversations: ChatSession[],
  query = ""
): GroupedChats {
  let list = sortChats(conversations);
  const trimmed = query.trim().toLowerCase();
  if (trimmed) {
    list = list.filter((c) => {
      if (c.title.toLowerCase().includes(trimmed)) return true;
      return c.messages.some(
        (m) => m.text && m.text.toLowerCase().includes(trimmed)
      );
    });
  }

  const now = new Date();
  const startOfToday = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate()
  ).getTime();
  const startOfYesterday = startOfToday - 86_400_000;

  const groups: GroupedChats = {
    pinned: [],
    today: [],
    yesterday: [],
    earlier: [],
  };

  for (const chat of list) {
    if (chat.pinned) {
      groups.pinned.push(chat);
      continue;
    }

    const time = chat.updatedAt || chat.createdAt || 0;
    if (time >= startOfToday) {
      groups.today.push(chat);
    } else if (time >= startOfYesterday) {
      groups.yesterday.push(chat);
    } else {
      groups.earlier.push(chat);
    }
  }

  return groups;
}

export function getTheme(): ThemeMode {
  if (!hasLocalStorage()) return "light";
  try {
    const stored = window.localStorage.getItem(STORAGE_KEYS.THEME);
    if (stored === "dark" || stored === "light") return stored;
  } catch {
    // ignore
  }
  return "light";
}

export function setTheme(theme: ThemeMode): ThemeMode {
  const normalized: ThemeMode = theme === "dark" ? "dark" : "light";
  if (hasLocalStorage()) {
    try {
      window.localStorage.setItem(STORAGE_KEYS.THEME, normalized);
    } catch (e) {
      console.error("Lỗi lưu theme:", e);
    }
  }
  return normalized;
}

export function getWallpaper(): WallpaperMode {
  if (!hasLocalStorage()) return "telegram-blue";
  try {
    const stored = window.localStorage.getItem(STORAGE_KEYS.WALLPAPER);
    if (stored && (VALID_WALLPAPERS as readonly string[]).includes(stored)) {
      return stored as WallpaperMode;
    }
  } catch {
    // ignore
  }
  return "telegram-blue";
}

export function setWallpaper(wallpaper: WallpaperMode): WallpaperMode {
  const normalized: WallpaperMode = (
    VALID_WALLPAPERS as readonly string[]
  ).includes(wallpaper)
    ? wallpaper
    : "telegram-blue";
  if (hasLocalStorage()) {
    try {
      window.localStorage.setItem(STORAGE_KEYS.WALLPAPER, normalized);
    } catch (e) {
      console.error("Lỗi lưu wallpaper:", e);
    }
  }
  return normalized;
}

export function isSidebarClosed(): boolean {
  if (!hasLocalStorage()) return false;
  try {
    return window.localStorage.getItem(STORAGE_KEYS.SIDEBAR_CLOSED) === "true";
  } catch {
    return false;
  }
}

export function setSidebarClosed(closed: boolean): void {
  if (!hasLocalStorage()) return;
  try {
    window.localStorage.setItem(STORAGE_KEYS.SIDEBAR_CLOSED, String(closed));
  } catch (e) {
    console.error("Lỗi lưu trạng thái sidebar:", e);
  }
}


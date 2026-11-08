/**
 * src/interfaces/static/js/storage.js
 * Storage & Chat State Manager Module - Version 2.0
 * Quản lý lịch sử hội thoại trên LocalStorage:
 * Hỗ trợ Ghim hội thoại (Pin), Phân nhóm (Hôm nay, Hôm qua, Trước đó),
 * Đổi tên, Xóa, Tìm kiếm, Đính kèm file và Theme.
 */

const STORAGE_KEY = "uet_ai_telegram_conversations_v3";
const ACTIVE_CHAT_KEY = "uet_ai_active_chat_id_v3";
const THEME_KEY = "uet_ai_theme_mode";
const WALLPAPER_KEY = "uet_ai_chat_wallpaper";
const SIDEBAR_CLOSED_KEY = "uet_ai_sidebar_closed";

export class UETChatStorage {
    constructor() {
        this.conversations = this._loadFromStorage();
        this.activeChatId = localStorage.getItem(ACTIVE_CHAT_KEY) || null;

        // Khởi tạo cuộc trò chuyện mẫu đầu tiên nếu danh sách trống
        if (this.conversations.length === 0) {
            const initialChat = this.createChat("Tư vấn học vụ UET");
            this.activeChatId = initialChat.id;
        } else if (!this.activeChatId || !this.getChat(this.activeChatId)) {
            this.activeChatId = this.conversations[0].id;
            localStorage.setItem(ACTIVE_CHAT_KEY, this.activeChatId);
        }
    }

    _loadFromStorage() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) {
                    return parsed;
                }
            }
        } catch (e) {
            console.error("Lỗi đọc LocalStorage:", e);
        }
        return [];
    }

    _saveToStorage() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(this.conversations));
        } catch (e) {
            console.error("Lỗi ghi LocalStorage:", e);
        }
    }

    getAllChats() {
        return [...this.conversations].sort((a, b) => {
            // Ghim lên đầu
            if (a.pinned && !b.pinned) return -1;
            if (!a.pinned && b.pinned) return 1;
            return (b.updatedAt || 0) - (a.updatedAt || 0);
        });
    }

    getChat(id) {
        return this.conversations.find((c) => c.id === id) || null;
    }

    getActiveChat() {
        return this.getChat(this.activeChatId);
    }

    setActiveChatId(id) {
        if (this.getChat(id)) {
            this.activeChatId = id;
            localStorage.setItem(ACTIVE_CHAT_KEY, id);
        }
    }

    createChat(title = "Cuộc trò chuyện mới") {
        // Tránh tạo trùng lặp hội thoại rỗng khi người dùng click liên tục mà chưa gửi tin nhắn
        if (this.conversations.length > 0) {
            const firstChat = this.conversations[0];
            if ((!firstChat.messages || firstChat.messages.length === 0) &&
                (firstChat.title === "Cuộc trò chuyện mới" || firstChat.title === "Tư vấn học vụ UET")) {
                this.setActiveChatId(firstChat.id);
                return firstChat;
            }
        }

        const id = "chat_" + Date.now() + "_" + Math.random().toString(36).substring(2, 6);
        const newChat = {
            id,
            title: title.trim() || "Cuộc trò chuyện mới",
            createdAt: Date.now(),
            updatedAt: Date.now(),
            pinned: false,
            unreadCount: 0,
            messages: []
        };
        this.conversations.unshift(newChat);
        this._saveToStorage();
        this.setActiveChatId(id);
        return newChat;
    }

    togglePinChat(id) {
        const chat = this.getChat(id);
        if (chat) {
            chat.pinned = !chat.pinned;
            this._saveToStorage();
            return chat.pinned;
        }
        return false;
    }

    renameChat(id, newTitle) {
        const chat = this.getChat(id);
        if (chat && newTitle && newTitle.trim()) {
            chat.title = newTitle.trim();
            chat.updatedAt = Date.now();
            this._saveToStorage();
            return true;
        }
        return false;
    }

    deleteChat(id) {
        const idx = this.conversations.findIndex((c) => c.id === id);
        if (idx !== -1) {
            this.conversations.splice(idx, 1);
            if (this.conversations.length === 0) {
                const fresh = this.createChat("Cuộc trò chuyện mới");
                this.activeChatId = fresh.id;
            } else if (this.activeChatId === id) {
                this.activeChatId = this.conversations[0].id;
                localStorage.setItem(ACTIVE_CHAT_KEY, this.activeChatId);
            }
            this._saveToStorage();
            return true;
        }
        return false;
    }

    clearChatMessages(id) {
        const chat = this.getChat(id);
        if (chat) {
            chat.messages = [];
            chat.updatedAt = Date.now();
            this._saveToStorage();
            return true;
        }
        return false;
    }

    addMessage(chatId, messageObj) {
        const chat = this.getChat(chatId);
        if (!chat) return null;

        const msg = {
            id: "msg_" + Date.now() + "_" + Math.random().toString(36).substring(2, 6),
            sender: messageObj.sender || "user", // "user" hoặc "bot"
            text: messageObj.text || "",
            timestamp: messageObj.timestamp || Date.now(),
            sources: messageObj.sources || [],
            latency: messageObj.latency || 0.0,
            model: messageObj.model || "",
            attachments: messageObj.attachments || [],
            status: messageObj.status || "delivered",
            error: messageObj.error || false
        };

        chat.messages.push(msg);
        chat.updatedAt = Date.now();

        // Tự động đặt tên theo câu hỏi đầu tiên của sinh viên
        if (chat.messages.length === 1 && msg.sender === "user" && (!chat.title || chat.title.startsWith("Cuộc trò chuyện mới") || chat.title === "Tư vấn học vụ UET")) {
            const autoTitle = msg.text.trim().substring(0, 36) + (msg.text.length > 36 ? "..." : "");
            chat.title = autoTitle;
        }

        this._saveToStorage();
        return msg;
    }

    /**
     * Phân nhóm các cuộc trò chuyện theo: Ghim, Hôm nay, Hôm qua, Trước đó
     */
    getGroupedChats(query = "") {
        let list = this.getAllChats();
        if (query && query.trim()) {
            const q = query.trim().toLowerCase();
            list = list.filter((c) => {
                if (c.title.toLowerCase().includes(q)) return true;
                return c.messages.some((m) => m.text && m.text.toLowerCase().includes(q));
            });
        }

        const now = new Date();
        const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
        const startOfYesterday = startOfToday - 86400000;

        const groups = {
            pinned: [],
            today: [],
            yesterday: [],
            earlier: []
        };

        list.forEach((chat) => {
            if (chat.pinned) {
                groups.pinned.push(chat);
                return;
            }

            const time = chat.updatedAt || chat.createdAt || 0;
            if (time >= startOfToday) {
                groups.today.push(chat);
            } else if (time >= startOfYesterday) {
                groups.yesterday.push(chat);
            } else {
                groups.earlier.push(chat);
            }
        });

        return groups;
    }

    getTheme() {
        const stored = localStorage.getItem(THEME_KEY);
        if (stored === "dark" || stored === "light") return stored;
        return "light"; // Mặc định Light Theme sạch sẽ chuẩn Telegram Web
    }

    setTheme(theme) {
        const t = theme === "dark" ? "dark" : "light";
        localStorage.setItem(THEME_KEY, t);
        return t;
    }

    getWallpaper() {
        const stored = localStorage.getItem(WALLPAPER_KEY);
        const valid = ["telegram-blue", "plain", "dark-night"];
        return (stored && valid.includes(stored)) ? stored : "telegram-blue";
    }

    setWallpaper(wallpaper) {
        const valid = ["telegram-blue", "plain", "dark-night"];
        const w = valid.includes(wallpaper) ? wallpaper : "telegram-blue";
        localStorage.setItem(WALLPAPER_KEY, w);
        return w;
    }

    isSidebarClosed() {
        return localStorage.getItem(SIDEBAR_CLOSED_KEY) === "true";
    }

    setSidebarClosed(closed) {
        localStorage.setItem(SIDEBAR_CLOSED_KEY, String(closed));
    }
}

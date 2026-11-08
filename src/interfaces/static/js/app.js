/**
 * src/interfaces/static/js/app.js
 * Main Controller - Version 2.0 (Telegram Desktop/Web Edition)
 * Hỗ trợ Ghim hội thoại, Phân nhóm thời gian, Context menu chuột phải,
 * Thu gọn sidebar, Đính kèm tài liệu (.pdf, .docx, .txt), Drag & drop,
 * LaTeX Math (KaTeX), Settings modal và Tối ưu trải nghiệm Telegram.
 */

import { UETApiClient } from "./api.js";
import { UETChatStorage } from "./storage.js";
import { UETChatRenderer } from "./renderer.js";

document.addEventListener("DOMContentLoaded", () => {
    // 1. Khởi tạo các phân hệ
    const api = new UETApiClient();
    const storage = new UETChatStorage();
    const renderer = new UETChatRenderer();

    let isGenerating = false;
    let selectedFiles = []; // Danh sách file đính kèm trước khi gửi
    let contextMenuTargetChatId = null; // ID hội thoại đang mở context menu
    let lastUserQueryText = ""; // Lưu câu hỏi gần nhất để hỗ trợ nút "Gửi lại"

    // 2. DOM Elements
    const appContainer = document.getElementById("app-container");
    const sidebar = document.getElementById("sidebar");
    const sidebarBackdrop = document.getElementById("sidebar-backdrop");
    const chatListContainer = document.getElementById("chat-list-container");
    const btnNewChat = document.getElementById("btn-new-chat");
    const inputSearchChats = document.getElementById("input-search-chats");
    const btnClearSearch = document.getElementById("btn-clear-search");
    const btnThemeToggle = document.getElementById("btn-theme-toggle");
    const btnSidebarHamburger = document.getElementById("btn-sidebar-hamburger");

    const chatHeader = document.getElementById("chat-header");
    const headerTitle = document.getElementById("header-chat-title");
    const headerStatus = document.getElementById("header-chat-status");
    const btnHeaderHamburger = document.getElementById("btn-header-hamburger");
    const btnHeaderSearch = document.getElementById("btn-header-search");
    const btnHeaderSettings = document.getElementById("btn-header-settings");
    const btnHeaderMenu = document.getElementById("btn-header-menu");
    const headerDropdownMenu = document.getElementById("header-dropdown-menu");
    const headerOptPin = document.getElementById("header-opt-pin");
    const headerOptPinText = document.getElementById("header-opt-pin-text");

    const chatMainArea = document.getElementById("chat-main-area");
    const chatWindow = document.getElementById("chat-window");
    const messagesContainer = document.getElementById("messages-container");
    const welcomeScreen = document.getElementById("welcome-screen");
    const dragDropOverlay = document.getElementById("drag-drop-overlay");

    const chatForm = document.getElementById("chat-form");
    const inputMessage = document.getElementById("input-message");
    const btnSend = document.getElementById("btn-send");
    const btnEmoji = document.getElementById("btn-emoji");
    const emojiPicker = document.getElementById("emoji-picker");
    const btnAttach = document.getElementById("btn-attach");
    const fileAttachmentInput = document.getElementById("file-attachment-input");
    const attachmentPreviewArea = document.getElementById("attachment-preview-area");

    // Context Menu
    const chatContextMenu = document.getElementById("chat-context-menu");
    const ctxOptPin = document.getElementById("ctx-opt-pin");
    const ctxPinLabel = document.getElementById("ctx-pin-label");
    const ctxOptRename = document.getElementById("ctx-opt-rename");
    const ctxOptDelete = document.getElementById("ctx-opt-delete");

    // Settings Modal
    const settingsModal = document.getElementById("settings-modal");
    const btnOpenSettings = document.getElementById("btn-open-settings");
    const btnCloseSettingsModal = document.getElementById("btn-close-settings-modal");
    const btnSettingsDone = document.getElementById("btn-settings-done");
    const settingsModelName = document.getElementById("settings-model-name");
    const settingsRagStatus = document.getElementById("settings-rag-status");
    const btnClearAllChats = document.getElementById("btn-clear-all-chats");

    // Source Modal
    const sourceModal = document.getElementById("source-modal");
    const modalSourceTitle = document.getElementById("modal-source-title");
    const modalSourceMeta = document.getElementById("modal-source-meta");
    const modalSourceContent = document.getElementById("modal-source-content");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnModalDone = document.getElementById("btn-modal-done");

    // Rename Modal (Telegram Popup)
    const renameModal = document.getElementById("rename-modal");
    const inputRenameChat = document.getElementById("input-rename-chat");
    const btnCloseRenameModal = document.getElementById("btn-close-rename-modal");
    const btnCancelRename = document.getElementById("btn-cancel-rename");
    const btnConfirmRename = document.getElementById("btn-confirm-rename");
    let currentRenameCallback = null;

    // Confirm Modal (Telegram Confirmation Popup)
    const confirmModal = document.getElementById("confirm-modal");
    const confirmModalIcon = document.getElementById("confirm-modal-icon");
    const confirmModalTitle = document.getElementById("confirm-modal-title");
    const confirmModalMessage = document.getElementById("confirm-modal-message");
    const btnCloseConfirmModal = document.getElementById("btn-close-confirm-modal");
    const btnCancelConfirm = document.getElementById("btn-cancel-confirm");
    const btnProceedConfirm = document.getElementById("btn-proceed-confirm");
    let currentConfirmCallback = null;

    // Scroll to Bottom Floating Button
    const btnScrollBottom = document.getElementById("btn-scroll-bottom");

    // In-chat search
    const inChatSearchBar = document.getElementById("in-chat-search-bar");
    const inChatSearchInput = document.getElementById("in-chat-search-input");
    const inChatSearchCount = document.getElementById("in-chat-search-count");
    const btnSearchPrev = document.getElementById("btn-search-prev");
    const btnSearchNext = document.getElementById("btn-search-next");
    const btnCloseInChatSearch = document.getElementById("btn-close-in-chat-search");

    // 3. Khởi tạo Theme
    const currentTheme = storage.getTheme();
    document.documentElement.setAttribute("data-theme", currentTheme);
    updateThemeIcon(currentTheme);

    function updateThemeIcon(theme) {
        if (!btnThemeToggle) return;
        if (theme === "dark") {
            btnThemeToggle.innerHTML = `
                <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="5"></circle>
                    <line x1="12" y1="1" x2="12" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="23"></line>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                    <line x1="1" y1="12" x2="3" y2="12"></line>
                    <line x1="21" y1="12" x2="23" y2="12"></line>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                </svg>
            `;
            btnThemeToggle.title = "Chuyển sang chế độ Sáng";
        } else {
            btnThemeToggle.innerHTML = `
                <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                </svg>
            `;
            btnThemeToggle.title = "Chuyển sang chế độ Tối (Telegram Dark)";
        }
        updateActiveThemeButtons(theme);
    }

    function updateActiveThemeButtons(theme) {
        document.querySelectorAll(".btn-theme-choice").forEach((btn) => {
            btn.classList.toggle("active", btn.getAttribute("data-theme") === theme);
        });
    }

    function updateActiveWallpaperButtons(wallpaper) {
        document.querySelectorAll(".btn-wallpaper-choice").forEach((btn) => {
            btn.classList.toggle("active", btn.getAttribute("data-wallpaper") === wallpaper);
        });
    }

    const initialWallpaper = storage.getWallpaper();
    document.documentElement.setAttribute("data-wallpaper", initialWallpaper);
    updateActiveWallpaperButtons(initialWallpaper);

    document.querySelectorAll(".btn-theme-choice").forEach((btn) => {
        btn.addEventListener("click", () => {
            const theme = btn.getAttribute("data-theme");
            if (theme) {
                storage.setTheme(theme);
                document.documentElement.setAttribute("data-theme", theme);
                updateThemeIcon(theme);
                renderer.showToast(`Đã chuyển sang giao diện ${theme === "dark" ? "Tối" : "Sáng"}`, "info", 1500);
            }
        });
    });

    document.querySelectorAll(".btn-wallpaper-choice").forEach((btn) => {
        btn.addEventListener("click", () => {
            const wp = btn.getAttribute("data-wallpaper");
            if (wp) {
                storage.setWallpaper(wp);
                document.documentElement.setAttribute("data-wallpaper", wp);
                updateActiveWallpaperButtons(wp);
                renderer.showToast("Đã cập nhật hình nền trò chuyện", "info", 1500);
            }
        });
    });

    if (btnThemeToggle) {
        btnThemeToggle.addEventListener("click", () => {
            const now = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
            storage.setTheme(now);
            document.documentElement.setAttribute("data-theme", now);
            updateThemeIcon(now);
            renderer.showToast(`Đã chuyển sang giao diện ${now === "dark" ? "Tối" : "Sáng"}`, "info", 1500);
        });
    }

    // 4. Quản lý Đóng / Mở Sidebar (Telegram Hamburger Drawer & Desktop Toggle)
    const isSidebarClosedStored = storage.isSidebarClosed();
    if (window.innerWidth > 768 && isSidebarClosedStored && sidebar) {
        sidebar.classList.add("sidebar-closed");
    }

    function toggleSidebar() {
        if (!sidebar) return;
        if (window.innerWidth <= 768) {
            // Trên Mobile: mở/đóng dạng Drawer trượt kèm backdrop
            const isOpen = sidebar.classList.toggle("open-mobile");
            if (sidebarBackdrop) {
                sidebarBackdrop.classList.toggle("show", isOpen);
            }
        } else {
            // Trên Desktop: thu gọn / mở rộng sang trái mượt mà
            const isClosed = sidebar.classList.toggle("sidebar-closed");
            storage.setSidebarClosed(isClosed);
        }
    }

    function closeSidebar() {
        if (!sidebar) return;
        if (window.innerWidth <= 768) {
            sidebar.classList.remove("open-mobile");
            if (sidebarBackdrop) sidebarBackdrop.classList.remove("show");
        }
    }

    if (btnHeaderHamburger) {
        btnHeaderHamburger.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleSidebar();
        });
    }

    if (btnSidebarHamburger) {
        btnSidebarHamburger.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleSidebar();
        });
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", () => {
            closeSidebar();
        });
    }

    // Phím tắt Ctrl + B để đóng/mở sidebar
    window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
            e.preventDefault();
            toggleSidebar();
        }
    });

    // 5. Kiểm tra kết nối Backend & Trạng thái hoạt động
    async function checkServerStatus() {
        const health = await api.getHealth();
        if (headerStatus) {
            if (health.online) {
                headerStatus.innerHTML = `<span class="online-indicator"></span> trực tuyến`;
                headerStatus.className = "chat-header-status online";
            } else {
                headerStatus.innerHTML = `<span class="offline-indicator"></span> ngoại tuyến (offline)`;
                headerStatus.className = "chat-header-status offline";
            }
        }
        if (settingsModelName) {
            settingsModelName.textContent = health.model_name || "Qwen / Gemini RAG";
        }
        if (settingsRagStatus) {
            if (health.online) {
                settingsRagStatus.textContent = "Đang hoạt động (Connected)";
                settingsRagStatus.className = "settings-val-badge badge-success";
            } else {
                settingsRagStatus.textContent = "Chưa kết nối (Offline)";
                settingsRagStatus.className = "settings-val-badge badge-warning";
            }
        }
    }
    checkServerStatus();

    // 6. PHẦN 1: RENDER SIDEBAR PHÂN NHÓM & GHIM (Hôm nay, Hôm qua, Trước đó, Đã ghim)
    function renderChatList(filterQuery = "") {
        if (!chatListContainer) return;
        chatListContainer.innerHTML = "";

        const groups = storage.getGroupedChats(filterQuery);
        const hasAny = groups.pinned.length > 0 || groups.today.length > 0 || groups.yesterday.length > 0 || groups.earlier.length > 0;

        if (!hasAny) {
            chatListContainer.innerHTML = `
                <div class="chat-list-empty">
                    <div class="empty-icon">💬</div>
                    <div class="empty-text">Không tìm thấy cuộc trò chuyện nào</div>
                </div>
            `;
            return;
        }

        const renderSection = (title, chats, isPinned = false) => {
            if (chats.length === 0) return;

            const secHeader = document.createElement("div");
            secHeader.className = "chat-group-header";
            secHeader.innerHTML = `<span>${title}</span> <span class="group-count">(${chats.length})</span>`;
            chatListContainer.appendChild(secHeader);

            chats.forEach((chat) => {
                const isActive = chat.id === storage.activeChatId;
                const item = document.createElement("div");
                item.className = `chat-list-item ${isActive ? "active" : ""} ${chat.pinned ? "pinned-chat" : ""}`;
                item.setAttribute("data-chat-id", chat.id);

                let lastSnippet = "Chưa có tin nhắn nào";
                let timeDisplay = "";
                if (chat.messages && chat.messages.length > 0) {
                    const lastMsg = chat.messages[chat.messages.length - 1];
                    const prefix = lastMsg.sender === "user" ? "Bạn: " : "";
                    lastSnippet = prefix + (lastMsg.text || "").replace(/\n/g, " ");
                    if (lastSnippet.length > 38) {
                        lastSnippet = lastSnippet.substring(0, 38) + "...";
                    }
                    timeDisplay = renderer.formatTime(lastMsg.timestamp);
                } else if (chat.createdAt) {
                    timeDisplay = renderer.formatTime(chat.createdAt);
                }

                const initials = chat.title ? chat.title.trim().charAt(0).toUpperCase() : "U";
                const colorClass = "avatar-grad-" + (Math.abs(hashCode(chat.id)) % 5);

                item.innerHTML = `
                    <div class="telegram-avatar chat-item-avatar ${colorClass}">
                        <span>${renderer.escapeHtml(initials)}</span>
                        ${chat.pinned ? `<span class="pin-badge-dot" title="Đã ghim">📌</span>` : ""}
                    </div>
                    <div class="chat-item-content">
                        <div class="chat-item-header">
                            <div class="chat-item-title">${renderer.escapeHtml(chat.title)}</div>
                            <div class="chat-item-time">${timeDisplay}</div>
                        </div>
                        <div class="chat-item-bottom-row">
                            <div class="chat-item-snippet">${renderer.escapeHtml(lastSnippet)}</div>
                            ${chat.unreadCount && chat.unreadCount > 0 ? `<span class="unread-badge">${chat.unreadCount}</span>` : ""}
                        </div>
                    </div>
                    <div class="chat-item-actions">
                        <button class="btn-chat-item-menu" title="Tùy chọn">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="1.2"></circle>
                                <circle cx="12" cy="5" r="1.2"></circle>
                                <circle cx="12" cy="19" r="1.2"></circle>
                            </svg>
                        </button>
                    </div>
                `;

                // Chọn hội thoại
                item.addEventListener("click", (e) => {
                    if (e.target.closest(".chat-item-actions")) return;
                    storage.setActiveChatId(chat.id);
                    if (chat.unreadCount) {
                        chat.unreadCount = 0;
                        storage._saveToStorage();
                    }
                    renderChatList(inputSearchChats ? inputSearchChats.value : "");
                    renderActiveChatMessages();
                    closeSidebar();
                });

                // Click chuột phải mở Context Menu
                item.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                    showContextMenu(e.clientX, e.clientY, chat.id);
                });

                // Nút 3 chấm mở Context Menu ngay tại vị trí nút
                const btnItemMenu = item.querySelector(".btn-chat-item-menu");
                if (btnItemMenu) {
                    btnItemMenu.addEventListener("click", (e) => {
                        e.stopPropagation();
                        const rect = btnItemMenu.getBoundingClientRect();
                        showContextMenu(rect.left - 120, rect.bottom + 4, chat.id);
                    });
                }

                chatListContainer.appendChild(item);
            });
        };

        // Render theo đúng thứ tự phân nhóm: Đã ghim -> Hôm nay -> Hôm qua -> Trước đó
        renderSection("📌 ĐÃ GHIM", groups.pinned, true);
        renderSection("HÔM NAY", groups.today);
        renderSection("HÔM QUA", groups.yesterday);
        renderSection("TRƯỚC ĐÓ", groups.earlier);
    }

    function hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = (hash << 5) - hash + str.charCodeAt(i);
            hash |= 0;
        }
        return hash;
    }

    // 7. Context Menu (Menu chuột phải / 3 chấm)
    function showContextMenu(x, y, chatId) {
        if (!chatContextMenu) return;
        contextMenuTargetChatId = chatId;
        const targetChat = storage.getChat(chatId);
        if (!targetChat) return;

        if (ctxPinLabel) {
            ctxPinLabel.textContent = targetChat.pinned ? "Bỏ ghim hội thoại" : "Ghim hội thoại";
        }

        // Đảm bảo menu không tràn màn hình
        const menuWidth = 180;
        const menuHeight = 130;
        let finalX = x;
        let finalY = y;
        if (finalX + menuWidth > window.innerWidth) finalX = window.innerWidth - menuWidth - 10;
        if (finalY + menuHeight > window.innerHeight) finalY = window.innerHeight - menuHeight - 10;

        chatContextMenu.style.left = `${finalX}px`;
        chatContextMenu.style.top = `${finalY}px`;
        chatContextMenu.classList.add("show");
    }

    function hideContextMenu() {
        if (chatContextMenu) chatContextMenu.classList.remove("show");
        contextMenuTargetChatId = null;
    }

    document.addEventListener("click", () => {
        hideContextMenu();
        if (headerDropdownMenu) headerDropdownMenu.classList.remove("show");
        if (emojiPicker) emojiPicker.classList.remove("show");
    });

    // Thao tác Ghim
    if (ctxOptPin) {
        ctxOptPin.addEventListener("click", () => {
            if (!contextMenuTargetChatId) return;
            const isNowPinned = storage.togglePinChat(contextMenuTargetChatId);
            renderChatList(inputSearchChats ? inputSearchChats.value : "");
            hideContextMenu();
            renderer.showToast(isNowPinned ? "Đã ghim cuộc trò chuyện lên đầu" : "Đã bỏ ghim cuộc trò chuyện", "info", 1500);
        });
    }

    // Thao tác Đổi tên
    if (ctxOptRename) {
        ctxOptRename.addEventListener("click", () => {
            if (!contextMenuTargetChatId) return;
            const chat = storage.getChat(contextMenuTargetChatId);
            hideContextMenu();
            if (!chat) return;

            openRenameModal(chat.title, (newTitle) => {
                storage.renameChat(chat.id, newTitle);
                renderChatList(inputSearchChats ? inputSearchChats.value : "");
                if (chat.id === storage.activeChatId && headerTitle) {
                    headerTitle.textContent = newTitle;
                }
                renderer.showToast("Đã đổi tên cuộc trò chuyện thành công!", "success");
            });
        });
    }

    // Thao tác Xóa
    if (ctxOptDelete) {
        ctxOptDelete.addEventListener("click", () => {
            if (!contextMenuTargetChatId) return;
            const chat = storage.getChat(contextMenuTargetChatId);
            hideContextMenu();
            if (!chat) return;

            openConfirmModal({
                title: "Xóa cuộc trò chuyện",
                message: `Bạn có chắc muốn xóa cuộc trò chuyện "${chat.title}"?`,
                icon: "🗑️",
                confirmText: "Xóa cuộc trò chuyện",
                isDanger: true,
                onConfirm: () => {
                    storage.deleteChat(chat.id);
                    renderChatList(inputSearchChats ? inputSearchChats.value : "");
                    renderActiveChatMessages();
                    renderer.showToast("Đã xóa cuộc trò chuyện", "info");
                }
            });
        });
    }

    // 8. PHẦN 2: CHAT WINDOW & RENDER ACTIVE MESSAGES
    function renderActiveChatMessages() {
        const activeChat = storage.getActiveChat();
        if (!activeChat) return;

        if (headerTitle) {
            headerTitle.textContent = activeChat.title || "UET AI Assistant";
        }
        if (headerOptPinText) {
            headerOptPinText.textContent = activeChat.pinned ? "Bỏ ghim hội thoại" : "Ghim hội thoại";
        }

        messagesContainer.innerHTML = "";

        // Nếu rỗng -> Hiện Welcome Screen
        if (!activeChat.messages || activeChat.messages.length === 0) {
            welcomeScreen.style.display = "flex";
            messagesContainer.appendChild(welcomeScreen);
            return;
        }

        welcomeScreen.style.display = "none";

        let lastDate = "";
        activeChat.messages.forEach((msg, idx) => {
            const dateStr = renderer.formatDateHeader(msg.timestamp);
            if (dateStr !== lastDate) {
                lastDate = dateStr;
                const dateBadge = document.createElement("div");
                dateBadge.className = "date-divider-badge";
                dateBadge.innerHTML = `<span>${dateStr}</span>`;
                messagesContainer.appendChild(dateBadge);
            }

            // Gửi lại câu hỏi khi bấm Retry
            const onRetry = () => {
                // Tìm câu hỏi của user gần nhất trước tin nhắn bot này
                let queryToRetry = lastUserQueryText;
                for (let i = idx - 1; i >= 0; i--) {
                    if (activeChat.messages[i].sender === "user") {
                        queryToRetry = activeChat.messages[i].text;
                        break;
                    }
                }
                if (queryToRetry) {
                    handleSendMessage(queryToRetry);
                }
            };

            const msgEl = renderer.renderMessage(msg, (source) => {
                openSourceModal(source);
            }, onRetry);

            messagesContainer.appendChild(msgEl);
        });

        scrollToBottom();
    }

    function scrollToBottom(smooth = true) {
        setTimeout(() => {
            if (chatWindow) {
                chatWindow.scrollTo({
                    top: chatWindow.scrollHeight,
                    behavior: smooth ? "smooth" : "auto"
                });
            }
        }, 50);
    }

    // Scroll To Bottom Floating Button (Telegram Web interaction)
    if (chatWindow && btnScrollBottom) {
        chatWindow.addEventListener("scroll", () => {
            const distanceToBottom = chatWindow.scrollHeight - chatWindow.scrollTop - chatWindow.clientHeight;
            if (distanceToBottom > 160) {
                btnScrollBottom.classList.add("show");
            } else {
                btnScrollBottom.classList.remove("show");
            }
        });

        btnScrollBottom.addEventListener("click", () => {
            scrollToBottom(true);
        });
    }

    // Custom Rename Modal Helpers
    function openRenameModal(initialTitle, onConfirm) {
        if (!renameModal || !inputRenameChat) return;
        currentRenameCallback = onConfirm;
        inputRenameChat.value = initialTitle || "";
        renameModal.classList.add("show");
        setTimeout(() => {
            inputRenameChat.focus();
            inputRenameChat.select();
        }, 100);
    }

    function closeRenameModal() {
        if (renameModal) renameModal.classList.remove("show");
        currentRenameCallback = null;
    }

    function handleSaveRename() {
        if (!inputRenameChat) return;
        const newTitle = inputRenameChat.value.trim();
        if (newTitle && typeof currentRenameCallback === "function") {
            currentRenameCallback(newTitle);
        }
        closeRenameModal();
    }

    if (btnCloseRenameModal) btnCloseRenameModal.addEventListener("click", closeRenameModal);
    if (btnCancelRename) btnCancelRename.addEventListener("click", closeRenameModal);
    if (btnConfirmRename) btnConfirmRename.addEventListener("click", handleSaveRename);
    if (inputRenameChat) {
        inputRenameChat.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleSaveRename();
            } else if (e.key === "Escape") {
                closeRenameModal();
            }
        });
    }
    if (renameModal) {
        renameModal.addEventListener("click", (e) => {
            if (e.target === renameModal) closeRenameModal();
        });
    }

    // Custom Confirm Modal Helpers
    function openConfirmModal({ title = "Xác nhận", message, icon = "⚠️", confirmText = "Tiếp tục", isDanger = true, onConfirm }) {
        if (!confirmModal) return;
        currentConfirmCallback = onConfirm;
        if (confirmModalTitle) confirmModalTitle.textContent = title;
        if (confirmModalMessage) confirmModalMessage.textContent = message;
        if (confirmModalIcon) confirmModalIcon.textContent = icon;
        if (btnProceedConfirm) {
            btnProceedConfirm.textContent = confirmText;
            btnProceedConfirm.className = isDanger ? "btn-modal-danger" : "btn-modal-primary";
        }
        confirmModal.classList.add("show");
        if (btnProceedConfirm) btnProceedConfirm.focus();
    }

    function closeConfirmModal() {
        if (confirmModal) confirmModal.classList.remove("show");
        currentConfirmCallback = null;
    }

    if (btnCloseConfirmModal) btnCloseConfirmModal.addEventListener("click", closeConfirmModal);
    if (btnCancelConfirm) btnCancelConfirm.addEventListener("click", closeConfirmModal);
    if (btnProceedConfirm) {
        btnProceedConfirm.addEventListener("click", () => {
            if (typeof currentConfirmCallback === "function") {
                currentConfirmCallback();
            }
            closeConfirmModal();
        });
    }
    if (confirmModal) {
        confirmModal.addEventListener("click", (e) => {
            if (e.target === confirmModal) closeConfirmModal();
        });
    }

    // Modal trích dẫn nguồn văn bản quy chế
    function openSourceModal(source) {
        if (!sourceModal) return;
        const meta = source.metadata || {};
        const title = meta.title || "Văn bản quy chế đào tạo UET";
        const score = source.similarity_score ? (source.similarity_score * 100).toFixed(1) + "%" : "Độ tin cậy cao";

        modalSourceTitle.textContent = title;
        modalSourceMeta.innerHTML = `
            <span class="meta-tag">Độ liên quan: ${score}</span>
            <span class="meta-tag">Mã chunk: ${source.chunk_id || "N/A"}</span>
            ${meta.page ? `<span class="meta-tag">Trang tham chiếu: ${meta.page}</span>` : ""}
            ${meta.source ? `<span class="meta-tag">Tài liệu: ${meta.source}</span>` : ""}
        `;
        modalSourceContent.textContent = source.text || "Không có trích đoạn nội dung.";

        sourceModal.classList.add("show");
    }

    if (btnCloseModal) btnCloseModal.addEventListener("click", () => sourceModal.classList.remove("show"));
    if (btnModalDone) btnModalDone.addEventListener("click", () => sourceModal.classList.remove("show"));
    if (sourceModal) {
        sourceModal.addEventListener("click", (e) => {
            if (e.target === sourceModal) sourceModal.classList.remove("show");
        });
    }

    // 9. PHẦN 3: MESSAGE INPUT & FILE ATTACHMENTS (.pdf, .docx, .txt)
    function renderAttachmentPreviews() {
        if (!attachmentPreviewArea) return;
        attachmentPreviewArea.innerHTML = "";

        if (selectedFiles.length === 0) {
            attachmentPreviewArea.style.display = "none";
            return;
        }

        attachmentPreviewArea.style.display = "flex";
        selectedFiles.forEach((file, index) => {
            const ext = file.name.split(".").pop().toLowerCase();
            let iconText = "📄";
            if (ext === "pdf") iconText = "📕";
            else if (ext === "docx" || ext === "doc") iconText = "📘";
            else if (ext === "txt") iconText = "📝";

            const chip = document.createElement("div");
            chip.className = "pre-attach-chip";
            chip.innerHTML = `
                <span class="chip-icon">${iconText}</span>
                <span class="chip-name" title="${renderer.escapeHtml(file.name)}">${renderer.escapeHtml(file.name)}</span>
                <span class="chip-size">(${renderer.formatFileSize(file.size)})</span>
                <button type="button" class="btn-remove-attach" data-index="${index}" title="Gỡ tệp">✕</button>
            `;

            chip.querySelector(".btn-remove-attach").addEventListener("click", (e) => {
                e.stopPropagation();
                selectedFiles.splice(index, 1);
                renderAttachmentPreviews();
                btnSend.disabled = inputMessage.value.trim().length === 0 && selectedFiles.length === 0;
            });

            attachmentPreviewArea.appendChild(chip);
        });
    }

    function addFiles(fileList) {
        const allowedExts = ["pdf", "docx", "doc", "txt"];
        let addedCount = 0;

        for (let i = 0; i < fileList.length; i++) {
            const f = fileList[i];
            const ext = f.name.split(".").pop().toLowerCase();
            if (allowedExts.includes(ext)) {
                // Giới hạn 10MB
                if (f.size > 10 * 1024 * 1024) {
                    renderer.showToast(`Tệp "${f.name}" vượt quá giới hạn 10MB`, "error");
                    continue;
                }
                selectedFiles.push(f);
                addedCount++;
            } else {
                renderer.showToast(`Định dạng .${ext} không được hỗ trợ. Vui lòng chọn .pdf, .docx, .txt`, "error");
            }
        }

        if (addedCount > 0) {
            renderAttachmentPreviews();
            btnSend.disabled = false;
            renderer.showToast(`Đã đính kèm ${addedCount} tệp`, "info", 1500);
        }
    }

    if (btnAttach && fileAttachmentInput) {
        btnAttach.addEventListener("click", () => {
            fileAttachmentInput.click();
        });

        fileAttachmentInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                addFiles(e.target.files);
                fileAttachmentInput.value = "";
            }
        });
    }

    // Drag and drop vào khu vực chat
    if (chatMainArea && dragDropOverlay) {
        let dragCounter = 0;

        window.addEventListener("dragenter", (e) => {
            e.preventDefault();
            dragCounter++;
            dragDropOverlay.classList.add("show");
        });

        window.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter <= 0) {
                dragCounter = 0;
                dragDropOverlay.classList.remove("show");
            }
        });

        window.addEventListener("dragover", (e) => {
            e.preventDefault();
        });

        window.addEventListener("drop", (e) => {
            e.preventDefault();
            dragCounter = 0;
            dragDropOverlay.classList.remove("show");
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                addFiles(e.dataTransfer.files);
            }
        });
    }

    // 10. XỬ LÝ GỬI TIN NHẮN (Submit Flow)
    async function handleSendMessage(overrideText = null) {
        const queryText = (overrideText !== null ? overrideText : (inputMessage ? inputMessage.value : "")).trim();
        const filesToSend = [...selectedFiles];

        if (!queryText && filesToSend.length === 0) return;
        if (isGenerating) return;

        let activeChat = storage.getActiveChat();
        if (!activeChat) {
            activeChat = storage.createChat("Tư vấn học vụ UET");
        }

        if (welcomeScreen) {
            welcomeScreen.style.display = "none";
        }
        lastUserQueryText = queryText;

        // Chuẩn bị đính kèm
        const attachmentsData = filesToSend.map((f) => ({
            name: f.name,
            size: f.size,
            type: f.name.split(".").pop().toLowerCase()
        }));

        // Thêm tin nhắn User
        const userMsg = storage.addMessage(activeChat.id, {
            sender: "user",
            text: queryText || (filesToSend.length > 0 ? `[Đã đính kèm ${filesToSend.length} tệp]` : ""),
            timestamp: Date.now(),
            attachments: attachmentsData
        });

        const userEl = renderer.renderMessage(userMsg);
        messagesContainer.appendChild(userEl);
        scrollToBottom();

        // Xóa sạch ô nhập và danh sách file đính kèm
        if (inputMessage) {
            inputMessage.value = "";
            inputMessage.style.height = "auto";
        }
        selectedFiles = [];
        renderAttachmentPreviews();
        if (btnSend) btnSend.disabled = true;
        isGenerating = true;

        if (headerStatus) {
            headerStatus.innerHTML = `<span class="typing-indicator-text">Đang soạn tin nhắn...</span>`;
        }

        const typingBubble = renderer.createTypingBubble();
        messagesContainer.appendChild(typingBubble);
        scrollToBottom();

        renderChatList(inputSearchChats ? inputSearchChats.value : "");

        try {
            // Trường hợp có file đính kèm: báo nhận file rõ ràng, không giả lập
            let effectiveQuery = queryText;
            if (filesToSend.length > 0 && !queryText) {
                effectiveQuery = `Tôi đã đính kèm tài liệu: ${filesToSend.map(f => f.name).join(", ")}. Hãy giới thiệu các quy chế liên quan.`;
            }

            const result = await api.sendQuery(effectiveQuery, activeChat.id);
            typingBubble.remove();

            let finalAnswer = result.answer;
            if (filesToSend.length > 0) {
                finalAnswer = `📎 *Đã nhận ${filesToSend.length} tệp đính kèm: ${filesToSend.map(f => f.name).join(", ")}.*\n*(Lưu ý: Tính năng trích xuất nội dung file người dùng tải lên đang được tích hợp vào pipeline backend. Dưới đây là câu trả lời đối chiếu với kho quy chế chính thức của UET:)*\n\n` + finalAnswer;
            }

            const botMsg = storage.addMessage(activeChat.id, {
                sender: "bot",
                text: finalAnswer,
                timestamp: Date.now(),
                sources: result.sources,
                latency: result.latency_seconds,
                model: result.model_name
            });

            const onRetry = () => handleSendMessage(lastUserQueryText);
            const botEl = renderer.renderMessage(botMsg, (source) => openSourceModal(source), onRetry);
            messagesContainer.appendChild(botEl);
            scrollToBottom();

        } catch (err) {
            typingBubble.remove();
            console.error("Lỗi gửi tin nhắn:", err);

            const errMsg = storage.addMessage(activeChat.id, {
                sender: "bot",
                text: `⚠️ **Không thể kết nối đến AI Assistant**: ${err.message}\n\n*Gợi ý: Kiểm tra lại máy chủ backend hoặc nhấn nút **Gửi lại** phía dưới để thử lại.*`,
                timestamp: Date.now(),
                error: true
            });

            const onRetry = () => handleSendMessage(lastUserQueryText);
            const errEl = renderer.renderMessage(errMsg, null, onRetry);
            messagesContainer.appendChild(errEl);
            scrollToBottom();

            renderer.showToast("Không nhận được phản hồi từ AI Assistant.", "error");
        } finally {
            isGenerating = false;
            if (btnSend && inputMessage) {
                btnSend.disabled = (inputMessage.value.trim().length === 0 && selectedFiles.length === 0);
            }
            checkServerStatus();
            renderChatList(inputSearchChats ? inputSearchChats.value : "");
            if (inputMessage) inputMessage.focus();
        }
    }

    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            handleSendMessage();
        });
    }

    if (inputMessage) {
        inputMessage.addEventListener("input", () => {
            inputMessage.style.height = "auto";
            inputMessage.style.height = Math.max(28, Math.min(inputMessage.scrollHeight, 180)) + "px";
            if (btnSend) {
                btnSend.disabled = ((inputMessage.value.trim().length === 0 && selectedFiles.length === 0) || isGenerating);
            }
        });

        inputMessage.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (btnSend && !btnSend.disabled && !isGenerating) {
                    handleSendMessage();
                }
            }
        });
    }

    // Nút New Chat
    if (btnNewChat) {
        btnNewChat.addEventListener("click", () => {
            const activeChat = storage.getActiveChat();
            if (activeChat && (!activeChat.messages || activeChat.messages.length === 0)) {
                renderActiveChatMessages();
                renderer.showToast("Bạn đang ở cuộc trò chuyện mới!", "info", 1500);
                if (inputMessage) inputMessage.focus();
                closeSidebar();
                return;
            }

            const newChat = storage.createChat("Cuộc trò chuyện mới");
            renderChatList();
            renderActiveChatMessages();
            renderer.showToast("Đã tạo cuộc trò chuyện mới!", "success", 1500);
            if (inputMessage) inputMessage.focus();
            closeSidebar();
        });
    }

    // Tìm kiếm Sidebar
    if (inputSearchChats) {
        inputSearchChats.addEventListener("input", (e) => {
            const val = e.target.value.trim();
            if (btnClearSearch) {
                btnClearSearch.style.display = val.length > 0 ? "block" : "none";
            }
            renderChatList(val);
        });
    }

    if (btnClearSearch) {
        btnClearSearch.addEventListener("click", () => {
            inputSearchChats.value = "";
            btnClearSearch.style.display = "none";
            renderChatList("");
            inputSearchChats.focus();
        });
    }

    // 11. PHẦN 4: THẺ CHỨC NĂNG WELCOME SCREEN (6 Cards) - Event Delegation
    document.addEventListener("click", (e) => {
        const card = e.target.closest(".telegram-welcome-card");
        if (card) {
            e.preventDefault();
            const query = card.getAttribute("data-query");
            if (query) {
                handleSendMessage(query);
            }
        }
    });

    // 12. Emoji Picker Telegram-style
    const EMOJI_LIST = ["👋", "🎓", "📚", "📅", "🏆", "⭐", "⚠️", "💡", "✅", "❌", "🔍", "📖", "📝", "🏛️", "🔥", "👍", "❤️", "🎯", "🚀", "🤖"];
    function initEmojiPicker() {
        if (!emojiPicker) return;
        emojiPicker.innerHTML = "";
        EMOJI_LIST.forEach((emo) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "emoji-cell";
            btn.textContent = emo;
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                insertAtCursor(inputMessage, emo);
                inputMessage.dispatchEvent(new Event("input"));
                emojiPicker.classList.remove("show");
            });
            emojiPicker.appendChild(btn);
        });
    }
    initEmojiPicker();

    if (btnEmoji && emojiPicker) {
        btnEmoji.addEventListener("click", (e) => {
            e.stopPropagation();
            emojiPicker.classList.toggle("show");
        });
    }

    function insertAtCursor(myField, myValue) {
        if (document.selection) {
            myField.focus();
            const sel = document.selection.createRange();
            sel.text = myValue;
        } else if (myField.selectionStart || myField.selectionStart === 0) {
            const startPos = myField.selectionStart;
            const endPos = myField.selectionEnd;
            myField.value = myField.value.substring(0, startPos) + myValue + myField.value.substring(endPos, myField.value.length);
            myField.selectionStart = startPos + myValue.length;
            myField.selectionEnd = startPos + myValue.length;
        } else {
            myField.value += myValue;
        }
        myField.focus();
    }

    // 13. Header Menu 3 chấm
    if (btnHeaderMenu && headerDropdownMenu) {
        btnHeaderMenu.addEventListener("click", (e) => {
            e.stopPropagation();
            headerDropdownMenu.classList.toggle("show");
        });

        // Ghim từ Header
        if (headerOptPin) {
            headerOptPin.addEventListener("click", () => {
                headerDropdownMenu.classList.remove("show");
                const activeChat = storage.getActiveChat();
                if (!activeChat) return;
                const isNowPinned = storage.togglePinChat(activeChat.id);
                renderChatList(inputSearchChats ? inputSearchChats.value : "");
                renderActiveChatMessages();
                renderer.showToast(isNowPinned ? "Đã ghim cuộc trò chuyện" : "Đã bỏ ghim", "info");
            });
        }

        // Đổi tên từ Header
        const optRenameHeader = document.getElementById("header-opt-rename");
        if (optRenameHeader) {
            optRenameHeader.addEventListener("click", () => {
                headerDropdownMenu.classList.remove("show");
                const activeChat = storage.getActiveChat();
                if (!activeChat) return;

                openRenameModal(activeChat.title, (newTitle) => {
                    storage.renameChat(activeChat.id, newTitle);
                    renderChatList(inputSearchChats ? inputSearchChats.value : "");
                    if (headerTitle) headerTitle.textContent = newTitle;
                    renderer.showToast("Đã đổi tên cuộc trò chuyện!", "success");
                });
            });
        }

        // Xóa toàn bộ tin nhắn trong hội thoại
        const optClearHeader = document.getElementById("header-opt-clear");
        if (optClearHeader) {
            optClearHeader.addEventListener("click", () => {
                headerDropdownMenu.classList.remove("show");
                const activeChat = storage.getActiveChat();
                if (!activeChat) return;

                openConfirmModal({
                    title: "Xóa toàn bộ tin nhắn",
                    message: "Bạn có chắc chắn muốn xóa toàn bộ tin nhắn trong cuộc trò chuyện này không? Cuộc trò chuyện vẫn được giữ lại trong danh sách, chỉ nội dung các tin nhắn sẽ bị xóa.",
                    icon: "🧹",
                    confirmText: "Xóa tin nhắn",
                    isDanger: true,
                    onConfirm: () => {
                        storage.clearChatMessages(activeChat.id);
                        renderActiveChatMessages();
                        renderChatList(inputSearchChats ? inputSearchChats.value : "");
                        renderer.showToast("Đã xóa toàn bộ tin nhắn trong hội thoại.", "info");
                    }
                });
            });
        }

        // Xóa cuộc trò chuyện
        const optDeleteHeader = document.getElementById("header-opt-delete");
        if (optDeleteHeader) {
            optDeleteHeader.addEventListener("click", () => {
                headerDropdownMenu.classList.remove("show");
                const activeChat = storage.getActiveChat();
                if (!activeChat) return;

                openConfirmModal({
                    title: "Xóa vĩnh viễn cuộc trò chuyện",
                    message: `CẢNH BÁO: Bạn có chắc chắn muốn xóa vĩnh viễn cuộc trò chuyện "${activeChat.title}"? Thao tác này không thể hoàn tác.`,
                    icon: "🗑️",
                    confirmText: "Xóa vĩnh viễn",
                    isDanger: true,
                    onConfirm: () => {
                        storage.deleteChat(activeChat.id);
                        renderChatList(inputSearchChats ? inputSearchChats.value : "");
                        renderActiveChatMessages();
                        renderer.showToast("Đã xóa cuộc trò chuyện thành công.", "info");
                    }
                });
            });
        }
    }

    // 14. Settings Modal
    function openSettings() {
        if (!settingsModal) return;
        checkServerStatus();
        updateActiveThemeButtons(storage.getTheme());
        settingsModal.classList.add("show");
    }

    if (btnOpenSettings) btnOpenSettings.addEventListener("click", openSettings);
    if (btnHeaderSettings) btnHeaderSettings.addEventListener("click", openSettings);
    if (btnCloseSettingsModal) btnCloseSettingsModal.addEventListener("click", () => settingsModal.classList.remove("show"));
    if (btnSettingsDone) btnSettingsDone.addEventListener("click", () => settingsModal.classList.remove("show"));
    if (settingsModal) {
        settingsModal.addEventListener("click", (e) => {
            if (e.target === settingsModal) settingsModal.classList.remove("show");
        });
    }

    // Theme selector inside settings modal
    document.querySelectorAll(".btn-theme-choice").forEach((btn) => {
        btn.addEventListener("click", () => {
            const t = btn.getAttribute("data-theme");
            storage.setTheme(t);
            document.documentElement.setAttribute("data-theme", t);
            updateThemeIcon(t);
            renderer.showToast(`Đã chuyển sang giao diện ${t === "dark" ? "Tối" : "Sáng"}`, "info", 1500);
        });
    });

    // Wallpaper selector inside settings modal
    document.querySelectorAll(".btn-wallpaper-choice").forEach((btn) => {
        btn.addEventListener("click", () => {
            const w = btn.getAttribute("data-wallpaper");
            storage.setWallpaper(w);
            document.documentElement.setAttribute("data-wallpaper", w);
            updateActiveWallpaperButtons(w);
            const nameMap = {
                "telegram-green": "Telegram Pastel Green",
                "telegram-blue": "UET AI Doodle (Telegram Desktop)",
                "dark-night": "Starry Night (Doodle cute)",
                "plain": "Tối giản (#F1F6FC Nền trơn)"
            };
            renderer.showToast(`Đã đổi hình nền: ${nameMap[w] || w}`, "info", 1500);
        });
    });

    // Clear all chats
    if (btnClearAllChats) {
        btnClearAllChats.addEventListener("click", () => {
            openConfirmModal({
                title: "Xóa toàn bộ lịch sử trò chuyện",
                message: "CẢNH BÁO NGUY HIỂM: Thao tác này sẽ xóa vĩnh viễn TOÀN BỘ các cuộc trò chuyện trên trình duyệt của bạn. Bạn có chắc chắn muốn tiếp tục?",
                icon: "⚠️",
                confirmText: "Xóa tất cả",
                isDanger: true,
                onConfirm: () => {
                    localStorage.removeItem("uet_ai_telegram_conversations_v3");
                    localStorage.removeItem("uet_ai_active_chat_id_v3");
                    localStorage.removeItem("uet_ai_telegram_conversations_v2");
                    localStorage.removeItem("uet_ai_active_chat_id_v2");
                    location.reload();
                }
            });
        });
    }

    // 15. In-chat Search (Tìm kiếm từ khóa trong tin nhắn với Next / Prev & đếm kết quả)
    let inChatSearchResults = [];
    let currentSearchIndex = -1;

    function openInChatSearch() {
        if (!inChatSearchBar) return;
        inChatSearchBar.classList.add("show");
        if (inChatSearchInput) {
            inChatSearchInput.focus();
            inChatSearchInput.select();
            if (inChatSearchInput.value.trim()) {
                performInChatSearch(inChatSearchInput.value.trim());
            }
        }
    }

    function closeInChatSearch() {
        if (!inChatSearchBar) return;
        inChatSearchBar.classList.remove("show");
        clearInChatHighlights();
        inChatSearchResults = [];
        currentSearchIndex = -1;
        if (inChatSearchCount) inChatSearchCount.textContent = "0/0";
    }

    if (btnHeaderSearch) {
        btnHeaderSearch.addEventListener("click", () => {
            if (inChatSearchBar.classList.contains("show")) {
                closeInChatSearch();
            } else {
                openInChatSearch();
            }
        });
    }

    if (btnCloseInChatSearch) {
        btnCloseInChatSearch.addEventListener("click", closeInChatSearch);
    }

    function clearInChatHighlights() {
        document.querySelectorAll("mark.search-highlight").forEach((el) => {
            const parent = el.parentNode;
            if (parent) {
                parent.replaceChild(document.createTextNode(el.textContent), el);
                parent.normalize();
            }
        });
    }

    function performInChatSearch(query) {
        clearInChatHighlights();
        inChatSearchResults = [];
        currentSearchIndex = -1;

        if (!query || !query.trim()) {
            if (inChatSearchCount) inChatSearchCount.textContent = "0/0";
            return;
        }

        const q = query.trim().toLowerCase();
        const textElements = messagesContainer.querySelectorAll(".message-bubble .message-text");

        textElements.forEach((el) => {
            const innerHtml = el.innerHTML;
            const plainText = el.innerText || "";
            if (!plainText.toLowerCase().includes(q)) return;

            // Highlight các cụm text khớp mà không làm hỏng thẻ HTML
            const regex = new RegExp(`(${renderer.escapeRegex(query.trim())})`, "gi");
            el.innerHTML = innerHtml.replace(regex, `<mark class="search-highlight">$1</mark>`);
        });

        inChatSearchResults = Array.from(messagesContainer.querySelectorAll("mark.search-highlight"));

        if (inChatSearchResults.length > 0) {
            currentSearchIndex = 0;
            focusSearchResult(currentSearchIndex);
        } else {
            if (inChatSearchCount) inChatSearchCount.textContent = "0/0";
        }
    }

    function focusSearchResult(index) {
        if (inChatSearchResults.length === 0 || index < 0 || index >= inChatSearchResults.length) return;

        inChatSearchResults.forEach((mark, i) => {
            if (i === index) {
                mark.classList.add("search-highlight-active");
                mark.scrollIntoView({ behavior: "smooth", block: "center" });
            } else {
                mark.classList.remove("search-highlight-active");
            }
        });

        if (inChatSearchCount) {
            inChatSearchCount.textContent = `${index + 1}/${inChatSearchResults.length}`;
        }
    }

    function nextSearchResult() {
        if (inChatSearchResults.length === 0) return;
        currentSearchIndex = (currentSearchIndex + 1) % inChatSearchResults.length;
        focusSearchResult(currentSearchIndex);
    }

    function prevSearchResult() {
        if (inChatSearchResults.length === 0) return;
        currentSearchIndex = (currentSearchIndex - 1 + inChatSearchResults.length) % inChatSearchResults.length;
        focusSearchResult(currentSearchIndex);
    }

    if (inChatSearchInput) {
        inChatSearchInput.addEventListener("input", (e) => {
            performInChatSearch(e.target.value);
        });

        inChatSearchInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                if (e.shiftKey) {
                    prevSearchResult();
                } else {
                    nextSearchResult();
                }
            } else if (e.key === "Escape") {
                closeInChatSearch();
            }
        });
    }

    if (btnSearchNext) {
        btnSearchNext.addEventListener("click", nextSearchResult);
    }

    if (btnSearchPrev) {
        btnSearchPrev.addEventListener("click", prevSearchResult);
    }

    // Phím tắt Ctrl + F mở tìm kiếm trong chat
    window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "f") {
            e.preventDefault();
            openInChatSearch();
        } else if (e.key === "Escape") {
            if (settingsModal && settingsModal.classList.contains("show")) {
                settingsModal.classList.remove("show");
            }
            if (sourceModal && sourceModal.classList.contains("show")) {
                sourceModal.classList.remove("show");
            }
            if (renameModal && renameModal.classList.contains("show")) {
                closeRenameModal();
            }
            if (confirmModal && confirmModal.classList.contains("show")) {
                closeConfirmModal();
            }
            if (inChatSearchBar && inChatSearchBar.classList.contains("show")) {
                closeInChatSearch();
            }
            hideContextMenu();
            if (headerDropdownMenu) headerDropdownMenu.classList.remove("show");
            if (emojiPicker) emojiPicker.classList.remove("show");
        }
    });

    // Khởi chạy ban đầu
    try {
        renderChatList();
        renderActiveChatMessages();
    } catch (initErr) {
        console.error("Lỗi khởi chạy UET AI Assistant:", initErr);
    }
});

/**
 * UET Chatbot Web Application Logic
 * Kết nối API Backend, render chat bubbles, markdown & source modal.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatMessages = document.getElementById("chat-messages");
    const welcomeScreen = document.getElementById("welcome-screen");
    const queryInput = document.getElementById("query-input");
    const chatForm = document.getElementById("chat-form");
    const btnSubmit = document.getElementById("btn-submit");
    const btnNewChat = document.getElementById("btn-new-chat");
    const btnToggleSidebar = document.getElementById("btn-toggle-sidebar");
    const sidebar = document.getElementById("sidebar");

    // Modal Elements
    const sourceModal = document.getElementById("source-modal");
    const modalTitle = document.getElementById("modal-source-title");
    const modalContent = document.getElementById("modal-source-content");
    const modalMeta = document.getElementById("modal-source-meta");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnModalDone = document.getElementById("btn-modal-done");

    let isGenerating = false;
    let sessionId = "session_" + Math.random().toString(36).substring(2, 9);

    // Auto-resize textarea
    queryInput.addEventListener("input", () => {
        queryInput.style.height = "auto";
        queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
        btnSubmit.disabled = queryInput.value.trim().length === 0 || isGenerating;
    });

    // Enter to submit
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!btnSubmit.disabled) {
                chatForm.dispatchEvent(new Event("submit"));
            }
        }
    });

    // Toggle Sidebar
    btnToggleSidebar.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
    });

    // New Chat
    btnNewChat.addEventListener("click", () => {
        chatMessages.innerHTML = "";
        chatMessages.appendChild(welcomeScreen);
        welcomeScreen.style.display = "block";
        sessionId = "session_" + Math.random().toString(36).substring(2, 9);
        queryInput.value = "";
        queryInput.style.height = "auto";
        btnSubmit.disabled = true;
        queryInput.focus();
    });

    // Prompt Cards Click
    document.querySelectorAll(".prompt-card, .topic-btn").forEach((card) => {
        card.addEventListener("click", () => {
            const query = card.getAttribute("data-query");
            if (query) {
                queryInput.value = query;
                queryInput.style.height = "auto";
                btnSubmit.disabled = false;
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    // Submit handler
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (!text || isGenerating) return;

        // Hide welcome screen
        if (welcomeScreen.parentNode) {
            welcomeScreen.style.display = "none";
        }

        // Add user message
        appendUserMessage(text);
        queryInput.value = "";
        queryInput.style.height = "auto";
        btnSubmit.disabled = true;
        isGenerating = true;

        // Add typing indicator
        const typingEl = appendTypingIndicator();
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: text,
                    session_id: sessionId,
                    top_k: 4,
                }),
            });

            const data = await response.json();
            typingEl.remove();

            if (response.ok) {
                appendBotMessage(data.answer, data.sources || [], data.latency_seconds);
            } else {
                appendBotMessage(
                    `⚠️ **Lỗi hệ thống**: ${data.detail || "Không thể xử lý yêu cầu lúc này. Vui lòng thử lại sau."}`,
                    []
                );
            }
        } catch (err) {
            typingEl.remove();
            appendBotMessage(
                "❌ **Lỗi kết nối**: Không thể kết nối tới máy chủ RAG. Vui lòng kiểm tra lại dịch vụ backend.",
                []
            );
        } finally {
            isGenerating = false;
            btnSubmit.disabled = queryInput.value.trim().length === 0;
            chatMessages.scrollTop = chatMessages.scrollHeight;
            queryInput.focus();
        }
    });

    // Render User Message
    function appendUserMessage(text) {
        const item = document.createElement("div");
        item.className = "message-item user";
        item.innerHTML = `
            <div class="message-content-wrapper">
                <div class="message-bubble">${escapeHtml(text)}</div>
            </div>
            <div class="message-avatar">Bạn</div>
        `;
        chatMessages.appendChild(item);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Render Bot Message
    function appendBotMessage(answer, sources, latency) {
        const item = document.createElement("div");
        item.className = "message-item bot";

        let sourcesHtml = "";
        if (sources && sources.length > 0) {
            const chips = sources.map((s, idx) => {
                const title = s.metadata?.title || s.metadata?.source || `Tài liệu #${idx + 1}`;
                const score = s.similarity_score ? (s.similarity_score * 100).toFixed(0) + "%" : "Khớp";
                return `
                    <button class="source-chip" data-index="${idx}">
                        <span>📄 ${escapeHtml(title)}</span>
                        <span class="source-badge">${score}</span>
                    </button>
                `;
            }).join("");

            sourcesHtml = `
                <div class="sources-container">
                    <div class="sources-label">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                        </svg>
                        <span>Căn Cứ Văn Bản Trích Xuất:</span>
                    </div>
                    <div class="sources-grid">${chips}</div>
                </div>
            `;
        }

        const metaText = latency ? `Thời gian phản hồi: ${latency.toFixed(2)}s` : "UET Knowledge Base";

        item.innerHTML = `
            <div class="message-avatar">UET</div>
            <div class="message-content-wrapper">
                <div class="message-bubble">
                    ${formatMarkdown(answer)}
                    ${sourcesHtml}
                </div>
                <div class="message-meta">
                    <span>${metaText}</span>
                </div>
            </div>
        `;

        // Attach event listeners to source chips
        item.querySelectorAll(".source-chip").forEach((chip) => {
            chip.addEventListener("click", () => {
                const idx = parseInt(chip.getAttribute("data-index"), 10);
                const source = sources[idx];
                if (source) {
                    openSourceModal(source);
                }
            });
        });

        chatMessages.appendChild(item);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Typing Indicator
    function appendTypingIndicator() {
        const item = document.createElement("div");
        item.className = "message-item bot typing";
        item.innerHTML = `
            <div class="message-avatar">UET</div>
            <div class="message-content-wrapper">
                <div class="message-bubble typing-bubble">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(item);
        return item;
    }

    // Open Modal
    function openSourceModal(source) {
        modalTitle.textContent = source.metadata?.title || source.metadata?.source || "Tài Liệu Quy Chế Đào Tạo";
        modalContent.textContent = source.text || "Không có nội dung trích xuất.";
        
        const page = source.metadata?.page ? `Trang: ${source.metadata.page} | ` : "";
        const score = source.similarity_score ? `Độ tương đồng: ${(source.similarity_score * 100).toFixed(1)}%` : "";
        modalMeta.textContent = `${page}${score} | Nguồn: ${source.metadata?.file_path || "UET Portal"}`;
        
        sourceModal.classList.add("active");
    }

    function closeModal() {
        sourceModal.classList.remove("active");
    }

    btnCloseModal.addEventListener("click", closeModal);
    btnModalDone.addEventListener("click", closeModal);
    sourceModal.addEventListener("click", (e) => {
        if (e.target === sourceModal) closeModal();
    });

    // Simple Markdown Formatter
    function formatMarkdown(text) {
        if (!text) return "";
        let html = escapeHtml(text);

        // Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

        // Bullet lists
        html = html.replace(/^[•\-\*]\s+(.*)$/gm, "<li>$1</li>");
        html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");

        // Line breaks & Paragraphs
        html = html.split("\n\n").map(p => {
            if (p.startsWith("<ul>") || p.startsWith("<li>")) return p;
            return `<p>${p.replace(/\n/g, "<br>")}</p>`;
        }).join("");

        return html;
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }
});

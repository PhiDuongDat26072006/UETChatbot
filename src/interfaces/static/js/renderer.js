/**
 * src/interfaces/static/js/renderer.js
 * UI Renderer Module - Version 2.0
 * Định dạng tin nhắn, Markdown, LaTeX Math (KaTeX), code blocks,
 * bong bóng chat Telegram, tài liệu đính kèm và khu vực trích dẫn văn bản quy chế.
 */

export class UETChatRenderer {
    constructor() {
        this.sourceModal = null;
        this.toastContainer = null;
        this._initToastContainer();
    }

    _initToastContainer() {
        let container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            container.className = "toast-container";
            document.body.appendChild(container);
        }
        this.toastContainer = container;
    }

    showToast(message, type = "info", duration = 3000) {
        const toast = document.createElement("div");
        toast.className = `toast-item toast-${type}`;

        let iconSvg = "";
        if (type === "success") {
            iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        } else if (type === "error") {
            iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
        } else {
            iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
        }

        toast.innerHTML = `
            <span class="toast-icon">${iconSvg}</span>
            <span class="toast-message">${this.escapeHtml(message)}</span>
        `;
        this.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add("toast-fade-out");
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    escapeHtml(text) {
        if (!text) return "";
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    escapeRegex(string) {
        if (!string) return "";
        return String(string).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    formatTime(timestamp) {
        if (!timestamp) return "";
        const d = new Date(timestamp);
        const hours = String(d.getHours()).padStart(2, "0");
        const minutes = String(d.getMinutes()).padStart(2, "0");
        return `${hours}:${minutes}`;
    }

    formatDateHeader(timestamp) {
        const d = new Date(timestamp);
        const today = new Date();
        if (d.toDateString() === today.toDateString()) {
            return "HÔM NAY";
        }
        const yesterday = new Date(today);
        yesterday.setDate(today.getDate() - 1);
        if (d.toDateString() === yesterday.toDateString()) {
            return "HÔM QUA";
        }
        return `${d.getDate()}/${d.getMonth() + 1}/${d.getFullYear()}`;
    }

    formatFileSize(bytes) {
        if (!bytes || bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    /**
     * Render công thức toán LaTeX (KaTeX)
     */
    renderLatex(mathStr, displayMode = false) {
        if (window.katex && typeof window.katex.renderToString === "function") {
            try {
                return window.katex.renderToString(mathStr, {
                    displayMode: displayMode,
                    throwOnError: false
                });
            } catch (e) {
                console.warn("KaTeX render error:", e);
            }
        }
        // Fallback hiển thị rõ ràng nếu KaTeX chưa tải
        const tag = displayMode ? "div" : "span";
        return `<${tag} class="math-fallback">${this.escapeHtml(mathStr)}</${tag}>`;
    }

    /**
     * Trình phân tích cú pháp Markdown nâng cao hỗ trợ Tiêu đề, Bảng, Danh sách, Code Blocks và LaTeX
     */
    parseMarkdown(markdownText) {
        if (!markdownText) return "";

        let text = markdownText;

        // 1. Block Math: $$...$$
        text = text.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
            return `<div class="katex-block-wrapper">${this.renderLatex(formula.trim(), true)}</div>`;
        });

        // 2. Inline Math: $...$ (không trùng với ký hiệu tiền tệ)
        text = text.replace(/(?<!\$)\$([^\$\n]+?)\$(?!\$)/g, (match, formula) => {
            return this.renderLatex(formula.trim(), false);
        });

        // 3. Code blocks (```language ... ```)
        text = text.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const escapedCode = this.escapeHtml(code.trim());
            const displayLang = (lang || "code").toUpperCase();
            return `
                <div class="code-block-wrapper">
                    <div class="code-block-header">
                        <span class="code-lang">${displayLang}</span>
                        <button class="btn-copy-code" onclick="window.__copyCode(this)" title="Sao chép mã">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                            <span>Copy</span>
                        </button>
                    </div>
                    <pre><code class="language-${lang || 'text'}">${escapedCode}</code></pre>
                </div>
            `;
        });

        // 4. Inline code (`code`)
        text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

        // 5. Headers
        text = text.replace(/^### (.*$)/gim, '<h4 class="md-h3">$1</h4>');
        text = text.replace(/^## (.*$)/gim, '<h3 class="md-h2">$1</h3>');
        text = text.replace(/^# (.*$)/gim, '<h2 class="md-h1">$1</h2>');

        // 6. Blockquotes
        text = text.replace(/^\> (.*$)/gim, '<blockquote class="md-quote">$1</blockquote>');

        // 7. Bold & Italic
        text = text.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // 8. Markdown Tables (Hỗ trợ chuẩn Github, có hoặc không có dấu | ở 2 đầu biên)
        text = text.replace(/((?:^[ \t]*\|?.*?\|.*?\|?[ \t]*(?:\r?\n|$))+)/gm, (match) => {
            const rawRows = match.trim().split(/\r?\n/).map(r => r.trim()).filter(r => r.length > 0);
            if (rawRows.length < 2) return match;

            // Kiểm tra dòng thứ 2 có phải dòng ngăn cách bảng (|---|:---|---:| hoặc ---|---)
            const sepRegex = /^\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?$/;
            let sepIdx = -1;
            if (rawRows.length > 1 && sepRegex.test(rawRows[1])) {
                sepIdx = 1;
            }
            if (sepIdx === -1) return match;

            const splitRow = (rowStr) => {
                let parts = rowStr.split("|").map(p => p.trim());
                if (parts.length > 0 && parts[0] === "") parts.shift();
                if (parts.length > 0 && parts[parts.length - 1] === "") parts.pop();
                return parts;
            };

            const headerCells = splitRow(rawRows[0]);
            if (headerCells.length === 0) return match;

            // Xác định căn lề từ dòng phân cách
            const alignments = splitRow(rawRows[sepIdx]).map(sep => {
                const left = sep.startsWith(":");
                const right = sep.endsWith(":");
                if (left && right) return "center";
                if (right) return "right";
                if (left) return "left";
                return "left";
            });

            let html = '<div class="table-container"><table class="md-table"><thead><tr>';
            headerCells.forEach((cell, idx) => {
                const alignStyle = alignments[idx] ? ` style="text-align: ${alignments[idx]};"` : "";
                html += `<th${alignStyle}>${cell}</th>`;
            });
            html += '</tr></thead><tbody>';

            for (let i = sepIdx + 1; i < rawRows.length; i++) {
                const rowCells = splitRow(rawRows[i]);
                if (rowCells.length > 0) {
                    html += '<tr>';
                    headerCells.forEach((_, idx) => {
                        const cellContent = (rowCells[idx] !== undefined) ? rowCells[idx] : "";
                        const alignStyle = alignments[idx] ? ` style="text-align: ${alignments[idx]};"` : "";
                        html += `<td${alignStyle}>${cellContent}</td>`;
                    });
                    html += '</tr>';
                }
            }
            html += '</tbody></table></div>';
            return html;
        });

        // 9. Lists (Loại bỏ newlines thừa bên trong ul/ol để không sinh tag br rác)
        text = text.replace(/^[\*\-\•]\s+(.*)$/gim, '<li class="md-li-bullet">$1</li>');
        text = text.replace(/(<li class="md-li-bullet">[\s\S]*?<\/li>[\s\n]*)+/g, (m) => `<ul class="md-ul">${m.replace(/\n/g, '')}</ul>`);

        text = text.replace(/^\d+\.\s+(.*)$/gim, '<li class="md-li-num">$1</li>');
        text = text.replace(/(<li class="md-li-num">[\s\S]*?<\/li>[\s\n]*)+/g, (m) => `<ol class="md-ol">${m.replace(/\n/g, '')}</ol>`);

        // 10. Links [text](url)
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$1</a>');

        // 11. Paragraphs & Line breaks (Xử lý mượt mà, dọn dẹp br cạnh các khối block)
        text = text.replace(/\n\n+/g, '<br/><br/>');
        text = text.replace(/\n/g, '<br/>');

        // Dọn sạch br dư thừa ngay trước hoặc ngay sau các thẻ block lớn (bảng, tiêu đề, danh sách, quote, code)
        text = text.replace(/(<\/(?:div|table|blockquote|ul|ol|h2|h3|h4|pre)>)\s*(?:<br\s*\/?>)+/gi, '$1');
        text = text.replace(/(?:<br\s*\/?>)+\s*(<(?:div|table|blockquote|ul|ol|h2|h3|h4|pre))/gi, '$1');

        return text;
    }

    /**
     * Render tin nhắn hoàn chỉnh trong cửa sổ chat
     */
    renderMessage(msg, onOpenSource, onRetry) {
        const item = document.createElement("div");
        const isUser = msg.sender === "user";
        item.className = `message-bubble-row ${isUser ? "user-row" : "bot-row"} message-fade-in`;
        item.id = msg.id;

        const timeStr = this.formatTime(msg.timestamp);

        if (isUser) {
            // Tệp đính kèm (nếu có)
            let attachmentsHtml = "";
            if (msg.attachments && msg.attachments.length > 0) {
                attachmentsHtml = '<div class="message-attachments-preview">';
                msg.attachments.forEach((att) => {
                    const ext = (att.name || "").split(".").pop().toLowerCase();
                    let iconEmoji = "📎";
                    if (ext === "pdf") iconEmoji = "📕";
                    else if (ext === "docx" || ext === "doc") iconEmoji = "📘";
                    else if (ext === "txt") iconEmoji = "📄";

                    attachmentsHtml += `
                        <div class="user-file-badge">
                            <span class="file-badge-icon">${iconEmoji}</span>
                            <div class="file-badge-info">
                                <span class="file-badge-name">${this.escapeHtml(att.name)}</span>
                                <span class="file-badge-size">${this.formatFileSize(att.size)}</span>
                            </div>
                        </div>
                    `;
                });
                attachmentsHtml += '</div>';
            }

            item.innerHTML = `
                <div class="message-bubble user-bubble">
                    ${attachmentsHtml}
                    <div class="message-text">${this.escapeHtml(msg.text)}</div>
                    <div class="message-meta">
                        <span class="message-time">${timeStr}</span>
                        <span class="read-receipt" title="Đã gửi">
                            <svg width="16" height="11" viewBox="0 0 16 11" fill="none">
                                <path d="M1 5.5L5 9.5L15 1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M5 5.5L9 9.5L15 1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </span>
                    </div>
                </div>
            `;
        } else {
            const formattedContent = this.parseMarkdown(msg.text);

            // Khu vực hiển thị tài liệu tham khảo quy chế UET
            let sourcesHtml = "";
            if (msg.sources && msg.sources.length > 0) {
                const totalSources = msg.sources.length;
                const showToggle = totalSources > 3;
                let cardsHtml = "";

                msg.sources.forEach((src, sIdx) => {
                    const title = src.metadata?.title || src.metadata?.source || `Văn bản quy chế #${sIdx + 1}`;
                    const isPdf = title.toLowerCase().includes(".pdf") || (!title.toLowerCase().includes(".doc") && !title.toLowerCase().includes(".txt"));
                    const iconBadge = isPdf
                        ? `<span class="source-format-tag format-pdf">PDF</span>`
                        : `<span class="source-format-tag format-doc">DOC</span>`;
                    const pageStr = src.metadata?.page ? `Trang ${src.metadata.page}` : "";
                    const score = src.similarity_score ? `${(src.similarity_score * 100).toFixed(0)}% liên quan` : "";
                    const metaParts = [pageStr, score].filter(Boolean).join(" • ");
                    const hiddenClass = (showToggle && sIdx >= 3) ? "source-card-hidden" : "";

                    cardsHtml += `
                        <div class="source-doc-card ${hiddenClass}" data-source-index="${sIdx}" role="button" tabindex="0" title="Bấm để xem trích đoạn: ${this.escapeHtml(title)}">
                            <div class="source-doc-icon-col">
                                ${iconBadge}
                            </div>
                            <div class="source-doc-info-col">
                                <div class="source-doc-title">${this.escapeHtml(title)}</div>
                                <div class="source-doc-meta">${this.escapeHtml(metaParts || "Căn cứ quy chế đào tạo UET")}</div>
                            </div>
                            <div class="source-doc-action-col">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <polyline points="9 18 15 12 9 6"></polyline>
                                </svg>
                            </div>
                        </div>
                    `;
                });

                const toggleBtnHtml = showToggle ? `
                    <button type="button" class="btn-toggle-sources" data-expanded="false">
                        <span class="toggle-text">Xem thêm ${totalSources - 3} tài liệu</span>
                        <svg class="toggle-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </button>
                ` : "";

                sourcesHtml = `
                    <div class="sources-reference-section">
                        <div class="sources-ref-header">
                            <div class="sources-ref-title">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                                </svg>
                                <span>Tài liệu tham khảo</span>
                                <span class="sources-count-badge">${totalSources}</span>
                            </div>
                        </div>
                        <div class="sources-doc-list">
                            ${cardsHtml}
                        </div>
                        ${toggleBtnHtml}
                    </div>
                `;
            }

            // Thanh thao tác tin nhắn AI
            const latencyStr = msg.latency ? `${msg.latency}s` : "";

            item.innerHTML = `
                <div class="bot-avatar-col">
                    <div class="telegram-avatar bot-avatar-mini" title="UET AI Assistant">
                        <span>UET</span>
                    </div>
                </div>
                <div class="message-bubble bot-bubble">
                    <div class="bot-name-tag">UET AI Assistant</div>
                    <div class="message-text markdown-body">${formattedContent}</div>
                    ${sourcesHtml}
                    <div class="message-meta-footer">
                        <div class="message-actions">
                            <button class="btn-msg-action btn-copy-msg" title="Sao chép câu trả lời">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                            </button>
                            <button class="btn-msg-action btn-like-msg" title="Câu trả lời hữu ích">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                                </svg>
                            </button>
                            <button class="btn-msg-action btn-dislike-msg" title="Chưa chính xác hoặc cần hoàn thiện">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
                                </svg>
                            </button>
                            <button class="btn-msg-action btn-retry-msg" title="Tạo lại câu trả lời">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path>
                                    <path d="M21 3v5h-5"></path>
                                    <path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path>
                                    <path d="M3 21v-5h5"></path>
                                </svg>
                            </button>
                            ${latencyStr ? `<span class="badge-latency">${latencyStr}</span>` : ""}
                        </div>
                        <span class="message-time">${timeStr}</span>
                    </div>
                </div>
            `;

            // Bấm xem chi tiết nguồn quy chế
            if (msg.sources && msg.sources.length > 0 && typeof onOpenSource === "function") {
                item.querySelectorAll(".source-doc-card, .source-chip").forEach((card) => {
                    const clickHandler = () => {
                        const idx = parseInt(card.getAttribute("data-source-index"), 10);
                        if (!isNaN(idx) && msg.sources[idx]) {
                            onOpenSource(msg.sources[idx]);
                        }
                    };
                    card.addEventListener("click", clickHandler);
                    card.addEventListener("keydown", (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            clickHandler();
                        }
                    });
                });

                // Xử lý nút thu gọn / mở rộng nguồn
                const toggleSourcesBtn = item.querySelector(".btn-toggle-sources");
                if (toggleSourcesBtn) {
                    toggleSourcesBtn.addEventListener("click", () => {
                        const isExpanded = toggleSourcesBtn.getAttribute("data-expanded") === "true";
                        const nextState = !isExpanded;
                        toggleSourcesBtn.setAttribute("data-expanded", String(nextState));
                        item.querySelectorAll(".source-doc-card.source-card-hidden").forEach((c) => {
                            c.style.display = nextState ? "flex" : "none";
                        });
                        const textSpan = toggleSourcesBtn.querySelector(".toggle-text");
                        if (textSpan) {
                            textSpan.textContent = nextState ? "Thu gọn danh sách" : `Xem thêm ${msg.sources.length - 3} tài liệu`;
                        }
                        const icon = toggleSourcesBtn.querySelector(".toggle-icon");
                        if (icon) {
                            icon.style.transform = nextState ? "rotate(180deg)" : "none";
                        }
                    });
                }
            }

            // Copy câu trả lời
            const copyBtn = item.querySelector(".btn-copy-msg");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    navigator.clipboard.writeText(msg.text).then(() => {
                        this.showToast("Đã sao chép nội dung vào khay nhớ tạm!", "success");
                    });
                });
            }

            // Gửi lại / Thử lại
            const retryBtn = item.querySelector(".btn-retry-msg");
            if (retryBtn && typeof onRetry === "function") {
                retryBtn.addEventListener("click", () => {
                    onRetry();
                });
            }

            // Like / Dislike
            const likeBtn = item.querySelector(".btn-like-msg");
            const dislikeBtn = item.querySelector(".btn-dislike-msg");
            if (likeBtn && dislikeBtn) {
                likeBtn.addEventListener("click", () => {
                    likeBtn.classList.toggle("active");
                    dislikeBtn.classList.remove("active");
                    if (likeBtn.classList.contains("active")) {
                        this.showToast("Cảm ơn đánh giá tích cực của bạn!", "success");
                    }
                });
                dislikeBtn.addEventListener("click", () => {
                    dislikeBtn.classList.toggle("active");
                    likeBtn.classList.remove("active");
                    if (dislikeBtn.classList.contains("active")) {
                        this.showToast("Đã ghi nhận phản hồi để cải thiện dữ liệu quy chế.", "info");
                    }
                });
            }
        }

        return item;
    }

    createTypingBubble() {
        const item = document.createElement("div");
        item.className = "message-bubble-row bot-row typing-indicator-row message-fade-in";
        item.id = "typing-bubble-indicator";
        item.innerHTML = `
            <div class="bot-avatar-col">
                <div class="telegram-avatar bot-avatar-mini">
                    <span>UET</span>
                </div>
            </div>
            <div class="message-bubble bot-bubble typing-bubble">
                <div class="telegram-typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        return item;
    }
}

// Khai báo hàm copy code toàn cục
window.__copyCode = function(button) {
    const wrapper = button.closest(".code-block-wrapper");
    if (!wrapper) return;
    const codeEl = wrapper.querySelector("code");
    if (!codeEl) return;
    navigator.clipboard.writeText(codeEl.innerText).then(() => {
        const span = button.querySelector("span");
        if (span) {
            const old = span.innerText;
            span.innerText = "Copied!";
            button.classList.add("copied");
            setTimeout(() => {
                span.innerText = old;
                button.classList.remove("copied");
            }, 2000);
        }
    });
};

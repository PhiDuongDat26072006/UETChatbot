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
        this._markedConfigured = false;
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

    _initMarked() {
        if (!window.marked || this._markedConfigured) return;
        try {
            const renderer = new window.marked.Renderer();

            // 1. Tables: Bao bọc trong .table-container với lớp .md-table chuẩn card
            renderer.table = (header, body) => {
                return `<div class="table-container"><table class="md-table"><thead>${header}</thead><tbody>${body}</tbody></table></div>`;
            };

            // 2. Code blocks: Giao diện thẻ đen hiện đại với nút Copy và huy hiệu ngôn ngữ
            renderer.code = (code, language) => {
                const lang = (language || "code").toLowerCase();
                const displayLang = (language || "code").toUpperCase();
                const escapedCode = this.escapeHtml(code);
                return `
                    <div class="code-block-wrapper">
                        <div class="code-block-header">
                            <span class="code-lang">${displayLang}</span>
                            <button class="btn-copy-code" onclick="window.__copyCode(this)" title="Sao chép mã">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                                <span>Copy</span>
                            </button>
                        </div>
                        <pre><code class="language-${lang}">${escapedCode}</code></pre>
                    </div>
                `;
            };

            // 3. Inline code
            renderer.codespan = (code) => {
                return `<code class="inline-code">${this.escapeHtml(code)}</code>`;
            };

            // 4. Blockquote
            renderer.blockquote = (quote) => {
                return `<blockquote class="md-quote">${quote}</blockquote>`;
            };

            // 5. Links: Mở tab mới an toàn
            renderer.link = (href, title, text) => {
                const titleAttr = title ? ` title="${this.escapeHtml(title)}"` : "";
                return `<a href="${href}" target="_blank" rel="noopener noreferrer" class="md-link"${titleAttr}>${text}</a>`;
            };

            // 6. Headings: Phân cấp h2, h3, h4
            renderer.heading = (text, level) => {
                const hTag = (level === 1) ? 'h2' : (level === 2) ? 'h3' : 'h4';
                const cls = `md-h${Math.min(level, 3)}`;
                return `<${hTag} class="${cls}">${text}</${hTag}>`;
            };

            // 7. Horizontal Rule
            renderer.hr = () => {
                return '<hr class="md-hr"/>';
            };

            // 8. Lists & List items
            renderer.list = (body, ordered, start) => {
                const tag = ordered ? 'ol' : 'ul';
                const cls = ordered ? 'md-ol' : 'md-ul';
                const startAttr = (ordered && start !== 1) ? ` start="${start}"` : '';
                return `<${tag} class="${cls}"${startAttr}>${body}</${tag}>`;
            };

            renderer.listitem = (text) => {
                return `<li>${text}</li>`;
            };

            window.marked.setOptions({
                renderer: renderer,
                gfm: true,
                breaks: true,
                pedantic: false
            });

            this._markedConfigured = true;
        } catch (e) {
            console.warn("Lỗi cấu hình marked.js:", e);
        }
    }

    /**
     * Trình phân tích cú pháp Markdown chuẩn công nghiệp kết hợp KaTeX và GFM.
     * Tự động bảo vệ công thức toán, lọc thẻ pseudo-HTML từ LLM,
     * render bảng biểu, danh sách phân cấp và code blocks hoàn hảo.
     */
    parseMarkdown(markdownText) {
        if (!markdownText) return "";

        this._initMarked();

        let text = String(markdownText);
        const mathTokens = new Map();
        let mathIndex = 0;

        // 1. Trích xuất và bảo vệ công thức toán LaTeX (Block Math: $$...$$ hoặc \[...\])
        text = text.replace(/(?:\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\])/g, (match, f1, f2) => {
            const formula = (f1 !== undefined ? f1 : f2).trim();
            const key = `XXUETMATHBLOCK${mathIndex++}XX`;
            mathTokens.set(key, `<div class="katex-block-wrapper">${this.renderLatex(formula, true)}</div>`);
            return `\n\n${key}\n\n`;
        });

        // 2. Trích xuất và bảo vệ công thức toán LaTeX (Inline Math: $...$ hoặc \(...\))
        // Tránh nhận nhầm giá tiền tệ $100 hoặc ký tự $ đơn độc
        text = text.replace(/(?:(?<![\w\$])\$(?!\s)([^\$\n]+?)(?<!\s)\$(?![\w\$])|\\\(([\s\S]*?)\\\))/g, (match, f1, f2) => {
            const formula = (f1 !== undefined ? f1 : f2).trim();
            const key = `XXUETMATHINLINE${mathIndex++}XX`;
            mathTokens.set(key, this.renderLatex(formula, false));
            return key;
        });

        // 3. Chuẩn hóa các thẻ placeholder từ LLM (Ví dụ: <Phòng 105-E3>, <Chứng chỉ>, <GPA>)
        // Để trình duyệt không hiểu nhầm là thẻ HTML bị ẩn mất nội dung
        text = text.replace(/<(?!\/?(?:b|i|strong|em|code|pre|table|thead|tbody|tr|th|td|ul|ol|li|blockquote|h[1-6]|p|a|span|div|hr|sub|sup|mark|img|br)\b)([^>\n]+)>/g, '&lt;$1&gt;');

        // 4. Nếu thư viện marked.js đã tải thành công, sử dụng marked (Chuẩn GitHub Flavored Markdown 100%)
        if (window.marked && typeof window.marked.parse === "function") {
            try {
                let html = window.marked.parse(text);

                // Khôi phục các token công thức toán LaTeX
                mathTokens.forEach((val, key) => {
                    // Gỡ bỏ thẻ <p> bao quanh công thức khối nếu marked tự bọc
                    html = html.replace(new RegExp(`<p>\\s*${key}\\s*<\\/p>`, 'g'), val);
                    html = html.split(key).join(val);
                });

                return html;
            } catch (err) {
                console.warn("Marked parser warning, switching to fallback:", err);
            }
        }

        // 5. FALLBACK PARSER (Khi không có mạng hoặc chưa tải xong marked.js)
        return this._fallbackParseMarkdown(text, mathTokens);
    }

    /**
     * Fallback Parser xử lý Markdown nội bộ bằng Regex nâng cao
     */
    _fallbackParseMarkdown(text, mathTokens) {
        const tokens = new Map();
        let tokenCount = 0;
        const addToken = (html) => {
            const key = `XXUETTOKEN${tokenCount++}XX`;
            tokens.set(key, html);
            return key;
        };

        const formatInline = (str) => {
            if (!str) return "";
            let s = str;
            s = s.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
            s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            s = s.replace(/(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)/g, '<em>$1</em>');
            s = s.replace(/(?<!_)\_(?!\s)([^_\n]+?)(?<!\s)\_(?!\_)/g, '<em>$1</em>');
            s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$1</a>');
            return s;
        };

        let t = text;

        // Code blocks
        t = t.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const escapedCode = this.escapeHtml(code.trim());
            const displayLang = (lang || "code").toUpperCase();
            const codeBlockHtml = `
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
            return addToken(codeBlockHtml);
        });

        // Inline code
        t = t.replace(/`([^`\n]+)`/g, (match, code) => {
            return addToken(`<code class="inline-code">${this.escapeHtml(code)}</code>`);
        });

        // Markdown Tables
        t = t.replace(/((?:^[ \t]*\|?.*?\|.*?\|?[ \t]*(?:\r?\n|$))+)/gm, (match) => {
            const rawRows = match.trim().split(/\r?\n/).map(r => r.trim()).filter(r => r.length > 0);
            if (rawRows.length < 2) return match;

            const sepRegex = /^\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?$/;
            if (!sepRegex.test(rawRows[1])) return match;

            const splitRow = (rowStr) => {
                let parts = rowStr.split("|").map(p => p.trim());
                if (parts.length > 0 && parts[0] === "") parts.shift();
                if (parts.length > 0 && parts[parts.length - 1] === "") parts.pop();
                return parts;
            };

            const headerCells = splitRow(rawRows[0]);
            if (headerCells.length === 0) return match;

            const alignments = splitRow(rawRows[1]).map(sep => {
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
                html += `<th${alignStyle}>${formatInline(this.escapeHtml(cell))}</th>`;
            });
            html += '</tr></thead><tbody>';

            for (let i = 2; i < rawRows.length; i++) {
                const rowCells = splitRow(rawRows[i]);
                if (rowCells.length > 0) {
                    html += '<tr>';
                    headerCells.forEach((_, idx) => {
                        const cellContent = (rowCells[idx] !== undefined) ? formatInline(this.escapeHtml(rowCells[idx])) : "";
                        const alignStyle = alignments[idx] ? ` style="text-align: ${alignments[idx]};"` : "";
                        html += `<td${alignStyle}>${cellContent}</td>`;
                    });
                    html += '</tr>';
                }
            }
            html += '</tbody></table></div>';
            return addToken(html);
        });

        // Headers
        t = t.replace(/^### (.*$)/gim, '<h4 class="md-h3">$1</h4>');
        t = t.replace(/^## (.*$)/gim, '<h3 class="md-h2">$1</h3>');
        t = t.replace(/^# (.*$)/gim, '<h2 class="md-h1">$1</h2>');

        // Blockquotes
        t = t.replace(/^\> (.*$)/gim, '<blockquote class="md-quote">$1</blockquote>');

        // Lists
        t = t.replace(/((?:^[ \t]*[\*\-\•]\s+[^\n]+(?:\r?\n|$))+)/gm, (match) => {
            const lines = match.trim().split(/\r?\n/).map(l => l.trim()).filter(Boolean);
            const items = lines.map(line => {
                const itemContent = line.replace(/^[\*\-\•]\s+/, '');
                return `<li>${formatInline(itemContent)}</li>`;
            }).join('');
            return addToken(`<ul class="md-ul">${items}</ul>`);
        });

        t = t.replace(/((?:^[ \t]*\d+\.\s+[^\n]+(?:\r?\n|$))+)/gm, (match) => {
            const lines = match.trim().split(/\r?\n/).map(l => l.trim()).filter(Boolean);
            const items = lines.map(line => {
                const itemContent = line.replace(/^\d+\.\s+/, '');
                return `<li>${formatInline(itemContent)}</li>`;
            }).join('');
            return addToken(`<ol class="md-ol">${items}</ol>`);
        });

        // Bold & Italic
        t = formatInline(t);

        // Horizontal rule
        t = t.replace(/^[ \t]*[-*_]{3,}[ \t]*$/gim, '<hr class="md-hr"/>');

        // Line breaks
        t = t.replace(/\n\n+/g, '<br/><br/>');
        t = t.replace(/\n/g, '<br/>');

        // Restore custom tokens
        tokens.forEach((val, key) => {
            t = t.replace(new RegExp(`(?:<br\\s*\\/?>)+\\s*(${key})`, 'g'), '$1');
            t = t.replace(new RegExp(`(${key})\\s*(?:<br\\s*\\/?>)+`, 'g'), '$1');
            t = t.split(key).join(val);
        });

        // Restore math tokens
        mathTokens.forEach((val, key) => {
            t = t.split(key).join(val);
        });

        return t;
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

            // Thanh thao tác tin nhắn AI: Tối giản icon, loại bỏ Like/Dislike rườm rà
            const latencyStr = msg.latency ? `${msg.latency}s` : "";

            let actionsHtml = "";
            if (msg.error) {
                // Tin nhắn lỗi: Chỉ hiện nút Thử lại gọn gàng, không icon rườm rà
                actionsHtml = `
                    <button class="btn-msg-action btn-retry-msg" title="Thử lại câu hỏi">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path>
                            <path d="M21 3v5h-5"></path>
                            <path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path>
                            <path d="M3 21v-5h5"></path>
                        </svg>
                        <span class="action-btn-text">Thử lại</span>
                    </button>
                `;
            } else {
                // Tin nhắn bình thường: Chỉ giữ Copy và Retry
                actionsHtml = `
                    <button class="btn-msg-action btn-copy-msg" title="Sao chép câu trả lời">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                    </button>
                    <button class="btn-msg-action btn-retry-msg" title="Tạo lại câu trả lời">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path>
                            <path d="M21 3v5h-5"></path>
                            <path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path>
                            <path d="M3 21v-5h5"></path>
                        </svg>
                    </button>
                    ${latencyStr ? `<span class="badge-latency">${latencyStr}</span>` : ""}
                `;
            }

            item.innerHTML = `
                <div class="bot-avatar-col">
                    <div class="telegram-avatar bot-avatar-mini" title="UET AI Assistant">
                        <img src="/static/uet-logo.png?v=v14_robot_mascot" alt="UET ChatBot Logo" class="uet-logo-img">
                    </div>
                </div>
                <div class="message-bubble bot-bubble ${msg.error ? 'bot-bubble-error' : ''}">
                    <div class="bot-name-tag">UET AI Assistant</div>
                    <div class="message-text markdown-body">${formattedContent}</div>
                    ${sourcesHtml}
                    <div class="message-meta-footer">
                        <div class="message-actions">
                            ${actionsHtml}
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
        }

        return item;
    }

    /**
     * Render câu trả lời của AI với hiệu ứng Typewriter trượt từng chữ/dòng như ChatGPT/Claude
     * Hỗ trợ tự động cuộn (auto-scroll), click-to-finish, và render bảng Markdown mượt mà.
     */
    renderStreamingBotMessage(msg, options = {}) {
        const {
            container,
            onOpenSource,
            onRetry,
            onScroll,
            onFinish
        } = options;

        const item = document.createElement("div");
        item.className = "message-bubble-row bot-row message-fade-in";
        item.id = msg.id;

        const timeStr = this.formatTime(msg.timestamp);

        // Nguồn tài liệu tham khảo (nếu có)
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
                <div class="sources-reference-section" style="display: none;">
                    <div class="sources-ref-header">
                        <div class="sources-ref-title">
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

        const latencyStr = msg.latency ? `${msg.latency}s` : "";
        let actionsHtml = `
            <button class="btn-msg-action btn-copy-msg" title="Sao chép câu trả lời">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
            </button>
            <button class="btn-msg-action btn-retry-msg" title="Tạo lại câu trả lời">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path>
                    <path d="M21 3v5h-5"></path>
                    <path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path>
                    <path d="M3 21v-5h5"></path>
                </svg>
            </button>
            ${latencyStr ? `<span class="badge-latency">${latencyStr}</span>` : ""}
        `;

        item.innerHTML = `
            <div class="bot-avatar-col">
                <div class="telegram-avatar bot-avatar-mini" title="UET AI Assistant">
                    <img src="/static/uet-logo.png?v=v14_robot_mascot" alt="UET ChatBot Logo" class="uet-logo-img">
                </div>
            </div>
            <div class="message-bubble bot-bubble">
                <div class="bot-name-tag">UET AI Assistant</div>
                <div class="message-text markdown-body streaming-active">
                    <span class="typing-cursor"></span>
                </div>
                ${sourcesHtml}
                <div class="message-meta-footer" style="display: none;">
                    <div class="message-actions">
                        ${actionsHtml}
                    </div>
                    <span class="message-time">${timeStr}</span>
                </div>
            </div>
        `;

        if (container) {
            container.appendChild(item);
        }

        const textEl = item.querySelector(".message-text.markdown-body");
        const sourcesEl = item.querySelector(".sources-reference-section");
        const footerEl = item.querySelector(".message-meta-footer");

        // Hoàn tất hiệu ứng: gắn sự kiện cho các nút
        const finalizeEvents = () => {
            if (sourcesEl) sourcesEl.style.display = "block";
            if (footerEl) footerEl.style.display = "flex";
            textEl.classList.remove("streaming-active");

            // Xử lý xem nguồn
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

            // Copy
            const copyBtn = item.querySelector(".btn-copy-msg");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    navigator.clipboard.writeText(msg.text).then(() => {
                        this.showToast("Đã sao chép nội dung vào khay nhớ tạm!", "success");
                    });
                });
            }

            // Retry
            const retryBtn = item.querySelector(".btn-retry-msg");
            if (retryBtn && typeof onRetry === "function") {
                retryBtn.addEventListener("click", () => {
                    onRetry();
                });
            }
        };

        const fullText = msg.text || "";
        const tokens = fullText.match(/\S+|\s+/g) || [fullText];
        const totalTokens = tokens.length;

        // Nếu nội dung quá ngắn, render ngay lập tức
        if (totalTokens <= 2) {
            textEl.innerHTML = this.parseMarkdown(fullText);
            finalizeEvents();
            if (typeof onFinish === "function") onFinish();
            return { element: item, finishImmediately: () => {} };
        }

        // Tốc độ: Chạy trong 1.0s - 3.2s
        const targetDuration = Math.min(3200, Math.max(1000, totalTokens * 14));
        const frameInterval = 22; // ~45 fps
        const totalFrames = Math.max(1, Math.floor(targetDuration / frameInterval));
        const tokensPerFrame = Math.max(1, Math.ceil(totalTokens / totalFrames));

        let currentIdx = 0;
        let isDone = false;
        let timer = null;

        const completeStream = () => {
            if (isDone) return;
            isDone = true;
            if (timer) clearInterval(timer);
            textEl.innerHTML = this.parseMarkdown(fullText);
            finalizeEvents();
            if (typeof onScroll === "function") onScroll();
            if (typeof onFinish === "function") onFinish();
        };

        // Click để bỏ qua hiệu ứng, hiện ngay toàn bộ
        textEl.style.cursor = "pointer";
        textEl.title = "Bấm để hiện ngay toàn bộ nội dung";
        const skipHandler = () => {
            textEl.style.cursor = "";
            textEl.title = "";
            completeStream();
            textEl.removeEventListener("click", skipHandler);
        };
        textEl.addEventListener("click", skipHandler);

        timer = setInterval(() => {
            if (currentIdx < totalTokens) {
                currentIdx = Math.min(totalTokens, currentIdx + tokensPerFrame);
                const partialText = tokens.slice(0, currentIdx).join("");
                textEl.innerHTML = this.parseMarkdown(partialText) + '<span class="typing-cursor"></span>';

                if (typeof onScroll === "function") {
                    onScroll();
                }
            } else {
                textEl.style.cursor = "";
                textEl.title = "";
                textEl.removeEventListener("click", skipHandler);
                completeStream();
            }
        }, frameInterval);

        return {
            element: item,
            finishImmediately: completeStream
        };
    }

    createTypingBubble() {
        const item = document.createElement("div");
        item.className = "message-bubble-row bot-row typing-indicator-row message-fade-in";
        item.id = "typing-bubble-indicator";
        item.innerHTML = `
            <div class="bot-avatar-col">
                <div class="telegram-avatar bot-avatar-mini">
                    <img src="/static/uet-logo.png?v=v14_robot_mascot" alt="UET ChatBot Logo" class="uet-logo-img">
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

/**
 * src/interfaces/static/js/api.js
 * API Client Module - Tầng kết nối REST API Backend (/api/chat, /api/health)
 * Hỗ trợ timeout, xử lý lỗi chi tiết và fallback an toàn khi offline.
 */

export class UETApiClient {
    constructor(baseUrl = "") {
        this.baseUrl = baseUrl;
        this.timeoutMs = 60000; // 60s timeout cho LLM RAG
    }

    /**
     * Kiểm tra trạng thái máy chủ và mô hình AI
     */
    async getHealth() {
        try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 6000);

            const resp = await fetch(`${this.baseUrl}/api/health`, {
                method: "GET",
                signal: controller.signal,
                headers: { "Cache-Control": "no-cache" }
            });
            clearTimeout(timer);

            if (!resp.ok) {
                return { status: "error", model_name: "UET RAG Model", online: false };
            }
            const data = await resp.json();
            return {
                status: data.status || "ok",
                model_name: data.model_name || "Qwen / Gemini",
                online: true,
                message: data.message || "Hệ thống hoạt động bình thường"
            };
        } catch (err) {
            console.warn("API health check failed:", err.message);
            return { status: "offline", model_name: "UET AI Assistant", online: false };
        }
    }

    /**
     * Gửi câu hỏi đến RAG Pipeline và nhận câu trả lời cùng trích dẫn nguồn
     */
    async sendQuery(queryText, sessionId = "default_session", topK = null) {
        if (!queryText || !queryText.trim()) {
            throw new Error("Nội dung câu hỏi không được để trống.");
        }

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeoutMs);

        try {
            const payload = {
                query: queryText.trim(),
                session_id: sessionId,
            };
            if (topK && Number.isInteger(topK)) {
                payload.top_k = topK;
            }

            const response = await fetch(`${this.baseUrl}/api/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload),
                signal: controller.signal
            });

            clearTimeout(timer);

            const responseData = await response.json().catch(() => ({}));

            if (!response.ok) {
                const detail = responseData.detail || `Lỗi máy chủ (${response.status})`;
                throw new Error(detail);
            }

            return {
                query: responseData.query || queryText,
                answer: responseData.answer || "Không nhận được phản hồi từ mô hình.",
                sources: Array.isArray(responseData.sources) ? responseData.sources : [],
                latency_seconds: responseData.latency_seconds || 0.0,
                model_name: responseData.model_name || "UET RAG"
            };
        } catch (error) {
            clearTimeout(timer);
            if (error.name === "AbortError") {
                throw new Error("Yêu cầu quá thời gian chờ (Timeout 60s). Vui lòng thử lại với câu hỏi ngắn gọn hơn.");
            }
            throw error;
        }
    }
}


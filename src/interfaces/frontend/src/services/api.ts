/**
 * src/interfaces/frontend/src/services/api.ts
 * Tầng giao tiếp REST & SSE API (/api/health, /api/chat, /api/chat/stream).
 * Tích hợp classifyApiError để đảm bảo không lỗi kỹ thuật thô nào lọt ra giao diện.
 */

import type {
  ChatApiResponse,
  ClassifiedError,
  HealthStatus,
  SourceItem,
  StreamCallbacks,
} from "../types/chat";
import {
  classifyApiError,
  containsRawTechnicalError,
} from "../utils/errorClassifier";

export class ApiClientError extends Error {
  public readonly classified: ClassifiedError;

  constructor(classified: ClassifiedError) {
    super(classified.message);
    this.name = "ApiClientError";
    this.classified = classified;
  }
}

interface RawStreamEvent {
  type?: "sources" | "token" | "done" | "error";
  sources?: SourceItem[];
  model_name?: string;
  content?: string;
  latency_seconds?: number;
  hide_sources?: boolean;
  error_type?: string;
  message?: string;
}

function linkAbortSignals(
  externalSignal: AbortSignal | undefined,
  controller: AbortController
): () => void {
  if (!externalSignal) return () => {};
  if (externalSignal.aborted) {
    controller.abort(externalSignal.reason);
    return () => {};
  }
  const onAbort = () => controller.abort(externalSignal.reason);
  externalSignal.addEventListener("abort", onAbort, { once: true });
  return () => externalSignal.removeEventListener("abort", onAbort);
}

export class UETApiClient {
  public readonly baseUrl: string;
  public readonly timeoutMs: number;

  constructor(baseUrl = "", timeoutMs = 120_000) {
    this.baseUrl = baseUrl;
    this.timeoutMs = timeoutMs;
  }

  async getHealth(): Promise<HealthStatus> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 6000);
    try {
      const resp = await fetch(`${this.baseUrl}/api/health`, {
        method: "GET",
        signal: controller.signal,
        headers: { "Cache-Control": "no-cache" },
      });
      clearTimeout(timer);

      if (!resp.ok) {
        return {
          status: "error",
          model_name: "UET RAG Model",
          online: false,
        };
      }

      const data = (await resp.json()) as Partial<HealthStatus>;
      return {
        status: data.status || "ok",
        model_name: data.model_name || "Qwen / Gemini",
        online: true,
        message: data.message || "Hệ thống hoạt động bình thường",
      };
    } catch (err) {
      clearTimeout(timer);
      console.warn("API health check failed:", err);
      return {
        status: "offline",
        model_name: "UET AI Assistant",
        online: false,
      };
    }
  }

  async sendQuery(
    queryText: string,
    sessionId = "default_session",
    topK: number | null = null,
    externalSignal?: AbortSignal,
    voiceMode = false
  ): Promise<ChatApiResponse> {
    const trimmed = (queryText || "").trim();
    if (!trimmed) {
      const classified = classifyApiError(
        "Nội dung câu hỏi không được để trống.",
        400
      );
      throw new ApiClientError({
        ...classified,
        message: "Nội dung câu hỏi không được để trống.",
      });
    }

    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);
    const unlink = linkAbortSignals(externalSignal, controller);

    try {
      const payload: {
        query: string;
        session_id: string;
        top_k?: number;
        voice_mode?: boolean;
      } = {
        query: trimmed,
        session_id: sessionId,
      };
      if (topK !== null && Number.isInteger(topK)) {
        payload.top_k = topK;
      }
      if (voiceMode) {
        payload.voice_mode = true;
      }

      const response = await fetch(`${this.baseUrl}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timer);
      unlink();

      const responseData = (await response.json().catch(() => ({}))) as Record<
        string,
        unknown
      >;

      if (!response.ok) {
        const classified = classifyApiError(responseData, response.status);
        throw new ApiClientError(classified);
      }

      const rawAnswer =
        typeof responseData.answer === "string"
          ? responseData.answer
          : "Không nhận được phản hồi từ mô hình.";

      if (containsRawTechnicalError(rawAnswer)) {
        const classified = classifyApiError(rawAnswer, 500);
        throw new ApiClientError(classified);
      }

      return {
        query:
          typeof responseData.query === "string" ? responseData.query : trimmed,
        answer: rawAnswer,
        sources: Array.isArray(responseData.sources)
          ? (responseData.sources as SourceItem[])
          : [],
        latency_seconds:
          typeof responseData.latency_seconds === "number"
            ? responseData.latency_seconds
            : 0.0,
        model_name:
          typeof responseData.model_name === "string"
            ? responseData.model_name
            : "UET RAG",
      };
    } catch (err) {
      clearTimeout(timer);
      unlink();
      if (err instanceof ApiClientError) {
        throw err;
      }
      if (timedOut) {
        throw new ApiClientError(
          classifyApiError(new Error("Request timeout"), 504)
        );
      }
      throw new ApiClientError(classifyApiError(err));
    }
  }

  async streamQuery(
    queryText: string,
    sessionId = "default_session",
    callbacks: StreamCallbacks = {},
    topK: number | null = null,
    voiceMode = false
  ): Promise<ChatApiResponse> {
    const trimmed = (queryText || "").trim();
    if (!trimmed) {
      const errObj: ClassifiedError = {
        category: "server",
        message: "Nội dung câu hỏi không được để trống.",
        retryable: false,
      };
      callbacks.onError?.(errObj);
      throw new ApiClientError(errObj);
    }

    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);
    const unlink = linkAbortSignals(callbacks.signal, controller);

    const emitError = (classified: ClassifiedError): never => {
      callbacks.onError?.(classified);
      throw new ApiClientError(classified);
    };

    try {
      const payload: {
        query: string;
        session_id: string;
        top_k?: number;
        voice_mode?: boolean;
      } = {
        query: trimmed,
        session_id: sessionId,
      };
      if (topK !== null && Number.isInteger(topK)) {
        payload.top_k = topK;
      }
      if (voiceMode) {
        payload.voice_mode = true;
      }

      const response = await fetch(`${this.baseUrl}/api/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      // Fallback sang /api/chat nếu endpoint stream chưa khả dụng hoặc browser không hỗ trợ stream body
      if (
        response.status === 404 ||
        response.status === 405 ||
        response.status === 501 ||
        (response.ok && !response.body)
      ) {
        clearTimeout(timer);
        unlink();
        const fallbackResult = await this.sendQuery(
          trimmed,
          sessionId,
          topK,
          callbacks.signal,
          voiceMode
        );
        callbacks.onSources?.(
          fallbackResult.sources,
          fallbackResult.model_name
        );
        callbacks.onToken?.(fallbackResult.answer, fallbackResult.answer);
        callbacks.onDone?.(fallbackResult);
        return fallbackResult;
      }

      if (!response.ok) {
        clearTimeout(timer);
        unlink();
        const errBody = await response.json().catch(() => ({}));
        return emitError(classifyApiError(errBody, response.status));
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder("utf-8");
      let sseBuffer = "";
      let accumulated = "";
      let sources: SourceItem[] = [];
      let modelName = "UET RAG";
      let latencySeconds = 0.0;
      let hideSources = false;
      let streamFinished = false;

      const processSseFrame = (rawFrame: string) => {
        const dataLines = rawFrame
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart());

        if (dataLines.length === 0) return;
        const dataStr = dataLines.join("\n").trim();
        if (!dataStr) return;

        if (dataStr === "[DONE]") {
          streamFinished = true;
          return;
        }

        let event: RawStreamEvent;
        try {
          event = JSON.parse(dataStr) as RawStreamEvent;
        } catch (parseErr) {
          console.error("[UET SSE Parse Error]", dataStr, parseErr);
          return;
        }

        if (event.type === "sources") {
          sources = Array.isArray(event.sources) ? event.sources : [];
          modelName = event.model_name || modelName;
          callbacks.onSources?.(sources, modelName);
        } else if (event.type === "token") {
          const chunk = typeof event.content === "string" ? event.content : "";
          if (!chunk) return;
          const nextAccumulated = accumulated + chunk;
          if (
            containsRawTechnicalError(chunk) ||
            containsRawTechnicalError(nextAccumulated)
          ) {
            emitError(classifyApiError(nextAccumulated, 500));
          }
          accumulated = nextAccumulated;
          callbacks.onToken?.(chunk, accumulated);
        } else if (event.type === "error") {
          emitError(classifyApiError(event));
        } else if (event.type === "done") {
          if (typeof event.latency_seconds === "number") {
            latencySeconds = event.latency_seconds;
          }
          if (typeof event.hide_sources === "boolean") {
            hideSources = event.hide_sources;
          }
        }
      };

      while (!streamFinished) {
        const { value, done } = await reader.read();
        if (done) break;

        sseBuffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        let boundaryIdx = sseBuffer.indexOf("\n\n");
        while (boundaryIdx !== -1) {
          const frame = sseBuffer.slice(0, boundaryIdx);
          sseBuffer = sseBuffer.slice(boundaryIdx + 2);
          processSseFrame(frame);
          if (streamFinished) break;
          boundaryIdx = sseBuffer.indexOf("\n\n");
        }
      }

      const trailing = (sseBuffer + decoder.decode()).replace(/\r\n/g, "\n").trim();
      if (!streamFinished && trailing) {
        processSseFrame(trailing);
      }

      clearTimeout(timer);
      unlink();

      if (containsRawTechnicalError(accumulated)) {
        return emitError(classifyApiError(accumulated, 500));
      }

      const result: ChatApiResponse = {
        query: trimmed,
        answer: accumulated,
        sources: hideSources ? [] : sources,
        latency_seconds: latencySeconds,
        model_name: modelName,
        hide_sources: hideSources,
      };
      callbacks.onDone?.(result);
      return result;
    } catch (err) {
      clearTimeout(timer);
      unlink();

      if (!timedOut && callbacks.signal?.aborted) {
        return {
          query: trimmed,
          answer: "",
          sources: [],
          latency_seconds: 0,
          model_name: "",
        };
      }

      if (err instanceof ApiClientError) {
        throw err;
      }

      const classified = timedOut
        ? classifyApiError(new Error("Request timeout"), 504)
        : classifyApiError(err);
      callbacks.onError?.(classified);
      throw new ApiClientError(classified);
    }
  }

  async transcribeAudio(
    audioBase64: string,
    mimeType = "audio/webm",
    externalSignal?: AbortSignal
  ): Promise<string> {
    const cleanB64 = (audioBase64 || "").trim();
    if (!cleanB64) return "";

    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 30_000);
    const unlink = linkAbortSignals(externalSignal, controller);

    try {
      const response = await fetch(`${this.baseUrl}/api/voice/transcribe`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          audio_base64: cleanB64,
          mime_type: mimeType,
          language: "vi",
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);
      unlink();

      const data = (await response.json().catch(() => ({}))) as {
        text?: unknown;
        detail?: unknown;
      };

      if (!response.ok) {
        const classified = classifyApiError(data, response.status);
        throw new ApiClientError(classified);
      }

      return typeof data.text === "string" ? data.text.trim() : "";
    } catch (err) {
      clearTimeout(timer);
      unlink();

      if (!timedOut && externalSignal?.aborted) {
        return "";
      }
      if (err instanceof ApiClientError) {
        throw err;
      }
      if (timedOut) {
        throw new ApiClientError(
          classifyApiError(new Error("Request timeout"), 504)
        );
      }
      throw new ApiClientError(classifyApiError(err));
    }
  }
}

export const defaultApiClient = new UETApiClient();

export function getHealth(): Promise<HealthStatus> {
  return defaultApiClient.getHealth();
}

export function transcribeAudio(
  audioBase64: string,
  mimeType = "audio/webm",
  signal?: AbortSignal
): Promise<string> {
  return defaultApiClient.transcribeAudio(audioBase64, mimeType, signal);
}

export function sendQuery(
  queryText: string,
  sessionId = "default_session",
  topK: number | null = null,
  signal?: AbortSignal,
  voiceMode = false
): Promise<ChatApiResponse> {
  return defaultApiClient.sendQuery(
    queryText,
    sessionId,
    topK,
    signal,
    voiceMode
  );
}

export function streamQuery(
  queryText: string,
  sessionId = "default_session",
  callbacks: StreamCallbacks = {},
  topK: number | null = null,
  voiceMode = false
): Promise<ChatApiResponse> {
  return defaultApiClient.streamQuery(
    queryText,
    sessionId,
    callbacks,
    topK,
    voiceMode
  );
}


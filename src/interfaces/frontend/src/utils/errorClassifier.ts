/**
 * src/interfaces/frontend/src/utils/errorClassifier.ts
 * Bộ phân loại và lọc lỗi API: tuyệt đối không để rò rỉ thông tin kỹ thuật
 * (org_id, ITPM/TPM, console.groq.com, stack trace, tên provider thô) lên giao diện chat.
 * Mọi chi tiết kỹ thuật chỉ được ghi qua console.error().
 */

import type { ClassifiedError, ErrorCategory } from "../types/chat";

const RAW_TECHNICAL_SUBSTRINGS = [
  "MÔ HÌNH AI TẠM THỜI TỪ CHỐI DỊCH VỤ",
  "Groq [",
  "SambaNova [",
  "Gemini [",
  "org_",
  "ITPM",
  "TPM",
  "rate_limit",
  "Request too large",
  "console.groq.com",
  "billing",
  "Traceback",
  "RESOURCE_EXHAUSTED",
  "HTTPException",
  "ConnectError",
  "ReadTimeout",
  "context_length",
  "maximum context",
];

const RAW_TECHNICAL_PATTERNS: RegExp[] = [
  /\borg_[a-zA-Z0-9_-]+\b/i,
  /\b(itpm|tpm|rpm|rpd)\b/i,
  /console\.[a-z0-9-]+\.com/i,
  /\b(groq|sambanova|gemini)\s*\[\d{3}\]/i,
  /traceback\s*\(most recent call last\)/i,
  /\bfile\s+"[^"]+",\s+line\s+\d+/i,
  /^\s*at\s+\S+\s+\([^)]+:\d+:\d+\)/m,
  /\b(runtimeerror|valueerror|connectionerror|timeouterror):/i,
];

const FRIENDLY_MESSAGES: Record<ErrorCategory, string> = {
  token_limit:
    "Nội dung câu hỏi hoặc tài liệu tham chiếu hiện hơi dài vượt quá giới hạn xử lý trong một lượt. Bạn hãy thử rút gọn câu hỏi hoặc chia nhỏ thành từng ý để mình hỗ trợ chính xác hơn nhé.",
  rate_limit:
    "Hệ thống AI hiện đang tiếp nhận nhiều yêu cầu cùng lúc nên tạm thời quá tải. Bạn vui lòng đợi trong giây lát rồi thử gửi lại câu hỏi nhé.",
  timeout:
    "Yêu cầu mất nhiều thời gian xử lý hơn dự kiến nên đã hết thời gian chờ. Bạn vui lòng thử lại hoặc đặt câu hỏi ngắn gọn hơn nhé.",
  network:
    "Không thể kết nối đến máy chủ hoặc đường truyền tạm thời bị gián đoạn. Bạn vui lòng kiểm tra lại kết nối mạng và thử gửi lại nhé.",
  server:
    "Hệ thống đang gặp sự cố kỹ thuật tạm thời khi xử lý yêu cầu của bạn. Bạn vui lòng thử lại sau ít phút nhé.",
};

const VALID_CATEGORIES = new Set<ErrorCategory>([
  "token_limit",
  "rate_limit",
  "network",
  "timeout",
  "server",
]);

export function containsRawTechnicalError(text: string): boolean {
  if (!text) return false;
  const normalized = String(text);
  if (RAW_TECHNICAL_SUBSTRINGS.some((marker) => normalized.includes(marker))) {
    return true;
  }
  return RAW_TECHNICAL_PATTERNS.some((pattern) => pattern.test(normalized));
}

function extractErrorContext(err: unknown): {
  rawText: string;
  explicitCategory?: ErrorCategory;
  explicitMessage?: string;
  explicitStatus?: number;
  errorName?: string;
} {
  if (typeof err === "string") {
    return { rawText: err, explicitMessage: err };
  }

  if (err instanceof Error) {
    const extra = err as Error & {
      statusCode?: number;
      status?: number;
      error_type?: string;
      category?: string;
      detail?: unknown;
      classified?: ClassifiedError;
    };
    if (extra.classified) {
      return {
        rawText: extra.message || extra.classified.message,
        explicitCategory: extra.classified.category,
        explicitMessage: extra.classified.message,
        explicitStatus: extra.classified.statusCode,
        errorName: err.name,
      };
    }
    const detailStr =
      typeof extra.detail === "string"
        ? extra.detail
        : extra.detail
        ? JSON.stringify(extra.detail)
        : "";
    return {
      rawText: `${err.name}: ${err.message} ${detailStr}`.trim(),
      explicitCategory: VALID_CATEGORIES.has(extra.error_type as ErrorCategory)
        ? (extra.error_type as ErrorCategory)
        : VALID_CATEGORIES.has(extra.category as ErrorCategory)
        ? (extra.category as ErrorCategory)
        : undefined,
      explicitMessage: err.message,
      explicitStatus: extra.statusCode ?? extra.status,
      errorName: err.name,
    };
  }

  if (err && typeof err === "object") {
    const obj = err as Record<string, unknown>;
    const detailObj =
      obj.detail && typeof obj.detail === "object"
        ? (obj.detail as Record<string, unknown>)
        : null;

    const rawCategory =
      (detailObj?.error_type as string) ||
      (obj.error_type as string) ||
      (obj.category as string) ||
      "";
    const explicitCategory = VALID_CATEGORIES.has(rawCategory as ErrorCategory)
      ? (rawCategory as ErrorCategory)
      : undefined;

    const explicitMessage =
      (typeof detailObj?.message === "string" && detailObj.message) ||
      (typeof obj.detail === "string" && obj.detail) ||
      (typeof obj.message === "string" && obj.message) ||
      undefined;

    const explicitStatus =
      typeof obj.statusCode === "number"
        ? obj.statusCode
        : typeof obj.status === "number"
        ? obj.status
        : undefined;

    let serialized = "";
    try {
      serialized = JSON.stringify(obj);
    } catch {
      serialized = String(obj);
    }

    return {
      rawText: `${explicitMessage ?? ""} ${serialized}`.trim(),
      explicitCategory,
      explicitMessage,
      explicitStatus,
    };
  }

  return { rawText: String(err ?? "") };
}

function detectCategory(
  rawText: string,
  statusCode?: number,
  errorName?: string
): ErrorCategory {
  const lower = rawText.toLowerCase();

  if (errorName === "AbortError" || statusCode === 408 || statusCode === 504) {
    return "timeout";
  }

  if (
    statusCode === 413 ||
    lower.includes("413") ||
    lower.includes("request too large") ||
    lower.includes("itpm") ||
    lower.includes("context_length") ||
    lower.includes("maximum context")
  ) {
    return "token_limit";
  }

  if (
    statusCode === 429 ||
    lower.includes("429") ||
    lower.includes("rate limit") ||
    lower.includes("rate_limit") ||
    lower.includes("quota") ||
    lower.includes("resource_exhausted") ||
    lower.includes("too many requests")
  ) {
    return "rate_limit";
  }

  if (lower.includes("tpm") || lower.includes("tokens")) {
    return "token_limit";
  }

  if (
    lower.includes("timeout") ||
    lower.includes("timed out") ||
    lower.includes("readtimeout") ||
    lower.includes("connecttimeout") ||
    lower.includes("quá thời gian")
  ) {
    return "timeout";
  }

  if (
    statusCode === 502 ||
    statusCode === 503 ||
    lower.includes("failed to fetch") ||
    lower.includes("networkerror") ||
    lower.includes("network") ||
    lower.includes("connecterror") ||
    lower.includes("connection") ||
    lower.includes("econnrefused") ||
    lower.includes("không phản hồi")
  ) {
    return "network";
  }

  return "server";
}

export function classifyApiError(
  err: unknown,
  statusCode?: number
): ClassifiedError {
  console.error("[UET API Error]", { statusCode, error: err });

  const ctx = extractErrorContext(err);
  const resolvedStatus = statusCode ?? ctx.explicitStatus;

  // Luôn ưu tiên phân loại chuẩn từ nội dung hoặc HTTP status code
  const category: ErrorCategory =
    ctx.explicitCategory ??
    detectCategory(ctx.rawText, resolvedStatus, ctx.errorName);

  // Chỉ sử dụng message từ server nếu server đã phân loại và hoàn toàn sạch dấu hiệu kỹ thuật
  const safeExplicitMessage =
    ctx.explicitCategory &&
    ctx.explicitMessage &&
    !containsRawTechnicalError(ctx.explicitMessage)
      ? ctx.explicitMessage
      : null;

  return {
    category,
    message: safeExplicitMessage ?? FRIENDLY_MESSAGES[category],
    statusCode: resolvedStatus,
    retryable: category !== "token_limit",
  };
}


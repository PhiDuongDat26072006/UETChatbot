/**
 * src/interfaces/frontend/src/types/chat.ts
 * Định nghĩa toàn bộ kiểu dữ liệu TypeScript cho UET Chatbot Web UI,
 * đồng bộ 100% với routes.py (Backend) và storage.js (LocalStorage schema v3).
 */

export interface SourceMetadata {
  title?: string;
  source?: string;
  final_url?: string;
  url?: string;
  resource_type?: string;
  domain?: string;
  matched_chunks_count?: number;
  page?: number | string;
  chapter?: string;
  article?: string;
  site?: string;
  faculty?: string;
  search_type?: string;
  search_source?: string;
  bm25_rank?: number;
  vector_rank?: number;
  [key: string]: unknown;
}

export interface SourceItem {
  chunk_id: string;
  text: string;
  similarity_score: number;
  rank: number;
  search_type?: string | null;
  search_source?: string | null;
  metadata: SourceMetadata;
}

export interface AttachmentItem {
  name: string;
  size: number;
  type: string;
}

export type MessageSender = "user" | "bot";

export type MessageStatus = "sending" | "streaming" | "delivered" | "error";

export type ErrorCategory =
  | "token_limit"
  | "rate_limit"
  | "network"
  | "timeout"
  | "server";

export interface ClassifiedError {
  category: ErrorCategory;
  message: string;
  statusCode?: number;
  retryable: boolean;
}

export interface ChatMessage {
  id: string;
  sender: MessageSender;
  text: string;
  timestamp: number;
  sources: SourceItem[];
  latency: number;
  model: string;
  attachments: AttachmentItem[];
  status: MessageStatus;
  error: boolean;
  errorType?: ErrorCategory;
}

export type CreateMessageInput = {
  sender?: MessageSender;
  text?: string;
  timestamp?: number;
  sources?: SourceItem[];
  latency?: number;
  model?: string;
  attachments?: AttachmentItem[];
  status?: MessageStatus;
  error?: boolean;
  errorType?: ErrorCategory;
};

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  pinned: boolean;
  unreadCount: number;
  messages: ChatMessage[];
}

export interface GroupedChats {
  pinned: ChatSession[];
  today: ChatSession[];
  yesterday: ChatSession[];
  earlier: ChatSession[];
}

export type ThemeMode = "light" | "dark";

export type WallpaperMode = "telegram-blue" | "plain" | "dark-night";

export interface HealthStatus {
  status: "ok" | "error" | "offline" | string;
  model_name: string;
  online: boolean;
  message?: string;
}

export interface ChatApiResponse {
  query: string;
  answer: string;
  sources: SourceItem[];
  latency_seconds: number;
  model_name: string;
  hide_sources?: boolean;
}

export type ToastType = "info" | "success" | "error" | "warning";

export interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

export interface StreamCallbacks {
  onSources?: (sources: SourceItem[], modelName: string) => void;
  onToken?: (chunk: string, accumulatedText: string) => void;
  onDone?: (result: ChatApiResponse) => void;
  onError?: (error: ClassifiedError) => void;
  signal?: AbortSignal;
}


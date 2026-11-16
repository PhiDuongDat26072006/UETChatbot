/**
 * src/interfaces/frontend/src/hooks/useChatStore.ts
 * Hook quản lý toàn bộ trạng thái hội thoại, theme, hình nền, sidebar,
 * đính kèm tệp, streaming SSE và toast notifications với LocalStorage uet_ai_*.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getHealth, streamQuery } from "../services/api";
import {
  addMessage,
  clearAllChats,
  clearChatMessages,
  createChat,
  deleteChat,
  getTheme,
  getWallpaper,
  groupChats,
  initializeConversations,
  isSidebarClosed as getStoredSidebarClosed,
  renameChat,
  saveConversations,
  setActiveChatId as persistActiveChatId,
  setSidebarClosed as persistSidebarClosed,
  setTheme as persistTheme,
  setWallpaper as persistWallpaper,
  togglePinChat,
} from "../services/storage";
import type {
  AttachmentItem,
  ChatSession,
  GroupedChats,
  HealthStatus,
  SourceItem,
  ThemeMode,
  ToastItem,
  ToastType,
  WallpaperMode,
} from "../types/chat";
import {
  classifyApiError,
  containsRawTechnicalError,
} from "../utils/errorClassifier";
import {
  getSoundEnabled,
  playNotificationSound,
  setSoundEnabled as persistSoundEnabled,
} from "../utils/sound";
import { isNoAnswerOrOutOfDomain } from "../utils/formatters";
import { useStreamBuffer } from "./useStreamBuffer";

const ALLOWED_EXTENSIONS = new Set(["pdf", "docx", "doc", "txt"]);
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

export interface UseChatStoreReturn {
  conversations: ChatSession[];
  activeChatId: string;
  activeChat: ChatSession | null;
  groupedChats: GroupedChats;
  searchChatQuery: string;
  setSearchChatQuery: (query: string) => void;

  theme: ThemeMode;
  toggleTheme: () => void;
  setThemeMode: (mode: ThemeMode) => void;

  wallpaper: WallpaperMode;
  setWallpaperMode: (mode: WallpaperMode) => void;

  soundEnabled: boolean;
  setSoundEnabled: (enabled: boolean) => void;

  isSidebarClosed: boolean;
  toggleSidebar: () => void;
  isMobileSidebarOpen: boolean;
  setMobileSidebarOpen: (open: boolean) => void;

  health: HealthStatus;
  refreshHealth: () => Promise<HealthStatus>;

  isGenerating: boolean;
  isWaitingFirstToken: boolean;
  streamingMessageId: string | null;
  streamingText: string;
  streamingSources: SourceItem[];
  streamingModel: string;
  skipStreaming: () => void;

  selectedFiles: File[];
  addFiles: (files: FileList | File[]) => void;
  removeSelectedFile: (index: number) => void;
  clearSelectedFiles: () => void;

  toasts: ToastItem[];
  showToast: (msg: string, type?: ToastType, duration?: number) => void;
  dismissToast: (id: string) => void;

  selectChat: (id: string) => void;
  createNewChat: () => ChatSession;
  togglePin: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  deleteConversation: (id: string) => void;
  clearMessages: (id: string) => void;
  clearAllConversations: () => void;
  sendMessage: (
    overrideText?: string,
    options?: { voiceMode?: boolean }
  ) => Promise<string | null>;
  retryMessage: (botMsgIndex: number) => Promise<void>;
  stopGeneration: () => void;
}

type AbortReason = "user_stop" | "user_skip" | "chat_switch" | null;

function applyThemeToDom(mode: ThemeMode): void {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("data-theme", mode);
  document.documentElement.classList.toggle("dark", mode === "dark");
}

function applyWallpaperToDom(mode: WallpaperMode): void {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("data-wallpaper", mode);
}

function buildAttachmentNotice(files: File[], answer: string): string {
  if (files.length === 0) return answer;
  const names = files.map((f) => f.name).join(", ");
  return (
    `*Đã nhận ${files.length} tệp đính kèm: ${names}.*\n` +
    `*(Lưu ý: Tính năng trích xuất nội dung file người dùng tải lên đang được tích hợp vào pipeline backend. Dưới đây là câu trả lời đối chiếu với kho quy chế chính thức của UET:)*\n\n` +
    answer
  );
}

export function useChatStore(): UseChatStoreReturn {
  const initialData = useMemo(() => initializeConversations(), []);
  const [conversations, setConversations] = useState<ChatSession[]>(
    initialData.conversations
  );
  const [activeChatId, setActiveChatId] = useState<string>(
    initialData.activeChatId
  );
  const [searchChatQuery, setSearchChatQuery] = useState<string>("");

  const [theme, setThemeState] = useState<ThemeMode>(() => getTheme());
  const [wallpaper, setWallpaperState] = useState<WallpaperMode>(() =>
    getWallpaper()
  );
  const [soundEnabled, setSoundEnabledState] = useState<boolean>(() =>
    getSoundEnabled()
  );
  const [isSidebarClosed, setIsSidebarClosed] = useState<boolean>(() =>
    getStoredSidebarClosed()
  );
  const [isMobileSidebarOpen, setMobileSidebarOpen] = useState<boolean>(false);

  const [health, setHealth] = useState<HealthStatus>({
    status: "checking",
    model_name: "Đang tải...",
    online: true,
  });

  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isWaitingFirstToken, setIsWaitingFirstToken] = useState<boolean>(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(
    null
  );
  const [streamingSources, setStreamingSources] = useState<SourceItem[]>([]);
  const [streamingModel, setStreamingModel] = useState<string>("");

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const streamBuffer = useStreamBuffer();

  // Refs giúp tránh stale closures khi streaming hoặc hủy request
  const conversationsRef = useRef(conversations);
  conversationsRef.current = conversations;
  const activeChatIdRef = useRef(activeChatId);
  activeChatIdRef.current = activeChatId;
  const selectedFilesRef = useRef(selectedFiles);
  selectedFilesRef.current = selectedFiles;
  const isGeneratingRef = useRef(isGenerating);
  isGeneratingRef.current = isGenerating;
  const healthRef = useRef(health);
  healthRef.current = health;
  const soundEnabledRef = useRef(soundEnabled);
  soundEnabledRef.current = soundEnabled;

  const abortControllerRef = useRef<AbortController | null>(null);
  const abortReasonRef = useRef<AbortReason>(null);
  const activeStreamContextRef = useRef<{
    chatId: string;
    files: File[];
    startTimeMs: number;
    sources: SourceItem[];
    model: string;
  } | null>(null);

  const toastTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map()
  );

  // Sync theme & wallpaper lên <html>
  useEffect(() => {
    applyThemeToDom(theme);
  }, [theme]);

  useEffect(() => {
    applyWallpaperToDom(wallpaper);
  }, [wallpaper]);

  // Cleanup toast timers & active stream khi unmount
  useEffect(() => {
    const timers = toastTimersRef.current;
    return () => {
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
      if (abortControllerRef.current) {
        abortReasonRef.current = "chat_switch";
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const dismissToast = useCallback((id: string) => {
    const timer = toastTimersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      toastTimersRef.current.delete(id);
    }
    setToasts((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const showToast = useCallback(
    (msg: string, type: ToastType = "info", duration = 3000) => {
      const id = `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      setToasts((prev) => [...prev.slice(-4), { id, message: msg, type, duration }]);
      if (duration > 0) {
        const timer = setTimeout(() => {
          toastTimersRef.current.delete(id);
          setToasts((prev) => prev.filter((item) => item.id !== id));
        }, duration);
        toastTimersRef.current.set(id, timer);
      }
    },
    []
  );

  const refreshHealth = useCallback(async (): Promise<HealthStatus> => {
    const status = await getHealth();
    setHealth(status);
    return status;
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const setThemeMode = useCallback(
    (mode: ThemeMode) => {
      const normalized = persistTheme(mode);
      setThemeState(normalized);
      applyThemeToDom(normalized);
      showToast(
        `Đã chuyển sang giao diện ${normalized === "dark" ? "Tối" : "Sáng"}`,
        "info",
        1500
      );
    },
    [showToast]
  );

  const toggleTheme = useCallback(() => {
    setThemeMode(theme === "dark" ? "light" : "dark");
  }, [theme, setThemeMode]);

  const setWallpaperMode = useCallback(
    (mode: WallpaperMode) => {
      const normalized = persistWallpaper(mode);
      setWallpaperState(normalized);
      applyWallpaperToDom(normalized);
    },
    []
  );

  const setSoundEnabled = useCallback(
    (enabled: boolean) => {
      const normalized = persistSoundEnabled(enabled);
      setSoundEnabledState(normalized);
      soundEnabledRef.current = normalized;
      showToast(
        normalized
          ? "Đã bật âm thanh thông báo"
          : "Đã tắt âm thanh thông báo",
        "info",
        1500
      );
    },
    [showToast]
  );

  const toggleSidebar = useCallback(() => {
    if (typeof window !== "undefined" && window.innerWidth < 768) {
      setMobileSidebarOpen((prev) => !prev);
      return;
    }
    setIsSidebarClosed((prev) => {
      const next = !prev;
      persistSidebarClosed(next);
      return next;
    });
  }, []);

  const clearStreamingState = useCallback(() => {
    abortControllerRef.current = null;
    abortReasonRef.current = null;
    activeStreamContextRef.current = null;
    streamBuffer.reset();
    setIsGenerating(false);
    setIsWaitingFirstToken(false);
    setStreamingMessageId(null);
    setStreamingSources([]);
    setStreamingModel("");
  }, [streamBuffer]);

  const finalizePartialOnAbort = useCallback(
    (commitPartial: boolean) => {
      const ctx = activeStreamContextRef.current;
      const partial = streamBuffer.flush();
      if (commitPartial && ctx && partial.trim().length > 0 && !containsRawTechnicalError(partial)) {
        const latency = Number(
          ((performance.now() - ctx.startTimeMs) / 1000).toFixed(2)
        );
        const finalAnswer = buildAttachmentNotice(ctx.files, partial);
        setConversations((prev) => {
          const res = addMessage(prev, ctx.chatId, {
            sender: "bot",
            text: finalAnswer,
            timestamp: Date.now(),
            sources: ctx.sources,
            latency,
            model: ctx.model,
            status: "delivered",
            error: false,
          });
          return res.conversations;
        });
      }
      clearStreamingState();
    },
    [clearStreamingState, streamBuffer]
  );

  const abortActiveStream = useCallback(
    (reason: Exclude<AbortReason, null>, commitPartial: boolean) => {
      if (!isGeneratingRef.current && !abortControllerRef.current) return;
      abortReasonRef.current = reason;
      const controller = abortControllerRef.current;
      abortControllerRef.current = null;
      if (controller) {
        controller.abort();
      }
      finalizePartialOnAbort(commitPartial);
    },
    [finalizePartialOnAbort]
  );

  const stopGeneration = useCallback(() => {
    if (!isGeneratingRef.current) return;
    abortActiveStream("user_stop", true);
    showToast("Đã dừng tạo câu trả lời", "info", 1800);
  }, [abortActiveStream, showToast]);

  const skipStreaming = useCallback(() => {
    if (!isGeneratingRef.current) return;
    abortActiveStream("user_skip", true);
  }, [abortActiveStream]);

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const incoming = Array.from(files);
      if (incoming.length === 0) return;

      const validFiles: File[] = [];
      for (const f of incoming) {
        const ext = f.name.split(".").pop()?.toLowerCase() || "";
        if (!ALLOWED_EXTENSIONS.has(ext)) {
          showToast(
            `Định dạng .${ext || "unknown"} không được hỗ trợ. Vui lòng chọn .pdf, .docx, .txt`,
            "error"
          );
          continue;
        }
        if (f.size > MAX_FILE_SIZE_BYTES) {
          showToast(`Tệp "${f.name}" vượt quá giới hạn 10MB`, "error");
          continue;
        }
        validFiles.push(f);
      }

      if (validFiles.length > 0) {
        setSelectedFiles((prev) => [...prev, ...validFiles]);
        showToast(`Đã đính kèm ${validFiles.length} tệp`, "info", 1500);
      }
    },
    [showToast]
  );

  const removeSelectedFile = useCallback((index: number) => {
    setSelectedFiles((prev) => prev.filter((_, idx) => idx !== index));
  }, []);

  const clearSelectedFiles = useCallback(() => {
    setSelectedFiles([]);
  }, []);

  const selectChat = useCallback(
    (id: string) => {
      if (id === activeChatIdRef.current) {
        setMobileSidebarOpen(false);
        return;
      }
      if (isGeneratingRef.current) {
        abortActiveStream("chat_switch", false);
      }
      persistActiveChatId(id);
      setActiveChatId(id);
      setConversations((prev) => {
        const target = prev.find((c) => c.id === id);
        if (!target || !target.unreadCount) return prev;
        const next = prev.map((c) =>
          c.id === id ? { ...c, unreadCount: 0 } : c
        );
        saveConversations(next);
        return next;
      });
      setMobileSidebarOpen(false);
    },
    [abortActiveStream]
  );

  const createNewChat = useCallback((): ChatSession => {
    if (isGeneratingRef.current) {
      abortActiveStream("chat_switch", false);
    }

    const currentActive = conversationsRef.current.find(
      (c) => c.id === activeChatIdRef.current
    );
    if (currentActive && currentActive.messages.length === 0) {
      showToast("Bạn đang ở cuộc trò chuyện mới!", "info", 1500);
      setMobileSidebarOpen(false);
      return currentActive;
    }

    const res = createChat(conversationsRef.current, "Cuộc trò chuyện mới");
    setConversations(res.conversations);
    setActiveChatId(res.chat.id);
    setMobileSidebarOpen(false);
    showToast("Đã tạo cuộc trò chuyện mới!", "success", 1500);
    return res.chat;
  }, [abortActiveStream, showToast]);

  const togglePin = useCallback(
    (id: string) => {
      const target = conversationsRef.current.find((c) => c.id === id);
      const next = togglePinChat(conversationsRef.current, id);
      setConversations(next);
      const isNowPinned = target ? !target.pinned : true;
      showToast(
        isNowPinned
          ? "Đã ghim cuộc trò chuyện lên đầu"
          : "Đã bỏ ghim cuộc trò chuyện",
        "info",
        1500
      );
    },
    [showToast]
  );

  const renameConversation = useCallback(
    (id: string, title: string) => {
      const next = renameChat(conversationsRef.current, id, title);
      setConversations(next);
      showToast("Đã đổi tên cuộc trò chuyện thành công!", "success");
    },
    [showToast]
  );

  const deleteConversation = useCallback(
    (id: string) => {
      if (isGeneratingRef.current && activeChatIdRef.current === id) {
        abortActiveStream("chat_switch", false);
      }
      const res = deleteChat(
        conversationsRef.current,
        id,
        activeChatIdRef.current
      );
      setConversations(res.conversations);
      setActiveChatId(res.activeChatId);
      showToast("Đã xóa cuộc trò chuyện", "info");
    },
    [abortActiveStream, showToast]
  );

  const clearMessages = useCallback(
    (id: string) => {
      if (isGeneratingRef.current && activeChatIdRef.current === id) {
        abortActiveStream("chat_switch", false);
      }
      const next = clearChatMessages(conversationsRef.current, id);
      setConversations(next);
      showToast("Đã xóa toàn bộ tin nhắn trong hội thoại.", "info");
    },
    [abortActiveStream, showToast]
  );

  const clearAllConversations = useCallback(() => {
    if (isGeneratingRef.current) {
      abortActiveStream("chat_switch", false);
    }
    try {
      window.localStorage.removeItem("uet_ai_telegram_conversations_v2");
      window.localStorage.removeItem("uet_ai_active_chat_id_v2");
    } catch {
      // ignore
    }
    const fresh = clearAllChats();
    setConversations(fresh.conversations);
    setActiveChatId(fresh.activeChatId);
    showToast("Đã xóa toàn bộ lịch sử trò chuyện.", "info");
  }, [abortActiveStream, showToast]);

  const sendMessage = useCallback(
    async (
      overrideText?: string,
      options?: { voiceMode?: boolean }
    ): Promise<string | null> => {
      const voiceMode = Boolean(options?.voiceMode);
      const queryText = (overrideText ?? "").trim();
      const filesToSend = [...selectedFilesRef.current];

      if (!queryText && filesToSend.length === 0) return null;
      if (isGeneratingRef.current) return null;

      let currentConversations = conversationsRef.current;
      let targetChatId = activeChatIdRef.current;

      if (
        !targetChatId ||
        !currentConversations.some((c) => c.id === targetChatId)
      ) {
        const created = createChat(currentConversations, "Tư vấn học vụ UET");
        currentConversations = created.conversations;
        targetChatId = created.chat.id;
        setActiveChatId(targetChatId);
      }

      const attachmentsData: AttachmentItem[] = filesToSend.map((f) => ({
        name: f.name,
        size: f.size,
        type: f.name.split(".").pop()?.toLowerCase() || "file",
      }));

      const userDisplayText =
        queryText ||
        (filesToSend.length > 0
          ? `[Đã đính kèm ${filesToSend.length} tệp]`
          : "");

      const afterUser = addMessage(currentConversations, targetChatId, {
        sender: "user",
        text: userDisplayText,
        timestamp: Date.now(),
        attachments: attachmentsData,
        status: "delivered",
      });
      setConversations(afterUser.conversations);
      setSelectedFiles([]);

      let effectiveQuery = queryText;
      if (filesToSend.length > 0 && !queryText) {
        effectiveQuery = `Tôi đã đính kèm tài liệu: ${filesToSend
          .map((f) => f.name)
          .join(", ")}. Hãy giới thiệu các quy chế liên quan.`;
      }

      const controller = new AbortController();
      const streamMsgId = `stream_${Date.now()}_${Math.random()
        .toString(36)
        .slice(2, 6)}`;
      abortControllerRef.current = controller;
      abortReasonRef.current = null;
      activeStreamContextRef.current = {
        chatId: targetChatId,
        files: filesToSend,
        startTimeMs: performance.now(),
        sources: [],
        model: "",
      };

      streamBuffer.reset();
      setIsGenerating(true);
      setIsWaitingFirstToken(true);
      setStreamingMessageId(streamMsgId);
      setStreamingSources([]);
      setStreamingModel("");

      let settled = false;
      let resolvedAnswer: string | null = null;

      const handleFailure = (rawErr: unknown) => {
        if (settled) return;
        settled = true;

        if (controller.signal.aborted || abortReasonRef.current !== null) {
          return;
        }

        const classified = classifyApiError(rawErr);
        clearStreamingState();

        setConversations((prev) => {
          const res = addMessage(prev, targetChatId, {
            sender: "bot",
            text: classified.message,
            timestamp: Date.now(),
            sources: [],
            latency: 0,
            model: "",
            status: "error",
            error: true,
            errorType: classified.category,
          });
          return res.conversations;
        });

        showToast(
          classified.message,
          classified.category === "rate_limit" ? "warning" : "error",
          4500
        );
      };

      try {
        await streamQuery(
          effectiveQuery,
          targetChatId,
          {
            signal: controller.signal,
            onSources: (sources, modelName) => {
              if (
                controller.signal.aborted ||
                abortControllerRef.current !== controller ||
                abortReasonRef.current !== null
              ) {
                return;
              }
              if (activeStreamContextRef.current) {
                activeStreamContextRef.current.sources = sources;
                activeStreamContextRef.current.model = modelName;
              }
              setStreamingSources(sources);
              setStreamingModel(modelName);
            },
            onToken: (chunk) => {
              if (
                controller.signal.aborted ||
                abortControllerRef.current !== controller ||
                abortReasonRef.current !== null
              ) {
                return;
              }
              setIsWaitingFirstToken(false);
              streamBuffer.pushChunk(chunk);
            },
            onDone: (res) => {
              if (settled) return;
              settled = true;

              if (controller.signal.aborted || abortReasonRef.current !== null) {
                return;
              }

              const flushed = streamBuffer.flush();
              const rawAnswer = flushed || res.answer || "";

              if (containsRawTechnicalError(rawAnswer)) {
                const classified = classifyApiError(rawAnswer, 500);
                clearStreamingState();
                setConversations((prev) => {
                  const errRes = addMessage(prev, targetChatId, {
                    sender: "bot",
                    text: classified.message,
                    timestamp: Date.now(),
                    sources: [],
                    latency: 0,
                    model: "",
                    status: "error",
                    error: true,
                    errorType: classified.category,
                  });
                  return errRes.conversations;
                });
                showToast(classified.message, "error", 4500);
                return;
              }

              const ctx = activeStreamContextRef.current;
              const finalAnswer = buildAttachmentNotice(filesToSend, rawAnswer);
              resolvedAnswer = finalAnswer;
              const shouldHideSources =
                Boolean(res.hide_sources) || isNoAnswerOrOutOfDomain(finalAnswer);
              const rawSources =
                res.sources && res.sources.length > 0
                  ? res.sources
                  : ctx?.sources || [];
              const finalSources = shouldHideSources ? [] : rawSources;
              const finalModel = res.model_name || ctx?.model || "";
              const latency =
                typeof res.latency_seconds === "number" &&
                res.latency_seconds > 0
                  ? res.latency_seconds
                  : Number(
                      (
                        (performance.now() -
                          (ctx?.startTimeMs ?? performance.now())) /
                        1000
                      ).toFixed(2)
                    );

              clearStreamingState();

              setConversations((prev) => {
                const botRes = addMessage(prev, targetChatId, {
                  sender: "bot",
                  text: finalAnswer,
                  timestamp: Date.now(),
                  sources: finalSources,
                  latency,
                  model: finalModel,
                  status: "delivered",
                  error: false,
                });
                return botRes.conversations;
              });

              if (soundEnabledRef.current && !voiceMode) {
                playNotificationSound();
              }

              if (!healthRef.current.online) {
                void refreshHealth();
              }
            },
            onError: (classified) => {
              handleFailure(classified);
            },
          },
          null,
          voiceMode
        );
        return resolvedAnswer;
      } catch (err) {
        handleFailure(err);
        return null;
      }
    },
    [clearStreamingState, refreshHealth, showToast, streamBuffer]
  );

  const retryMessage = useCallback(
    async (botMsgIndex: number): Promise<void> => {
      const active = conversationsRef.current.find(
        (c) => c.id === activeChatIdRef.current
      );
      if (!active || active.messages.length === 0) return;

      const startIdx = Math.min(botMsgIndex - 1, active.messages.length - 1);
      for (let i = startIdx; i >= 0; i -= 1) {
        if (active.messages[i].sender === "user" && active.messages[i].text.trim()) {
          await sendMessage(active.messages[i].text);
          return;
        }
      }
    },
    [sendMessage]
  );

  const activeChat = useMemo(
    () => conversations.find((c) => c.id === activeChatId) ?? null,
    [conversations, activeChatId]
  );

  const groupedChats = useMemo(
    () => groupChats(conversations, searchChatQuery),
    [conversations, searchChatQuery]
  );

  return {
    conversations,
    activeChatId,
    activeChat,
    groupedChats,
    searchChatQuery,
    setSearchChatQuery,

    theme,
    toggleTheme,
    setThemeMode,

    wallpaper,
    setWallpaperMode,

    soundEnabled,
    setSoundEnabled,

    isSidebarClosed,
    toggleSidebar,
    isMobileSidebarOpen,
    setMobileSidebarOpen,

    health,
    refreshHealth,

    isGenerating,
    isWaitingFirstToken,
    streamingMessageId,
    streamingText: streamBuffer.text,
    streamingSources,
    streamingModel,
    skipStreaming,

    selectedFiles,
    addFiles,
    removeSelectedFile,
    clearSelectedFiles,

    toasts,
    showToast,
    dismissToast,

    selectChat,
    createNewChat,
    togglePin,
    renameConversation,
    deleteConversation,
    clearMessages,
    clearAllConversations,
    sendMessage,
    retryMessage,
    stopGeneration,
  };
}


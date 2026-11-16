/**
 * src/interfaces/frontend/src/components/chat/VirtualizedMessageList.tsx
 * Kết hợp useStickToBottom + @tanstack/react-virtual để render hàng nghìn tin nhắn ở 60fps,
 * phân nhóm Telegram + Date Divider, tự ngắt auto-scroll khi người dùng cuộn lên,
 * nút Scroll-to-Bottom kèm unread badge, và cuộn mượt tới kết quả In-Chat Search.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useStickToBottom } from "use-stick-to-bottom";
import { AnimatePresence, motion } from "motion/react";
import { ArrowDown } from "lucide-react";
import type { ChatMessage, SourceItem } from "../../types/chat";
import { formatDateHeader } from "../../utils/formatters";
import { MessageBubble } from "./MessageBubble";
import { StreamingMessage } from "./StreamingMessage";
import { ThinkingIndicator } from "./ThinkingIndicator";
import { WelcomeScreen } from "./WelcomeScreen";

export interface VirtualizedMessageListProps {
  chatId: string;
  messages: ChatMessage[];
  isGenerating: boolean;
  isWaitingFirstToken: boolean;
  streamingText: string;
  streamingSources: SourceItem[];
  streamingModel?: string;
  searchQuery?: string;
  activeSearchMessageIndex?: number | null;
  onSelectSuggestion: (query: string) => void;
  onOpenSource: (source: SourceItem) => void;
  onRetryMessage: (botMsgIndex: number) => void;
  onCopyMessage?: (text: string) => void;
  onStopGeneration?: () => void;
  onSkipStreaming?: () => void;
}

const GROUP_GAP_MS = 5 * 60 * 1000;

interface MessageRowMeta {
  dateHeader: string | null;
  isFirstInGroup: boolean;
  isLastInGroup: boolean;
}

function DateDividerBadge({ label }: { label: string }) {
  return (
    <div className="date-divider-badge my-2.5 text-center">
      <span className="inline-block rounded-full bg-black/15 px-3 py-1 text-[11px] font-bold tracking-wider text-white backdrop-blur-sm dark:bg-white/15">
        {label}
      </span>
    </div>
  );
}

export function VirtualizedMessageList({
  chatId,
  messages,
  isGenerating,
  isWaitingFirstToken,
  streamingText,
  streamingSources,
  streamingModel,
  searchQuery,
  activeSearchMessageIndex,
  onSelectSuggestion,
  onOpenSource,
  onRetryMessage,
  onCopyMessage,
  onStopGeneration,
  onSkipStreaming,
}: VirtualizedMessageListProps) {
  const { scrollRef, contentRef, scrollToBottom, stopScroll, isAtBottom } =
    useStickToBottom({
      initial: "instant",
      resize: "smooth",
    });

  const [unreadWhileScrolledUp, setUnreadWhileScrolledUp] = useState(0);

  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const onRetryRef = useRef(onRetryMessage);
  onRetryRef.current = onRetryMessage;
  const onOpenSourceRef = useRef(onOpenSource);
  onOpenSourceRef.current = onOpenSource;
  const onCopyRef = useRef(onCopyMessage);
  onCopyRef.current = onCopyMessage;

  const stableOpenSource = useCallback((source: SourceItem) => {
    onOpenSourceRef.current(source);
  }, []);

  const stableCopyMessage = useCallback((text: string) => {
    onCopyRef.current?.(text);
  }, []);
  const hasCopyHandler = Boolean(onCopyMessage);

  // Giữ ổn định callback retry theo message.id để không phá React.memo của MessageBubble khi đang stream
  const retryHandlerMapRef = useRef<Map<string, () => void>>(new Map());
  const getRetryHandler = useCallback((messageId: string) => {
    const existing = retryHandlerMapRef.current.get(messageId);
    if (existing) return existing;

    const handler = () => {
      const idx = messagesRef.current.findIndex((m) => m.id === messageId);
      if (idx !== -1) {
        onRetryRef.current(idx);
      }
    };
    retryHandlerMapRef.current.set(messageId, handler);
    return handler;
  }, []);

  // Dọn dẹp map khi đổi hội thoại
  useEffect(() => {
    retryHandlerMapRef.current.clear();
  }, [chatId]);

  const rowMetas = useMemo<MessageRowMeta[]>(() => {
    let lastDateStr = "";
    return messages.map((msg, idx) => {
      const dateStr = formatDateHeader(msg.timestamp);
      const dateHeader = dateStr !== lastDateStr ? dateStr : null;
      lastDateStr = dateStr;

      const prev = idx > 0 ? messages[idx - 1] : null;
      const next = idx < messages.length - 1 ? messages[idx + 1] : null;

      const sameAsPrev =
        Boolean(prev) &&
        prev!.sender === msg.sender &&
        formatDateHeader(prev!.timestamp) === dateStr &&
        msg.timestamp - prev!.timestamp < GROUP_GAP_MS;

      const sameAsNext =
        Boolean(next) &&
        next!.sender === msg.sender &&
        formatDateHeader(next!.timestamp) === dateStr &&
        next!.timestamp - msg.timestamp < GROUP_GAP_MS;

      return {
        dateHeader,
        isFirstInGroup: !sameAsPrev,
        isLastInGroup: !sameAsNext,
      };
    });
  }, [messages]);

  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 136,
    overscan: 6,
    getItemKey: (index) => messages[index]?.id ?? index,
  });

  // Reset unread & cuộn về cuối khi chuyển cuộc trò chuyện
  const prevChatIdRef = useRef(chatId);
  const prevCountRef = useRef(messages.length);

  useEffect(() => {
    if (prevChatIdRef.current !== chatId) {
      prevChatIdRef.current = chatId;
      prevCountRef.current = messages.length;
      setUnreadWhileScrolledUp(0);
      const raf = requestAnimationFrame(() => {
        void scrollToBottom({ animation: "instant" });
      });
      return () => cancelAnimationFrame(raf);
    }

    if (messages.length > prevCountRef.current) {
      const newlyAdded = messages.slice(prevCountRef.current);
      const hasUserMsg = newlyAdded.some((m) => m.sender === "user");
      if (hasUserMsg) {
        setUnreadWhileScrolledUp(0);
        void scrollToBottom({ animation: "smooth" });
      } else if (!isAtBottom) {
        const botAddedCount = newlyAdded.filter((m) => m.sender === "bot").length;
        if (botAddedCount > 0) {
          setUnreadWhileScrolledUp((prev) => prev + botAddedCount);
        }
      }
    }
    prevCountRef.current = messages.length;
  }, [chatId, messages, isAtBottom, scrollToBottom]);

  useEffect(() => {
    if (isAtBottom) {
      setUnreadWhileScrolledUp(0);
    }
  }, [isAtBottom]);

  // Khi user bấm tìm kiếm trong cuộc trò chuyện -> ngắt stick-to-bottom & cuộn tới đúng message index
  useEffect(() => {
    if (
      activeSearchMessageIndex === null ||
      activeSearchMessageIndex === undefined ||
      activeSearchMessageIndex < 0 ||
      activeSearchMessageIndex >= messages.length
    ) {
      return;
    }

    stopScroll();
    virtualizer.scrollToIndex(activeSearchMessageIndex, {
      align: "center",
      behavior: "smooth",
    });
  }, [activeSearchMessageIndex, messages.length, stopScroll, virtualizer]);

  const handleScrollToBottomClick = useCallback(() => {
    setUnreadWhileScrolledUp(0);
    void scrollToBottom({ animation: "smooth" });
  }, [scrollToBottom]);

  const isEmpty = messages.length === 0 && !isGenerating;
  const virtualItems = virtualizer.getVirtualItems();

  return (
    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
      <div
        ref={scrollRef}
        className="telegram-chat-window relative z-[2] min-h-0 flex-1 overflow-y-auto px-3 py-4 sm:px-5"
      >
        <div
          ref={contentRef}
          className={`telegram-messages-feed mx-auto flex w-full max-w-[960px] flex-col ${
            isEmpty ? "min-h-full justify-center" : ""
          }`}
        >
          {isEmpty ? (
            <WelcomeScreen onSelectSuggestion={onSelectSuggestion} />
          ) : (
            <>
              {messages.length > 0 && (
                <div
                  style={{
                    height: `${virtualizer.getTotalSize()}px`,
                    width: "100%",
                    position: "relative",
                  }}
                >
                  {virtualItems.map((virtualRow) => {
                    const msg = messages[virtualRow.index];
                    if (!msg) return null;
                    const meta = rowMetas[virtualRow.index] ?? {
                      dateHeader: null,
                      isFirstInGroup: true,
                      isLastInGroup: true,
                    };
                    const isHighlighted =
                      activeSearchMessageIndex === virtualRow.index;

                    return (
                      <div
                        key={virtualRow.key}
                        data-index={virtualRow.index}
                        ref={virtualizer.measureElement}
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          transform: `translateY(${virtualRow.start}px)`,
                        }}
                        className={meta.isLastInGroup ? "pb-3.5" : "pb-1.5"}
                      >
                        {meta.dateHeader && (
                          <DateDividerBadge label={meta.dateHeader} />
                        )}
                        <MessageBubble
                          message={msg}
                          isFirstInGroup={meta.isFirstInGroup}
                          isLastInGroup={meta.isLastInGroup}
                          isHighlighted={isHighlighted}
                          searchQuery={searchQuery}
                          onOpenSource={stableOpenSource}
                          onRetry={
                            msg.sender === "bot"
                              ? getRetryHandler(msg.id)
                              : undefined
                          }
                          onCopy={hasCopyHandler ? stableCopyMessage : undefined}
                        />
                      </div>
                    );
                  })}
                </div>
              )}

              {isWaitingFirstToken && (
                <div className="pt-1 pb-3">
                  <ThinkingIndicator />
                </div>
              )}

              {isGenerating && !isWaitingFirstToken && (
                <div className="pt-1 pb-3">
                  <StreamingMessage
                    text={streamingText}
                    sources={streamingSources}
                    modelName={streamingModel}
                    onOpenSource={onOpenSource}
                    onStop={onStopGeneration}
                    onSkip={onSkipStreaming}
                  />
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <AnimatePresence>
        {!isAtBottom && !isEmpty && (
          <motion.button
            type="button"
            initial={{ opacity: 0, y: 12, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 420, damping: 28 }}
            whileHover={{ y: -2, scale: 1.05 }}
            whileTap={{ scale: 0.94 }}
            style={{ willChange: "transform, opacity" }}
            onClick={handleScrollToBottomClick}
            className="btn-scroll-bottom show absolute bottom-4 right-5 z-30 flex h-10 w-10 items-center justify-center rounded-full border border-[rgba(8,117,201,0.22)] bg-[var(--bg-surface)] text-[var(--primary)] shadow-[0_4px_14px_rgba(8,117,201,0.2)] transition-colors hover:bg-[var(--primary)] hover:text-white dark:border-white/15 dark:bg-[#1E2430] dark:text-[#38BDF8] dark:hover:bg-[#38BDF8] dark:hover:text-[#0F172A]"
            title="Cuộn xuống tin nhắn mới nhất"
            aria-label="Cuộn xuống tin nhắn mới nhất"
          >
            <ArrowDown className="h-5 w-5" />
            {unreadWhileScrolledUp > 0 && (
              <span className="unread-scroll-badge absolute -right-1.5 -top-1.5 flex h-5 min-w-[20px] items-center justify-center rounded-full border-2 border-[var(--bg-surface)] bg-red-500 px-1 text-[11px] font-bold text-white shadow">
                {unreadWhileScrolledUp}
              </span>
            )}
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  );
}


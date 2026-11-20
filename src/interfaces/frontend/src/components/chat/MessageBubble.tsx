/**
 * src/interfaces/frontend/src/components/chat/MessageBubble.tsx
 * Bong bóng tin nhắn Telegram thích ứng (18px bo tròn, co còn 4px khi gom nhóm,
 * SVG tail ở tin cuối nhóm, GPU-accelerated entrance animation, memoized tuyệt đối).
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import {
  Check,
  ChevronDown,
  Copy,
  ExternalLink,
  FileText,
  Globe,
  RotateCw,
  User,
} from "lucide-react";
import type { ChatMessage, SourceItem } from "../../types/chat";
import {
  escapeRegex,
  formatFileSize,
  formatTime,
} from "../../utils/formatters";
import { StreamdownRenderer } from "./StreamdownRenderer";

export interface MessageBubbleProps {
  message: ChatMessage;
  isFirstInGroup: boolean;
  isLastInGroup: boolean;
  isHighlighted?: boolean;
  searchQuery?: string;
  onOpenSource: (source: SourceItem) => void;
  onRetry?: () => void;
  onCopy?: (text: string) => void;
}

export function BubbleTail({ sender }: { sender: "user" | "bot" }) {
  if (sender === "user") {
    return (
      <svg
        className="bubble-tail-svg tail-user"
        viewBox="0 0 11 18"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M0 0V18H11C6.4 15.5 3.6 10.6 2.4 0H0Z"
          fill="currentColor"
        />
      </svg>
    );
  }

  return (
    <svg
      className="bubble-tail-svg tail-bot"
      viewBox="0 0 11 18"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M11 0V18H0C4.6 15.5 7.4 10.6 8.6 0H11Z"
        fill="currentColor"
      />
    </svg>
  );
}

export interface SourceReferenceListProps {
  sources: SourceItem[];
  onOpenSource?: (source: SourceItem) => void;
}

export function SourceReferenceList({
  sources,
  onOpenSource: _onOpenSource,
}: SourceReferenceListProps) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  const total = sources.length;
  const badgeText = total > 2 ? "2+" : String(total);

  return (
    <div className="sources-reference-section mt-3 border-t border-[var(--border-divider)] pt-2.5">
      <div className="sources-ref-header flex items-center justify-between">
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          aria-expanded={expanded}
          title={`${total} tài liệu tham khảo — bấm để ${expanded ? "thu gọn" : "xem"}`}
          className="sources-ref-toggle inline-flex items-center gap-2 rounded-lg border border-transparent px-2 py-1 text-[13px] font-bold text-[var(--text-primary)] transition-colors hover:border-[var(--border-color)] hover:bg-[var(--bg-hover-chat)] dark:hover:bg-white/5"
        >
          <span>Tài liệu tham khảo</span>
          <span className="sources-count-badge rounded-full bg-[var(--user-bubble)] px-2 py-0.5 text-[11px] font-bold text-[var(--primary)] dark:bg-[rgba(8,117,201,0.3)] dark:text-[#38BDF8]">
            {badgeText}
          </span>
          <ChevronDown
            className={`h-3.5 w-3.5 text-[var(--text-muted)] transition-transform duration-200 ${
              expanded ? "rotate-180" : ""
            }`}
          />
        </button>
      </div>

      {expanded && (
        <div className="sources-doc-list mt-2 flex flex-col gap-1.5">
          {sources.map((src, idx) => {
            const title =
              src.metadata?.title ||
              src.metadata?.source ||
              `Tài liệu tham khảo #${idx + 1}`;
            const lowerTitle = title.toLowerCase();

            const rawUrl =
              typeof src.metadata?.final_url === "string" && src.metadata.final_url.trim()
                ? src.metadata.final_url.trim()
                : typeof src.metadata?.url === "string" && src.metadata.url.trim()
                ? src.metadata.url.trim()
                : typeof src.metadata?.source === "string" && src.metadata.source.startsWith("http")
                ? src.metadata.source.trim()
                : "";
            const isHttp = rawUrl.startsWith("http");

            let domain = typeof src.metadata?.domain === "string" ? src.metadata.domain : "";
            if (!domain && isHttp) {
              try {
                domain = new URL(rawUrl).hostname.replace(/^www\./, "");
              } catch {
                domain = "";
              }
            }

            const resType = String(src.metadata?.resource_type || "").toLowerCase();
            const isPdf =
              resType === "pdf" ||
              lowerTitle.endsWith(".pdf") ||
              rawUrl.toLowerCase().endsWith(".pdf");

            const scoreStr =
              typeof src.similarity_score === "number" && src.similarity_score > 0
                ? `${(src.similarity_score * 100).toFixed(0)}% liên quan`
                : "";
            const metaText =
              [domain || (src.metadata?.site ? String(src.metadata.site) : ""), scoreStr]
                .filter(Boolean)
                .join(" • ") || "Trang web trường ĐH Công nghệ";

            if (isHttp) {
              return (
                <a
                  key={`${src.chunk_id || idx}_${idx}`}
                  href={rawUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={`Mở trang web gốc: ${title}`}
                  className="source-doc-card group flex w-full cursor-pointer items-center gap-3 rounded-[10px] border border-[var(--border-color)] bg-[var(--bg-hover-chat)] px-3 py-2 text-left no-underline transition-all hover:-translate-y-px hover:border-[var(--primary)] hover:bg-[var(--user-bubble)] dark:bg-[#141922] dark:hover:border-[#38BDF8] dark:hover:bg-[#1E3A5F]"
                >
                  <div className="source-doc-icon-col shrink-0">
                    <span
                      className={`source-format-tag inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-extrabold tracking-wide ${
                        isPdf
                          ? "format-pdf bg-red-100 text-red-600 dark:bg-[#3D1A1A] dark:text-red-400"
                          : "format-web bg-emerald-100 text-emerald-700 dark:bg-[#064E3B] dark:text-emerald-300"
                      }`}
                    >
                      {isPdf ? <FileText size={10} /> : <Globe size={10} />}
                      {isPdf ? "PDF" : "WEB"}
                    </span>
                  </div>

                  <div className="source-doc-info-col min-w-0 flex-1">
                    <div className="source-doc-title truncate text-[13px] font-semibold text-[var(--text-primary)] group-hover:text-[var(--primary)] dark:group-hover:text-[#38BDF8]">
                      {title}
                    </div>
                    <div className="source-doc-meta mt-0.5 truncate text-[11.5px] text-[var(--text-muted)]">
                      {metaText}
                    </div>
                  </div>

                  <ExternalLink className="h-4 w-4 shrink-0 text-[var(--text-muted)] transition-transform group-hover:translate-x-0.5 group-hover:text-[var(--primary)] dark:group-hover:text-[#38BDF8]" />
                </a>
              );
            }

            return (
              <div
                key={`${src.chunk_id || idx}_${idx}`}
                title={title}
                className="source-doc-card group flex w-full items-center gap-3 rounded-[10px] border border-[var(--border-color)] bg-[var(--bg-hover-chat)] px-3 py-2 text-left transition-all hover:bg-[var(--user-bubble)] dark:bg-[#141922]"
              >
                <div className="source-doc-icon-col shrink-0">
                  <span className="source-format-tag rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-extrabold text-blue-700 dark:bg-[#172554] dark:text-blue-400">
                    DOC
                  </span>
                </div>
                <div className="source-doc-info-col min-w-0 flex-1">
                  <div className="source-doc-title truncate text-[13px] font-semibold text-[var(--text-primary)]">
                    {title}
                  </div>
                  <div className="source-doc-meta mt-0.5 truncate text-[11.5px] text-[var(--text-muted)]">
                    {metaText}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function HighlightedPlainText({
  text,
  query,
  isActiveMatch,
}: {
  text: string;
  query?: string;
  isActiveMatch?: boolean;
}) {
  const trimmed = (query || "").trim();
  const segments = useMemo(() => {
    if (!trimmed) return null;
    const regex = new RegExp(`(${escapeRegex(trimmed)})`, "gi");
    return text.split(regex);
  }, [text, trimmed]);

  if (!segments) {
    return <>{text}</>;
  }

  const lowerQuery = trimmed.toLowerCase();
  return (
    <>
      {segments.map((part, idx) =>
        part.toLowerCase() === lowerQuery ? (
          <mark
            key={idx}
            className={
              isActiveMatch
                ? "search-highlight search-highlight-active"
                : "search-highlight"
            }
          >
            {part}
          </mark>
        ) : (
          <span key={idx}>{part}</span>
        )
      )}
    </>
  );
}

function getAttachmentBadge(fileName: string): "PDF" | "DOCX" | "TXT" | "FILE" {
  const ext = (fileName || "").split(".").pop()?.toLowerCase() || "";
  if (ext === "pdf") return "PDF";
  if (ext === "docx" || ext === "doc") return "DOCX";
  if (ext === "txt") return "TXT";
  return "FILE";
}

function getGroupClasses(
  isFirstInGroup: boolean,
  isLastInGroup: boolean
): string {
  if (isFirstInGroup && isLastInGroup) {
    return "bubble-group-first bubble-group-last";
  }
  if (isFirstInGroup) return "bubble-group-first";
  if (isLastInGroup) return "bubble-group-middle bubble-group-last";
  return "bubble-group-middle";
}

const MessageBubbleInner = ({
  message,
  isFirstInGroup,
  isLastInGroup,
  isHighlighted = false,
  searchQuery,
  onOpenSource: _onOpenSource,
  onRetry,
  onCopy,
}: MessageBubbleProps) => {
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) {
        clearTimeout(copyTimerRef.current);
      }
    };
  }, []);

  const handleCopy = useCallback(() => {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(message.text).then(() => {
        setCopied(true);
        if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
        copyTimerRef.current = setTimeout(() => setCopied(false), 1600);
        onCopy?.(message.text);
      });
    } else {
      onCopy?.(message.text);
    }
  }, [message.text, onCopy]);

  const isUser = message.sender === "user";
  const timeStr = formatTime(message.timestamp);
  const groupClasses = getGroupClasses(isFirstInGroup, isLastInGroup);
  const trimmedSearch = (searchQuery || "").trim().toLowerCase();
  const hasSearchMatch =
    Boolean(trimmedSearch) &&
    message.text.toLowerCase().includes(trimmedSearch);

  const highlightRingClass = isHighlighted
    ? "ring-2 ring-amber-400/90 shadow-md"
    : hasSearchMatch
    ? "ring-1 ring-amber-300/80"
    : "";

  if (isUser) {
    return (
      <motion.div
        id={message.id}
        initial={{ opacity: 0, y: 8, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: "spring", stiffness: 420, damping: 30 }}
        style={{ willChange: "transform, opacity" }}
        className="message-bubble-row user-row flex w-full items-end justify-end gap-2.5"
      >
        <div
          className={`message-bubble user-bubble relative max-w-[80%] md:max-w-[76%] ${groupClasses} ${highlightRingClass}`}
        >
          {message.attachments && message.attachments.length > 0 && (
            <div className="message-attachments-preview mb-1.5 flex flex-col gap-1">
              {message.attachments.map((att, idx) => {
                const badgeLabel = getAttachmentBadge(att.name);
                const badgeColorClass =
                  badgeLabel === "PDF"
                    ? "bg-red-100 text-red-600 dark:bg-[#3D1A1A] dark:text-red-400"
                    : badgeLabel === "DOCX"
                    ? "bg-blue-100 text-blue-700 dark:bg-[#172554] dark:text-blue-400"
                    : "bg-slate-200/80 text-slate-700 dark:bg-white/15 dark:text-slate-200";

                return (
                  <div
                    key={`${att.name}_${idx}`}
                    className="user-file-badge flex items-center gap-2 rounded-md bg-[rgba(0,84,166,0.08)] px-2.5 py-1.5 dark:bg-white/10"
                  >
                    <span
                      className={`file-badge-icon rounded px-1.5 py-0.5 text-[10px] font-extrabold tracking-wide ${badgeColorClass}`}
                    >
                      {badgeLabel}
                    </span>
                    <div className="file-badge-info flex min-w-0 flex-col text-[12px]">
                      <span className="file-badge-name truncate font-semibold">
                        {att.name}
                      </span>
                      <span className="file-badge-size text-[10.5px] text-[var(--text-muted)]">
                        {formatFileSize(att.size)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <div className="message-text whitespace-pre-wrap text-[14.5px] leading-[1.6] text-[var(--bubble-user-text)]">
            <HighlightedPlainText
              text={message.text}
              query={searchQuery}
              isActiveMatch={isHighlighted}
            />
          </div>

          <div className="message-meta mt-1 flex items-center justify-end gap-1 text-[11px] text-[var(--text-muted)]">
            <span className="message-time">{timeStr}</span>
          </div>

          {isLastInGroup && <BubbleTail sender="user" />}
        </div>

        <div className="user-avatar-col mb-0.5 w-[34px] shrink-0">
          {isLastInGroup && (
            <div
              className="student-avatar flex h-[34px] w-[34px] items-center justify-center rounded-full shadow-[0_2px_6px_rgba(8,117,201,0.2)]"
              title="Sinh viên UET"
            >
              <User size={18} strokeWidth={2.2} />
            </div>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      id={message.id}
      initial={{ opacity: 0, y: 8, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 420, damping: 30 }}
      style={{ willChange: "transform, opacity" }}
      className="message-bubble-row bot-row flex w-full items-end justify-start gap-2.5"
    >
      <div className="bot-avatar-col mb-0.5 w-[34px] shrink-0">
        {isLastInGroup && (
          <div
            className="telegram-avatar bot-avatar-mini flex h-[34px] w-[34px] items-center justify-center overflow-hidden rounded-full border border-[rgba(8,117,201,0.18)] bg-white shadow-[0_2px_6px_rgba(8,117,201,0.2)]"
            title="UET AI Assistant"
          >
            <img
              src="/static/uet-logo.png?v=v14_robot_mascot"
              alt="UET ChatBot Logo"
              className="uet-logo-img"
            />
          </div>
        )}
      </div>

      <div
        className={`message-bubble bot-bubble relative max-w-[92%] md:max-w-[88%] ${groupClasses} ${
          message.error ? "bot-bubble-error" : ""
        } ${highlightRingClass}`}
      >
        {(message.error || isHighlighted) && (
          <div className="mb-1.5 flex items-center justify-end gap-1.5">
            {isHighlighted && (
              <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10.5px] font-semibold text-amber-800 dark:bg-amber-900/50 dark:text-amber-200">
                Kết quả tìm kiếm
              </span>
            )}
            {message.error && (
              <span className="inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-[10.5px] font-bold text-amber-800 dark:bg-amber-900/40 dark:text-amber-300">
                Tạm gián đoạn
              </span>
            )}
          </div>
        )}

        <div className="message-text">
          <StreamdownRenderer
            content={message.text}
            isStreaming={false}
            searchQuery={searchQuery}
            isActiveMatch={isHighlighted}
          />
        </div>

        <div className="message-meta-footer mt-2.5 flex items-center justify-between gap-3 border-t border-[var(--border-divider)] pt-2 text-[11px] text-[var(--text-muted)]">
          <div className="message-actions flex items-center gap-1.5">
            {message.error ? (
              onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="btn-msg-action btn-retry-msg inline-flex h-7 items-center gap-1.5 rounded-lg bg-[var(--primary)] px-3 text-[12px] font-semibold text-white shadow-sm transition-colors hover:bg-[var(--primary-hover)]"
                  title="Thử lại câu hỏi"
                >
                  <RotateCw className="h-3.5 w-3.5" />
                  <span>Thử lại</span>
                </button>
              )
            ) : (
              <>
                <button
                  type="button"
                  onClick={handleCopy}
                  className="btn-msg-action btn-copy-msg inline-flex h-7 items-center justify-center gap-1 rounded-md px-1.5 text-[var(--text-muted)] transition-colors hover:bg-[var(--user-bubble)] hover:text-[var(--primary)] dark:hover:bg-[rgba(8,117,201,0.25)] dark:hover:text-[#38BDF8]"
                  title="Sao chép câu trả lời"
                  aria-label="Sao chép câu trả lời"
                >
                  {copied ? (
                    <>
                      <Check className="h-4 w-4 text-emerald-500" />
                      <span className="text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
                        Đã chép
                      </span>
                    </>
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </button>

                {onRetry && (
                  <button
                    type="button"
                    onClick={onRetry}
                    className="btn-msg-action btn-retry-msg inline-flex h-7 w-7 items-center justify-center rounded-md text-[var(--text-muted)] transition-colors hover:bg-[var(--user-bubble)] hover:text-[var(--primary)] dark:hover:bg-[rgba(8,117,201,0.25)] dark:hover:text-[#38BDF8]"
                    title="Tạo lại câu trả lời"
                    aria-label="Tạo lại câu trả lời"
                  >
                    <RotateCw className="h-4 w-4" />
                  </button>
                )}

                {message.latency > 0 && (
                  <span className="badge-latency ml-1 rounded-full bg-[var(--bg-search)] px-2 py-0.5 text-[10px] font-semibold text-[var(--text-muted)]">
                    {message.latency}s
                  </span>
                )}
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="message-time">{timeStr}</span>
          </div>
        </div>

        {isLastInGroup && <BubbleTail sender="bot" />}
      </div>
    </motion.div>
  );
};

function areMessagePropsEqual(
  prev: MessageBubbleProps,
  next: MessageBubbleProps
): boolean {
  return (
    prev.message.id === next.message.id &&
    prev.message.text === next.message.text &&
    prev.message.status === next.message.status &&
    prev.message.error === next.message.error &&
    prev.message.errorType === next.message.errorType &&
    prev.message.latency === next.message.latency &&
    prev.message.model === next.message.model &&
    prev.message.timestamp === next.message.timestamp &&
    prev.message.sources === next.message.sources &&
    prev.message.attachments === next.message.attachments &&
    prev.isFirstInGroup === next.isFirstInGroup &&
    prev.isLastInGroup === next.isLastInGroup &&
    prev.isHighlighted === next.isHighlighted &&
    prev.searchQuery === next.searchQuery &&
    prev.onOpenSource === next.onOpenSource &&
    prev.onRetry === next.onRetry &&
    prev.onCopy === next.onCopy
  );
}

export const MessageBubble = memo(MessageBubbleInner, areMessagePropsEqual);


/**
 * src/interfaces/frontend/src/components/chat/ChatComposer.tsx
 * Thanh soạn thảo dạng Capsule Bar Glassmorphism nổi trên hình nền,
 * hỗ trợ textarea tự co dãn tối đa ~144px, đính kèm tệp và nút gửi/dừng spring bounce.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  ArrowUp,
  AudioLines,
  Loader2,
  Mic,
  MicOff,
  Paperclip,
  Plus,
  Square,
  X,
} from "lucide-react";
import { transcribeAudio } from "../../services/api";
import { formatFileSize } from "../../utils/formatters";
import { playVoiceRecordStartSound } from "../../utils/sound";
import {
  detectBrowserName,
  getSpeechRecognitionConstructor,
  requestMicrophoneStream,
  startAudioSilenceRecorder,
  VoiceAccessError,
  type AudioSilenceRecorderHandle,
  type SpeechRecognitionCtrl,
  type SpeechRecognitionErrEvtLike,
  type SpeechRecognitionEvtLike,
} from "../../utils/voice";
import type { ToastType } from "../../types/chat";

export interface ChatComposerProps {
  isGenerating: boolean;
  selectedFiles: File[];
  onAddFiles: (files: FileList | File[]) => void;
  onRemoveFile: (idx: number) => void;
  onSendMessage: (text: string) => void;
  onStopGeneration?: () => void;
  onOpenVoiceChat?: () => void;
  onShowToast?: (message: string, type?: ToastType, durationMs?: number) => void;
  draftInsertTrigger?: { text: string; id: number } | null;
}

const MAX_TEXTAREA_HEIGHT_PX = 144;

function getAttachmentIcon(fileName: string): "PDF" | "DOCX" | "TXT" | "FILE" {
  const ext = (fileName || "").split(".").pop()?.toLowerCase() || "";
  if (ext === "pdf") return "PDF";
  if (ext === "docx" || ext === "doc") return "DOCX";
  if (ext === "txt") return "TXT";
  return "FILE";
}

export function ChatComposer({
  isGenerating,
  selectedFiles,
  onAddFiles,
  onRemoveFile,
  onSendMessage,
  onStopGeneration,
  onOpenVoiceChat,
  onShowToast,
  draftInsertTrigger,
}: ChatComposerProps) {
  const [text, setText] = useState("");
  const [isMultiline, setIsMultiline] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isAttachMenuOpen, setIsAttachMenuOpen] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const attachMenuRef = useRef<HTMLDivElement | null>(null);
  const recognitionRef = useRef<SpeechRecognitionCtrl | null>(null);
  const audioRecorderRef = useRef<AudioSilenceRecorderHandle | null>(null);
  const sttAbortRef = useRef<AbortController | null>(null);
  const baseSpeechTextRef = useRef("");
  const isMountedRef = useRef(true);

  useEffect(() => {
    if (!isAttachMenuOpen) return;
    const handleOutsideClick = (e: MouseEvent) => {
      if (
        attachMenuRef.current &&
        !attachMenuRef.current.contains(e.target as Node)
      ) {
        setIsAttachMenuOpen(false);
      }
    };
    const handleEsc = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsAttachMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", handleOutsideClick);
    window.addEventListener("keydown", handleEsc);
    return () => {
      window.removeEventListener("mousedown", handleOutsideClick);
      window.removeEventListener("keydown", handleEsc);
    };
  }, [isAttachMenuOpen]);

  const resizeTextarea = useCallback((el?: HTMLTextAreaElement | null) => {
    const target = el ?? textareaRef.current;
    if (!target) return;
    target.style.height = "auto";
    const nextHeight = Math.min(target.scrollHeight, MAX_TEXTAREA_HEIGHT_PX);
    target.style.height = `${Math.max(24, nextHeight)}px`;
    setIsMultiline(target.scrollHeight > 44);
  }, []);

  // Nhận trigger chèn mẫu câu hỏi từ bên ngoài (vd: MiniRail Quy chế đào tạo)
  useEffect(() => {
    if (!draftInsertTrigger || !draftInsertTrigger.text) return;
    setText(draftInsertTrigger.text);
    const raf = requestAnimationFrame(() => {
      resizeTextarea();
      textareaRef.current?.focus();
    });
    return () => cancelAnimationFrame(raf);
  }, [draftInsertTrigger, resizeTextarea]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {
          // ignore cleanup abort errors
        }
        recognitionRef.current = null;
      }
      if (audioRecorderRef.current) {
        audioRecorderRef.current.cancel();
        audioRecorderRef.current = null;
      }
      if (sttAbortRef.current) {
        sttAbortRef.current.abort();
        sttAbortRef.current = null;
      }
    };
  }, []);

  const canSend =
    !isGenerating &&
    !isTranscribing &&
    (text.trim().length > 0 || selectedFiles.length > 0);
  const showMicButton =
    !isGenerating && (text.length === 0 || isRecording || isTranscribing);

  const applySpokenText = useCallback(
    (spokenText: string) => {
      const cleaned = spokenText.trim();
      const prefix = baseSpeechTextRef.current;
      const nextText = prefix
        ? cleaned
          ? `${prefix} ${cleaned}`
          : prefix
        : cleaned;
      setText(nextText);
      requestAnimationFrame(() => resizeTextarea());
    },
    [resizeTextarea]
  );

  const startWebSpeechDictation = useCallback(
    (SpeechRecognitionCtor: NonNullable<ReturnType<typeof getSpeechRecognitionConstructor>>) => {
      try {
        const recognition = new SpeechRecognitionCtor();
        recognition.lang = "vi-VN";
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;

        recognitionRef.current = recognition;

        recognition.onstart = () => {
          setIsRecording(true);
          playVoiceRecordStartSound();
        };

        recognition.onresult = (event: SpeechRecognitionEvtLike) => {
          const parts: string[] = [];
          for (let i = 0; i < event.results.length; i += 1) {
            const transcript = event.results[i]?.[0]?.transcript?.trim();
            if (transcript) parts.push(transcript);
          }
          applySpokenText(parts.join(" "));
        };

        recognition.onerror = (event: SpeechRecognitionErrEvtLike) => {
          if (event.error === "aborted") {
            setIsRecording(false);
            recognitionRef.current = null;
            return;
          }

          if (
            event.error === "not-allowed" ||
            event.error === "service-not-allowed"
          ) {
            onShowToast?.(
              "Vui lòng cấp quyền truy cập micro cho trình duyệt để thu âm.",
              "warning",
              3500
            );
          } else if (event.error === "audio-capture") {
            onShowToast?.(
              "Không tìm thấy thiết bị micro khả dụng.",
              "warning",
              3200
            );
          } else if (event.error === "no-speech") {
            onShowToast?.(
              "Không phát hiện giọng nói, vui lòng thử lại.",
              "info",
              2500
            );
          } else if (event.error === "network") {
            onShowToast?.(
              "Không thể kết nối dịch vụ nhận dạng giọng nói của trình duyệt.",
              "warning",
              3200
            );
          } else {
            onShowToast?.(
              "Không thể khởi tạo nhận dạng giọng nói trên trình duyệt.",
              "warning",
              3000
            );
          }
          setIsRecording(false);
          recognitionRef.current = null;
        };

        recognition.onend = () => {
          setIsRecording(false);
          recognitionRef.current = null;
          requestAnimationFrame(() => {
            resizeTextarea();
            textareaRef.current?.focus();
          });
        };

        recognition.start();
      } catch {
        setIsRecording(false);
        recognitionRef.current = null;
        onShowToast?.(
          "Không thể bắt đầu thu âm giọng nói trên trình duyệt.",
          "warning",
          3000
        );
      }
    },
    [applySpokenText, onShowToast, resizeTextarea]
  );

  const handleToggleVoiceInput = useCallback(async () => {
    if (isTranscribing) return;

    if (isRecording) {
      try {
        recognitionRef.current?.stop();
        audioRecorderRef.current?.stop();
      } catch {
        setIsRecording(false);
        recognitionRef.current = null;
        audioRecorderRef.current = null;
      }
      return;
    }

    baseSpeechTextRef.current = text.trim();
    const SpeechRecognitionCtor = getSpeechRecognitionConstructor();
    const browser = detectBrowserName();

    // Đường dẫn A: Chrome / Edge / Safari dùng Web Speech API
    if (SpeechRecognitionCtor) {
      // Safari yêu cầu gọi recognition.start() đồng bộ trong click handler; Safari tự bật popup xin quyền mic
      if (browser === "Safari") {
        startWebSpeechDictation(SpeechRecognitionCtor);
        return;
      }

      // Chrome & Edge: chủ động hỏi quyền Micro trước qua getUserMedia ở lần đầu
      try {
        await requestMicrophoneStream({ keepStream: false });
      } catch (err) {
        const msg =
          err instanceof VoiceAccessError
            ? err.message
            : "Vui lòng cấp quyền truy cập micro cho trình duyệt để thu âm.";
        onShowToast?.(msg, "warning", 3500);
        return;
      }

      if (!isMountedRef.current) return;
      startWebSpeechDictation(SpeechRecognitionCtor);
      return;
    }

    // Đường dẫn B: Firefox hoặc trình duyệt không có SpeechRecognition -> xin quyền Mic + MediaRecorder + Whisper STT backend
    let stream: MediaStream | null = null;
    try {
      stream = await requestMicrophoneStream({ keepStream: true });
    } catch (err) {
      const msg =
        err instanceof VoiceAccessError
          ? err.message
          : "Không thể truy cập micro trên trình duyệt.";
      onShowToast?.(msg, "warning", 3600);
      return;
    }

    if (!stream || !isMountedRef.current) {
      return;
    }

    try {
      setIsRecording(true);
      playVoiceRecordStartSound();

      audioRecorderRef.current = startAudioSilenceRecorder(stream, {
        autoStopOnSilence: true,
        silenceDurationMs: 1400,
        noSpeechTimeoutMs: 10000,
        onNoSpeech: () => {
          if (!isMountedRef.current) return;
          setIsRecording(false);
          audioRecorderRef.current = null;
          onShowToast?.(
            "Không phát hiện giọng nói, vui lòng thử lại.",
            "info",
            2500
          );
        },
        onError: (msg) => {
          if (!isMountedRef.current) return;
          setIsRecording(false);
          audioRecorderRef.current = null;
          onShowToast?.(msg, "warning", 3200);
        },
        onComplete: (audioBase64, mimeType) => {
          if (!isMountedRef.current) return;
          setIsRecording(false);
          audioRecorderRef.current = null;

          if (!audioBase64) return;

          setIsTranscribing(true);
          const controller = new AbortController();
          sttAbortRef.current = controller;

          void transcribeAudio(audioBase64, mimeType, controller.signal)
            .then((transcript) => {
              if (!isMountedRef.current || controller.signal.aborted) return;
              if (!transcript) {
                onShowToast?.(
                  "Không nghe rõ nội dung, vui lòng thử nói lại.",
                  "info",
                  2400
                );
                return;
              }
              applySpokenText(transcript);
              requestAnimationFrame(() => {
                resizeTextarea();
                textareaRef.current?.focus();
              });
            })
            .catch((err: unknown) => {
              if (!isMountedRef.current || controller.signal.aborted) return;
              const msg =
                err instanceof Error && err.message
                  ? err.message
                  : "Không thể chuyển giọng nói thành văn bản.";
              onShowToast?.(msg, "warning", 3500);
            })
            .finally(() => {
              if (sttAbortRef.current === controller) {
                sttAbortRef.current = null;
              }
              if (isMountedRef.current) {
                setIsTranscribing(false);
              }
            });
        },
      });
    } catch (err) {
      setIsRecording(false);
      audioRecorderRef.current = null;
      const msg =
        err instanceof VoiceAccessError
          ? err.message
          : "Không thể bắt đầu thu âm trên trình duyệt.";
      onShowToast?.(msg, "warning", 3200);
    }
  }, [
    applySpokenText,
    isRecording,
    isTranscribing,
    onShowToast,
    resizeTextarea,
    startWebSpeechDictation,
    text,
  ]);

  const submitCurrent = useCallback(() => {
    if (!canSend) return;
    if (isRecording) {
      try {
        recognitionRef.current?.stop();
        audioRecorderRef.current?.cancel();
        audioRecorderRef.current = null;
      } catch {
        // ignore stop errors when submitting
      }
    }
    const trimmed = text.trim();
    onSendMessage(trimmed);
    setText("");
    setIsMultiline(false);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.focus();
    }
  }, [canSend, isRecording, onSendMessage, text]);

  const handleFormSubmit = (e: FormEvent) => {
    e.preventDefault();
    submitCurrent();
  };

  const handleTextareaChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    resizeTextarea(e.target);
  };

  const handleTextareaKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submitCurrent();
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onAddFiles(e.target.files);
      e.target.value = "";
    }
  };

  return (
    <footer className="telegram-input-pane telegram-input-dock relative z-[12] shrink-0 px-2.5 pb-[max(16px,env(safe-area-inset-bottom))] pt-2 sm:px-5 sm:pb-7">
      <AnimatePresence>
        {selectedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            transition={{ type: "spring", stiffness: 420, damping: 28 }}
            className="attachment-preview-area mx-auto mb-2 flex max-w-[860px] flex-wrap gap-2"
          >
            {selectedFiles.map((file, index) => {
              const badgeLabel = getAttachmentIcon(file.name);
              const badgeColorClass =
                badgeLabel === "PDF"
                  ? "bg-red-100 text-red-600 dark:bg-[#3D1A1A] dark:text-red-400"
                  : badgeLabel === "DOCX"
                  ? "bg-blue-100 text-blue-700 dark:bg-[#172554] dark:text-blue-400"
                  : "bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-slate-300";

              return (
                <motion.div
                  key={`${file.name}_${file.size}_${index}`}
                  initial={{ opacity: 0, scale: 0.92 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="pre-attach-chip inline-flex items-center gap-2 rounded-lg border border-[var(--border-card)] bg-[var(--bg-card)] px-2.5 py-1.5 text-[12px] shadow-sm"
                >
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-extrabold tracking-wide ${badgeColorClass}`}
                  >
                    {badgeLabel}
                  </span>
                  <span
                    className="chip-name max-w-[170px] truncate font-semibold text-[var(--text-primary)]"
                    title={file.name}
                  >
                    {file.name}
                  </span>
                  <span className="chip-size text-[10.5px] text-[var(--text-muted)]">
                    ({formatFileSize(file.size)})
                  </span>
                  <button
                    type="button"
                    onClick={() => onRemoveFile(index)}
                    className="btn-remove-attach rounded-full p-0.5 text-[var(--text-muted)] transition-colors hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40 dark:hover:text-red-400"
                    title="Gỡ tệp"
                    aria-label={`Gỡ tệp ${file.name}`}
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.doc,.txt"
        multiple
        className="hidden"
        onChange={handleFileChange}
      />

      <form
        onSubmit={handleFormSubmit}
        className="telegram-input-form chatgpt-input-bar relative mx-auto flex min-h-[52px] w-full max-w-[860px] items-end gap-1.5 border border-[rgba(8,117,201,0.22)] bg-[var(--bg-composer-glass)] px-2.5 py-1.5 shadow-[0_4px_18px_rgba(15,23,42,0.08)] backdrop-blur-md transition-[border-radius,border-color,box-shadow] duration-200 sm:gap-2"
        style={{
          borderRadius: isMultiline ? "24px" : "9999px",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          transform: "translate3d(0, 0, 0)",
        }}
      >
        <div ref={attachMenuRef} className="relative mb-0.5 shrink-0">
          <AnimatePresence>
            {isAttachMenuOpen && (
              <motion.div
                initial={{ opacity: 0, y: 8, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 6, scale: 0.95 }}
                transition={{ type: "spring", stiffness: 460, damping: 30 }}
                style={{ willChange: "transform, opacity" }}
                role="menu"
                aria-label="Tùy chọn đính kèm"
                className="composer-attach-menu absolute bottom-12 left-0 z-30 min-w-[180px] rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-sidebar)] p-1.5 shadow-[var(--shadow-dropdown)]"
              >
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setIsAttachMenuOpen(false);
                    fileInputRef.current?.click();
                  }}
                  className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13.5px] font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover-chat)] hover:text-[var(--primary)] dark:hover:text-[#6AB3F3]"
                >
                  <Paperclip className="h-4 w-4 shrink-0 text-[var(--text-secondary)]" />
                  <span>Tải tệp lên</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            type="button"
            onClick={() => setIsAttachMenuOpen((prev) => !prev)}
            aria-expanded={isAttachMenuOpen}
            aria-haspopup="menu"
            className={`btn-bar-action btn-attach flex h-9 w-9 items-center justify-center rounded-full transition-colors ${
              isAttachMenuOpen
                ? "bg-[rgba(8,117,201,0.14)] text-[var(--primary)] dark:bg-white/15 dark:text-[#6AB3F3]"
                : "text-[var(--text-secondary)] hover:bg-[rgba(8,117,201,0.1)] hover:text-[var(--primary)] dark:hover:bg-white/10 dark:hover:text-[#6AB3F3]"
            }`}
            title="Thêm nội dung"
            aria-label="Mở menu đính kèm"
          >
            <Plus
              className={`h-5 w-5 transition-transform duration-200 ${
                isAttachMenuOpen ? "rotate-45" : ""
              }`}
            />
          </button>
        </div>

        <div className="bar-textarea-wrapper flex min-h-[36px] min-w-0 flex-1 items-center px-1">
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={handleTextareaChange}
            onKeyDown={handleTextareaKeyDown}
            placeholder="Hỏi UET Chatbot"
            autoComplete="off"
            aria-label="Nhập câu hỏi cho UET Chatbot"
            className="telegram-textarea max-h-[144px] min-h-[24px] w-full resize-none border-0 bg-transparent py-1 text-[15px] leading-[1.45] text-[var(--text-primary)] shadow-none outline-none placeholder:text-[var(--text-muted)] placeholder:opacity-85 focus:border-0 focus:shadow-none focus:outline-none focus:ring-0 focus-visible:outline-none"
          />
        </div>

        <div className="bar-actions-right mb-0.5 flex shrink-0 items-center gap-1">
          {isGenerating && onStopGeneration ? (
            <motion.button
              type="button"
              onClick={onStopGeneration}
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.92 }}
              transition={{ type: "spring", stiffness: 460, damping: 25 }}
              style={{ willChange: "transform" }}
              className="btn-bar-send flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-full bg-red-500 text-white shadow-[0_2px_8px_rgba(239,68,68,0.35)] hover:bg-red-600"
              title="Dừng tạo phản hồi"
              aria-label="Dừng tạo phản hồi"
            >
              <Square className="h-4 w-4 fill-current" />
            </motion.button>
          ) : (
            <>
              {showMicButton && (
                <motion.button
                  type="button"
                  disabled={isTranscribing}
                  onClick={() => void handleToggleVoiceInput()}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  whileHover={!isTranscribing ? { scale: 1.06 } : undefined}
                  whileTap={!isTranscribing ? { scale: 0.92 } : undefined}
                  transition={{ type: "spring", stiffness: 460, damping: 25 }}
                  style={{ willChange: "transform" }}
                  className={`btn-bar-action btn-voice-input flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-full transition-colors ${
                    isRecording
                      ? "bg-red-500 text-white shadow-[0_2px_10px_rgba(239,68,68,0.4)] animate-pulse hover:bg-red-600"
                      : isTranscribing
                      ? "cursor-wait bg-[rgba(8,117,201,0.12)] text-[var(--primary)] dark:text-[#6AB3F3]"
                      : "text-[var(--text-secondary)] hover:bg-[rgba(8,117,201,0.1)] hover:text-[var(--primary)] dark:hover:bg-white/10 dark:hover:text-[#6AB3F3]"
                  }`}
                  title={
                    isTranscribing
                      ? "Đang nhận dạng giọng nói..."
                      : isRecording
                      ? "Đang nghe... Bấm để dừng thu âm"
                      : "Nhập câu hỏi bằng giọng nói"
                  }
                  aria-label={
                    isTranscribing
                      ? "Đang nhận dạng giọng nói"
                      : isRecording
                      ? "Dừng thu âm giọng nói"
                      : "Nhập câu hỏi bằng giọng nói"
                  }
                >
                  {isTranscribing ? (
                    <Loader2 className="h-[18px] w-[18px] animate-spin" />
                  ) : isRecording ? (
                    <MicOff className="h-[18px] w-[18px]" strokeWidth={2.2} />
                  ) : (
                    <Mic className="h-[18px] w-[18px]" strokeWidth={2.2} />
                  )}
                </motion.button>
              )}

              {onOpenVoiceChat &&
                text.length === 0 &&
                !isRecording &&
                !isTranscribing && (
                <motion.button
                  type="button"
                  onClick={() => onOpenVoiceChat()}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  whileHover={{ scale: 1.06 }}
                  whileTap={{ scale: 0.92 }}
                  transition={{ type: "spring", stiffness: 460, damping: 25 }}
                  style={{ willChange: "transform" }}
                  className="btn-bar-action btn-live-voice flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-full bg-[rgba(8,117,201,0.1)] text-[var(--primary)] transition-colors hover:bg-[rgba(8,117,201,0.18)] dark:bg-white/10 dark:text-[#38BDF8] dark:hover:bg-white/20"
                  title="Đàm thoại giọng nói trực tiếp"
                  aria-label="Đàm thoại giọng nói trực tiếp"
                >
                  <AudioLines className="h-[18px] w-[18px]" strokeWidth={2.2} />
                </motion.button>
              )}

              <motion.button
                type="submit"
                disabled={!canSend}
                animate={{ scale: canSend ? 1 : 0.94 }}
                whileHover={canSend ? { scale: 1.07 } : undefined}
                whileTap={canSend ? { scale: 0.92 } : undefined}
                transition={{ type: "spring", stiffness: 460, damping: 24 }}
                style={{ willChange: "transform" }}
                className={`btn-bar-send btn-telegram-send flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-full transition-colors ${
                  canSend
                    ? "bg-gradient-to-br from-[#0054A6] to-[#0875C9] text-white shadow-[0_2px_8px_rgba(8,117,201,0.35)] hover:shadow-[0_4px_12px_rgba(8,117,201,0.45)]"
                    : "cursor-not-allowed bg-[var(--border-subtle)] text-[var(--text-muted)] opacity-55 dark:bg-white/10 dark:text-white/30"
                }`}
                title="Gửi câu hỏi (Enter)"
                aria-label="Gửi câu hỏi"
              >
                <ArrowUp className="h-[19px] w-[19px]" strokeWidth={2.5} />
              </motion.button>
            </>
          )}
        </div>
      </form>
    </footer>
  );
}


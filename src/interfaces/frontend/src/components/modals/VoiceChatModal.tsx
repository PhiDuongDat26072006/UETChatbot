/**
 * src/interfaces/frontend/src/components/modals/VoiceChatModal.tsx
 * Chế độ đàm thoại giọng nói trực tiếp (ChatGPT Live Voice Chat style):
 * Tự động lắng nghe bằng Web Speech API (vi-VN), nhận biết ngắt câu tự nhiên
 * sau khoảng lặng, gửi truy vấn ở chế độ voice_mode (trả lời ngắn gọn 1–3 câu)
 * và đọc phản hồi qua SpeechSynthesis rồi tự động nghe lượt tiếp theo.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  Mic,
  MicOff,
  Send,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import {
  playVoiceConnectSound,
  playVoiceEndSound,
} from "../../utils/sound";

type VoicePhase = "idle" | "listening" | "thinking" | "speaking" | "error";

interface SpeechRecognitionAlt {
  transcript: string;
}

interface SpeechRecognitionRes {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionAlt;
}

interface SpeechRecognitionEvt extends Event {
  results: {
    length: number;
    [index: number]: SpeechRecognitionRes;
  };
}

interface SpeechRecognitionErrEvt extends Event {
  error: string;
}

interface SpeechRecognitionCtrl extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: ((ev: Event) => void) | null;
  onresult: ((ev: SpeechRecognitionEvt) => void) | null;
  onspeechend: ((ev: Event) => void) | null;
  onerror: ((ev: SpeechRecognitionErrEvt) => void) | null;
  onend: ((ev: Event) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

type SpeechRecognitionClass = new () => SpeechRecognitionCtrl;

export interface VoiceChatModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSendVoiceQuery: (text: string) => Promise<string | null>;
  onStopGeneration?: () => void;
}

const SILENCE_SUBMIT_MS = 1350;

function getSpeechRecognitionCtor(): SpeechRecognitionClass | null {
  if (typeof window === "undefined") return null;
  const win = window as unknown as {
    SpeechRecognition?: SpeechRecognitionClass;
    webkitSpeechRecognition?: SpeechRecognitionClass;
  };
  return win.SpeechRecognition || win.webkitSpeechRecognition || null;
}

function detectBrowser(): "Firefox" | "Safari" | "Chrome" | "Edge" | "Other" {
  if (typeof navigator === "undefined") return "Other";
  const ua = navigator.userAgent;
  if (/Edg\//i.test(ua)) return "Edge";
  if (/Firefox\//i.test(ua)) return "Firefox";
  if (/Chrome\//i.test(ua) || /CriOS\//i.test(ua)) return "Chrome";
  if (/Safari\//i.test(ua)) return "Safari";
  return "Other";
}

function stripMarkdownForSpeech(raw: string): string {
  return (raw || "")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\$[^$\n]+\$/g, " ")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/^[#>*\-+]+\s*/gm, "")
    .replace(/[*_~]+/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function splitSpeechChunks(text: string): string[] {
  const cleaned = stripMarkdownForSpeech(text);
  if (!cleaned) return [];
  const rawParts = cleaned.split(/(?<=[.!?…])\s+/);
  const chunks: string[] = [];
  let buffer = "";
  for (const part of rawParts) {
    const p = part.trim();
    if (!p) continue;
    if ((buffer + " " + p).length <= 180) {
      buffer = buffer ? `${buffer} ${p}` : p;
    } else {
      if (buffer) chunks.push(buffer);
      buffer = p;
    }
  }
  if (buffer) chunks.push(buffer);
  return chunks;
}

function pickVietnameseVoice(): SpeechSynthesisVoice | null {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return null;
  }
  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return null;
  return (
    voices.find((v) => v.lang.toLowerCase() === "vi-vn") ||
    voices.find((v) => v.lang.toLowerCase().startsWith("vi")) ||
    null
  );
}

export function VoiceChatModal({
  isOpen,
  onClose,
  onSendVoiceQuery,
  onStopGeneration,
}: VoiceChatModalProps) {
  const [phase, setPhase] = useState<VoicePhase>("idle");
  const [liveTranscript, setLiveTranscript] = useState("");
  const [replyText, setReplyText] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isMuted, setIsMuted] = useState(false);

  const isOpenRef = useRef(isOpen);
  isOpenRef.current = isOpen;
  const isMutedRef = useRef(isMuted);
  isMutedRef.current = isMuted;
  const onSendVoiceQueryRef = useRef(onSendVoiceQuery);
  onSendVoiceQueryRef.current = onSendVoiceQuery;

  const recognitionRef = useRef<SpeechRecognitionCtrl | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const transcriptRef = useRef("");
  const submittedRef = useRef(false);
  const intentionalAbortRef = useRef(false);
  const speechTokenRef = useRef(0);

  const clearTimers = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (resumeTimerRef.current) {
      clearTimeout(resumeTimerRef.current);
      resumeTimerRef.current = null;
    }
  }, []);

  const stopSpeechSynthesis = useCallback(() => {
    speechTokenRef.current += 1;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore cancel errors
      }
    }
  }, []);

  const stopRecognition = useCallback((abort = true) => {
    if (recognitionRef.current) {
      const rec = recognitionRef.current;
      recognitionRef.current = null;
      intentionalAbortRef.current = abort;
      try {
        if (abort) rec.abort();
        else rec.stop();
      } catch {
        // ignore stop/abort errors
      }
    }
  }, []);

  const startListeningRef = useRef<() => void>(() => {});

  const speakAndContinue = useCallback(
    (rawReply: string) => {
      const spoken = stripMarkdownForSpeech(rawReply);
      setReplyText(spoken);

      if (
        isMutedRef.current ||
        !spoken ||
        typeof window === "undefined" ||
        !("speechSynthesis" in window)
      ) {
        setPhase("idle");
        if (isOpenRef.current) {
          resumeTimerRef.current = setTimeout(() => {
            if (isOpenRef.current) {
              startListeningRef.current();
            }
          }, 1800);
        }
        return;
      }

      stopSpeechSynthesis();
      const myToken = speechTokenRef.current;
      const chunks = splitSpeechChunks(spoken);
      if (chunks.length === 0) {
        if (isOpenRef.current) startListeningRef.current();
        return;
      }

      setPhase("speaking");
      const preferredVoice = pickVietnameseVoice();

      const speakNext = (idx: number) => {
        if (
          !isOpenRef.current ||
          speechTokenRef.current !== myToken ||
          isMutedRef.current
        ) {
          return;
        }
        if (idx >= chunks.length) {
          if (isOpenRef.current) {
            resumeTimerRef.current = setTimeout(() => {
              if (isOpenRef.current && speechTokenRef.current === myToken) {
                startListeningRef.current();
              }
            }, 350);
          }
          return;
        }

        const utterance = new SpeechSynthesisUtterance(chunks[idx]);
        utterance.lang = "vi-VN";
        utterance.rate = 1.05;
        if (preferredVoice) {
          utterance.voice = preferredVoice;
        }
        utterance.onend = () => {
          speakNext(idx + 1);
        };
        utterance.onerror = () => {
          if (isOpenRef.current && speechTokenRef.current === myToken) {
            setPhase("idle");
          }
        };

        try {
          window.speechSynthesis.speak(utterance);
        } catch {
          setPhase("idle");
        }
      };

      speakNext(0);
    },
    [stopSpeechSynthesis]
  );

  const submitTranscript = useCallback(
    async (rawText: string) => {
      const cleaned = rawText.trim();
      if (!cleaned || submittedRef.current || !isOpenRef.current) return;

      submittedRef.current = true;
      clearTimers();
      stopRecognition(false);
      setPhase("thinking");
      setReplyText("");

      const reply = await onSendVoiceQueryRef.current(cleaned);
      if (!isOpenRef.current) return;

      if (!reply) {
        setPhase("error");
        setErrorMessage("Chưa nhận được phản hồi từ hệ thống. Bạn hãy thử nói lại.");
        return;
      }

      speakAndContinue(reply);
    },
    [clearTimers, speakAndContinue, stopRecognition]
  );

  const startListening = useCallback(() => {
    if (!isOpenRef.current) return;

    clearTimers();
    stopSpeechSynthesis();
    stopRecognition(true);

    const SpeechCtor = getSpeechRecognitionCtor();
    if (!SpeechCtor) {
      const browser = detectBrowser();
      setPhase("error");
      setErrorMessage(
        browser === "Firefox"
          ? "Trình duyệt Firefox chưa hỗ trợ sẵn đàm thoại giọng nói trực tiếp (Web Speech API). Vui lòng mở bằng Chrome, Edge hoặc Safari."
          : "Trình duyệt hiện tại chưa hỗ trợ đàm thoại giọng nói."
      );
      return;
    }

    if (typeof window !== "undefined" && !window.isSecureContext) {
      setPhase("error");
      setErrorMessage(
        "Đàm thoại giọng nói yêu cầu kết nối HTTPS hoặc localhost."
      );
      return;
    }

    transcriptRef.current = "";
    submittedRef.current = false;
    intentionalAbortRef.current = false;
    setLiveTranscript("");
    setErrorMessage("");
    setPhase("listening");

    try {
      const rec = new SpeechCtor();
      rec.lang = "vi-VN";
      rec.continuous = true;
      rec.interimResults = true;
      rec.maxAlternatives = 1;
      recognitionRef.current = rec;

      rec.onstart = () => {
        if (!isOpenRef.current) return;
        setPhase("listening");
      };

      rec.onresult = (event: SpeechRecognitionEvt) => {
        if (!isOpenRef.current || submittedRef.current) return;

        const parts: string[] = [];
        for (let i = 0; i < event.results.length; i += 1) {
          const t = event.results[i]?.[0]?.transcript?.trim();
          if (t) parts.push(t);
        }
        const merged = parts.join(" ").trim();
        transcriptRef.current = merged;
        setLiveTranscript(merged);

        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
        }
        if (merged) {
          silenceTimerRef.current = setTimeout(() => {
            if (isOpenRef.current && !submittedRef.current) {
              void submitTranscript(transcriptRef.current);
            }
          }, SILENCE_SUBMIT_MS);
        }
      };

      rec.onspeechend = () => {
        if (
          isOpenRef.current &&
          !submittedRef.current &&
          transcriptRef.current.trim()
        ) {
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
          }
          silenceTimerRef.current = setTimeout(() => {
            void submitTranscript(transcriptRef.current);
          }, 550);
        }
      };

      rec.onerror = (event: SpeechRecognitionErrEvt) => {
        if (intentionalAbortRef.current || event.error === "aborted") {
          return;
        }
        clearTimers();
        recognitionRef.current = null;
        if (!isOpenRef.current) return;

        if (
          event.error === "not-allowed" ||
          event.error === "service-not-allowed"
        ) {
          setPhase("error");
          setErrorMessage(
            "Vui lòng cấp quyền truy cập micro cho trình duyệt để sử dụng đàm thoại trực tiếp."
          );
        } else if (event.error === "audio-capture") {
          setPhase("error");
          setErrorMessage("Không tìm thấy thiết bị micro trên máy tính.");
        } else if (event.error === "no-speech") {
          setPhase("idle");
          setErrorMessage("Chưa nghe thấy giọng nói, bấm micro để nói lại.");
        } else if (event.error === "network") {
          setPhase("error");
          setErrorMessage(
            "Mất kết nối dịch vụ nhận dạng giọng nói của trình duyệt."
          );
        } else {
          setPhase("error");
          setErrorMessage("Không thể tiếp tục thu âm. Bấm micro để thử lại.");
        }
      };

      rec.onend = () => {
        recognitionRef.current = null;
        if (
          !intentionalAbortRef.current &&
          isOpenRef.current &&
          !submittedRef.current &&
          transcriptRef.current.trim()
        ) {
          void submitTranscript(transcriptRef.current);
        } else if (
          !intentionalAbortRef.current &&
          isOpenRef.current &&
          !submittedRef.current
        ) {
          setPhase((prev) => (prev === "listening" ? "idle" : prev));
        }
      };

      rec.start();
    } catch {
      setPhase("error");
      setErrorMessage("Không thể khởi chạy micro trên trình duyệt.");
    }
  }, [clearTimers, stopRecognition, stopSpeechSynthesis, submitTranscript]);

  startListeningRef.current = startListening;

  useEffect(() => {
    if (isOpen) {
      setReplyText("");
      setLiveTranscript("");
      setErrorMessage("");
      playVoiceConnectSound();
      startListening();
    } else {
      clearTimers();
      stopRecognition(true);
      stopSpeechSynthesis();
      setPhase("idle");
    }
  }, [clearTimers, isOpen, startListening, stopRecognition, stopSpeechSynthesis]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        playVoiceEndSound();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onClose]);

  const handleCloseModal = () => {
    clearTimers();
    stopRecognition(true);
    stopSpeechSynthesis();
    playVoiceEndSound();
    if (phase === "thinking") {
      onStopGeneration?.();
    }
    onClose();
  };

  const handleToggleMute = () => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    if (nextMuted && phase === "speaking") {
      stopSpeechSynthesis();
      startListening();
    }
  };

  const handlePrimaryOrbAction = () => {
    if (phase === "speaking") {
      stopSpeechSynthesis();
      startListening();
      return;
    }
    if (phase === "listening") {
      if (liveTranscript.trim()) {
        void submitTranscript(liveTranscript);
      } else {
        stopRecognition(true);
        setPhase("idle");
      }
      return;
    }
    if (phase === "idle" || phase === "error") {
      startListening();
    }
  };

  const statusHeadline =
    phase === "listening"
      ? "Đang nghe..."
      : phase === "thinking"
      ? "Đang suy nghĩ..."
      : phase === "speaking"
      ? "Đang trả lời..."
      : phase === "error"
      ? "Tạm dừng thu âm"
      : "Sẵn sàng lắng nghe";

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          key="voice-chat-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-[1200] flex flex-col items-center justify-between bg-[rgba(10,16,28,0.86)] px-5 py-6 text-white backdrop-blur-xl"
          role="dialog"
          aria-modal="true"
          aria-labelledby="voice-chat-title"
        >
          {/* Header */}
          <div className="flex w-full max-w-[680px] items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full bg-white p-0.5 shadow">
                <img
                  src="/static/uet-logo.png?v=v14_robot_mascot"
                  alt="UET AI"
                  className="h-full w-full rounded-full object-cover"
                />
              </div>
              <div>
                <h2 id="voice-chat-title" className="text-[15px] font-bold">
                  Đàm thoại trực tiếp UET AI
                </h2>
                <p className="text-[11.5px] text-slate-300">
                  Ngừng nói hết câu để AI tự động trả lời ngắn gọn
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleCloseModal}
              className="flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-slate-200 transition-colors hover:bg-white/20 hover:text-white"
              title="Đóng (Esc)"
              aria-label="Đóng đàm thoại giọng nói"
            >
              <X size={18} />
            </button>
          </div>

          {/* Center Animated Voice Sphere + Subtitles */}
          <div className="my-auto flex w-full max-w-[640px] flex-col items-center justify-center px-2 text-center">
            <button
              type="button"
              onClick={handlePrimaryOrbAction}
              aria-label={statusHeadline}
              className="group relative flex h-44 w-44 items-center justify-center rounded-full focus:outline-none sm:h-52 sm:w-52"
            >
              {/* Outer Ambient Pulse Rings */}
              <motion.span
                aria-hidden="true"
                animate={
                  phase === "listening"
                    ? { scale: [1, 1.24, 1], opacity: [0.45, 0.14, 0.45] }
                    : phase === "speaking"
                    ? { scale: [1, 1.32, 1], opacity: [0.55, 0.18, 0.55] }
                    : phase === "thinking"
                    ? { rotate: 360, scale: [1, 1.08, 1], opacity: 0.4 }
                    : { scale: 1, opacity: 0.18 }
                }
                transition={
                  phase === "thinking"
                    ? { duration: 2.2, repeat: Infinity, ease: "linear" }
                    : { duration: 1.6, repeat: Infinity, ease: "easeInOut" }
                }
                className={`absolute inset-0 rounded-full blur-xl ${
                  phase === "speaking"
                    ? "bg-emerald-400/60"
                    : phase === "error"
                    ? "bg-amber-500/45"
                    : "bg-[#38BDF8]/60"
                }`}
              />

              {/* Main Voice Sphere */}
              <motion.span
                aria-hidden="true"
                animate={
                  phase === "listening"
                    ? { scale: [1, 1.07, 1] }
                    : phase === "speaking"
                    ? { scale: [0.96, 1.1, 0.98, 1.08, 0.96] }
                    : phase === "thinking"
                    ? { scale: [0.95, 1.02, 0.95] }
                    : { scale: 1 }
                }
                transition={{
                  duration: phase === "speaking" ? 1.1 : 1.5,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                className={`relative flex h-32 w-32 items-center justify-center rounded-full shadow-[0_0_48px_rgba(8,117,201,0.55)] transition-colors sm:h-36 sm:w-36 ${
                  phase === "speaking"
                    ? "bg-gradient-to-br from-emerald-400 via-[#0875C9] to-cyan-500"
                    : phase === "thinking"
                    ? "bg-gradient-to-tr from-[#0054A6] via-indigo-500 to-[#38BDF8]"
                    : phase === "error"
                    ? "bg-gradient-to-br from-slate-600 to-slate-800"
                    : "bg-gradient-to-br from-[#38BDF8] via-[#0875C9] to-[#003F88]"
                }`}
              >
                {phase === "listening" ? (
                  <Mic className="h-11 w-11 text-white drop-shadow" />
                ) : phase === "error" ? (
                  <MicOff className="h-10 w-10 text-amber-300" />
                ) : phase === "speaking" ? (
                  <Volume2 className="h-11 w-11 text-white drop-shadow" />
                ) : (
                  <Mic className="h-10 w-10 text-white/85" />
                )}
              </motion.span>
            </button>

            <div className="mt-6 text-[18px] font-bold tracking-tight text-white">
              {statusHeadline}
            </div>

            {/* Live Transcript / Concise AI Reply / Guidance */}
            <div className="mt-3 min-h-[72px] max-w-[540px] px-2">
              {phase === "error" && errorMessage ? (
                <p className="text-[14px] leading-relaxed text-amber-200">
                  {errorMessage}
                </p>
              ) : liveTranscript ? (
                <p className="text-[15px] leading-relaxed text-sky-100">
                  &ldquo;{liveTranscript}&rdquo;
                </p>
              ) : replyText && (phase === "speaking" || phase === "idle") ? (
                <p className="text-[14.5px] leading-relaxed text-slate-200">
                  {replyText}
                </p>
              ) : (
                <p className="text-[13.5px] text-slate-300">
                  {phase === "listening"
                    ? "Hãy đặt câu hỏi về quy chế, điểm số hoặc học vụ UET..."
                    : phase === "thinking"
                    ? "Đang tìm thông tin ngắn gọn nhất cho bạn..."
                    : "Bấm vào biểu tượng micro để bắt đầu nói."}
                </p>
              )}
            </div>
          </div>

          {/* Bottom Dock Controls */}
          <div className="flex items-center justify-center gap-4 pb-2">
            <button
              type="button"
              onClick={handleToggleMute}
              className={`flex h-12 w-12 items-center justify-center rounded-full border transition-colors ${
                isMuted
                  ? "border-amber-400/40 bg-amber-500/20 text-amber-200"
                  : "border-white/15 bg-white/10 text-white hover:bg-white/20"
              }`}
              title={isMuted ? "Bật giọng đọc AI" : "Tắt tiếng giọng đọc AI"}
              aria-label={isMuted ? "Bật giọng đọc AI" : "Tắt tiếng giọng đọc AI"}
            >
              {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
            </button>

            {phase === "listening" && liveTranscript.trim().length > 0 ? (
              <button
                type="button"
                onClick={() => void submitTranscript(liveTranscript)}
                className="inline-flex h-12 items-center gap-2 rounded-full bg-[var(--primary)] px-6 text-[14px] font-semibold text-white shadow-lg transition-colors hover:bg-[var(--primary-hover)]"
              >
                <Send size={17} />
                <span>Gửi ngay</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handlePrimaryOrbAction}
                className="inline-flex h-12 items-center gap-2 rounded-full bg-white/15 px-6 text-[14px] font-semibold text-white transition-colors hover:bg-white/25"
              >
                {phase === "speaking" ? (
                  <>
                    <Mic size={17} />
                    <span>Ngắt lời &amp; nói</span>
                  </>
                ) : phase === "listening" ? (
                  <>
                    <MicOff size={17} />
                    <span>Tạm dừng</span>
                  </>
                ) : (
                  <>
                    <Mic size={17} />
                    <span>Bắt đầu nói</span>
                  </>
                )}
              </button>
            )}

            <button
              type="button"
              onClick={handleCloseModal}
              className="flex h-12 w-12 items-center justify-center rounded-full bg-red-500 text-white shadow-lg transition-colors hover:bg-red-600"
              title="Tắt micro & kết thúc đàm thoại"
              aria-label="Tắt micro và kết thúc đàm thoại"
            >
              <MicOff size={20} />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}


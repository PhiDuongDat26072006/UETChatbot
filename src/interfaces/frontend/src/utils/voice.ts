/**
 * src/interfaces/frontend/src/utils/voice.ts
 * Tiện ích thu âm và xin quyền Micro thống nhất trên mọi trình duyệt
 * (Chrome, Edge, Safari dùng Web Speech API hoặc MediaRecorder; Firefox dùng
 * MediaRecorder + AudioContext phát hiện ngừng nói hết câu + Whisper STT backend).
 */

export interface SpeechRecognitionAltLike {
  transcript: string;
  confidence?: number;
}

export interface SpeechRecognitionResLike {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionAltLike;
}

export interface SpeechRecognitionEvtLike extends Event {
  resultIndex?: number;
  results: {
    length: number;
    [index: number]: SpeechRecognitionResLike;
  };
}

export interface SpeechRecognitionErrEvtLike extends Event {
  error: string;
  message?: string;
}

export interface SpeechRecognitionCtrl extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: ((this: SpeechRecognitionCtrl, ev: Event) => void) | null;
  onresult:
    | ((this: SpeechRecognitionCtrl, ev: SpeechRecognitionEvtLike) => void)
    | null;
  onspeechend: ((this: SpeechRecognitionCtrl, ev: Event) => void) | null;
  onerror:
    | ((this: SpeechRecognitionCtrl, ev: SpeechRecognitionErrEvtLike) => void)
    | null;
  onend: ((this: SpeechRecognitionCtrl, ev: Event) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

export type SpeechRecognitionCtor = new () => SpeechRecognitionCtrl;

export type SupportedBrowser =
  | "Edge"
  | "Chrome"
  | "Firefox"
  | "Safari"
  | "Other";

export function detectBrowserName(): SupportedBrowser {
  if (typeof navigator === "undefined") return "Other";
  const ua = navigator.userAgent;
  if (/Edg\//i.test(ua)) return "Edge";
  if (/Firefox\//i.test(ua)) return "Firefox";
  if (/Chrome\//i.test(ua) || /CriOS\//i.test(ua)) return "Chrome";
  if (/Safari\//i.test(ua)) return "Safari";
  return "Other";
}

export function getSpeechRecognitionConstructor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const win = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return win.SpeechRecognition || win.webkitSpeechRecognition || null;
}

export class VoiceAccessError extends Error {
  public readonly code:
    | "insecure"
    | "unsupported"
    | "denied"
    | "no_device"
    | "busy"
    | "unknown";

  constructor(
    code: VoiceAccessError["code"],
    message: string
  ) {
    super(message);
    this.name = "VoiceAccessError";
    this.code = code;
  }
}

export function stopMediaStream(stream: MediaStream | null | undefined): void {
  if (!stream) return;
  try {
    stream.getTracks().forEach((track) => {
      try {
        track.stop();
      } catch {
        // ignore stop track errors
      }
    });
  } catch {
    // ignore stream cleanup errors
  }
}

/**
 * Kích hoạt hộp thoại xin quyền sử dụng Micro của trình duyệt ở lần đầu
 * người dùng bấm vào tính năng giọng nói (áp dụng cho cả Firefox, Chrome, Edge, Safari).
 */
export async function requestMicrophoneStream(options?: {
  keepStream?: boolean;
}): Promise<MediaStream | null> {
  if (typeof window === "undefined") {
    throw new VoiceAccessError(
      "unsupported",
      "Môi trường hiện tại không hỗ trợ thu âm."
    );
  }

  if (!window.isSecureContext) {
    throw new VoiceAccessError(
      "insecure",
      "Tính năng giọng nói yêu cầu kết nối HTTPS hoặc localhost."
    );
  }

  if (
    typeof navigator === "undefined" ||
    !navigator.mediaDevices ||
    typeof navigator.mediaDevices.getUserMedia !== "function"
  ) {
    throw new VoiceAccessError(
      "unsupported",
      "Trình duyệt hiện tại không hỗ trợ API truy cập micro."
    );
  }

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    const domErr = err as { name?: string };
    const name = domErr?.name || "";
    if (name === "NotAllowedError" || name === "PermissionDeniedError") {
      throw new VoiceAccessError(
        "denied",
        "Vui lòng cấp quyền truy cập micro cho trình duyệt để sử dụng tính năng giọng nói."
      );
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      throw new VoiceAccessError(
        "no_device",
        "Không tìm thấy thiết bị micro khả dụng trên máy tính."
      );
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      throw new VoiceAccessError(
        "busy",
        "Micro đang được sử dụng bởi ứng dụng khác hoặc không thể đọc tín hiệu."
      );
    }
    throw new VoiceAccessError(
      "unknown",
      "Không thể khởi chạy micro trên trình duyệt."
    );
  }

  if (options?.keepStream) {
    return stream;
  }

  stopMediaStream(stream);
  return null;
}

export function pickSupportedAudioMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  for (const candidate of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(candidate)) {
        return candidate;
      }
    } catch {
      // ignore unsupported check
    }
  }
  return "";
}

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const commaIdx = result.indexOf(",");
      resolve(commaIdx >= 0 ? result.slice(commaIdx + 1) : result);
    };
    reader.onerror = () => {
      reject(new Error("Không thể đọc dữ liệu âm thanh."));
    };
    reader.readAsDataURL(blob);
  });
}

export interface AudioSilenceRecorderOptions {
  silenceDurationMs?: number;
  speechThreshold?: number;
  noSpeechTimeoutMs?: number;
  maxDurationMs?: number;
  autoStopOnSilence?: boolean;
  onSpeechStart?: () => void;
  onSilenceTimeout?: () => void;
  onNoSpeech?: () => void;
  onComplete: (audioBase64: string, mimeType: string, hasSpoken: boolean) => void;
  onError?: (message: string) => void;
}

export interface AudioSilenceRecorderHandle {
  stop: () => void;
  cancel: () => void;
}

/**
 * Thu âm từ MediaStream kết hợp Web Audio API phân tích năng lượng giọng nói (RMS)
 * để tự động nhận biết khi người dùng nói xong câu trên Firefox và mọi trình duyệt hỗ trợ MediaRecorder.
 */
export function startAudioSilenceRecorder(
  stream: MediaStream,
  options: AudioSilenceRecorderOptions
): AudioSilenceRecorderHandle {
  if (typeof MediaRecorder === "undefined") {
    stopMediaStream(stream);
    throw new VoiceAccessError(
      "unsupported",
      "Trình duyệt không hỗ trợ ghi âm MediaRecorder."
    );
  }

  const silenceDurationMs = options.silenceDurationMs ?? 1350;
  const speechThreshold = options.speechThreshold ?? 0.016;
  const noSpeechTimeoutMs = options.noSpeechTimeoutMs ?? 12000;
  const maxDurationMs = options.maxDurationMs ?? 35000;
  const autoStopOnSilence = options.autoStopOnSilence ?? true;

  const preferredMime = pickSupportedAudioMimeType();
  const recorder = preferredMime
    ? new MediaRecorder(stream, { mimeType: preferredMime })
    : new MediaRecorder(stream);
  const actualMime =
    recorder.mimeType || preferredMime || "audio/webm";

  const chunks: BlobPart[] = [];
  const startedAt = performance.now();
  let lastSpeechAt = startedAt;
  let hasSpoken = false;
  let cancelled = false;
  let finished = false;
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  let maxTimer: ReturnType<typeof setTimeout> | null = null;
  let audioCtx: AudioContext | null = null;

  const cleanupMonitoring = () => {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
    if (maxTimer) {
      clearTimeout(maxTimer);
      maxTimer = null;
    }
    if (audioCtx) {
      const ctx = audioCtx;
      audioCtx = null;
      void ctx.close().catch(() => {});
    }
    stopMediaStream(stream);
  };

  const stopInternal = () => {
    if (finished || cancelled) return;
    if (recorder.state === "recording") {
      try {
        recorder.stop();
      } catch {
        cleanupMonitoring();
      }
    }
  };

  recorder.ondataavailable = (ev: BlobEvent) => {
    if (ev.data && ev.data.size > 0) {
      chunks.push(ev.data);
    }
  };

  recorder.onerror = () => {
    if (finished) return;
    finished = true;
    cleanupMonitoring();
    if (!cancelled) {
      options.onError?.("Có lỗi xảy ra khi thu âm từ micro.");
    }
  };

  recorder.onstop = () => {
    if (finished) return;
    finished = true;
    cleanupMonitoring();
    if (cancelled) return;

    const blob = new Blob(chunks, { type: actualMime });
    if (blob.size < 400) {
      options.onComplete("", actualMime, hasSpoken);
      return;
    }

    void blobToBase64(blob)
      .then((b64) => {
        if (!cancelled) {
          options.onComplete(b64, actualMime, hasSpoken);
        }
      })
      .catch(() => {
        if (!cancelled) {
          options.onError?.("Không thể xử lý dữ liệu âm thanh vừa thu.");
        }
      });
  };

  // Setup Web Audio API RMS silence / speech endpoint detector
  try {
    const AudioCtxCtor =
      typeof window !== "undefined"
        ? window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext })
            .webkitAudioContext
        : undefined;

    if (AudioCtxCtor) {
      audioCtx = new AudioCtxCtor();
      if (audioCtx.state === "suspended") {
        void audioCtx.resume().catch(() => {});
      }
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.fftSize);
      pollInterval = setInterval(() => {
        if (finished || cancelled || recorder.state !== "recording") return;

        analyser.getByteTimeDomainData(dataArray);
        let sumSq = 0;
        for (let i = 0; i < dataArray.length; i += 1) {
          const sample = (dataArray[i] - 128) / 128;
          sumSq += sample * sample;
        }
        const rms = Math.sqrt(sumSq / dataArray.length);
        const now = performance.now();

        if (rms >= speechThreshold) {
          if (!hasSpoken) {
            hasSpoken = true;
            options.onSpeechStart?.();
          }
          lastSpeechAt = now;
        } else if (hasSpoken && autoStopOnSilence) {
          if (now - lastSpeechAt >= silenceDurationMs) {
            options.onSilenceTimeout?.();
            stopInternal();
          }
        } else if (!hasSpoken && noSpeechTimeoutMs > 0) {
          if (now - startedAt >= noSpeechTimeoutMs) {
            cancelled = true;
            if (recorder.state === "recording") {
              try {
                recorder.stop();
              } catch {
                cleanupMonitoring();
              }
            } else {
              cleanupMonitoring();
            }
            options.onNoSpeech?.();
          }
        }
      }, 60);
    }
  } catch {
    // Fallback cleanly without Web Audio VAD if AudioContext creation fails
  }

  maxTimer = setTimeout(() => {
    stopInternal();
  }, maxDurationMs);

  recorder.start(250);

  return {
    stop: stopInternal,
    cancel: () => {
      if (finished) return;
      cancelled = true;
      if (recorder.state === "recording") {
        try {
          recorder.stop();
        } catch {
          cleanupMonitoring();
        }
      } else {
        cleanupMonitoring();
      }
    },
  };
}


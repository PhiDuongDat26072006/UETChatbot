/**
 * src/interfaces/frontend/src/utils/sound.ts
 * Quản lý tùy chọn âm thanh thông báo (LocalStorage uet_ai_sound_enabled)
 * và phát hiệu ứng âm thanh Telegram an toàn trước chính sách autoplay của trình duyệt.
 */

export const SOUND_STORAGE_KEY = "uet_ai_sound_enabled";

const NOTIFICATION_AUDIO_SRC = "/static/audio/msg_incoming.mp3";
const MESSAGE_SENT_AUDIO_SRC = "/static/audio/message_sent.mp3";
const VOICE_CONNECT_AUDIO_SRC = "/static/audio/group_call_connect.mp3";
const VOICE_END_AUDIO_SRC = "/static/audio/group_call_end.mp3";
const VOICE_RECORD_START_AUDIO_SRC = "/static/audio/group_call_recording_start.mp3";
const DEFAULT_NOTIFICATION_VOLUME = 0.55;
const DEFAULT_MESSAGE_SENT_VOLUME = 0.5;
const DEFAULT_VOICE_VOLUME = 0.42;

let memorySoundEnabled: boolean | null = null;
const audioCache = new Map<string, HTMLAudioElement>();

function hasLocalStorage(): boolean {
  try {
    return (
      typeof window !== "undefined" &&
      typeof window.localStorage !== "undefined"
    );
  } catch {
    return false;
  }
}

export function getSoundEnabled(): boolean {
  if (hasLocalStorage()) {
    try {
      const stored = window.localStorage.getItem(SOUND_STORAGE_KEY);
      if (stored === "false") return false;
      if (stored === "true") return true;
    } catch {
      // ignore storage access errors
    }
  }
  return memorySoundEnabled !== null ? memorySoundEnabled : true;
}

export function setSoundEnabled(enabled: boolean): boolean {
  const normalized = Boolean(enabled);
  memorySoundEnabled = normalized;
  if (hasLocalStorage()) {
    try {
      window.localStorage.setItem(SOUND_STORAGE_KEY, String(normalized));
    } catch {
      // ignore storage write errors
    }
  }
  return normalized;
}

function playAudioClip(src: string, volume: number): void {
  if (!getSoundEnabled()) return;
  if (typeof window === "undefined" || typeof Audio === "undefined") return;

  try {
    let audio = audioCache.get(src);
    if (!audio) {
      audio = new Audio(src);
      audio.preload = "auto";
      audioCache.set(src, audio);
    }

    let target = audio;
    if (!audio.paused && !audio.ended) {
      target = new Audio(src);
    } else if (audio.currentTime > 0) {
      try {
        audio.currentTime = 0;
      } catch {
        target = new Audio(src);
      }
    }

    target.volume = Math.min(1, Math.max(0, volume));
    const playPromise = target.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
  } catch {
    // ignore browser autoplay or audio decoding errors
  }
}

export function playNotificationSound(): void {
  playAudioClip(NOTIFICATION_AUDIO_SRC, DEFAULT_NOTIFICATION_VOLUME);
}

export function playMessageSentSound(): void {
  playAudioClip(MESSAGE_SENT_AUDIO_SRC, DEFAULT_MESSAGE_SENT_VOLUME);
}

export function playVoiceConnectSound(): void {
  playAudioClip(VOICE_CONNECT_AUDIO_SRC, DEFAULT_VOICE_VOLUME);
}

export function playVoiceEndSound(): void {
  playAudioClip(VOICE_END_AUDIO_SRC, DEFAULT_VOICE_VOLUME);
}

export function playVoiceRecordStartSound(): void {
  playAudioClip(VOICE_RECORD_START_AUDIO_SRC, DEFAULT_VOICE_VOLUME);
}

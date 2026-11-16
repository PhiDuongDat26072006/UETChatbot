/**
 * src/interfaces/frontend/src/components/chat/ChatWallpaper.tsx
 * Lớp hình nền cố định (position: fixed, pointer-events: none, z-index: 0)
 * sử dụng gradient + CSS mask-image SVG pattern không repaint khi cuộn.
 */

import type { ThemeMode, WallpaperMode } from "../../types/chat";

export interface ChatWallpaperProps {
  wallpaper: WallpaperMode;
  theme: ThemeMode;
}

interface WallpaperConfig {
  lightBackground: string;
  darkBackground: string;
  maskUrl: string | null;
  maskColor: string;
  maskOpacity: number;
}

function resolveWallpaperConfig(
  wallpaper: WallpaperMode,
  theme: ThemeMode
): WallpaperConfig {
  if (wallpaper === "plain") {
    return {
      lightBackground: "#EFF7FE",
      darkBackground: "#0E1621",
      maskUrl: null,
      maskColor: "transparent",
      maskOpacity: 0,
    };
  }

  if (wallpaper === "dark-night") {
    return {
      lightBackground: "linear-gradient(135deg, #0E1621 0%, #17212B 100%)",
      darkBackground: "linear-gradient(135deg, #0E1621 0%, #17212B 100%)",
      maskUrl: "url('/static/tdesktop-background.svg')",
      maskColor: "#6AB3F3",
      maskOpacity: 0.11,
    };
  }

  return {
    lightBackground: "linear-gradient(145deg, #EFF7FE 0%, #E3F1FA 100%)",
    darkBackground: "linear-gradient(145deg, #0E1621 0%, #131D29 100%)",
    maskUrl: "url('/static/tdesktop-background.svg')",
    maskColor: theme === "dark" ? "#6AB3F3" : "#4D7BA8",
    maskOpacity: theme === "dark" ? 0.11 : 0.16,
  };
}

export function ChatWallpaper({ wallpaper, theme }: ChatWallpaperProps) {
  const cfg = resolveWallpaperConfig(wallpaper, theme);
  const isDark = theme === "dark";

  return (
    <div
      className="chat-wallpaper-layer"
      data-wallpaper={wallpaper}
      data-theme={theme}
      aria-hidden="true"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100dvh",
        pointerEvents: "none",
        zIndex: 0,
        overflow: "hidden",
        contain: "strict",
        transform: "translate3d(0, 0, 0)",
      }}
    >
      <div
        className="wallpaper-gradient-bg wallpaper-gradient-layer"
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          zIndex: 0,
          background: cfg.lightBackground,
          opacity: isDark ? 0 : 1,
          transition: "opacity 0.38s cubic-bezier(0.22, 1, 0.36, 1)",
        }}
      />
      <div
        className="wallpaper-gradient-bg wallpaper-gradient-layer"
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          zIndex: 0,
          background: cfg.darkBackground,
          opacity: isDark ? 1 : 0,
          transition: "opacity 0.38s cubic-bezier(0.22, 1, 0.36, 1)",
        }}
      />
      {cfg.maskUrl && (
        <div
          className="wallpaper-doodle-pattern"
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            zIndex: 0,
            backgroundColor: cfg.maskColor,
            maskImage: cfg.maskUrl,
            WebkitMaskImage: cfg.maskUrl,
            maskSize: "460px auto",
            WebkitMaskSize: "460px auto",
            maskRepeat: "repeat",
            WebkitMaskRepeat: "repeat",
            maskPosition: "0 0",
            WebkitMaskPosition: "0 0",
            opacity: cfg.maskOpacity,
            transform: "translate3d(0, 0, 0)",
            backfaceVisibility: "hidden",
            transition:
              "opacity 0.38s cubic-bezier(0.22, 1, 0.36, 1), background-color 0.38s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      )}
    </div>
  );
}


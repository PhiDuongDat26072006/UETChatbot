# AGENT.md — Hướng dẫn cho AI Agent làm việc tại `src/interfaces/`

> Tài liệu tham chiếu chuẩn cho mọi Agent khi đọc, bảo trì hoặc mở rộng phân hệ Giao diện Web & API (`src/interfaces/`) trên nhánh `feat/hoangthang/interfaces`.

---

## 1. Tổng quan kiến trúc

Phân hệ `src/interfaces/` cung cấp API FastAPI + Giao diện Web hiện đại cho trợ lý **UET AI Assistant**:

- **Backend API (`src/interfaces/routes.py`, `src/interfaces/cli.py`)**:
  - Khởi tạo FastAPI app (`create_app()`), warmup `get_rag_pipeline()` khi startup.
  - Phục vụ bundle React build tại `src/interfaces/static/dist/index.html` ở `GET /` (fallback về `src/interfaces/static/index.html`), mount `/assets` -> `static/dist/assets` và `/static` -> `static/`.
  - Hợp đồng API chính:
    - `GET /api/health` -> `{ status, message, model_name }`.
    - `POST /api/chat` -> `ChatRequest { query, session_id, top_k, voice_mode }` -> `ChatResponse`.
    - `POST /api/chat/stream` -> SSE Stream trả các frame JSON `{"type": "sources" | "token" | "done" | "error"}` và kết thúc bằng `data: [DONE]`.
- **Frontend SPA (`src/interfaces/frontend/`)**:
  - Stack: **React 19 + TypeScript 5.8 + Vite 8 + Tailwind CSS v4** (`vite.config.ts` proxy `/api` & `/static` về `http://127.0.0.1:8000` khi chạy `npm run dev`, xuất build ra `../static/dist` khi chạy `npm run build`).
  - Thư viện cốt lõi (`package.json`):
    - `streamdown@2.7.1` + `katex@0.16.27`: render Markdown, bảng GFM, code block chưa đóng và công thức Toán học khi đang stream.
    - `motion@12.x`: chuyển động spring cho bong bóng chat, mobile drawer, modal và toast.
    - `use-stick-to-bottom@1.1.6` + `@tanstack/react-virtual@3.14.26`: cuộn bám đáy tự động (ngắt ngay khi người dùng cuộn lên) và ảo hóa danh sách tin nhắn dài ở 60fps.
    - `lucide-react`: hệ thống icon tối giản.

---

## 2. Cấu trúc thư mục `src/interfaces/`

```text
src/interfaces/
├── __init__.py
├── AGENT.md                  # Tài liệu hướng dẫn Agent (file này)
├── contexts/
│   ├── context.md            # Tổng mục lục lịch sử thay đổi & kiến trúc
│   └── context-DD-MM-YYYY-HH-MM-SS.md  # Nhật ký chi tiết từng đợt chỉnh sửa
├── cli.py                    # Giao diện CLI & entrypoint run_web
├── routes.py                 # FastAPI endpoints, SSE stream, /api/voice/transcribe, phân loại lỗi, format rules
├── static/
│   ├── dist/                 # Bản build production do Vite sinh ra (KHÔNG sửa tay)
│   ├── audio/                # Âm thanh tdesktop & tweb (msg_incoming.mp3, group_call_*.mp3)
│   ├── themes/               # Bảng màu trích xuất từ tdesktop (day-blue, night)
│   ├── tdesktop-background.svg # Họa tiết nền giải nén từ tdesktop/art/background.tgv
│   ├── pattern.svg           # Họa tiết nền từ morethanwords/tweb (CSS mask-image)
│   ├── uet-logo.png          # Logo UET AI Assistant
│   └── ...                   # Assets gốc cũ (uet-doodle-*, telegram-doodle.svg, html/css/js legacy)
└── frontend/
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx
        ├── main.tsx
        ├── index.css         # Tailwind v4 + CSS variables Light/Dark + Responsive breakpoints
        ├── types/chat.ts
        ├── services/
        │   ├── api.ts        # UETApiClient (REST + SSE parser + classifyApiError)
        │   └── storage.ts    # LocalStorage schema v3 (uet_ai_*)
        ├── hooks/
        │   ├── useChatStore.ts    # State tổng: hội thoại, theme, sound, stream, toast, abort
        │   └── useStreamBuffer.ts # Gom token theo nhịp requestAnimationFrame (~60fps)
        ├── utils/
        │   ├── errorClassifier.ts # Lọc & phân loại lỗi kỹ thuật -> tiếng Việt thân thiện
        │   ├── formatters.ts
        │   ├── sound.ts
        │   └── voice.ts           # Xin quyền Micro thống nhất + MediaRecorder/AudioContext VAD cho Firefox
        └── components/
            ├── chat/
            │   ├── ChatWallpaper.tsx       # Hình nền CSS mask-image viewport cố định 100vw x 100dvh
            │   ├── InChatSearchBar.tsx     # Thanh tìm kiếm nổi bán trong suốt (position: absolute)
            │   ├── MessageBubble.tsx       # Memoized bubble + Avatar user/bot + Tài liệu tham khảo thu gọn
            │   ├── StreamdownRenderer.tsx  # Bảo vệ công thức KaTeX ($...$, $$...$$, $ \text{...} $), bảng GFM
            │   ├── StreamingMessage.tsx    # Bong bóng riêng biệt khi đang stream
            │   ├── ThinkingIndicator.tsx   # Chấm nhảy Telegram khi đợi token đầu
            │   ├── VirtualizedMessageList.tsx
            │   ├── WelcomeScreen.tsx
            │   └── ChatComposer.tsx        # Capsule composer + Đính kèm + Speech Mic + Live Voice Chat
            ├── sidebar/
            │   ├── Sidebar.tsx             # Layout Gemini (tác vụ nhanh + ĐÃ GHIM / GẦN ĐÂY + ghim nhanh)
            │   ├── MiniRail.tsx            # Rail thu gọn 58px khi sidebar đóng
            │   └── ContextMenu.tsx
            └── modals/
                ├── VoiceChatModal.tsx      # Đàm thoại giọng nói trực tiếp + auto-send cuối câu + TTS
                ├── SettingsModal.tsx       # Thông tin model, RAG health, Dark/Light, Âm thanh, Xóa dữ liệu
                ├── SourceModal.tsx         # Xem trích đoạn tài liệu tham khảo
                ├── RenameModal.tsx
                ├── ConfirmModal.tsx
                └── ToastContainer.tsx      # Chỉ hiện icon cho cảnh báo/lỗi (warning/error)
```

---

## 3. Các quy tắc bất biến khi chỉnh sửa

1. **Phạm vi chỉnh sửa**: Chỉ chỉnh sửa các file bên trong `src/interfaces/` khi làm nhiệm vụ giao diện. **Không** sửa đổi `src/pipeline.py`, `src/llm/`, `src/retrieval/`, `src/prompts/` trừ khi có yêu cầu mở rộng sang các phân hệ đó.
2. **Không bao giờ rò rỉ lỗi kỹ thuật thô ra UI**:
   - Mọi phản hồi lỗi chứa `org_...`, `ITPM`, `TPM`, `console.groq.com`, `Groq [...]`, `SambaNova [...]`, `Gemini [...]`, `Traceback` phải bị chặn tại **cả 2 tầng**: backend (`_detect_and_classify_llm_error`, `_classify_exception` trong `routes.py`) và frontend (`containsRawTechnicalError`, `classifyApiError` trong `utils/errorClassifier.ts` & `services/api.ts`).
   - Chi tiết kỹ thuật chỉ ghi qua `logger.error` (server) / `console.error` (client); UI chỉ hiển thị thông báo tiếng Việt thân thiện kèm nút **Thử lại**.
3. **Đồng bộ Responsive Breakpoint tại `768px`**:
   - Tailwind CSS dùng `md:` từ `>= 768px`, do đó logic JS trong `Sidebar.tsx` và `useChatStore.ts` **bắt buộc dùng `< 768`** (`window.innerWidth < 768`), còn CSS dùng `@media (max-width: 767px)`. Tuyệt đối không đặt `<= 768` để tránh mất nút mở drawer ở màn hình đúng `768px`.
4. **Hiệu năng Sidebar đa trình duyệt (đặc biệt Firefox Gecko)**:
   - Không dùng render điều kiện kiểu `{isCollapsed ? <MiniRail /> : <Full />}` vì gây unmount/remount DOM mỗi lần bấm. Luôn giữ cả `.sidebar-mini-rail-wrapper` và `.sidebar-full-clip` trong DOM, khóa chiều rộng `.sidebar-full-content` (`width: var(--sidebar-width)`) kết hợp `contain: layout style`.
   - Giữ `.chat-wallpaper-layer` ở kích thước viewport cố định (`width: 100vw; height: 100dvh; contain: strict`) bên trong `.telegram-chat-pane { overflow: hidden }` để khi sidebar đổi width, SVG mask `pattern.svg` không bị rasterize lại.
5. **Tương thích LocalStorage Schema**:
   - Bảo toàn các khóa trong `STORAGE_KEYS` (`uet_ai_telegram_conversations_v3`, `uet_ai_active_chat_id_v3`, `uet_ai_theme_mode`, `uet_ai_chat_wallpaper`, `uet_ai_sidebar_closed`).
6. **Quy ước Render Toán & Bảng (`StreamdownRenderer.tsx`)**:
   - Đã tắt bọc bảng phụ mặc định của `streamdown` (`controls={{ code: true, table: false }}`, `tableMaxHeight={Infinity}`) để tránh "bảng trong bảng".
   - Mọi xử lý KaTeX đều chuyển qua tag giả `<uet-math>` dùng `encodeMathPayload` / `decodeMathPayload` nhằm tránh bị bộ parser Markdown làm hỏng dấu `_`, `*`, `|`.
7. **Tính năng Voice Chat trực tiếp & Hỗ trợ đa trình duyệt (Chrome, Edge, Safari, Firefox)**:
   - **Chrome / Edge / Safari**: Dùng **Web Speech API** (`SpeechRecognition`) để nhận dạng giọng nói; trong đó **Chrome & Edge** chủ động gọi `requestMicrophoneStream({ keepStream: false })` (`getUserMedia`) trước khi mở mic để kích hoạt popup xin quyền ở lần đầu truy cập, còn **Safari** phải gọi `recognition.start()` đồng bộ trực tiếp trong sự kiện click để không mất user-gesture.
   - **Firefox (và trình duyệt không có `SpeechRecognition`)**: Hoạt động bằng `requestMicrophoneStream({ keepStream: true })` (giữ nguyên 1 `MediaStream` để tránh bị Firefox hỏi quyền 2 lần) kết hợp `startAudioSilenceRecorder` trong `utils/voice.ts` (`MediaRecorder` + Web Audio API `AudioContext`/`AnalyserNode` đo RMS mỗi 60ms để tự động dừng khi ngừng nói hết câu) và gửi âm thanh qua `POST /api/voice/transcribe` (tái sử dụng `groq_keys` qua Groq Whisper `whisper-large-v3-turbo` / `whisper-large-v3` ngay trong `routes.py`), giữ nguyên không sửa module lõi ngoài `src/interfaces/`.
   - **Phản hồi giọng nói**: Dùng `voice_mode: true` (`_VOICE_RESPONSE_FORMAT_RULES`, 1–3 câu ngắn gọn) kết hợp `speechSynthesis`. Nếu tương lai nâng cấp sang server TTS hoặc Realtime WebSocket audio stream mới cần mở rộng sang module ngoài `src/interfaces/`.
8. **KHÔNG tự ý/ngẫu hứng sinh ra icon khi viết code**:
   - Tuyệt đối **không** tự ý thêm icon, emoji hay biểu tượng trang trí ngẫu hứng vào giao diện, thông báo đẩy (toast), tiêu đề, thẻ gợi ý, modal hoặc nội dung phản hồi của AI khi viết code, trừ khi được người dùng yêu cầu tường minh.
   - Với các lưu ý hoặc thông tin quan trọng trong nội dung chat, chỉ dùng **chữ in đậm** (đã có `stripDecorativeEmojis` + `boldNoticeLabels` hỗ trợ); với thông báo đẩy (`ToastContainer.tsx`), các thông báo bình thường hoặc tích cực (`info`, `success`) không gắn icon, chỉ giữ icon cảnh báo cho thông báo quan trọng/lỗi (`warning`, `error`).
   - Trong modal Voice Chat (`VoiceChatModal.tsx`), sử dụng cụm icon micro (`Mic` / `MicOff`), tuyệt đối không dùng icon điện thoại (`PhoneOff`).
9. **BẮT BUỘC cập nhật tài liệu `contexts/` và `AGENT.md` sau mỗi lần chỉnh sửa code**:
   - Mọi đợt chỉnh sửa hoặc thay đổi mã nguồn trong `src/interfaces/` **bắt buộc** phải ghi nhận lại đầy đủ mục đích, chi tiết file thay đổi, nguyên nhân gốc và tác dụng vào file `src/interfaces/contexts/context-DD-MM-YYYY-HH-MM-SS.md` mới nhất và cập nhật mục lục trong `src/interfaces/contexts/context.md`.
   - Nếu phát sinh quy tắc, ràng buộc hoặc giới hạn mới, phải bổ sung ngay vào `src/interfaces/AGENT.md`.
10. **Ẩn tài liệu tham khảo cho câu hỏi lỗi, không có câu trả lời hoặc ngoài domain**:
   - Các câu trả lời báo lỗi, từ chối trả lời do không có dữ liệu quy chế hoặc câu hỏi ngoài phạm vi domain UET (`_is_no_answer_or_out_of_domain` ở `routes.py`, `isNoAnswerOrOutOfDomain` ở `formatters.ts`, `MessageBubble.tsx`, `StreamingMessage.tsx`, `useChatStore.ts`) tuyệt đối **không** hiển thị khối `Tài liệu tham khảo`.

---

## 4. Quy trình Build & Kiểm thử bắt buộc sau khi sửa

```bash
# 1. Kiểm tra TypeScript & build bundle React vào src/interfaces/static/dist/
cd src/interfaces/frontend && npx tsc --noEmit && npm run build

# 2. Quay về thư mục gốc và chạy unit test kiến trúc backend
cd /home/hoangthang/Documents/UETChatbot
.venv/bin/python -m unittest tests/test_app.py -v
```

---

## 5. Tài liệu ngữ cảnh chi tiết (`src/interfaces/contexts/`)

Xem [context.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context.md) để tra cứu toàn bộ mục lục và chi tiết theo từng mốc thời gian:

- [context-16-11-2026-21-50-00.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context-16-11-2026-21-50-00.md) — Refactor nền tảng React + TypeScript + Vite + Tailwind CSS v4 & Error Classification SSE
- [context-16-11-2026-23-15-00.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context-16-11-2026-23-15-00.md) — Tinh giản giao diện, tích hợp pattern/audio Telegram, sửa bảng & KaTeX
- [context-17-11-2026-00-05-00.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context-17-11-2026-00-05-00.md) — 12 tinh chỉnh UI/UX, tối ưu Firefox 60fps & Speech Mic
- [context-17-11-2026-00-35-00.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context-17-11-2026-00-35-00.md) — Tìm kiếm nổi trong suốt, sửa công thức SAT, Responsive đa thiết bị, Sidebar Gemini & Live Voice Chat
- [context-17-11-2026-01-00-00.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context-17-11-2026-01-00-00.md) — Khắc phục abort khi chuyển chat, tìm kiếm & bôi đậm trong tin nhắn AI, ẩn nguồn ngoài domain/lỗi & thay icon Voice Chat sang Mic
- [context-17-11-2026-01-10-00.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context-17-11-2026-01-10-00.md) — Menu đính kèm nút `+`, bỏ `...` placeholder & Tích hợp theme, art, sounds từ Telegram Desktop (`tdesktop`)
- [context-18-11-2026-11-30-00.md](file:///home/hoangthang/Documents/UETChatbot/src/interfaces/contexts/context-18-11-2026-11-30-00.md) — Hỗ trợ Voice Chat & Mic trên Firefox (`MediaRecorder` + RMS VAD + Groq Whisper) & xin quyền Micro thống nhất trên mọi trình duyệt

---

## 6. Ghi chú cải tiến tương lai (Future Improvements Backlog)

- **Trả lời đa đoạn chat song song (Multi-chat concurrent background streaming)**:
  - **Hiện trạng**: `useChatStore.ts` hiện sử dụng một `AbortController` và một bộ state stream duy nhất (`abortControllerRef`, `activeStreamContextRef`, `streamingText`, `streamingSources`). Khi người dùng gửi câu hỏi ở đoạn chat A rồi chuyển ngay sang đoạn chat B (hoặc tạo đoạn chat mới), hệ thống chủ động hủy phiên stream cũ (`abortActiveStream("chat_switch", false)`) và **chỉ giữ lại câu hỏi của người dùng ở đoạn chat A**, không in phần AI đang dở hay thông báo lỗi timeout.
  - **Hướng cải tiến về sau**: Chuyển cấu trúc stream sang `Map<chatId, StreamSessionState>` (mỗi đoạn chat sở hữu `AbortController`, `streamBuffer`, `sources`, `status` độc lập), cho phép request SSE tiếp tục hoàn tất ở nền khi người dùng chuyển sang đoạn chat khác và tăng `unreadCount` trên sidebar khi câu trả lời nền hoàn tất.


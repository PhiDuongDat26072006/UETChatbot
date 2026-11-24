import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
} from "react";
import { AnimatePresence, motion } from "motion/react";
import { FolderUp, Menu } from "lucide-react";
import { useChatStore } from "./hooks/useChatStore";
import type { ChatMessage, ChatSession, SourceItem } from "./types/chat";
import { ChatWallpaper } from "./components/chat/ChatWallpaper";
import { InChatSearchBar } from "./components/chat/InChatSearchBar";
import { VirtualizedMessageList } from "./components/chat/VirtualizedMessageList";
import { ChatComposer } from "./components/chat/ChatComposer";
import { Sidebar } from "./components/sidebar/Sidebar";
import {
  ContextMenu,
  type ContextMenuState,
} from "./components/sidebar/ContextMenu";
import { SourceModal } from "./components/modals/SourceModal";
import { SettingsModal } from "./components/modals/SettingsModal";
import { RenameModal } from "./components/modals/RenameModal";
import {
  ConfirmModal,
  type ConfirmDialogState,
} from "./components/modals/ConfirmModal";
import { ToastContainer } from "./components/modals/ToastContainer";
import { VoiceChatModal } from "./components/modals/VoiceChatModal";
import { stripMarkdownForSearch } from "./utils/formatters";

const EMPTY_MESSAGES: ChatMessage[] = [];

function DragDropOverlay({ isVisible }: { isVisible: boolean }) {
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          key="drag-drop-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16 }}
          className="absolute inset-0 z-50 flex items-center justify-center bg-[rgba(0,84,166,0.38)] backdrop-blur-[4px]"
          aria-live="polite"
        >
          <motion.div
            initial={{ scale: 0.94, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.94, opacity: 0 }}
            transition={{ type: "spring", stiffness: 420, damping: 30 }}
            className="mx-4 rounded-[14px] border-2 border-dashed border-[var(--primary)] bg-[var(--bg-sidebar)] px-10 py-8 text-center shadow-[var(--shadow-md)] gpu-layer"
          >
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--primary-light)] text-[var(--primary)]">
              <FolderUp size={30} />
            </div>
            <div className="mb-1 text-[17px] font-bold text-[var(--text-primary)]">
              Thả tài liệu vào đây để đính kèm
            </div>
            <div className="text-[13px] text-[var(--text-secondary)]">
              Hỗ trợ các định dạng .PDF, .DOCX, .TXT
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default function App() {
  const {
    conversations,
    activeChatId,
    activeChat,
    groupedChats,
    searchChatQuery,
    setSearchChatQuery,
    theme,
    toggleTheme,
    setThemeMode,
    soundEnabled,
    setSoundEnabled,
    wallpaper,
    isSidebarClosed,
    toggleSidebar,
    isMobileSidebarOpen,
    setMobileSidebarOpen,
    health,
    refreshHealth,
    isGenerating,
    isWaitingFirstToken,
    streamingText,
    streamingSources,
    streamingModel,
    skipStreaming,
    selectedFiles,
    addFiles,
    removeSelectedFile,
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
  } = useChatStore();

  const [isInChatSearchOpen, setIsInChatSearchOpen] = useState(false);
  const [inChatSearchQuery, setInChatSearchQuery] = useState("");
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);

  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [activeSourceModal, setActiveSourceModal] = useState<SourceItem | null>(
    null
  );
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [renameTargetChat, setRenameTargetChat] = useState<ChatSession | null>(
    null
  );
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState | null>(
    null
  );
  const [isVoiceChatOpen, setIsVoiceChatOpen] = useState(false);
  const [draftInsertTrigger, setDraftInsertTrigger] = useState<{
    text: string;
    id: number;
  } | null>(null);

  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const dragCounterRef = useRef(0);

  const activeMessages = activeChat?.messages ?? EMPTY_MESSAGES;

  const matchedMessageIndices = useMemo(() => {
    const q = inChatSearchQuery.trim().toLowerCase();
    if (!isInChatSearchOpen || !q) return [];
    const indices: number[] = [];
    activeMessages.forEach((msg, idx) => {
      if (!msg.text) return;
      const rawLower = msg.text.toLowerCase();
      const plainLower = stripMarkdownForSearch(msg.text).toLowerCase();
      if (rawLower.includes(q) || plainLower.includes(q)) {
        indices.push(idx);
      }
    });
    return indices;
  }, [activeMessages, inChatSearchQuery, isInChatSearchOpen]);

  const totalMatches = matchedMessageIndices.length;
  const safeMatchIndex =
    totalMatches > 0 ? Math.min(currentMatchIndex, totalMatches - 1) : 0;
  const activeSearchMessageIndex =
    isInChatSearchOpen && totalMatches > 0
      ? matchedMessageIndices[safeMatchIndex]
      : null;

  useEffect(() => {
    setCurrentMatchIndex(0);
  }, [activeChatId]);

  const handleSearchQueryChange = useCallback((q: string) => {
    setInChatSearchQuery(q);
    setCurrentMatchIndex(0);
  }, []);

  const handlePrevMatch = useCallback(() => {
    if (totalMatches === 0) return;
    setCurrentMatchIndex((prev) => (prev - 1 + totalMatches) % totalMatches);
  }, [totalMatches]);

  const handleNextMatch = useCallback(() => {
    if (totalMatches === 0) return;
    setCurrentMatchIndex((prev) => (prev + 1) % totalMatches);
  }, [totalMatches]);

  const handleCloseInChatSearch = useCallback(() => {
    setIsInChatSearchOpen(false);
    setInChatSearchQuery("");
    setCurrentMatchIndex(0);
  }, []);

  // Global keyboard shortcuts: Ctrl+B (Sidebar), Ctrl+F (In-chat Search), Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        toggleSidebar();
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setIsInChatSearchOpen(true);
        return;
      }

      if (e.key === "Escape") {
        if (
          activeSourceModal ||
          isSettingsOpen ||
          renameTargetChat ||
          confirmDialog ||
          contextMenu
        ) {
          return;
        }
        if (isInChatSearchOpen) {
          handleCloseInChatSearch();
          return;
        }
        if (isMobileSidebarOpen) {
          setMobileSidebarOpen(false);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    activeSourceModal,
    confirmDialog,
    contextMenu,
    handleCloseInChatSearch,
    isInChatSearchOpen,
    isMobileSidebarOpen,
    isSettingsOpen,
    renameTargetChat,
    setMobileSidebarOpen,
    toggleSidebar,
  ]);

  // Prevent browser from opening dropped files outside chat pane
  useEffect(() => {
    const preventWindowDrag = (e: globalThis.DragEvent) => {
      e.preventDefault();
    };
    window.addEventListener("dragover", preventWindowDrag);
    window.addEventListener("drop", preventWindowDrag);
    return () => {
      window.removeEventListener("dragover", preventWindowDrag);
      window.removeEventListener("drop", preventWindowDrag);
    };
  }, []);

  const hasDraggedFiles = (e: DragEvent<HTMLElement>) => {
    const types = e.dataTransfer?.types;
    if (!types) return false;
    return Array.from(types).includes("Files");
  };

  const handlePaneDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    if (!hasDraggedFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current += 1;
    setIsDraggingFiles(true);
  }, []);

  const handlePaneDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    if (!hasDraggedFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handlePaneDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    if (!hasDraggedFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current <= 0) {
      dragCounterRef.current = 0;
      setIsDraggingFiles(false);
    }
  }, []);

  const handlePaneDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounterRef.current = 0;
      setIsDraggingFiles(false);
      if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
        addFiles(e.dataTransfer.files);
      }
    },
    [addFiles]
  );

  const requestClearMessages = useCallback(
    (chat: ChatSession) => {
      setConfirmDialog({
        title: "Xóa toàn bộ tin nhắn",
        message:
          "Bạn có chắc chắn muốn xóa toàn bộ tin nhắn trong cuộc trò chuyện này không? Cuộc trò chuyện vẫn được giữ lại trong danh sách, chỉ nội dung các tin nhắn sẽ bị xóa.",
        confirmText: "Xóa tin nhắn",
        isDanger: true,
        onConfirm: () => clearMessages(chat.id),
      });
    },
    [clearMessages]
  );

  const requestDeleteChat = useCallback(
    (chat: ChatSession) => {
      setConfirmDialog({
        title: "Xóa cuộc trò chuyện",
        message: `Bạn có chắc muốn xóa cuộc trò chuyện "${chat.title}"? Thao tác này không thể hoàn tác.`,
        confirmText: "Xóa cuộc trò chuyện",
        isDanger: true,
        onConfirm: () => deleteConversation(chat.id),
      });
    },
    [deleteConversation]
  );

  const requestClearAllChats = useCallback(() => {
    setConfirmDialog({
      title: "Xóa toàn bộ lịch sử trò chuyện",
      message:
        "CẢNH BÁO: Thao tác này sẽ xóa vĩnh viễn TOÀN BỘ các cuộc trò chuyện trên trình duyệt của bạn. Bạn có chắc chắn muốn tiếp tục?",
      confirmText: "Xóa tất cả",
      isDanger: true,
      onConfirm: () => {
        clearAllConversations();
        setIsSettingsOpen(false);
      },
    });
  }, [clearAllConversations]);

  const handleInsertRegulationsPrompt = useCallback(() => {
    setDraftInsertTrigger({
      text: "Tóm tắt các quy định quan trọng nhất trong Quy chế đào tạo UET",
      id: Date.now(),
    });
  }, []);

  const handleShowModulesInfo = useCallback(() => {
    showToast("Tính năng Tiện ích UET đang được cập nhật, vui lòng chờ!", "info", 2200);
  }, [showToast]);

  const handleCopyMessage = useCallback(() => {
    showToast("Đã sao chép nội dung vào khay nhớ tạm!", "success", 1500);
  }, [showToast]);

  const handleSelectSuggestion = useCallback(
    (query: string) => {
      void sendMessage(query);
    },
    [sendMessage]
  );

  const handleRetryMessage = useCallback(
    (botMsgIndex: number) => {
      void retryMessage(botMsgIndex);
    },
    [retryMessage]
  );

  const handleSendMessage = useCallback(
    (text: string) => {
      void sendMessage(text);
    },
    [sendMessage]
  );

  const handleSendVoiceQuery = useCallback(
    (text: string) => sendMessage(text, { voiceMode: true }),
    [sendMessage]
  );

  const handleOpenVoiceChat = useCallback(() => {
    setIsVoiceChatOpen(true);
  }, []);

  const handleCloseVoiceChat = useCallback(() => {
    setIsVoiceChatOpen(false);
  }, []);

  const handleOpenMobileSidebar = useCallback(() => {
    setMobileSidebarOpen(true);
  }, [setMobileSidebarOpen]);

  const handleCloseMobileSidebar = useCallback(() => {
    setMobileSidebarOpen(false);
  }, [setMobileSidebarOpen]);

  const handleOpenSettings = useCallback(() => {
    setIsSettingsOpen(true);
  }, []);

  const handleCloseSettings = useCallback(() => {
    setIsSettingsOpen(false);
  }, []);

  const handleOpenInChatSearch = useCallback(() => {
    setIsInChatSearchOpen(true);
  }, []);

  const handleNewChat = useCallback(() => {
    createNewChat();
  }, [createNewChat]);

  const handleOpenContextMenu = useCallback(
    (x: number, y: number, chat: ChatSession) => {
      setContextMenu({ x, y, chatId: chat.id, pinned: chat.pinned });
    },
    []
  );

  return (
    <div className="telegram-app" id="app-container">
      {/* Sidebar + MiniRail + Mobile Drawer */}
      <Sidebar
        groupedChats={groupedChats}
        activeChatId={activeChatId}
        searchQuery={searchChatQuery}
        onSearchQueryChange={setSearchChatQuery}
        isCollapsed={isSidebarClosed}
        isMobileOpen={isMobileSidebarOpen}
        onCloseMobile={handleCloseMobileSidebar}
        onToggleCollapse={toggleSidebar}
        theme={theme}
        onToggleTheme={toggleTheme}
        onSelectChat={selectChat}
        onNewChat={handleNewChat}
        onTogglePin={togglePin}
        onOpenSettings={handleOpenSettings}
        onOpenInChatSearch={handleOpenInChatSearch}
        onOpenContextMenu={handleOpenContextMenu}
        onInsertRegulationsPrompt={handleInsertRegulationsPrompt}
        onShowModulesInfo={handleShowModulesInfo}
      />

      {/* Main Chat Pane */}
      <main className="telegram-main" id="chat-main-area">
        <div
          className="telegram-chat-pane"
          id="chat-pane"
          onDragEnter={handlePaneDragEnter}
          onDragOver={handlePaneDragOver}
          onDragLeave={handlePaneDragLeave}
          onDrop={handlePaneDrop}
        >
          <ChatWallpaper wallpaper={wallpaper} theme={theme} />

          <button
            type="button"
            onClick={handleOpenMobileSidebar}
            className="btn-mobile-menu absolute left-3 top-3 z-[36] flex h-9 w-9 items-center justify-center rounded-full border border-[var(--border-subtle)] bg-[var(--bg-chat-header)] text-[var(--text-secondary)] shadow-[var(--shadow-sm)] backdrop-blur-md transition-colors hover:bg-[var(--bg-search)] hover:text-[var(--primary)] md:hidden"
            style={{
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
            }}
            title="Mở danh sách hội thoại"
            aria-label="Mở danh sách hội thoại"
          >
            <Menu className="h-5 w-5" />
          </button>

          <InChatSearchBar
            isOpen={isInChatSearchOpen}
            query={inChatSearchQuery}
            currentMatchIndex={safeMatchIndex}
            totalMatches={totalMatches}
            onQueryChange={handleSearchQueryChange}
            onPrev={handlePrevMatch}
            onNext={handleNextMatch}
            onClose={handleCloseInChatSearch}
          />

          <DragDropOverlay isVisible={isDraggingFiles} />

          <VirtualizedMessageList
            chatId={activeChatId}
            messages={activeMessages}
            isGenerating={isGenerating}
            isWaitingFirstToken={isWaitingFirstToken}
            streamingText={streamingText}
            streamingSources={streamingSources}
            streamingModel={streamingModel}
            searchQuery={isInChatSearchOpen ? inChatSearchQuery : ""}
            activeSearchMessageIndex={activeSearchMessageIndex}
            onSelectSuggestion={handleSelectSuggestion}
            onOpenSource={setActiveSourceModal}
            onRetryMessage={handleRetryMessage}
            onCopyMessage={handleCopyMessage}
            onStopGeneration={stopGeneration}
            onSkipStreaming={skipStreaming}
          />

          <ChatComposer
            isGenerating={isGenerating}
            selectedFiles={selectedFiles}
            onAddFiles={addFiles}
            onRemoveFile={removeSelectedFile}
            onSendMessage={handleSendMessage}
            onStopGeneration={stopGeneration}
            onOpenVoiceChat={handleOpenVoiceChat}
            onShowToast={showToast}
            draftInsertTrigger={draftInsertTrigger}
          />
        </div>
      </main>

      {/* Floating Overlays: Context Menu, Modals & Toasts */}
      <ContextMenu
        menu={contextMenu}
        onClose={() => setContextMenu(null)}
        onTogglePin={togglePin}
        onRename={(chatId) => {
          const target = conversations.find((c) => c.id === chatId);
          if (target) setRenameTargetChat(target);
        }}
        onSearchInChat={(chatId) => {
          selectChat(chatId);
          setIsInChatSearchOpen(true);
        }}
        onClearMessages={(chatId) => {
          const target = conversations.find((c) => c.id === chatId);
          if (target) requestClearMessages(target);
        }}
        onDelete={(chatId) => {
          const target = conversations.find((c) => c.id === chatId);
          if (target) requestDeleteChat(target);
        }}
      />

      <SourceModal
        source={activeSourceModal}
        onClose={() => setActiveSourceModal(null)}
      />

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={handleCloseSettings}
        health={health}
        onRefreshHealth={refreshHealth}
        theme={theme}
        onSelectTheme={setThemeMode}
        soundEnabled={soundEnabled}
        onToggleSound={setSoundEnabled}
        onRequestClearAllChats={requestClearAllChats}
      />

      <RenameModal
        isOpen={Boolean(renameTargetChat)}
        initialTitle={renameTargetChat?.title || ""}
        onSave={(newTitle) => {
          if (renameTargetChat) {
            renameConversation(renameTargetChat.id, newTitle);
          }
        }}
        onClose={() => setRenameTargetChat(null)}
      />

      <ConfirmModal
        dialog={confirmDialog}
        onClose={() => setConfirmDialog(null)}
      />

      <VoiceChatModal
        isOpen={isVoiceChatOpen}
        onClose={handleCloseVoiceChat}
        onSendVoiceQuery={handleSendVoiceQuery}
        onStopGeneration={stopGeneration}
      />

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

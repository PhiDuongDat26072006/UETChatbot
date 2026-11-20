(() => {
  "use strict";

  const THEME_STORAGE_KEY = "uet_review_theme";
  const ACTIVE_SITE_STORAGE_KEY = "uet_review_active_site";
  const MAX_UPLOAD_BYTES = 30 * 1024 * 1024;
  const ALLOWED_UPLOAD_EXTS = [".pdf", ".html", ".htm"];

  const ICONS = {
    check:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    cross:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"></path></svg>',
    list:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>',
    eye:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle></svg>',
    doc:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
    reingest:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"></path><path d="M16 16h5v5"></path></svg>',
    trash:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>',
    external:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>'
  };

  const state = {
    sites: [],
    totals: {},
    offlineDocsCount: 0,
    activeSite: null,
    items: [],
    totalFiltered: 0,
    discoveredFromOptions: [],
    filters: {
      status: "all",
      resource_type: "all",
      search: "",
      discovered_from: "",
      sort: "priority",
      page: 1,
      page_size: 50
    },
    selectedIds: new Set(),
    loadingSites: false,
    loadingSources: false,
    pendingUploadFile: null,
    previewSourceId: null,
    processedSourceId: null,
    removeTargetIds: [],
    ingestPollTimer: null,
    ingestWasRunning: false,
    lastJobIdSeen: null,
    dismissedJobId: null
  };

  let searchDebounceTimer = null;
  let themeTransitionTimer = null;

  /* ========================================================================
     DOM REFERENCES
     ======================================================================== */
  const el = {
    sidebar: document.getElementById("sidebar"),
    sidebarBackdrop: document.getElementById("sidebar-backdrop"),
    mobileMenuBtn: document.getElementById("mobile-menu-btn"),
    sidebarCloseBtn: document.getElementById("sidebar-close-btn"),
    themeToggleBtn: document.getElementById("theme-toggle-btn"),
    globalTotal: document.getElementById("global-total"),
    globalPending: document.getElementById("global-pending"),
    globalRatio: document.getElementById("global-ratio"),
    globalOffline: document.getElementById("global-offline"),
    btnOfflineScan: document.getElementById("btn-offline-scan"),
    btnSidebarUpload: document.getElementById("btn-sidebar-upload"),
    btnIngestAll: document.getElementById("btn-ingest-all"),
    siteCountPill: document.getElementById("site-count-pill"),
    siteList: document.getElementById("site-list"),

    activeSiteName: document.getElementById("active-site-name"),
    activeSiteHosts: document.getElementById("active-site-hosts"),
    btnRefresh: document.getElementById("btn-refresh"),
    btnOpenAddUrls: document.getElementById("btn-open-add-urls"),
    btnOpenUpload: document.getElementById("btn-open-upload"),
    btnIngestSite: document.getElementById("btn-ingest-site"),

    ingestBanner: document.getElementById("ingest-banner"),
    ingestSpinner: document.getElementById("ingest-spinner"),
    ingestStatusBadge: document.getElementById("ingest-status-badge"),
    ingestTitle: document.getElementById("ingest-title"),
    ingestPhase: document.getElementById("ingest-phase"),
    btnDismissIngest: document.getElementById("btn-dismiss-ingest"),
    ingestSkipped: document.getElementById("ingest-skipped"),
    ingestDownloaded: document.getElementById("ingest-downloaded"),
    ingestProcessed: document.getElementById("ingest-processed"),
    ingestNewPending: document.getElementById("ingest-new-pending"),
    ingestRejectedPreserved: document.getElementById("ingest-rejected-preserved"),
    ingestErrors: document.getElementById("ingest-errors"),
    ingestErrorMsg: document.getElementById("ingest-error-msg"),

    statCards: document.getElementById("stat-cards"),
    statTotal: document.getElementById("stat-total"),
    statPending: document.getElementById("stat-pending"),
    statQueued: document.getElementById("stat-queued"),
    statIngested: document.getElementById("stat-ingested"),
    statListing: document.getElementById("stat-listing"),
    statRejected: document.getElementById("stat-rejected"),
    statRejectedSub: document.getElementById("stat-rejected-sub"),
    statFailed: document.getElementById("stat-failed"),

    filterSearch: document.getElementById("filter-search"),
    filterDiscoveredFrom: document.getElementById("filter-discovered-from"),
    filterSort: document.getElementById("filter-sort"),
    btnResetFilters: document.getElementById("btn-reset-filters"),

    bulkBar: document.getElementById("bulk-bar"),
    bulkCountText: document.getElementById("bulk-count-text"),
    btnClearSelection: document.getElementById("btn-clear-selection"),
    btnBulkRemove: document.getElementById("btn-bulk-remove"),

    selectAllCheckbox: document.getElementById("select-all-checkbox"),
    sourcesTbody: document.getElementById("sources-tbody"),
    paginationSummary: document.getElementById("pagination-summary"),
    pageSizeSelect: document.getElementById("page-size-select"),
    btnPrevPage: document.getElementById("btn-prev-page"),
    btnNextPage: document.getElementById("btn-next-page"),
    pageIndicator: document.getElementById("page-indicator"),

    modalAddUrls: document.getElementById("modal-add-urls"),
    modalAddSubtitle: document.getElementById("modal-add-subtitle"),
    formAddUrls: document.getElementById("form-add-urls"),
    addUrlsInput: document.getElementById("add-urls-input"),
    addUrlsDecision: document.getElementById("add-urls-decision"),
    addUrlsNote: document.getElementById("add-urls-note"),
    addUrlsResult: document.getElementById("add-urls-result"),
    btnSubmitAddUrls: document.getElementById("btn-submit-add-urls"),

    modalUploadDoc: document.getElementById("modal-upload-doc"),
    formUploadDoc: document.getElementById("form-upload-doc"),
    uploadDropzone: document.getElementById("upload-dropzone"),
    uploadFileInput: document.getElementById("upload-file-input"),
    uploadFileInfo: document.getElementById("upload-file-info"),
    uploadDecision: document.getElementById("upload-decision"),
    btnSubmitUpload: document.getElementById("btn-submit-upload"),

    modalPreview: document.getElementById("modal-preview"),
    previewSourceUrl: document.getElementById("preview-source-url"),
    previewBody: document.getElementById("preview-body"),
    previewBtnReject: document.getElementById("preview-btn-reject"),
    previewBtnListing: document.getElementById("preview-btn-listing"),
    previewBtnAccept: document.getElementById("preview-btn-accept"),

    modalProcessed: document.getElementById("modal-processed"),
    processedSubtitle: document.getElementById("processed-subtitle"),
    processedBody: document.getElementById("processed-body"),
    processedBtnReject: document.getElementById("processed-btn-reject"),
    processedBtnReingest: document.getElementById("processed-btn-reingest"),

    modalRemove: document.getElementById("modal-remove"),
    removeConfirmMessage: document.getElementById("remove-confirm-message"),
    removePurgeCheckbox: document.getElementById("remove-purge-checkbox"),
    btnConfirmRemove: document.getElementById("btn-confirm-remove"),

    toastContainer: document.getElementById("toast-container")
  };

  /* ========================================================================
     UTILITIES
     ======================================================================== */
  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function safeHttpUrl(rawUrl) {
    if (!rawUrl || typeof rawUrl !== "string") return "";
    const trimmed = rawUrl.trim();
    if (/^https?:\/\//i.test(trimmed)) return trimmed;
    return "";
  }

  function formatNumber(num) {
    const val = Number(num || 0);
    return Number.isFinite(val) ? val.toLocaleString("vi-VN") : "0";
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function extractUrlFallbackLabel(uri) {
    if (!uri) return "(Không có tiêu đề)";
    try {
      const parsed = new URL(uri);
      const path = parsed.pathname.replace(/\/+$/, "");
      if (!path) return parsed.hostname;
      const last = path.split("/").filter(Boolean).pop();
      return decodeURIComponent(last || parsed.hostname);
    } catch {
      return uri;
    }
  }

  async function apiFetch(url, options = {}) {
    const config = {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      },
      ...options
    };

    const res = await fetch(url, config);
    const contentType = res.headers.get("content-type") || "";
    let data = null;
    if (contentType.includes("application/json")) {
      data = await res.json();
    } else {
      const text = await res.text();
      data = text ? { detail: text } : null;
    }

    if (!res.ok) {
      const err = new Error(
        (data && (data.detail?.message || data.detail || data.error || data.message)) ||
          `HTTP ${res.status}`
      );
      err.status = res.status;
      err.payload = data;
      throw err;
    }
    return data;
  }

  /* ========================================================================
     TOAST NOTIFICATIONS (Clean text, zero decorative icons)
     ======================================================================== */
  function showToast(message, type = "info", durationMs = 3800) {
    const toast = document.createElement("div");
    toast.className = `toast-item is-${type}`;

    const textSpan = document.createElement("span");
    textSpan.textContent = message;

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "toast-close";
    closeBtn.setAttribute("aria-label", "Đóng thông báo");
    closeBtn.textContent = "\u00D7";

    const removeToast = () => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    };

    closeBtn.addEventListener("click", removeToast);
    toast.appendChild(textSpan);
    toast.appendChild(closeBtn);
    el.toastContainer.appendChild(toast);

    if (durationMs > 0) {
      window.setTimeout(removeToast, durationMs);
    }
  }

  /* ========================================================================
     THEME MANAGEMENT (380ms Smooth Transition)
     ======================================================================== */
  function toggleTheme() {
    const root = document.documentElement;
    const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";

    if (themeTransitionTimer) {
      window.clearTimeout(themeTransitionTimer);
    }
    root.classList.add("theme-transitioning");
    root.setAttribute("data-theme", next);

    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Ignore storage failures
    }

    themeTransitionTimer = window.setTimeout(() => {
      root.classList.remove("theme-transitioning");
      themeTransitionTimer = null;
    }, 420);
  }

  /* ========================================================================
     MOBILE DRAWER & MODALS
     ======================================================================== */
  function setSidebarDrawerOpen(open) {
    el.sidebar.classList.toggle("is-open", open);
    el.sidebarBackdrop.classList.toggle("is-open", open);
  }

  function openModal(modalId) {
    const node = document.getElementById(modalId);
    if (!node) return;
    node.classList.remove("hidden");
  }

  function closeModal(modalId) {
    const node = document.getElementById(modalId);
    if (!node) return;
    node.classList.add("hidden");
  }

  function closeTopmostModalOrDrawer() {
    const modalIds = [
      "modal-remove",
      "modal-processed",
      "modal-preview",
      "modal-upload-doc",
      "modal-add-urls"
    ];
    for (const id of modalIds) {
      const node = document.getElementById(id);
      if (node && !node.classList.contains("hidden")) {
        closeModal(id);
        return true;
      }
    }
    if (el.sidebar.classList.contains("is-open")) {
      setSidebarDrawerOpen(false);
      return true;
    }
    return false;
  }

  /* ========================================================================
     LOAD SITES & SIDEBAR RENDERING
     ======================================================================== */
  async function loadSites({ refreshSources = false } = {}) {
    state.loadingSites = true;
    try {
      const data = await apiFetch("/api/sites");
      state.sites = Array.isArray(data.sites) ? data.sites : [];
      state.totals = data.totals || {};
      state.offlineDocsCount = Number(data.offline_docs_count || 0);

      if (!state.activeSite || !state.sites.some((s) => s.site === state.activeSite)) {
        let savedSite = null;
        try {
          savedSite = localStorage.getItem(ACTIVE_SITE_STORAGE_KEY);
        } catch {
          savedSite = null;
        }
        if (savedSite && state.sites.some((s) => s.site === savedSite)) {
          state.activeSite = savedSite;
        } else if (state.sites.some((s) => s.site === "UET")) {
          state.activeSite = "UET";
        } else if (state.sites.length > 0) {
          state.activeSite = state.sites[0].site;
        }
      }

      renderSidebar();
      renderActiveSiteHeader();
      updateStatCardsFromActiveSite();

      if (refreshSources && state.activeSite) {
        await loadSources();
      }
    } catch (err) {
      showToast(`Không tải được danh sách site: ${err.message}`, "error");
    } finally {
      state.loadingSites = false;
    }
  }

  function renderSidebar() {
    const totals = state.totals || {};
    const totalCount =
      totals.total ??
      state.sites.reduce((acc, s) => acc + Number(s.total || 0), 0);
    const totalPending =
      totals.pending ??
      state.sites.reduce((acc, s) => acc + Number(s.pending || 0), 0);
    const totalIngested =
      totals.accepted_ingested ??
      state.sites.reduce((acc, s) => acc + Number(s.accepted_ingested || 0), 0);
    const totalAccepted =
      totals.accepted_total ??
      state.sites.reduce((acc, s) => acc + Number(s.accepted_total || 0), 0);

    el.globalTotal.textContent = formatNumber(totalCount);
    el.globalPending.textContent = formatNumber(totalPending);
    el.globalRatio.textContent = `${formatNumber(totalIngested)} / ${formatNumber(totalAccepted)}`;
    el.globalOffline.textContent = formatNumber(state.offlineDocsCount);
    el.siteCountPill.textContent = String(state.sites.length);

    if (!state.sites.length) {
      el.siteList.innerHTML = '<div class="site-list-empty">Chưa có site nào được cấu hình.</div>';
      return;
    }

    el.siteList.innerHTML = state.sites
      .map((site) => {
        const isActive = site.site === state.activeSite;
        const pending = Number(site.pending || 0);
        const ingested = Number(site.accepted_ingested || 0);
        const accepted = Number(site.accepted_total || 0);
        const failed = Number(site.failed || 0);
        const pendingClass = pending === 0 ? "pending-badge is-zero" : "pending-badge";

        return `
          <button
            type="button"
            class="site-item ${isActive ? "is-active" : ""}"
            data-site="${escapeHtml(site.site)}"
          >
            <div class="site-item-main">
              <div class="site-item-name-row">
                <span class="site-item-name">${escapeHtml(site.site)}</span>
                ${failed > 0 ? `<span class="failed-dot" title="${failed} lỗi"></span>` : ""}
              </div>
              <span class="site-item-ratio">Đã cào ${formatNumber(ingested)} / ${formatNumber(accepted)} · Tổng ${formatNumber(site.total || 0)}</span>
            </div>
            <div class="site-item-badges">
              <span class="${pendingClass}" title="Chờ duyệt">${formatNumber(pending)}</span>
            </div>
          </button>
        `;
      })
      .join("");
  }

  function getActiveSiteSummary() {
    return state.sites.find((s) => s.site === state.activeSite) || null;
  }

  function renderActiveSiteHeader() {
    const summary = getActiveSiteSummary();
    el.activeSiteName.textContent = state.activeSite || "-";
    if (summary && Array.isArray(summary.hosts) && summary.hosts.length > 0) {
      el.activeSiteHosts.textContent = summary.hosts.join(" · ");
      el.activeSiteHosts.title = summary.hosts.join(", ");
    } else {
      el.activeSiteHosts.textContent = state.activeSite === "OTHER" ? "Tài liệu offline và nguồn bổ sung" : "Không có host cấu hình";
      el.activeSiteHosts.title = "";
    }
  }

  function updateStatCardsFromActiveSite(statsOverride = null) {
    const s = statsOverride || getActiveSiteSummary() || {};
    el.statTotal.textContent = formatNumber(s.total || 0);
    el.statPending.textContent = formatNumber(s.pending || 0);
    el.statQueued.textContent = formatNumber(s.accepted_queued || 0);
    el.statIngested.textContent = formatNumber(s.accepted_ingested || 0);
    el.statListing.textContent = formatNumber(s.listing || 0);
    el.statRejected.textContent = formatNumber(s.rejected_total || 0);
    el.statRejectedSub.textContent = `Thủ công: ${formatNumber(s.rejected_manual || 0)} · Tự động: ${formatNumber(s.rejected_auto || 0)}`;
    el.statFailed.textContent = formatNumber(s.failed || 0);

    const cardButtons = el.statCards.querySelectorAll(".stat-card");
    cardButtons.forEach((btn) => {
      const cardStatus = btn.getAttribute("data-status");
      const active = cardStatus === state.filters.status;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  /* ========================================================================
     LOAD & RENDER SITE SOURCES
     ======================================================================== */
  async function loadSources() {
    if (!state.activeSite) return;
    state.loadingSources = true;
    renderTableLoading();

    const params = new URLSearchParams();
    if (state.filters.status && state.filters.status !== "all") {
      params.set("status", state.filters.status);
    }
    if (state.filters.resource_type && state.filters.resource_type !== "all") {
      params.set("resource_type", state.filters.resource_type);
    }
    if (state.filters.search.trim()) {
      params.set("search", state.filters.search.trim());
    }
    if (state.filters.discovered_from) {
      params.set("discovered_from", state.filters.discovered_from);
    }
    params.set("sort", state.filters.sort || "priority");
    params.set("page", String(state.filters.page));
    params.set("page_size", String(state.filters.page_size));

    try {
      const data = await apiFetch(
        `/api/sites/${encodeURIComponent(state.activeSite)}/sources?${params.toString()}`
      );
      state.items = Array.isArray(data.items) ? data.items : [];
      state.totalFiltered = Number(data.total_filtered || 0);
      state.discoveredFromOptions = Array.isArray(data.discovered_from_options)
        ? data.discovered_from_options
        : [];

      if (data.stats) {
        const idx = state.sites.findIndex((s) => s.site === state.activeSite);
        if (idx >= 0) {
          state.sites[idx] = { ...state.sites[idx], ...data.stats };
          renderSidebar();
        }
        updateStatCardsFromActiveSite(data.stats);
      }

      syncDiscoveredFromDropdown();
      syncFilterControlsUI();
      renderTable();
      renderPagination();
      syncBulkBar();
    } catch (err) {
      el.sourcesTbody.innerHTML = `<tr><td colspan="7" class="table-empty-cell">Lỗi tải danh sách nguồn: ${escapeHtml(err.message)}</td></tr>`;
      showToast(`Lỗi tải nguồn ${state.activeSite}: ${err.message}`, "error");
    } finally {
      state.loadingSources = false;
    }
  }

  function renderTableLoading() {
    el.sourcesTbody.innerHTML =
      '<tr><td colspan="7" class="table-empty-cell">Đang tải dữ liệu...</td></tr>';
  }

  function syncDiscoveredFromDropdown() {
    const current = state.filters.discovered_from;
    const options = [...state.discoveredFromOptions];
    if (current && !options.includes(current)) {
      options.unshift(current);
    }

    const html = ['<option value="">Trang nguồn: Tất cả</option>'].concat(
      options.map((url) => {
        const selected = url === current ? "selected" : "";
        let label = url;
        try {
          const parsed = new URL(url);
          label = `${parsed.pathname || "/"} (${parsed.hostname})`;
        } catch {
          label = url;
        }
        return `<option value="${escapeHtml(url)}" ${selected} title="${escapeHtml(url)}">${escapeHtml(label)}</option>`;
      })
    );
    el.filterDiscoveredFrom.innerHTML = html.join("");
  }

  function syncFilterControlsUI() {
    const segmentBtns = document.querySelectorAll(".type-segment .segment-btn");
    segmentBtns.forEach((btn) => {
      btn.classList.toggle(
        "is-active",
        btn.getAttribute("data-type") === state.filters.resource_type
      );
    });

    el.filterSort.value = state.filters.sort;
    el.pageSizeSelect.value = String(state.filters.page_size);

    const hasActiveFilters =
      state.filters.status !== "all" ||
      state.filters.resource_type !== "all" ||
      state.filters.search.trim() !== "" ||
      state.filters.discovered_from !== "" ||
      state.filters.sort !== "priority";

    el.btnResetFilters.classList.toggle("hidden", !hasActiveFilters);
  }

  function getItemStatusPresentation(item) {
    if (item.fail_error && !item.ingested) {
      return { cls: "status-failed", label: "Failed", sub: item.decision === "accepted" ? "Accepted" : "" };
    }
    if (item.decision === "accepted") {
      if (item.ingested) {
        const chars = item.char_count ? `${formatNumber(item.char_count)} ký tự` : "Đã parse";
        return { cls: "status-ingested", label: "Ingested", sub: chars };
      }
      return { cls: "status-queued", label: "Queued", sub: "Chờ cào" };
    }
    if (item.decision === "pending") {
      return { cls: "status-pending", label: "Pending", sub: "" };
    }
    if (item.decision === "listing") {
      return { cls: "status-listing", label: "Listing", sub: "Mục lục" };
    }
    if (item.decision === "rejected") {
      const reason = item.decision_reason || "";
      const isManual = reason === "manual_curation" || reason === "manual_override";
      return {
        cls: "status-rejected",
        label: "Rejected",
        sub: reason ? (isManual ? "Thủ công" : "Tự động") : ""
      };
    }
    if (item.decision === "alias") {
      return { cls: "status-listing", label: "Alias", sub: "Redirect" };
    }
    return { cls: "status-listing", label: item.decision || "Unknown", sub: "" };
  }

  function buildRowDecisionButtons(item) {
    const d = item.decision;
    if (d === "alias") return "";
    const isHtml = (item.resource_type || "html") === "html";

    const btns = [];
    if (d !== "accepted") {
      btns.push(
        `<button type="button" class="btn btn-accept btn-xs" data-action="decision" data-decision="accepted" data-sid="${escapeHtml(item.source_id)}" title="Duyệt (Accept)">${ICONS.check}<span>Accept</span></button>`
      );
    }
    if (d !== "listing" && isHtml) {
      btns.push(
        `<button type="button" class="btn btn-secondary btn-xs" data-action="decision" data-decision="listing" data-sid="${escapeHtml(item.source_id)}" title="Đặt Listing">${ICONS.list}<span>Listing</span></button>`
      );
    }
    if (d !== "rejected") {
      btns.push(
        `<button type="button" class="btn btn-reject btn-xs" data-action="decision" data-decision="rejected" data-sid="${escapeHtml(item.source_id)}" title="Từ chối (Reject)">${ICONS.cross}<span>Reject</span></button>`
      );
    }
    if (d !== "pending") {
      btns.push(
        `<button type="button" class="btn btn-secondary btn-xs" data-action="decision" data-decision="pending" data-sid="${escapeHtml(item.source_id)}" title="Chuyển về Pending"><span>Pending</span></button>`
      );
    }
    return btns.join("");
  }

  function renderTable() {
    if (!state.items.length) {
      el.sourcesTbody.innerHTML =
        '<tr><td colspan="7" class="table-empty-cell">Không tìm thấy nguồn nào phù hợp với bộ lọc hiện tại.</td></tr>';
      el.selectAllCheckbox.checked = false;
      el.selectAllCheckbox.indeterminate = false;
      return;
    }

    el.sourcesTbody.innerHTML = state.items
      .map((item) => {
        const sid = item.source_id;
        const checked = state.selectedIds.has(sid);
        const primaryTitle =
          item.title || item.link_text || extractUrlFallbackLabel(item.uri);
        const showAnchorSub =
          item.link_text && item.title && item.link_text.trim() !== item.title.trim();

        const safeUrl = safeHttpUrl(item.uri);
        const urlHtml = safeUrl
          ? `<a class="url-link" href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(item.uri)}"><span>${escapeHtml(item.uri)}</span>${ICONS.external}</a>`
          : `<span class="mono">${escapeHtml(item.uri || "-")}</span>`;

        const parentHtml = item.discovered_from
          ? `<span class="parent-url" title="${escapeHtml(item.discovered_from)}">${escapeHtml(item.discovered_from)}</span>`
          : `<span class="note-text">Seed / Thủ công</span>`;

        const depthText =
          item.crawl_depth !== null && item.crawl_depth !== undefined
            ? `D${item.crawl_depth}`
            : "D0";

        let typeLabel = (item.resource_type || "html").toUpperCase();
        let typeClass = "type-badge";
        if (item.is_local) {
          typeLabel = "LOCAL";
          typeClass = "type-badge is-local";
        } else if (typeLabel === "PDF") {
          typeClass = "type-badge is-pdf";
        }

        const statusPres = getItemStatusPresentation(item);
        const decisionBtns = buildRowDecisionButtons(item);

        const inspectBtn = item.ingested
          ? `<button type="button" class="btn btn-secondary btn-xs" data-action="processed" data-sid="${escapeHtml(sid)}" title="Xem nội dung đã Parse">${ICONS.doc}<span>Đã Parse</span></button>`
          : `<button type="button" class="btn btn-secondary btn-xs" data-action="preview" data-sid="${escapeHtml(sid)}" title="Xem nhanh Live">${ICONS.eye}<span>Xem nhanh</span></button>`;

        const reingestBtn =
          item.decision === "accepted" || (item.fail_error && item.decision === "listing")
            ? `<button type="button" class="btn btn-secondary btn-xs" data-action="reingest" data-sid="${escapeHtml(sid)}" title="Cào lại nguồn này">${ICONS.reingest}<span>Cào lại</span></button>`
            : "";

        const removeBtn = `<button type="button" class="btn btn-secondary btn-xs" data-action="remove" data-sid="${escapeHtml(sid)}" title="Xóa khỏi tập nguồn">${ICONS.trash}</button>`;

        return `
          <tr class="${checked ? "is-selected" : ""}" data-sid="${escapeHtml(sid)}">
            <td class="col-check">
              <input type="checkbox" class="row-checkbox" data-sid="${escapeHtml(sid)}" ${checked ? "checked" : ""} aria-label="Chọn ${escapeHtml(primaryTitle)}" />
            </td>
            <td class="col-title">
              <div class="cell-title-text" title="${escapeHtml(primaryTitle)}">${escapeHtml(primaryTitle)}</div>
              <div class="cell-meta-row">
                ${item.topic_hint ? `<span class="topic-badge ${item.topic_hint === "Thông báo cũ" ? "is-old" : ""}" title="Độ ưu tiên: ${escapeHtml(item.topic_priority ?? 0)}">${escapeHtml(item.topic_hint)}</span>` : ""}
                ${item.is_local ? '<span class="note-text">offline_docs</span>' : ""}
                ${showAnchorSub ? `<span class="note-text">Anchor: ${escapeHtml(item.link_text)}</span>` : ""}
                ${item.override_note ? `<span class="note-text">Ghi chú: ${escapeHtml(item.override_note)}</span>` : ""}
              </div>
              ${item.fail_error ? `<div class="fail-inline">${escapeHtml(item.fail_error)}</div>` : ""}
            </td>
            <td class="col-url">${urlHtml}</td>
            <td class="col-parent">
              <div class="parent-cell">
                ${parentHtml}
                <span class="depth-badge">${escapeHtml(depthText)}</span>
              </div>
            </td>
            <td class="col-type">
              <span class="${typeClass}">${escapeHtml(typeLabel)}</span>
            </td>
            <td class="col-status">
              <span class="status-badge ${statusPres.cls}">${escapeHtml(statusPres.label)}</span>
              ${statusPres.sub ? `<div class="status-sub">${escapeHtml(statusPres.sub)}</div>` : ""}
            </td>
            <td class="col-actions">
              <div class="row-actions">
                ${decisionBtns}
                ${inspectBtn}
                ${reingestBtn}
                ${removeBtn}
              </div>
            </td>
          </tr>
        `;
      })
      .join("");

    syncSelectAllCheckbox();
  }

  function syncSelectAllCheckbox() {
    if (!state.items.length) {
      el.selectAllCheckbox.checked = false;
      el.selectAllCheckbox.indeterminate = false;
      return;
    }
    const selectedOnPage = state.items.filter((i) => state.selectedIds.has(i.source_id)).length;
    el.selectAllCheckbox.checked = selectedOnPage === state.items.length;
    el.selectAllCheckbox.indeterminate =
      selectedOnPage > 0 && selectedOnPage < state.items.length;
  }

  function syncBulkBar() {
    const count = state.selectedIds.size;
    el.bulkBar.classList.toggle("hidden", count === 0);
    el.bulkCountText.textContent = `Đã chọn ${formatNumber(count)} mục`;
  }

  function renderPagination() {
    const total = state.totalFiltered;
    const page = state.filters.page;
    const pageSize = state.filters.page_size;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));

    const start = total === 0 ? 0 : (page - 1) * pageSize + 1;
    const end = Math.min(total, page * pageSize);

    el.paginationSummary.textContent = `Hiển thị ${formatNumber(start)}–${formatNumber(end)} / ${formatNumber(total)} mục`;
    el.pageIndicator.textContent = `Trang ${formatNumber(page)} / ${formatNumber(totalPages)}`;
    el.btnPrevPage.disabled = page <= 1;
    el.btnNextPage.disabled = page >= totalPages;
  }

  /* ========================================================================
     MUTATIONS: DECISION, ADD URLS, REMOVE, OFFLINE SCAN & UPLOAD
     ======================================================================== */
  async function updateDecision(sourceIds, decision, note = "") {
    if (!state.activeSite || !sourceIds.length) return;
    try {
      const res = await apiFetch(
        `/api/sites/${encodeURIComponent(state.activeSite)}/decision`,
        {
          method: "POST",
          body: JSON.stringify({ source_ids: sourceIds, decision, note })
        }
      );
      const updated = Number(res.updated ?? sourceIds.length);
      showToast(`Đã cập nhật ${updated} nguồn sang trạng thái ${decision}`, "success");
      sourceIds.forEach((id) => state.selectedIds.delete(id));
      syncBulkBar();
      await Promise.all([loadSites(), loadSources()]);
    } catch (err) {
      showToast(`Không thể cập nhật trạng thái: ${err.message}`, "error");
    }
  }

  function openAddUrlsModal() {
    el.modalAddSubtitle.textContent = `Thêm trực tiếp URL vào site ${state.activeSite || ""}`;
    el.addUrlsInput.value = "";
    el.addUrlsNote.value = "";
    el.addUrlsDecision.value = "accepted";
    el.addUrlsResult.classList.add("hidden");
    el.addUrlsResult.innerHTML = "";
    openModal("modal-add-urls");
    window.setTimeout(() => el.addUrlsInput.focus(), 40);
  }

  async function handleAddUrlsSubmit(evt) {
    evt.preventDefault();
    if (!state.activeSite) return;

    const rawLines = el.addUrlsInput.value
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    if (!rawLines.length) {
      showToast("Vui lòng nhập ít nhất một URL hợp lệ.", "warning");
      return;
    }

    el.btnSubmitAddUrls.disabled = true;
    el.btnSubmitAddUrls.textContent = "Đang thêm...";
    try {
      const res = await apiFetch(
        `/api/sites/${encodeURIComponent(state.activeSite)}/add`,
        {
          method: "POST",
          body: JSON.stringify({
            urls: rawLines,
            decision: el.addUrlsDecision.value,
            note: el.addUrlsNote.value.trim()
          })
        }
      );

      const added = Array.isArray(res.added) ? res.added : [];
      const duplicates = Array.isArray(res.duplicates) ? res.duplicates : [];
      const invalid = Array.isArray(res.invalid) ? res.invalid : [];

      const parts = [
        `<div><strong>Đã thêm:</strong> ${added.length}</div>`,
        `<div><strong>Trùng lặp:</strong> ${duplicates.length}</div>`,
        `<div><strong>Không hợp lệ:</strong> ${invalid.length}</div>`
      ];

      if (duplicates.length > 0) {
        parts.push(
          `<div>Trùng lặp: ${duplicates
            .map((d) => `${escapeHtml(d.uri || d)} (${escapeHtml(d.existing_decision || "")})`)
            .join(", ")}</div>`
        );
      }
      if (invalid.length > 0) {
        parts.push(
          `<div>Lỗi URL: ${invalid
            .map((inv) => `${escapeHtml(inv.uri || inv)}: ${escapeHtml(inv.error || "invalid")}`)
            .join("; ")}</div>`
        );
      }

      el.addUrlsResult.innerHTML = parts.join("");
      el.addUrlsResult.classList.remove("hidden");

      if (added.length > 0) {
        showToast(`Đã thêm ${added.length} URL vào ${state.activeSite}`, "success");
        await Promise.all([loadSites(), loadSources()]);
      } else {
        showToast("Không có URL mới nào được thêm.", "warning");
      }
    } catch (err) {
      showToast(`Thêm URL thất bại: ${err.message}`, "error");
    } finally {
      el.btnSubmitAddUrls.disabled = false;
      el.btnSubmitAddUrls.textContent = "Thêm URL";
    }
  }

  function openRemoveModal(sourceIds) {
    if (!sourceIds.length) return;
    state.removeTargetIds = [...sourceIds];
    el.removePurgeCheckbox.checked = false;
    if (sourceIds.length === 1) {
      const target = state.items.find((i) => i.source_id === sourceIds[0]);
      const label = target ? target.uri : sourceIds[0];
      el.removeConfirmMessage.textContent = `Bạn có chắc muốn xóa nguồn "${label}" khỏi site ${state.activeSite}?`;
    } else {
      el.removeConfirmMessage.textContent = `Bạn có chắc muốn xóa ${sourceIds.length} nguồn đã chọn khỏi site ${state.activeSite}?`;
    }
    openModal("modal-remove");
  }

  async function handleConfirmRemove() {
    if (!state.activeSite || !state.removeTargetIds.length) return;
    const purgeData = Boolean(el.removePurgeCheckbox.checked);
    el.btnConfirmRemove.disabled = true;
    el.btnConfirmRemove.textContent = "Đang xóa...";

    try {
      const res = await apiFetch(
        `/api/sites/${encodeURIComponent(state.activeSite)}/remove`,
        {
          method: "POST",
          body: JSON.stringify({
            source_ids: state.removeTargetIds,
            purge_data: purgeData
          })
        }
      );
      state.removeTargetIds.forEach((id) => state.selectedIds.delete(id));
      state.removeTargetIds = [];
      closeModal("modal-remove");
      syncBulkBar();
      showToast(
        `Đã xóa ${res.removed ?? 0} nguồn${purgeData ? " (kèm raw & processed)" : ""}`,
        "success"
      );
      await Promise.all([loadSites(), loadSources()]);
    } catch (err) {
      showToast(`Xóa nguồn thất bại: ${err.message}`, "error");
    } finally {
      el.btnConfirmRemove.disabled = false;
      el.btnConfirmRemove.textContent = "Xác nhận xóa";
    }
  }

  async function handleOfflineScan() {
    el.btnOfflineScan.disabled = true;
    try {
      const res = await apiFetch("/api/offline/scan", { method: "POST" });
      showToast(
        `Quét Offline: ${res.scanned ?? 0} tệp, ${res.new_pending ?? 0} tệp mới (Pending), ${res.existing ?? 0} đã tồn tại`,
        "success"
      );
      await Promise.all([loadSites(), loadSources()]);
    } catch (err) {
      showToast(`Quét thư mục Offline thất bại: ${err.message}`, "error");
    } finally {
      el.btnOfflineScan.disabled = false;
    }
  }

  /* ========================================================================
     UPLOAD LOCAL DOCUMENT (FileReader.readAsDataURL -> pure base64 JSON)
     ======================================================================== */
  function openUploadModal() {
    state.pendingUploadFile = null;
    el.uploadFileInput.value = "";
    el.uploadFileInfo.textContent = "Chưa chọn tệp nào (.pdf, .html, .htm)";
    el.uploadDecision.value = "pending";
    el.btnSubmitUpload.disabled = true;
    openModal("modal-upload-doc");
  }

  function handleSelectedUploadFile(file) {
    if (!file) return;
    const lowerName = file.name.toLowerCase();
    const validExt = ALLOWED_UPLOAD_EXTS.some((ext) => lowerName.endsWith(ext));
    if (!validExt) {
      showToast("Chỉ hỗ trợ tệp .pdf, .html hoặc .htm", "error");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      showToast("Kích thước tệp vượt quá giới hạn 30MB", "error");
      return;
    }
    state.pendingUploadFile = file;
    el.uploadFileInfo.textContent = `${file.name} (${formatBytes(file.size)})`;
    el.btnSubmitUpload.disabled = false;
  }

  function readFileAsPureBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || "");
        const commaIndex = result.indexOf(",");
        const pureBase64 = commaIndex >= 0 ? result.slice(commaIndex + 1) : result;
        resolve(pureBase64);
      };
      reader.onerror = () => reject(new Error("Không thể đọc nội dung tệp"));
      reader.readAsDataURL(file);
    });
  }

  async function handleUploadSubmit(evt) {
    evt.preventDefault();
    const file = state.pendingUploadFile;
    if (!file) return;

    el.btnSubmitUpload.disabled = true;
    el.btnSubmitUpload.textContent = "Đang tải lên...";
    try {
      const contentBase64 = await readFileAsPureBase64(file);
      const res = await apiFetch("/api/offline/upload", {
        method: "POST",
        body: JSON.stringify({
          filename: file.name,
          content_base64: contentBase64,
          decision: el.uploadDecision.value
        })
      });
      closeModal("modal-upload-doc");
      showToast(
        `Đã upload ${res.filename || file.name} vào ${res.site || "OTHER"} (${res.decision || el.uploadDecision.value})`,
        "success"
      );
      await Promise.all([loadSites(), loadSources()]);
    } catch (err) {
      showToast(`Upload thất bại: ${err.message}`, "error");
    } finally {
      el.btnSubmitUpload.disabled = !state.pendingUploadFile;
      el.btnSubmitUpload.textContent = "Tải lên";
    }
  }

  /* ========================================================================
     INSPECT MODALS: LIVE PREVIEW & PROCESSED DOCUMENT VIEWER
     ======================================================================== */
  async function openPreviewModal(sourceId) {
    if (!state.activeSite) return;
    state.previewSourceId = sourceId;
    const item = state.items.find((i) => i.source_id === sourceId);
    const initialIsHtml = !item || (item.resource_type || "html") === "html";
    el.previewBtnListing.classList.toggle("hidden", !initialIsHtml);

    el.previewSourceUrl.textContent = item ? item.uri : sourceId;
    el.previewBody.innerHTML = '<div class="modal-loading">Đang tải xem nhanh từ máy chủ đích...</div>';
    openModal("modal-preview");

    try {
      const data = await apiFetch(
        `/api/sites/${encodeURIComponent(state.activeSite)}/preview/${encodeURIComponent(sourceId)}`
      );
      if (data.is_pdf) {
        el.previewBtnListing.classList.add("hidden");
      }

      if (data.error) {
        el.previewBody.innerHTML = `
          <div class="fail-inline">Lỗi tải preview: ${escapeHtml(data.error)}</div>
          <div class="inspect-meta-grid">
            <div class="inspect-meta-item">
              <div class="inspect-meta-label">HTTP Status</div>
              <div class="inspect-meta-val">${escapeHtml(data.status_code ?? "-")}</div>
            </div>
            <div class="inspect-meta-item">
              <div class="inspect-meta-label">Final URL</div>
              <div class="inspect-meta-val mono">${escapeHtml(data.final_url || (item && item.uri) || "-")}</div>
            </div>
          </div>
        `;
        return;
      }

      const finalUrlSafe = safeHttpUrl(data.final_url);
      const finalUrlHtml = finalUrlSafe
        ? `<a href="${escapeHtml(finalUrlSafe)}" target="_blank" rel="noopener noreferrer">${escapeHtml(data.final_url)}</a>`
        : escapeHtml(data.final_url || "-");

      const pdfLinks = Array.isArray(data.pdf_links) ? data.pdf_links : [];
      const pdfListHtml = pdfLinks.length
        ? `
          <div>
            <div class="inspect-section-title">Tệp PDF / Đính kèm phát hiện (${pdfLinks.length})</div>
            <ul class="pdf-links-list">
              ${pdfLinks
                .map((link) => {
                  const linkHref = safeHttpUrl(link.url);
                  return `
                    <li class="pdf-link-item">
                      <div>
                        <div class="pdf-link-text">${escapeHtml(link.link_text || "Tài liệu PDF")}</div>
                        <div class="pdf-link-url">${
                          linkHref
                            ? `<a href="${escapeHtml(linkHref)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.url)}</a>`
                            : escapeHtml(link.url || "")
                        }</div>
                      </div>
                      ${
                        linkHref
                          ? `<button type="button" class="btn btn-secondary btn-xs" data-preview-add-url="${escapeHtml(linkHref)}">+ Thêm</button>`
                          : ""
                      }
                    </li>
                  `;
                })
                .join("")}
            </ul>
          </div>
        `
        : "";

      const excerptText = data.is_pdf
        ? "Tài liệu dạng PDF (không trích xuất văn bản live; chọn Accept để tải và parse PDF qua pipeline)."
        : data.excerpt || "(Không có nội dung trích đoạn)";

      el.previewBody.innerHTML = `
        <div class="inspect-meta-grid">
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">Tiêu đề</div>
            <div class="inspect-meta-val">${escapeHtml(data.title || "(Không có tiêu đề)")}</div>
          </div>
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">Ngày đăng</div>
            <div class="inspect-meta-val">${escapeHtml(data.published_at || "-")}</div>
          </div>
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">HTTP / Loại</div>
            <div class="inspect-meta-val">${escapeHtml(data.status_code ?? "-")} · ${data.is_pdf ? "PDF" : "HTML"} · ${formatBytes(data.content_length || 0)}</div>
          </div>
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">Final URL</div>
            <div class="inspect-meta-val mono">${finalUrlHtml}</div>
          </div>
        </div>
        <div>
          <div class="inspect-section-title">Trích đoạn nội dung</div>
          <div class="inspect-content-box">${escapeHtml(excerptText)}</div>
        </div>
        ${pdfListHtml}
      `;
    } catch (err) {
      el.previewBody.innerHTML = `<div class="fail-inline">Không thể tải preview: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function openProcessedModal(sourceId) {
    if (!state.activeSite) return;
    state.processedSourceId = sourceId;
    const item = state.items.find((i) => i.source_id === sourceId);

    el.processedSubtitle.textContent = item ? item.uri : sourceId;
    el.processedBody.innerHTML = '<div class="modal-loading">Đang tải nội dung đã parse...</div>';
    openModal("modal-processed");

    try {
      const doc = await apiFetch(
        `/api/sites/${encodeURIComponent(state.activeSite)}/processed/${encodeURIComponent(sourceId)}`
      );
      const meta = doc.metadata && typeof doc.metadata === "object" ? doc.metadata : {};
      const extractionMethod =
        doc.extraction_method ||
        meta.extraction_method ||
        meta.extractor ||
        meta.parser ||
        "-";
      const charCount = doc.char_count ?? (doc.content ? doc.content.length : 0);

      el.processedBody.innerHTML = `
        <div class="inspect-meta-grid">
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">Tiêu đề</div>
            <div class="inspect-meta-val">${escapeHtml(doc.title || "(Không có tiêu đề)")}</div>
          </div>
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">Số ký tự</div>
            <div class="inspect-meta-val">${formatNumber(charCount)} ký tự</div>
          </div>
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">Phương thức trích xuất</div>
            <div class="inspect-meta-val mono">${escapeHtml(extractionMethod)}</div>
          </div>
          <div class="inspect-meta-item">
            <div class="inspect-meta-label">Document ID</div>
            <div class="inspect-meta-val mono">${escapeHtml(doc.id || sourceId)}</div>
          </div>
        </div>
        <div>
          <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;">
            <div class="inspect-section-title" style="margin-bottom:0;">Nội dung Markdown đã làm sạch</div>
            <button type="button" id="btn-copy-processed" class="btn btn-secondary btn-xs">Sao chép nội dung</button>
          </div>
          <div id="processed-content-text" class="inspect-content-box">${escapeHtml(doc.content || "(Trống)")}</div>
        </div>
        <div>
          <div class="inspect-section-title">Metadata</div>
          <div class="inspect-content-box mono">${escapeHtml(JSON.stringify(meta, null, 2))}</div>
        </div>
      `;

      const copyBtn = document.getElementById("btn-copy-processed");
      if (copyBtn) {
        copyBtn.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(doc.content || "");
            showToast("Đã sao chép nội dung Markdown", "success");
          } catch {
            showToast("Không thể truy cập clipboard", "warning");
          }
        });
      }
    } catch (err) {
      el.processedBody.innerHTML = `<div class="fail-inline">Không tải được nội dung đã parse: ${escapeHtml(err.message)}</div>`;
    }
  }

  /* ========================================================================
     INCREMENTAL INGEST & LIVE PROGRESS POLLING
     ======================================================================== */
  async function startIncrementalIngest(sites = null) {
    try {
      const res = await apiFetch("/api/ingest/incremental", {
        method: "POST",
        body: JSON.stringify({ sites })
      });
      state.dismissedJobId = null;
      state.ingestWasRunning = true;
      showToast(
        sites && sites.length
          ? `Đã khởi chạy Incremental Ingest cho ${sites.join(", ")}`
          : "Đã khởi chạy Incremental Ingest cho tất cả site",
        "info"
      );
      if (res && res.job_id) {
        state.lastJobIdSeen = res.job_id;
      }
      await pollIngestStatusOnce();
      ensureIngestPolling();
    } catch (err) {
      if (err.status === 409) {
        showToast("Đang có một tiến trình ingest khác chạy, đang theo dõi tiến độ...", "warning");
        state.dismissedJobId = null;
        await pollIngestStatusOnce();
        ensureIngestPolling();
      } else {
        showToast(`Không thể chạy Incremental Ingest: ${err.message}`, "error");
      }
    }
  }

  async function startReingest(sourceId) {
    if (!state.activeSite || !sourceId) return;
    try {
      const res = await apiFetch(
        `/api/sites/${encodeURIComponent(state.activeSite)}/reingest/${encodeURIComponent(sourceId)}`,
        { method: "POST" }
      );
      state.dismissedJobId = null;
      state.ingestWasRunning = true;
      if (res && res.job_id) {
        state.lastJobIdSeen = res.job_id;
      }
      showToast("Đã gửi yêu cầu cào lại nguồn dữ liệu", "info");
      await pollIngestStatusOnce();
      ensureIngestPolling();
    } catch (err) {
      if (err.status === 409) {
        showToast("Đang có tiến trình ingest chạy, vui lòng đợi hoàn tất", "warning");
        ensureIngestPolling();
      } else {
        showToast(`Cào lại thất bại: ${err.message}`, "error");
      }
    }
  }

  function ensureIngestPolling() {
    if (state.ingestPollTimer) return;
    state.ingestPollTimer = window.setInterval(pollIngestStatusOnce, 1000);
  }

  function stopIngestPolling() {
    if (state.ingestPollTimer) {
      window.clearInterval(state.ingestPollTimer);
      state.ingestPollTimer = null;
    }
  }

  function renderIngestBanner(job) {
    if (!job || !job.job_id || job.status === "idle") {
      el.ingestBanner.classList.add("hidden");
      return;
    }
    if (state.dismissedJobId === job.job_id) {
      el.ingestBanner.classList.add("hidden");
      return;
    }

    const isRunning = job.status === "queued" || job.status === "running";
    const isFailed = job.status === "failed";
    const isCompleted = job.status === "completed";

    el.ingestBanner.classList.remove("hidden", "is-failed", "is-completed");
    if (isFailed) el.ingestBanner.classList.add("is-failed");
    if (isCompleted) el.ingestBanner.classList.add("is-completed");

    el.ingestSpinner.classList.toggle("hidden", !isRunning);
    el.btnDismissIngest.classList.toggle("hidden", isRunning);

    el.ingestStatusBadge.className = "status-badge";
    if (isRunning) {
      el.ingestStatusBadge.classList.add("status-queued");
      el.ingestStatusBadge.textContent = job.status === "queued" ? "Đang xếp hàng" : "Đang chạy";
    } else if (isCompleted) {
      el.ingestStatusBadge.classList.add("status-ingested");
      el.ingestStatusBadge.textContent = "Hoàn tất";
    } else {
      el.ingestStatusBadge.classList.add("status-failed");
      el.ingestStatusBadge.textContent = "Thất bại";
    }

    const scopeText = Array.isArray(job.sites) && job.sites.length
      ? `Site: ${job.sites.join(", ")}`
      : "Tất cả Site";
    el.ingestTitle.textContent = job.force_source_id
      ? `Cào lại nguồn (${scopeText})`
      : `Incremental Ingest (${scopeText})`;

    const progress = job.progress || {};
    const phaseParts = [];
    if (progress.current_site) phaseParts.push(`Site hiện tại: ${progress.current_site}`);
    if (progress.phase) phaseParts.push(`Giai đoạn: ${progress.phase}`);
    el.ingestPhase.textContent = phaseParts.join(" · ");

    el.ingestSkipped.textContent = formatNumber(progress.skipped || 0);
    el.ingestDownloaded.textContent = formatNumber(progress.downloaded || 0);
    el.ingestProcessed.textContent = formatNumber(progress.processed_saved || 0);
    el.ingestNewPending.textContent = formatNumber(progress.new_pending || 0);
    el.ingestRejectedPreserved.textContent = formatNumber(progress.rejected_preserved || 0);
    el.ingestErrors.textContent = formatNumber(progress.errors || 0);

    if (job.error) {
      el.ingestErrorMsg.textContent = job.error;
      el.ingestErrorMsg.classList.remove("hidden");
    } else {
      el.ingestErrorMsg.classList.add("hidden");
      el.ingestErrorMsg.textContent = "";
    }
  }

  function setIngestButtonsBusy(busy) {
    el.btnIngestAll.disabled = busy;
    el.btnIngestSite.disabled = busy;
  }

  async function pollIngestStatusOnce() {
    try {
      const data = await apiFetch("/api/ingest/status");
      const job = data.active_job || data;
      const status = job.status || "idle";
      const isRunning = status === "queued" || status === "running";

      setIngestButtonsBusy(isRunning);

      if (isRunning) {
        state.ingestWasRunning = true;
        state.lastJobIdSeen = job.job_id || state.lastJobIdSeen;
        renderIngestBanner(job);
        ensureIngestPolling();
        return;
      }

      stopIngestPolling();

      if (state.ingestWasRunning) {
        state.ingestWasRunning = false;
        renderIngestBanner(job);
        if (status === "completed") {
          showToast("Incremental Ingest đã hoàn tất!", "success");
        } else if (status === "failed") {
          showToast(`Incremental Ingest lỗi: ${job.error || "Không xác định"}`, "error");
        }
        await Promise.all([loadSites(), loadSources()]);
      } else {
        el.ingestBanner.classList.add("hidden");
      }
    } catch {
      setIngestButtonsBusy(false);
      stopIngestPolling();
    }
  }

  /* ========================================================================
     EVENT BINDINGS
     ======================================================================== */
  function bindEvents() {
    el.themeToggleBtn.addEventListener("click", toggleTheme);

    el.mobileMenuBtn.addEventListener("click", () => setSidebarDrawerOpen(true));
    el.sidebarCloseBtn.addEventListener("click", () => setSidebarDrawerOpen(false));
    el.sidebarBackdrop.addEventListener("click", () => setSidebarDrawerOpen(false));

    // Sidebar site switching
    el.siteList.addEventListener("click", (evt) => {
      const btn = evt.target.closest(".site-item");
      if (!btn) return;
      const site = btn.getAttribute("data-site");
      if (!site || site === state.activeSite) {
        setSidebarDrawerOpen(false);
        return;
      }
      state.activeSite = site;
      try {
        localStorage.setItem(ACTIVE_SITE_STORAGE_KEY, site);
      } catch {
        // Ignore storage failures
      }
      state.filters.page = 1;
      state.filters.discovered_from = "";
      state.selectedIds.clear();
      syncBulkBar();
      renderSidebar();
      renderActiveSiteHeader();
      updateStatCardsFromActiveSite();
      setSidebarDrawerOpen(false);
      loadSources();
    });

    // Global & Topbar actions
    el.btnRefresh.addEventListener("click", () => loadSites({ refreshSources: true }));
    el.btnOfflineScan.addEventListener("click", handleOfflineScan);
    el.btnSidebarUpload.addEventListener("click", openUploadModal);
    el.btnOpenUpload.addEventListener("click", openUploadModal);
    el.btnOpenAddUrls.addEventListener("click", openAddUrlsModal);
    el.btnIngestAll.addEventListener("click", () => startIncrementalIngest(null));
    el.btnIngestSite.addEventListener("click", () => {
      if (state.activeSite) startIncrementalIngest([state.activeSite]);
    });
    el.btnDismissIngest.addEventListener("click", () => {
      state.dismissedJobId = state.lastJobIdSeen;
      el.ingestBanner.classList.add("hidden");
    });

    // Stat cards filter
    el.statCards.addEventListener("click", (evt) => {
      const card = evt.target.closest(".stat-card");
      if (!card) return;
      const status = card.getAttribute("data-status") || "all";
      state.filters.status = status;
      state.filters.page = 1;
      state.selectedIds.clear();
      syncBulkBar();
      loadSources();
    });

    // Search input (debounced)
    el.filterSearch.addEventListener("input", () => {
      if (searchDebounceTimer) window.clearTimeout(searchDebounceTimer);
      searchDebounceTimer = window.setTimeout(() => {
        state.filters.search = el.filterSearch.value;
        state.filters.page = 1;
        state.selectedIds.clear();
        syncBulkBar();
        loadSources();
      }, 280);
    });

    el.filterSearch.addEventListener("keydown", (evt) => {
      if (evt.key === "Enter") {
        if (searchDebounceTimer) window.clearTimeout(searchDebounceTimer);
        state.filters.search = el.filterSearch.value;
        state.filters.page = 1;
        state.selectedIds.clear();
        syncBulkBar();
        loadSources();
      }
    });

    // Resource type segmented buttons
    document.querySelector(".type-segment").addEventListener("click", (evt) => {
      const btn = evt.target.closest(".segment-btn");
      if (!btn) return;
      state.filters.resource_type = btn.getAttribute("data-type") || "all";
      state.filters.page = 1;
      state.selectedIds.clear();
      syncBulkBar();
      loadSources();
    });

    el.filterDiscoveredFrom.addEventListener("change", () => {
      state.filters.discovered_from = el.filterDiscoveredFrom.value;
      state.filters.page = 1;
      state.selectedIds.clear();
      syncBulkBar();
      loadSources();
    });

    el.filterSort.addEventListener("change", () => {
      state.filters.sort = el.filterSort.value || "priority";
      state.filters.page = 1;
      loadSources();
    });

    el.btnResetFilters.addEventListener("click", () => {
      state.filters.status = "all";
      state.filters.resource_type = "all";
      state.filters.search = "";
      state.filters.discovered_from = "";
      state.filters.sort = "priority";
      state.filters.page = 1;
      el.filterSearch.value = "";
      state.selectedIds.clear();
      syncBulkBar();
      loadSources();
    });

    // Pagination
    el.pageSizeSelect.addEventListener("change", () => {
      state.filters.page_size = Number(el.pageSizeSelect.value) || 50;
      state.filters.page = 1;
      state.selectedIds.clear();
      syncBulkBar();
      loadSources();
    });

    el.btnPrevPage.addEventListener("click", () => {
      if (state.filters.page > 1) {
        state.filters.page -= 1;
        state.selectedIds.clear();
        syncBulkBar();
        loadSources();
      }
    });

    el.btnNextPage.addEventListener("click", () => {
      const totalPages = Math.max(1, Math.ceil(state.totalFiltered / state.filters.page_size));
      if (state.filters.page < totalPages) {
        state.filters.page += 1;
        state.selectedIds.clear();
        syncBulkBar();
        loadSources();
      }
    });

    // Select-all and Row checkboxes & actions
    el.selectAllCheckbox.addEventListener("change", () => {
      const check = el.selectAllCheckbox.checked;
      state.items.forEach((item) => {
        if (check) {
          state.selectedIds.add(item.source_id);
        } else {
          state.selectedIds.delete(item.source_id);
        }
      });
      renderTable();
      syncBulkBar();
    });

    el.sourcesTbody.addEventListener("change", (evt) => {
      const cb = evt.target.closest(".row-checkbox");
      if (!cb) return;
      const sid = cb.getAttribute("data-sid");
      if (!sid) return;
      if (cb.checked) {
        state.selectedIds.add(sid);
      } else {
        state.selectedIds.delete(sid);
      }
      const row = cb.closest("tr");
      if (row) row.classList.toggle("is-selected", cb.checked);
      syncSelectAllCheckbox();
      syncBulkBar();
    });

    el.sourcesTbody.addEventListener("click", (evt) => {
      const btn = evt.target.closest("[data-action]");
      if (!btn) return;
      const action = btn.getAttribute("data-action");
      const sid = btn.getAttribute("data-sid");
      if (!sid) return;

      if (action === "decision") {
        const decision = btn.getAttribute("data-decision");
        if (decision) updateDecision([sid], decision);
      } else if (action === "preview") {
        openPreviewModal(sid);
      } else if (action === "processed") {
        openProcessedModal(sid);
      } else if (action === "reingest") {
        startReingest(sid);
      } else if (action === "remove") {
        openRemoveModal([sid]);
      }
    });

    // Bulk bar actions
    el.btnClearSelection.addEventListener("click", () => {
      state.selectedIds.clear();
      renderTable();
      syncBulkBar();
    });

    el.bulkBar.addEventListener("click", (evt) => {
      const btn = evt.target.closest("[data-bulk-decision]");
      if (!btn) return;
      const decision = btn.getAttribute("data-bulk-decision");
      const ids = Array.from(state.selectedIds);
      if (decision && ids.length) {
        updateDecision(ids, decision);
      }
    });

    el.btnBulkRemove.addEventListener("click", () => {
      const ids = Array.from(state.selectedIds);
      if (ids.length) openRemoveModal(ids);
    });

    // Add URLs form
    el.formAddUrls.addEventListener("submit", handleAddUrlsSubmit);

    // Upload dropzone & form
    el.uploadDropzone.addEventListener("click", () => el.uploadFileInput.click());
    el.uploadDropzone.addEventListener("keydown", (evt) => {
      if (evt.key === "Enter" || evt.key === " ") {
        evt.preventDefault();
        el.uploadFileInput.click();
      }
    });
    el.uploadFileInput.addEventListener("change", () => {
      if (el.uploadFileInput.files && el.uploadFileInput.files[0]) {
        handleSelectedUploadFile(el.uploadFileInput.files[0]);
      }
    });
    el.uploadDropzone.addEventListener("dragover", (evt) => {
      evt.preventDefault();
      el.uploadDropzone.classList.add("is-dragover");
    });
    el.uploadDropzone.addEventListener("dragleave", () => {
      el.uploadDropzone.classList.remove("is-dragover");
    });
    el.uploadDropzone.addEventListener("drop", (evt) => {
      evt.preventDefault();
      el.uploadDropzone.classList.remove("is-dragover");
      if (evt.dataTransfer && evt.dataTransfer.files && evt.dataTransfer.files[0]) {
        handleSelectedUploadFile(evt.dataTransfer.files[0]);
      }
    });
    el.formUploadDoc.addEventListener("submit", handleUploadSubmit);

    // Preview modal actions
    el.previewBody.addEventListener("click", async (evt) => {
      const addBtn = evt.target.closest("[data-preview-add-url]");
      if (!addBtn || !state.activeSite) return;
      const targetUrl = addBtn.getAttribute("data-preview-add-url");
      if (!targetUrl) return;
      addBtn.disabled = true;
      try {
        const res = await apiFetch(
          `/api/sites/${encodeURIComponent(state.activeSite)}/add`,
          {
            method: "POST",
            body: JSON.stringify({
              urls: [targetUrl],
              decision: "pending",
              note: "Added from live preview"
            })
          }
        );
        const addedCount = Array.isArray(res.added) ? res.added.length : 0;
        const dupCount = Array.isArray(res.duplicates) ? res.duplicates.length : 0;
        if (addedCount > 0) {
          addBtn.textContent = "Đã thêm";
          showToast("Đã thêm URL PDF vào danh sách Pending", "success");
          await Promise.all([loadSites(), loadSources()]);
        } else if (dupCount > 0) {
          addBtn.textContent = "Đã tồn tại";
          showToast("URL này đã có trong danh sách nguồn", "info");
        } else {
          addBtn.disabled = false;
          const errMsg =
            Array.isArray(res.invalid) && res.invalid[0] && res.invalid[0].error
              ? res.invalid[0].error
              : "URL không thuộc host của site này";
          showToast(errMsg, "warning");
        }
      } catch (err) {
        addBtn.disabled = false;
        showToast(`Không thể thêm URL: ${err.message}`, "error");
      }
    });

    el.previewBtnAccept.addEventListener("click", async () => {
      if (!state.previewSourceId) return;
      const sid = state.previewSourceId;
      closeModal("modal-preview");
      await updateDecision([sid], "accepted");
    });
    el.previewBtnListing.addEventListener("click", async () => {
      if (!state.previewSourceId) return;
      const sid = state.previewSourceId;
      closeModal("modal-preview");
      await updateDecision([sid], "listing");
    });
    el.previewBtnReject.addEventListener("click", async () => {
      if (!state.previewSourceId) return;
      const sid = state.previewSourceId;
      closeModal("modal-preview");
      await updateDecision([sid], "rejected");
    });

    // Processed viewer actions
    el.processedBtnReject.addEventListener("click", async () => {
      if (!state.processedSourceId) return;
      const sid = state.processedSourceId;
      closeModal("modal-processed");
      await updateDecision([sid], "rejected");
    });
    el.processedBtnReingest.addEventListener("click", async () => {
      if (!state.processedSourceId) return;
      const sid = state.processedSourceId;
      closeModal("modal-processed");
      await startReingest(sid);
    });

    // Remove confirmation
    el.btnConfirmRemove.addEventListener("click", handleConfirmRemove);

    // Modal close buttons and backdrop click
    document.querySelectorAll("[data-close-modal]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-close-modal");
        if (id) closeModal(id);
      });
    });

    document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
      backdrop.addEventListener("click", (evt) => {
        if (evt.target === backdrop) {
          closeModal(backdrop.id);
        }
      });
    });

    // Keyboard shortcuts (Esc to close modal/drawer)
    document.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape") {
        closeTopmostModalOrDrawer();
      }
    });
  }

  /* ========================================================================
     INITIALIZATION
     ======================================================================== */
  async function init() {
    bindEvents();
    await loadSites({ refreshSources: true });
    await pollIngestStatusOnce();
  }

  document.addEventListener("DOMContentLoaded", init);
})();

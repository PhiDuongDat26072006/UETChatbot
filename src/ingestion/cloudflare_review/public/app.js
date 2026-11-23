const PIN_KEY = "uet_cf_review_pin";
const THEME_KEY = "uet_cf_review_theme";

const STATUS_LABELS = {
  accepted: "Đã duyệt",
  listing: "Listing",
  rejected: "Từ chối",
  pending: "Chờ duyệt",
};

const state = {
  pin: "",
  site: "UET",
  status: "pending",
  resourceType: "all",
  sort: "priority",
  search: "",
  view: "focus",
  sitesSummary: [],
  totals: { pending: 0, dirty_count: 0 },
  items: [],
  totalFiltered: 0,
  page: 1,
  pageSize: 30,
  hasMore: false,
  focusIndex: 0,
  selected: new Map(),
  undoStack: [],
  busy: false,
  noteOpen: false,
};

const el = (id) => document.getElementById(id);

let toastTimer = null;
function showToast(msg) {
  const node = el("toast");
  node.textContent = msg;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, 2800);
}

function isAcceptedType(item) {
  return item.resource_type === "html" || item.resource_type === "pdf" || Boolean(item.is_docx);
}

async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}), "X-Review-Pin": state.pin };
  if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const resp = await fetch(path, { ...options, headers });
  const data = await resp.json().catch(() => ({}));
  if (resp.status === 401) {
    clearPin();
    throw new Error(data.error || "Mã PIN không hợp lệ");
  }
  if (!resp.ok) throw new Error(data.error || `Lỗi API (${resp.status})`);
  return data;
}

function clearPin() {
  state.pin = "";
  try {
    localStorage.removeItem(PIN_KEY);
  } catch {}
  el("app-root").hidden = true;
  el("auth-screen").hidden = false;
  el("pin-input").value = "";
}

async function tryUnlock(pin) {
  state.pin = pin.trim();
  if (!state.pin) throw new Error("Vui lòng nhập mã PIN");
  await apiFetch("/api/auth/check");
  try {
    localStorage.setItem(PIN_KEY, state.pin);
  } catch {}
  el("auth-error").hidden = true;
  el("auth-screen").hidden = true;
  el("app-root").hidden = false;
  await reloadAll();
}

function renderSitesSelect() {
  const select = el("site-select");
  const current = state.site;
  select.innerHTML = "";

  const allOpt = document.createElement("option");
  allOpt.value = "ALL";
  allOpt.textContent = `Tất cả site (${state.totals.pending || 0} chờ duyệt)`;
  select.appendChild(allOpt);

  for (const s of state.sitesSummary) {
    const opt = document.createElement("option");
    opt.value = s.site;
    opt.textContent = `${s.site} (${s.pending} chờ / ${s.accepted_total} duyệt)`;
    select.appendChild(opt);
  }
  select.value = state.sitesSummary.some((s) => s.site === current) ? current : "ALL";
  state.site = select.value;

  const dirty = Number(state.totals.dirty_count || 0);
  const pill = el("dirty-pill");
  pill.textContent = `Chưa sync: ${dirty}`;
  pill.classList.toggle("has-dirty", dirty > 0);
}

function applySiteStatsUpdate(updatedSiteStats) {
  if (!updatedSiteStats || !updatedSiteStats.site) return;
  const idx = state.sitesSummary.findIndex((s) => s.site === updatedSiteStats.site);
  if (idx >= 0) state.sitesSummary[idx] = updatedSiteStats;
  else state.sitesSummary.push(updatedSiteStats);
  const totals = { total: 0, pending: 0, accepted_total: 0, accepted_ingested: 0, listing: 0, rejected_total: 0, dirty_count: 0 };
  for (const s of state.sitesSummary) {
    for (const k of Object.keys(totals)) totals[k] += Number(s[k] || 0);
  }
  state.totals = totals;
  renderSitesSelect();
}

async function loadSites() {
  const data = await apiFetch("/api/sites");
  state.sitesSummary = data.sites || [];
  state.totals = data.totals || { pending: 0, dirty_count: 0 };
  renderSitesSelect();
}

async function loadSources({ append = false } = {}) {
  const qs = new URLSearchParams({
    site: state.site,
    status: state.status,
    resource_type: state.resourceType,
    search: state.search,
    sort: state.sort,
    page: String(state.page),
    page_size: String(state.pageSize),
  });
  const data = await apiFetch(`/api/sources?${qs.toString()}`);
  const incoming = data.items || [];
  state.items = append ? [...state.items, ...incoming] : incoming;
  state.totalFiltered = Number(data.total_filtered || 0);
  state.hasMore = Boolean(data.has_more);
  if (!append) state.focusIndex = 0;
  renderCurrentView();
}

async function reloadAll() {
  state.page = 1;
  state.selected.clear();
  await loadSites();
  await loadSources({ append: false });
}

function makeBadge(text, cls = "") {
  const span = document.createElement("span");
  span.className = `badge ${cls}`.trim();
  span.textContent = text;
  return span;
}

function renderFocusView() {
  const item = state.items[state.focusIndex] || null;
  const counter = el("focus-counter");
  const fill = el("focus-progress-fill");
  const undoBtn = el("undo-btn");
  undoBtn.hidden = state.undoStack.length === 0;

  if (!item) {
    el("focus-card").hidden = true;
    el("focus-dock").hidden = true;
    el("focus-empty").hidden = false;
    counter.textContent = `0 / ${state.totalFiltered} nguồn`;
    fill.style.width = "100%";
    return;
  }

  el("focus-empty").hidden = true;
  el("focus-card").hidden = false;
  el("focus-dock").hidden = false;

  const shownIndex = Math.min(state.focusIndex + 1, Math.max(1, state.totalFiltered));
  counter.textContent = `Thẻ ${shownIndex} / ${state.totalFiltered}`;
  const pct = state.totalFiltered ? Math.round((shownIndex / state.totalFiltered) * 100) : 0;
  fill.style.width = `${pct}%`;

  const badges = el("card-badges");
  badges.innerHTML = "";
  badges.appendChild(makeBadge(item.site, "badge-site"));
  const kindLabel = item.is_docx ? "DOCX" : (item.resource_type || "html").toUpperCase();
  badges.appendChild(makeBadge(kindLabel));
  if (item.is_local) badges.appendChild(makeBadge("Offline"));
  if (item.topic_hint) badges.appendChild(makeBadge(item.topic_hint));
  badges.appendChild(makeBadge(STATUS_LABELS[item.decision] || item.decision, `badge-${item.decision}`));
  if (item.dirty) badges.appendChild(makeBadge("Chờ sync về máy", "badge-listing"));

  el("card-title").textContent = item.title || item.link_text || "(Chưa có tiêu đề)";
  const uriNode = el("card-uri");
  uriNode.textContent = item.uri;
  uriNode.href = item.uri;
  el("open-link-btn").href = item.uri;

  const metaLines = [];
  if (item.link_text && item.link_text !== item.title) metaLines.push(`Neo link: ${item.link_text}`);
  if (item.discovered_from) metaLines.push(`Phát hiện từ: ${item.discovered_from}`);
  metaLines.push(`Độ sâu: ${item.crawl_depth} | Ưu tiên: ${item.topic_priority}`);
  if (item.override_note) metaLines.push(`Ghi chú: ${item.override_note}`);
  el("card-meta").innerHTML = "";
  for (const line of metaLines) {
    const div = document.createElement("div");
    div.textContent = line;
    el("card-meta").appendChild(div);
  }

  el("card-note-input").value = item.override_note || "";
  el("note-box").hidden = !state.noteOpen;
  el("preview-box").hidden = true;

  el("act-listing").disabled = state.busy || item.resource_type !== "html";
  el("act-accept").disabled = state.busy || !isAcceptedType(item);
  el("act-reject").disabled = state.busy;
  el("act-skip").disabled = state.busy;
}

function renderListView() {
  const container = el("list-cards");
  container.innerHTML = "";
  el("list-summary").textContent = `${state.totalFiltered} nguồn`;
  const totalPages = Math.max(1, Math.ceil(state.totalFiltered / state.pageSize));
  el("page-indicator").textContent = `Trang ${state.page} / ${totalPages}`;
  el("prev-page-btn").disabled = state.page <= 1;
  el("next-page-btn").disabled = !state.hasMore;

  if (!state.items.length) {
    el("list-empty").hidden = false;
    el("bulk-dock").hidden = true;
    return;
  }
  el("list-empty").hidden = true;

  for (const item of state.items) {
    const card = document.createElement("div");
    card.className = "list-item";
    if (state.selected.has(item.source_id)) card.classList.add("selected");

    const check = document.createElement("input");
    check.type = "checkbox";
    check.className = "list-check";
    check.checked = state.selected.has(item.source_id);

    const body = document.createElement("div");
    body.className = "list-item-body";

    const badgeRow = document.createElement("div");
    badgeRow.className = "badge-row";
    badgeRow.appendChild(makeBadge(item.site, "badge-site"));
    badgeRow.appendChild(makeBadge(item.is_docx ? "DOCX" : (item.resource_type || "html").toUpperCase()));
    if (item.topic_hint) badgeRow.appendChild(makeBadge(item.topic_hint));
    badgeRow.appendChild(makeBadge(STATUS_LABELS[item.decision] || item.decision, `badge-${item.decision}`));

    const title = document.createElement("div");
    title.className = "list-item-title";
    title.textContent = item.title || item.link_text || "(Chưa có tiêu đề)";

    const uri = document.createElement("a");
    uri.className = "list-item-uri";
    uri.href = item.uri;
    uri.target = "_blank";
    uri.rel = "noopener noreferrer";
    uri.textContent = item.uri;

    body.append(badgeRow, title, uri);
    card.append(check, body);

    const toggleSelect = (forceVal) => {
      const next = forceVal !== undefined ? forceVal : !state.selected.has(item.source_id);
      if (next) state.selected.set(item.source_id, item);
      else state.selected.delete(item.source_id);
      check.checked = next;
      card.classList.toggle("selected", next);
      renderBulkDock();
    };

    check.addEventListener("change", () => toggleSelect(check.checked));
    card.addEventListener("click", (evt) => {
      if (evt.target.closest("a, input")) return;
      toggleSelect();
    });
    container.appendChild(card);
  }
  renderBulkDock();
}

function renderBulkDock() {
  const count = state.selected.size;
  const show = state.view === "list" && count > 0;
  el("bulk-dock").hidden = !show;
  el("bulk-count").textContent = `Đã chọn: ${count}`;
}

function renderCurrentView() {
  const isFocus = state.view === "focus";
  el("focus-view").hidden = !isFocus;
  el("list-view").hidden = isFocus;
  if (isFocus) {
    el("bulk-dock").hidden = true;
    renderFocusView();
  } else {
    el("focus-dock").hidden = true;
    renderListView();
  }
}

async function maybePrefetchNextPage() {
  if (state.hasMore && state.focusIndex >= state.items.length - 5) {
    state.page += 1;
    await loadSources({ append: true });
  }
}

async function submitFocusDecision(decision) {
  const item = state.items[state.focusIndex];
  if (!item || state.busy) return;
  const note = state.noteOpen ? el("card-note-input").value.trim() : item.override_note || "";
  state.busy = true;
  renderFocusView();
  try {
    const res = await apiFetch("/api/decision", {
      method: "POST",
      body: JSON.stringify({
        site: item.site,
        source_ids: [item.source_id],
        decision,
        note,
      }),
    });
    state.undoStack.push({
      item: { ...item },
      prevDecision: item.decision,
      prevNote: item.override_note || "",
      removedAt: state.focusIndex,
    });
    if (state.undoStack.length > 15) state.undoStack.shift();

    state.noteOpen = false;
    if (state.status === "all" || state.status === decision || state.status === "dirty") {
      item.decision = decision;
      item.override_note = note;
      item.dirty = 1;
      state.focusIndex += 1;
    } else {
      state.items.splice(state.focusIndex, 1);
      state.totalFiltered = Math.max(0, state.totalFiltered - 1);
    }
    showToast(`${item.site}: đã chuyển sang ${STATUS_LABELS[decision] || decision}`);
    if (res.stats) applySiteStatsUpdate(res.stats);
    else await loadSites();
    await maybePrefetchNextPage();
  } catch (err) {
    showToast(err.message);
  } finally {
    state.busy = false;
    renderCurrentView();
  }
}

async function handleUndo() {
  const last = state.undoStack.pop();
  if (!last || state.busy) return;
  state.busy = true;
  try {
    await apiFetch("/api/decision", {
      method: "POST",
      body: JSON.stringify({
        site: last.item.site,
        source_ids: [last.item.source_id],
        decision: last.prevDecision,
        note: last.prevNote,
      }),
    });
    await reloadAll();
    showToast("Đã hoàn tác quyết định vừa rồi");
  } catch (err) {
    showToast(err.message);
  } finally {
    state.busy = false;
    renderCurrentView();
  }
}

async function handleBulkDecision(decision) {
  if (!state.selected.size || state.busy) return;
  const bySite = new Map();
  for (const item of state.selected.values()) {
    if (!bySite.has(item.site)) bySite.set(item.site, []);
    bySite.get(item.site).push(item.source_id);
  }
  state.busy = true;
  let updatedTotal = 0;
  try {
    for (const [site, ids] of bySite.entries()) {
      const res = await apiFetch("/api/decision", {
        method: "POST",
        body: JSON.stringify({ site, source_ids: ids, decision, note: "" }),
      });
      updatedTotal += Number(res.updated || 0);
    }
    state.selected.clear();
    await reloadAll();
    showToast(`Đã cập nhật ${updatedTotal} nguồn sang ${STATUS_LABELS[decision]}`);
  } catch (err) {
    showToast(err.message);
  } finally {
    state.busy = false;
  }
}

async function handlePreview() {
  const item = state.items[state.focusIndex];
  if (!item) return;
  const box = el("preview-box");
  el("preview-title").textContent = "Đang tải xem nhanh...";
  el("preview-body").textContent = "";
  box.hidden = false;
  try {
    const data = await apiFetch(`/api/preview?url=${encodeURIComponent(item.uri)}`);
    el("preview-title").textContent = data.title || "Nội dung xem nhanh";
    el("preview-body").textContent = data.excerpt || "(Trang không có nội dung văn bản)";
  } catch (err) {
    el("preview-title").textContent = "Không xem nhanh được";
    el("preview-body").textContent = err.message;
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {}
}

function bindEvents() {
  el("auth-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await tryUnlock(el("pin-input").value);
    } catch (err) {
      el("auth-error").textContent = err.message;
      el("auth-error").hidden = false;
    }
  });

  el("logout-btn").addEventListener("click", clearPin);
  el("theme-toggle-btn").addEventListener("click", toggleTheme);
  el("refresh-btn").addEventListener("click", () => reloadAll().catch((e) => showToast(e.message)));

  el("site-select").addEventListener("change", (e) => {
    state.site = e.target.value;
    reloadAll().catch((err) => showToast(err.message));
  });

  for (const btn of document.querySelectorAll(".view-btn")) {
    btn.addEventListener("click", () => {
      state.view = btn.dataset.view;
      document.querySelectorAll(".view-btn").forEach((b) => b.classList.toggle("active", b === btn));
      renderCurrentView();
    });
  }

  for (const pill of document.querySelectorAll(".status-pill")) {
    pill.addEventListener("click", () => {
      state.status = pill.dataset.status;
      document.querySelectorAll(".status-pill").forEach((p) => p.classList.toggle("active", p === pill));
      if (state.status === "dirty") {
        state.site = "ALL";
        el("site-select").value = "ALL";
      }
      state.page = 1;
      state.selected.clear();
      loadSources({ append: false }).catch((err) => showToast(err.message));
    });
  }

  let searchTimer = null;
  el("search-input").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.search = e.target.value.trim();
      state.page = 1;
      loadSources({ append: false }).catch((err) => showToast(err.message));
    }, 260);
  });

  el("type-select").addEventListener("change", (e) => {
    state.resourceType = e.target.value;
    state.page = 1;
    loadSources({ append: false }).catch((err) => showToast(err.message));
  });

  el("sort-select").addEventListener("change", (e) => {
    state.sort = e.target.value;
    state.page = 1;
    loadSources({ append: false }).catch((err) => showToast(err.message));
  });

  el("act-accept").addEventListener("click", () => submitFocusDecision("accepted"));
  el("act-listing").addEventListener("click", () => submitFocusDecision("listing"));
  el("act-reject").addEventListener("click", () => submitFocusDecision("rejected"));
  el("act-skip").addEventListener("click", async () => {
    if (state.focusIndex < state.items.length - 1) {
      state.focusIndex += 1;
      renderFocusView();
      await maybePrefetchNextPage();
    } else if (state.hasMore) {
      state.page += 1;
      await loadSources({ append: true });
      if (state.focusIndex < state.items.length - 1) {
        state.focusIndex += 1;
        renderFocusView();
      }
    } else {
      showToast("Đã tới cuối danh sách đang tải");
    }
  });

  el("undo-btn").addEventListener("click", handleUndo);
  el("note-toggle-btn").addEventListener("click", () => {
    state.noteOpen = !state.noteOpen;
    el("note-box").hidden = !state.noteOpen;
  });
  el("preview-btn").addEventListener("click", handlePreview);
  el("preview-close-btn").addEventListener("click", () => {
    el("preview-box").hidden = true;
  });
  el("focus-reload-btn").addEventListener("click", () => reloadAll().catch((e) => showToast(e.message)));

  el("list-select-page-btn").addEventListener("click", () => {
    for (const item of state.items) state.selected.set(item.source_id, item);
    renderListView();
  });
  el("prev-page-btn").addEventListener("click", () => {
    if (state.page > 1) {
      state.page -= 1;
      loadSources({ append: false }).catch((e) => showToast(e.message));
    }
  });
  el("next-page-btn").addEventListener("click", () => {
    if (state.hasMore) {
      state.page += 1;
      loadSources({ append: false }).catch((e) => showToast(e.message));
    }
  });

  el("bulk-accept").addEventListener("click", () => handleBulkDecision("accepted"));
  el("bulk-listing").addEventListener("click", () => handleBulkDecision("listing"));
  el("bulk-reject").addEventListener("click", () => handleBulkDecision("rejected"));
  el("bulk-clear").addEventListener("click", () => {
    state.selected.clear();
    renderListView();
  });
}

async function init() {
  bindEvents();
  let savedPin = "";
  try {
    savedPin = localStorage.getItem(PIN_KEY) || "";
  } catch {}
  if (savedPin) {
    try {
      await tryUnlock(savedPin);
      return;
    } catch {
      clearPin();
    }
  }
}

init();

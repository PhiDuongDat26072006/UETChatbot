/**
 * Cloudflare Worker API + Static Assets router for UET Ingestion Mobile Review.
 */

const VALID_DECISIONS = new Set(["accepted", "listing", "pending", "rejected"]);
const ALLOWED_PREVIEW_HOSTS = new Set([
  "uet.edu.vn",
  "uet.vnu.edu.vn",
  "cdnportal.vnu.edu.vn",
  "2489013871.e.cdneverest.net",
  "2679558515.e.cdneverest.net",
]);
const BINARY_PREVIEW_EXTS = [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt"];
const MAX_BATCH = 200;
const SITES_CACHE_TTL_MS = 15000;
let sitesCache = { data: null, expiresAt: 0 };

function invalidateSitesCache() {
  sitesCache = { data: null, expiresAt: 0 };
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

function safeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  const enc = new TextEncoder();
  const ba = enc.encode(a);
  const bb = enc.encode(b);
  const maxLen = Math.max(ba.length, bb.length);
  let diff = ba.length ^ bb.length;
  for (let i = 0; i < maxLen; i += 1) {
    diff |= (ba[i] ?? 0) ^ (bb[i] ?? 0);
  }
  return diff === 0;
}

function requireSecret(request, expected, headerName, label) {
  if (!expected) {
    return jsonResponse({ error: `Chưa cấu hình biến bí mật ${label} trên Worker` }, 503);
  }
  const provided = request.headers.get(headerName) || "";
  if (!provided || !safeEqual(provided, String(expected))) {
    return jsonResponse({ error: `Xác thực ${label} không hợp lệ` }, 401);
  }
  return null;
}

function cleanPathname(uri) {
  try {
    return decodeURIComponent(new URL(String(uri || "")).pathname).toLowerCase();
  } catch {
    return String(uri || "").split(/[?#]/, 1)[0].toLowerCase();
  }
}

function isDocxUri(uri) {
  return cleanPathname(uri).endsWith(".docx");
}

function isAcceptedType(resourceType, uri) {
  return resourceType === "html" || resourceType === "pdf" || isDocxUri(uri);
}

export function isAllowedPreviewUrl(rawUrl) {
  try {
    const parsed = new URL(String(rawUrl || "").trim());
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return false;
    if (parsed.username || parsed.password) return false;
    const host = parsed.hostname.toLowerCase().replace(/\.+$/, "");
    if (!host) return false;
    if (ALLOWED_PREVIEW_HOSTS.has(host)) return true;
    return host.endsWith(".uet.edu.vn") || host.endsWith(".uet.vnu.edu.vn");
  } catch {
    return false;
  }
}

function escapeLike(term) {
  return term.replace(/\\/g, "\\\\").replace(/%/g, "\\%").replace(/_/g, "\\_");
}

function sortSiteRows(rows) {
  return rows.sort((a, b) => {
    if (a.site === "UET" && b.site !== "UET") return -1;
    if (b.site === "UET" && a.site !== "UET") return 1;
    if (a.site === "OTHER" && b.site !== "OTHER") return 1;
    if (b.site === "OTHER" && a.site !== "OTHER") return -1;
    return a.site.localeCompare(b.site);
  });
}

function emptyTotals() {
  return {
    total: 0,
    pending: 0,
    accepted_total: 0,
    accepted_ingested: 0,
    listing: 0,
    rejected_total: 0,
    dirty_count: 0,
  };
}

async function querySitesSummary(db, onlySite = null) {
  if (!onlySite && sitesCache.data && Date.now() < sitesCache.expiresAt) {
    return sitesCache.data;
  }
  const statsSql = onlySite
    ? `SELECT site, total, pending, accepted_total, accepted_ingested, listing, rejected_total, dirty_count
       FROM site_stats WHERE site = ?`
    : `SELECT site, total, pending, accepted_total, accepted_ingested, listing, rejected_total, dirty_count
       FROM site_stats`;
  let results = [];
  try {
    const stmt = onlySite ? db.prepare(statsSql).bind(onlySite) : db.prepare(statsSql);
    ({ results } = await stmt.all());
  } catch {
    results = [];
  }
  if (!results || !results.length) {
    const whereClause = onlySite ? "WHERE site = ? " : "";
    const fallbackSql = `SELECT site,
         COUNT(*) AS total,
         SUM(CASE WHEN decision = 'pending' THEN 1 ELSE 0 END) AS pending,
         SUM(CASE WHEN decision = 'accepted' THEN 1 ELSE 0 END) AS accepted_total,
         SUM(CASE WHEN decision = 'accepted' AND ingested = 1 THEN 1 ELSE 0 END) AS accepted_ingested,
         SUM(CASE WHEN decision = 'listing' THEN 1 ELSE 0 END) AS listing,
         SUM(CASE WHEN decision = 'rejected' THEN 1 ELSE 0 END) AS rejected_total,
         SUM(CASE WHEN dirty = 1 THEN 1 ELSE 0 END) AS dirty_count
       FROM sources ${whereClause}GROUP BY site`;
    const fbStmt = onlySite ? db.prepare(fallbackSql).bind(onlySite) : db.prepare(fallbackSql);
    ({ results } = await fbStmt.all());
  }
  const sites = sortSiteRows(
    (results || []).map((r) => ({
      site: r.site,
      total: Number(r.total || 0),
      pending: Number(r.pending || 0),
      accepted_total: Number(r.accepted_total || 0),
      accepted_ingested: Number(r.accepted_ingested || 0),
      listing: Number(r.listing || 0),
      rejected_total: Number(r.rejected_total || 0),
      dirty_count: Number(r.dirty_count || 0),
    }))
  );
  const totals = emptyTotals();
  for (const s of sites) {
    for (const k of Object.keys(totals)) totals[k] += s[k];
  }
  const payload = { sites, totals };
  if (!onlySite && sites.length) {
    sitesCache = { data: payload, expiresAt: Date.now() + SITES_CACHE_TTL_MS };
  }
  return payload;
}

function countFromStatsRow(row, status) {
  if (!row) return 0;
  if (status === "pending") return row.pending;
  if (status === "accepted") return row.accepted_total;
  if (status === "listing") return row.listing;
  if (status === "rejected") return row.rejected_total;
  if (status === "dirty") return row.dirty_count;
  return row.total;
}

async function handleListSources(db, url) {
  const site = (url.searchParams.get("site") || "ALL").trim().toUpperCase();
  const status = (url.searchParams.get("status") || "pending").trim().toLowerCase();
  const resourceType = (url.searchParams.get("resource_type") || "all").trim().toLowerCase();
  const search = (url.searchParams.get("search") || "").trim().toLowerCase();
  const sort = (url.searchParams.get("sort") || "priority").trim().toLowerCase();
  const page = Math.max(1, parseInt(url.searchParams.get("page") || "1", 10) || 1);
  const pageSize = Math.min(100, Math.max(1, parseInt(url.searchParams.get("page_size") || "30", 10) || 30));

  const clauses = [];
  const params = [];
  if (site && site !== "ALL") {
    clauses.push("site = ?");
    params.push(site);
  }
  if (status === "pending") clauses.push("decision = 'pending'");
  else if (status === "accepted") clauses.push("decision = 'accepted'");
  else if (status === "listing") clauses.push("decision = 'listing'");
  else if (status === "rejected") clauses.push("decision = 'rejected'");
  else if (status === "dirty") clauses.push("dirty = 1");

  if (resourceType === "html" || resourceType === "pdf" || resourceType === "office") {
    clauses.push("resource_type = ?");
    params.push(resourceType);
  }
  if (search) {
    const pattern = `%${escapeLike(search)}%`;
    clauses.push(
      "(LOWER(uri) LIKE ? ESCAPE '\\' OR LOWER(title) LIKE ? ESCAPE '\\' OR LOWER(link_text) LIKE ? ESCAPE '\\' OR LOWER(discovered_from) LIKE ? ESCAPE '\\' OR LOWER(topic_hint) LIKE ? ESCAPE '\\' OR LOWER(override_note) LIKE ? ESCAPE '\\')"
    );
    for (let i = 0; i < 6; i += 1) params.push(pattern);
  }

  const whereSql = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
  let orderSql = "ORDER BY topic_priority DESC, crawl_depth ASC, uri ASC";
  if (sort === "newest") orderSql = "ORDER BY updated_at DESC, topic_priority DESC, uri ASC";
  else if (sort === "depth") orderSql = "ORDER BY crawl_depth ASC, topic_priority DESC, uri ASC";
  else if (sort === "url") orderSql = "ORDER BY LOWER(uri) ASC, source_id ASC";

  let totalFiltered = null;
  if (!search && resourceType === "all" && ["pending", "accepted", "listing", "rejected", "dirty", "all"].includes(status)) {
    const summary = await querySitesSummary(db);
    if (summary.sites.length) {
      const targetRow = site === "ALL" ? summary.totals : summary.sites.find((s) => s.site === site);
      totalFiltered = countFromStatsRow(targetRow, status);
    }
  }
  if (totalFiltered === null) {
    const countRow = await db.prepare(`SELECT COUNT(*) AS cnt FROM sources ${whereSql}`).bind(...params).first();
    totalFiltered = Number(countRow?.cnt || 0);
  }
  const offset = (page - 1) * pageSize;

  const { results } = await db
    .prepare(
      `SELECT source_id, site, uri, title, link_text, discovered_from, resource_type, crawl_depth,
              topic_hint, topic_priority, decision, decision_reason, ingested, override_note, updated_at, dirty
       FROM sources ${whereSql} ${orderSql} LIMIT ? OFFSET ?`
    )
    .bind(...params, pageSize, offset)
    .all();

  const items = (results || []).map((r) => ({
    ...r,
    crawl_depth: Number(r.crawl_depth || 0),
    topic_priority: Number(r.topic_priority || 0),
    ingested: Number(r.ingested || 0),
    dirty: Number(r.dirty || 0),
    is_docx: isDocxUri(r.uri),
    is_local: cleanPathname(r.uri).includes("/offline/"),
  }));

  return jsonResponse({
    site,
    items,
    total_filtered: totalFiltered,
    page,
    page_size: pageSize,
    has_more: offset + items.length < totalFiltered,
  });
}

async function fetchRowsByIds(db, site, sourceIds) {
  const rows = [];
  for (let i = 0; i < sourceIds.length; i += 50) {
    const chunk = sourceIds.slice(i, i + 50);
    const placeholders = chunk.map(() => "?").join(",");
    const { results } = await db
      .prepare(`SELECT source_id, site, uri, resource_type, decision, dirty FROM sources WHERE site = ? AND source_id IN (${placeholders})`)
      .bind(site, ...chunk)
      .all();
    rows.push(...(results || []));
  }
  return rows;
}

async function handleDecision(db, request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Payload JSON không hợp lệ" }, 400);
  }
  const site = String(body?.site || "").trim().toUpperCase();
  const decision = String(body?.decision || "").trim().toLowerCase();
  const note = String(body?.note || "");
  const sourceIds = Array.isArray(body?.source_ids)
    ? [...new Set(body.source_ids.map((s) => String(s || "").trim()).filter(Boolean))]
    : [];

  if (!site) return jsonResponse({ error: "Thiếu mã site" }, 400);
  if (!VALID_DECISIONS.has(decision)) return jsonResponse({ error: `Quyết định không hợp lệ: ${decision}` }, 400);
  if (!sourceIds.length || sourceIds.length > MAX_BATCH) {
    return jsonResponse({ error: `Số lượng source_ids phải từ 1 đến ${MAX_BATCH}` }, 400);
  }

  const foundRows = await fetchRowsByIds(db, site, sourceIds);
  for (const row of foundRows) {
    if (decision === "listing" && row.resource_type !== "html") {
      return jsonResponse({ error: `Chỉ URL HTML mới có thể đặt làm listing: ${row.uri}` }, 400);
    }
    if (decision === "accepted" && !isAcceptedType(row.resource_type, row.uri)) {
      return jsonResponse({ error: `Chỉ URL HTML, PDF hoặc DOCX mới được duyệt: ${row.uri}` }, 400);
    }
  }

  const timestamp = new Date().toISOString();
  if (foundRows.length) {
    const statusCol = { pending: "pending", accepted: "accepted_total", listing: "listing", rejected: "rejected_total" };
    let dPending = 0, dAccepted = 0, dListing = 0, dRejected = 0, dDirty = 0;
    const addColDelta = (col, delta) => {
      if (col === "pending") dPending += delta;
      else if (col === "accepted_total") dAccepted += delta;
      else if (col === "listing") dListing += delta;
      else if (col === "rejected_total") dRejected += delta;
    };
    const newCol = statusCol[decision];
    for (const row of foundRows) {
      const oldDec = statusCol[row.decision] ? row.decision : "pending";
      const oldCol = statusCol[oldDec];
      if (oldCol !== newCol) {
        addColDelta(oldCol, -1);
        addColDelta(newCol, 1);
      }
      if (!Number(row.dirty)) dDirty += 1;
    }

    const stmts = foundRows.map((row) =>
      db
        .prepare(
          `UPDATE sources
           SET decision = ?, decision_reason = 'manual_curation', override_note = ?, updated_at = ?, dirty = 1
           WHERE site = ? AND source_id = ?`
        )
        .bind(decision, note, timestamp, site, row.source_id)
    );
    stmts.push(
      db
        .prepare(
          `UPDATE site_stats
           SET pending = MAX(0, pending + ?),
               accepted_total = MAX(0, accepted_total + ?),
               listing = MAX(0, listing + ?),
               rejected_total = MAX(0, rejected_total + ?),
               dirty_count = MAX(0, dirty_count + ?)
           WHERE site = ?`
        )
        .bind(dPending, dAccepted, dListing, dRejected, dDirty, site)
    );
    await db.batch(stmts);
    invalidateSitesCache();
  }

  const summary = await querySitesSummary(db, site);
  return jsonResponse({
    updated: foundRows.length,
    site,
    decision,
    updated_at: timestamp,
    stats: summary.sites[0] || null,
  });
}

function decodeHtmlEntities(text) {
  return text
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#0*39;|&apos;/gi, "'");
}

async function handlePreview(url) {
  const target = (url.searchParams.get("url") || "").trim();
  if (!target) return jsonResponse({ error: "Thiếu tham số url" }, 400);
  if (!isAllowedPreviewUrl(target)) {
    return jsonResponse({ error: "Tên miền URL không nằm trong danh sách cho phép xem nhanh" }, 403);
  }
  const pathLower = cleanPathname(target);
  if (pathLower.includes("/offline/")) {
    const baseName = pathLower.split("/").pop() || target;
    return jsonResponse({
      ok: true,
      is_offline: true,
      title: baseName,
      excerpt: "Tài liệu offline nằm trong thư mục data/offline_docs/ trên máy chủ cục bộ.",
    });
  }
  const matchedExt = BINARY_PREVIEW_EXTS.find((ext) => pathLower.endsWith(ext));
  if (matchedExt) {
    return jsonResponse({
      ok: true,
      is_binary: true,
      resource_type: matchedExt === ".pdf" ? "pdf" : "office",
      title: pathLower.split("/").pop() || target,
      excerpt: "Tài liệu nhị phân (PDF/Office) — bấm 'Mở link gốc' để xem trực tiếp trên trình duyệt điện thoại.",
    });
  }
  try {
    const resp = await fetch(target, {
      redirect: "follow",
      signal: AbortSignal.timeout(8000),
      headers: { "User-Agent": "UETChatbot-ReviewPreview/1.0" },
    });
    const finalUrl = resp.url || target;
    if (!isAllowedPreviewUrl(finalUrl)) {
      return jsonResponse({ error: "URL chuyển hướng ra ngoài phạm vi tên miền UET" }, 403);
    }
    const contentType = (resp.headers.get("content-type") || "").toLowerCase();
    if (!contentType.includes("html") && !contentType.includes("text")) {
      return jsonResponse({
        ok: true,
        status: resp.status,
        final_url: finalUrl,
        content_type: contentType,
        title: "",
        excerpt: `Nội dung kiểu ${contentType || "binary"} — vui lòng mở link gốc để xem.`,
      });
    }
    const rawHtml = (await resp.text()).slice(0, 120000);
    const titleMatch = rawHtml.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
    const title = decodeHtmlEntities((titleMatch?.[1] || "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim());
    const bodyMatch = rawHtml.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
    const stripped = (bodyMatch ? bodyMatch[1] : rawHtml)
      .replace(/<(script|style|noscript|svg|nav|header|footer)[^>]*>[\s\S]*?<\/\1>/gi, " ")
      .replace(/<[^>]+>/g, " ");
    const excerpt = decodeHtmlEntities(stripped).replace(/\s+/g, " ").trim().slice(0, 900);
    return jsonResponse({
      ok: true,
      status: resp.status,
      final_url: finalUrl,
      content_type: contentType,
      title,
      excerpt,
    });
  } catch (err) {
    return jsonResponse({ error: `Không tải được trang xem nhanh: ${err?.message || "timeout"}` }, 502);
  }
}

async function handleSyncPush(db, request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Payload JSON không hợp lệ" }, 400);
  }
  const items = body?.items;
  const force = Boolean(body?.force);
  if (!Array.isArray(items) || items.length > MAX_BATCH) {
    return jsonResponse({ error: `items phải là mảng tối đa ${MAX_BATCH} phần tử` }, 400);
  }
  if (!items.length) return jsonResponse({ received: 0, changed: 0 });

  const sql = force
    ? `INSERT INTO sources (
         source_id, site, uri, title, link_text, discovered_from, resource_type,
         crawl_depth, topic_hint, topic_priority, decision, decision_reason,
         ingested, override_note, updated_at, dirty
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
       ON CONFLICT(source_id) DO UPDATE SET
         site = excluded.site, uri = excluded.uri, title = excluded.title,
         link_text = excluded.link_text, discovered_from = excluded.discovered_from,
         resource_type = excluded.resource_type, crawl_depth = excluded.crawl_depth,
         topic_hint = excluded.topic_hint, topic_priority = excluded.topic_priority,
         decision = excluded.decision, decision_reason = excluded.decision_reason,
         ingested = excluded.ingested, override_note = excluded.override_note,
         updated_at = excluded.updated_at, dirty = 0`
    : `INSERT INTO sources (
         source_id, site, uri, title, link_text, discovered_from, resource_type,
         crawl_depth, topic_hint, topic_priority, decision, decision_reason,
         ingested, override_note, updated_at, dirty
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
       ON CONFLICT(source_id) DO UPDATE SET
         site = excluded.site, uri = excluded.uri, title = excluded.title,
         link_text = excluded.link_text, discovered_from = excluded.discovered_from,
         resource_type = excluded.resource_type, crawl_depth = excluded.crawl_depth,
         topic_hint = excluded.topic_hint, topic_priority = excluded.topic_priority,
         decision = excluded.decision, decision_reason = excluded.decision_reason,
         ingested = excluded.ingested, override_note = excluded.override_note,
         updated_at = excluded.updated_at
       WHERE sources.dirty = 0 AND (
         sources.site != excluded.site OR sources.uri != excluded.uri OR
         sources.title != excluded.title OR sources.link_text != excluded.link_text OR
         sources.discovered_from != excluded.discovered_from OR sources.resource_type != excluded.resource_type OR
         sources.crawl_depth != excluded.crawl_depth OR sources.topic_hint != excluded.topic_hint OR
         sources.topic_priority != excluded.topic_priority OR sources.decision != excluded.decision OR
         sources.decision_reason != excluded.decision_reason OR sources.ingested != excluded.ingested OR
         sources.override_note != excluded.override_note
       )`;

  const stmts = items.map((item) => {
    const sid = String(item?.source_id || "").trim();
    const site = String(item?.site || "").trim().toUpperCase();
    const uri = String(item?.uri || "").trim();
    if (!sid || !site || !uri) {
      throw new Error("Mỗi item push phải có đủ source_id, site, uri");
    }
    const dec = String(item?.decision || "pending").trim().toLowerCase() || "pending";
    return db.prepare(sql).bind(
      sid,
      site,
      uri,
      String(item?.title || ""),
      String(item?.link_text || ""),
      String(item?.discovered_from || ""),
      String(item?.resource_type || "html"),
      Number(item?.crawl_depth || 0),
      String(item?.topic_hint || ""),
      Number(item?.topic_priority || 0),
      dec,
      String(item?.decision_reason || ""),
      item?.ingested ? 1 : 0,
      String(item?.override_note || ""),
      String(item?.updated_at || "")
    );
  });

  const results = await db.batch(stmts);
  let changed = 0;
  for (const res of results) changed += Number(res?.meta?.changes || 0);
  if (changed > 0) invalidateSitesCache();
  return jsonResponse({ received: items.length, changed });
}

async function handleSyncStats(db, request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Payload JSON không hợp lệ" }, 400);
  }
  const sites = body?.sites;
  if (!Array.isArray(sites) || sites.length > 50) {
    return jsonResponse({ error: "sites phải là mảng tối đa 50 phần tử" }, 400);
  }
  if (!sites.length) return jsonResponse({ updated: 0 });

  const sql = `INSERT INTO site_stats (
      site, total, pending, accepted_total, accepted_ingested, listing, rejected_total, dirty_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(site) DO UPDATE SET
      total = excluded.total,
      pending = excluded.pending,
      accepted_total = excluded.accepted_total,
      accepted_ingested = excluded.accepted_ingested,
      listing = excluded.listing,
      rejected_total = excluded.rejected_total,
      dirty_count = excluded.dirty_count`;
  const stmts = sites.map((s) =>
    db
      .prepare(sql)
      .bind(
        String(s?.site || "").trim().toUpperCase(),
        Number(s?.total || 0),
        Number(s?.pending || 0),
        Number(s?.accepted_total || 0),
        Number(s?.accepted_ingested || 0),
        Number(s?.listing || 0),
        Number(s?.rejected_total || 0),
        Number(s?.dirty_count || 0)
      )
  );
  await db.batch(stmts);
  invalidateSitesCache();
  return jsonResponse({ updated: sites.length });
}

async function handleSyncPull(db) {
  const { results } = await db
    .prepare(
      `SELECT source_id, site, uri, decision, override_note, updated_at
       FROM sources WHERE dirty = 1 ORDER BY updated_at ASC, site ASC`
    )
    .all();
  const items = results || [];
  return jsonResponse({ items, total: items.length });
}

async function handleSyncAck(db, request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Payload JSON không hợp lệ" }, 400);
  }
  const items = body?.items;
  if (!Array.isArray(items) || items.length > MAX_BATCH) {
    return jsonResponse({ error: `items phải là mảng tối đa ${MAX_BATCH} phần tử` }, 400);
  }
  if (!items.length) return jsonResponse({ requested: 0, acked: 0 });

  const stmts = items.map((item) =>
    db
      .prepare("UPDATE sources SET dirty = 0 WHERE source_id = ? AND updated_at = ?")
      .bind(String(item?.source_id || "").trim(), String(item?.updated_at || ""))
  );
  const results = await db.batch(stmts);
  let acked = 0;
  for (const res of results) acked += Number(res?.meta?.changes || 0);
  if (acked > 0) {
    try {
      const { results: dirtyRows } = await db
        .prepare("SELECT site, COUNT(*) AS cnt FROM sources WHERE dirty = 1 GROUP BY site")
        .all();
      const dirtyMap = new Map((dirtyRows || []).map((r) => [r.site, Number(r.cnt || 0)]));
      const { results: statSites } = await db.prepare("SELECT site FROM site_stats").all();
      if (statSites?.length) {
        await db.batch(
          statSites.map((r) =>
            db.prepare("UPDATE site_stats SET dirty_count = ? WHERE site = ?").bind(dirtyMap.get(r.site) || 0, r.site)
          )
        );
      }
    } catch {}
    invalidateSitesCache();
  }
  return jsonResponse({ requested: items.length, acked });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { pathname } = url;

    if (!pathname.startsWith("/api/")) {
      if (env.ASSETS) return env.ASSETS.fetch(request);
      return new Response("Not Found", { status: 404 });
    }

    const db = env.DB || env.uet_ingestion_review_db;

    if (pathname.startsWith("/api/sync/")) {
      const authErr = requireSecret(request, env.SYNC_SECRET, "X-Sync-Secret", "SYNC_SECRET");
      if (authErr) return authErr;
      if (!db) {
        return jsonResponse({ error: "Chưa cấu hình D1 database binding (DB) trên Worker" }, 503);
      }
      try {
        if (pathname === "/api/sync/push" && request.method === "POST") return await handleSyncPush(db, request);
        if (pathname === "/api/sync/pull" && request.method === "GET") return await handleSyncPull(db);
        if (pathname === "/api/sync/ack" && request.method === "POST") return await handleSyncAck(db, request);
        if (pathname === "/api/sync/stats" && request.method === "POST") return await handleSyncStats(db, request);
        return jsonResponse({ error: "Endpoint sync không tồn tại" }, 404);
      } catch (err) {
        return jsonResponse({ error: err?.message || "Lỗi xử lý sync" }, 400);
      }
    }

    const pinErr = requireSecret(request, env.REVIEW_PIN, "X-Review-Pin", "REVIEW_PIN");
    if (pinErr) return pinErr;

    if (!db) {
      return jsonResponse({ error: "Chưa cấu hình D1 database binding (DB) trên Worker" }, 503);
    }

    try {
      if (pathname === "/api/auth/check" && request.method === "GET") return jsonResponse({ ok: true });
      if (pathname === "/api/sites" && request.method === "GET") return jsonResponse(await querySitesSummary(db));
      if (pathname === "/api/sources" && request.method === "GET") return await handleListSources(db, url);
      if (pathname === "/api/decision" && request.method === "POST") return await handleDecision(db, request);
      if (pathname === "/api/preview" && request.method === "GET") return await handlePreview(url);
      return jsonResponse({ error: "API endpoint không tồn tại" }, 404);
    } catch (err) {
      return jsonResponse({ error: err?.message || "Lỗi máy chủ Worker" }, 500);
    }
  },
};

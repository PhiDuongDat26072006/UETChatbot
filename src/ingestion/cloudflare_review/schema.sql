-- Cloudflare D1 schema for UET Ingestion Mobile Review

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    site TEXT NOT NULL,
    uri TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    link_text TEXT NOT NULL DEFAULT '',
    discovered_from TEXT NOT NULL DEFAULT '',
    resource_type TEXT NOT NULL DEFAULT 'html',
    crawl_depth INTEGER NOT NULL DEFAULT 0,
    topic_hint TEXT NOT NULL DEFAULT '',
    topic_priority INTEGER NOT NULL DEFAULT 0,
    decision TEXT NOT NULL DEFAULT 'pending',
    decision_reason TEXT NOT NULL DEFAULT '',
    ingested INTEGER NOT NULL DEFAULT 0,
    override_note TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    dirty INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sources_site_dec_prio
    ON sources (site, decision, topic_priority DESC, crawl_depth ASC, uri ASC);

CREATE INDEX IF NOT EXISTS idx_sources_dec_prio
    ON sources (decision, topic_priority DESC, crawl_depth ASC, uri ASC);

CREATE INDEX IF NOT EXISTS idx_sources_dirty
    ON sources (dirty, site);

CREATE INDEX IF NOT EXISTS idx_sources_dirty_prio
    ON sources (dirty, topic_priority DESC, crawl_depth ASC, uri ASC);

CREATE TABLE IF NOT EXISTS site_stats (
    site TEXT PRIMARY KEY,
    total INTEGER NOT NULL DEFAULT 0,
    pending INTEGER NOT NULL DEFAULT 0,
    accepted_total INTEGER NOT NULL DEFAULT 0,
    accepted_ingested INTEGER NOT NULL DEFAULT 0,
    listing INTEGER NOT NULL DEFAULT 0,
    rejected_total INTEGER NOT NULL DEFAULT 0,
    dirty_count INTEGER NOT NULL DEFAULT 0
);

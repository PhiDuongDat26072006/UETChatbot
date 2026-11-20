"""Telegram Bot workflows for daily UET category discovery and two-way source curation."""

from __future__ import annotations

from html import escape
import logging
import time
from urllib.parse import urlsplit
import requests
from src.base import DataSource
from .config.profiles import load_profiles
from .crawler.parser import parse_page
from .crawler.policy import in_discovery_scope, resource_type
from .curation import (
    get_all_sites_summary, list_site_sources, run_incremental_ingest, set_sources_decision,
)
from .curation_helpers import load_corpus_policies, topic_info
from .curation_yaml import ordered_site_names
from .downloader.http import HTMLFetcher
from .identity import normalize_site_url, source_id
from .storage import Storage, now, writer_lock
from .telegram_client import (
    TelegramClient, _clip, _format_source_card, _format_status_html,
    _parse_callback, load_telegram_config, save_telegram_config,
)
from .utils import completion_map

log = logging.getLogger(__name__)
__all__ = [
    "TelegramClient", "load_telegram_config", "save_telegram_config",
    "setup_telegram_bot", "send_test_status", "run_daily_category_watch",
    "send_watch_notification", "process_telegram_update", "run_telegram_polling",
]


def setup_telegram_bot(data_dir, token=None, chat_id=None, wait_seconds: int = 60, profile_dir=None) -> dict:
    cfg = load_telegram_config(data_dir, token, chat_id)
    client = TelegramClient(cfg["bot_token"])
    me = client.get_me()
    bot_username = str(me.get("username") or "")
    target_chat, owner_name = cfg["chat_id"], cfg["owner_name"]
    if not target_chat:
        log.info("Mo Telegram, tim @%s va bam Start (/start) trong %ss...", bot_username, wait_seconds)
        deadline, offset = time.monotonic() + max(5, wait_seconds), None
        while time.monotonic() < deadline and not target_chat:
            updates = client.get_updates(offset=offset, timeout=min(10, max(1, int(deadline - time.monotonic()))))
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or (upd.get("callback_query") or {}).get("message")
                if msg and msg.get("chat"):
                    target_chat = str(msg["chat"]["id"])
                    user = (upd.get("message") or upd.get("callback_query") or {}).get("from") or {}
                    owner_name = " ".join(x for x in (user.get("first_name", ""), user.get("last_name", "")) if x).strip()
                    break
        if not target_chat:
            raise TimeoutError(f"Chua nhan duoc tin nhan /start toi @{bot_username} trong {wait_seconds}s")
    config_file = save_telegram_config(data_dir, cfg["bot_token"], target_chat, bot_username, owner_name)
    client.send_message(
        target_chat,
        f"<b>Ket noi thanh cong voi @{escape(bot_username)}</b>\nDung /status, /pending, /watch, /ingest.\n\n"
        + _format_status_html(get_all_sites_summary(data_dir, profile_dir)),
    )
    return {"status": "ok", "bot_username": bot_username, "chat_id": target_chat, "owner_name": owner_name, "config_file": str(config_file)}


def send_test_status(data_dir, token=None, chat_id=None, profile_dir=None) -> dict:
    cfg = load_telegram_config(data_dir, token, chat_id)
    if not cfg["chat_id"]:
        raise ValueError("Missing chat_id; run telegram --setup first")
    res = TelegramClient(cfg["bot_token"]).send_message(cfg["chat_id"], _format_status_html(get_all_sites_summary(data_dir, profile_dir)))
    return {"status": "sent", "chat_id": cfg["chat_id"], "message_id": res.get("message_id")}


def run_daily_category_watch(data_dir, sites=None, profile_dir=None, max_entry_urls_per_site: int = 25, timeout: int = 15) -> dict:
    profiles = load_profiles(profile_dir)
    selected = [s for s in ordered_site_names(profiles) if (not sites or s in sites) and s != "OTHER"]
    storage = Storage(data_dir)
    policies = load_corpus_policies(profiles)
    entry_jobs = []
    for site in selected:
        profile = profiles[site]
        sources = storage.records("sources", site)
        raw_entries = list(profile.get("discovery", {}).get("start_urls", [])) + [
            r["uri"] for r in sources.values() if r.get("metadata", {}).get("decision") == "listing"
        ]
        seen_entry, site_entries = set(), []
        for raw_u in raw_entries:
            try:
                norm_u = normalize_site_url(raw_u, profile)
            except ValueError:
                continue
            if norm_u in seen_entry or resource_type(norm_u) != "html" or urlsplit(norm_u).path.startswith("/offline/"):
                continue
            seen_entry.add(norm_u)
            site_entries.append((site, norm_u))
            if len(site_entries) >= max_entry_urls_per_site:
                break
        entry_jobs.extend(site_entries)

    fetcher = HTMLFetcher(timeout=timeout, retries=1, delay=0.1)
    fetched_pages, errors = [], []

    def _fetch_one(job):
        site_name, entry_url = job
        prof = profiles[site_name]
        try:
            page = fetcher.fetch(entry_url, lambda u: in_discovery_scope(u, prof) and resource_type(u) == "html")
            return page, None
        except Exception as exc:
            return None, str(exc)

    try:
        if entry_jobs:
            for (site_name, entry_url), (page, err) in completion_map(_fetch_one, entry_jobs, min(12, len(entry_jobs))):
                if err:
                    errors.append({"site": site_name, "url": entry_url, "error": err})
                    continue
                if resource_type(page["final_url"]) != "html" or page.get("encoding") == "binary":
                    continue
                html = page["body"].decode(page.get("encoding") or "utf-8", errors="replace")
                _, links = parse_page(html, page["final_url"], profiles[site_name])
                fetched_pages.append((site_name, page["final_url"], links))
    finally:
        fetcher.close()

    by_site: dict[str, list[dict]] = {}
    new_pending_total = 0
    with writer_lock(storage.root):
        storage = Storage(data_dir)
        discovered_ts = now()
        known_by_site = {s: set(storage.records("sources", s).keys()) for s in selected}
        with storage.batched():
            for site_name, parent_url, links in fetched_pages:
                prof = profiles[site_name]
                known = known_by_site[site_name]
                corp = policies.get(site_name)
                for link in links:
                    target = link["url"]
                    upper_t = target.upper()
                    if any(ch in target for ch in ('"', "'", "\u201d", "\u2019", "<", ">")) or any(
                        tok in upper_t for tok in ("%E2%80%9D", "%E2%80%99", "%22", "%3C", "%3E")
                    ):
                        continue
                    kind = resource_type(target)
                    if kind == "asset":
                        continue
                    sid = source_id(target)
                    if sid in known or not in_discovery_scope(target, prof):
                        continue
                    known.add(sid)
                    src = DataSource(
                        source_id=sid, source_type="web", uri=target,
                        metadata={
                            "site": site_name, "faculty": site_name, "resource_type": kind,
                            "discovered_from": parent_url, "link_text": link.get("link_text", ""),
                            "crawl_depth": 1, "decision": "pending",
                            "decision_reason": "daily_watch_pending", "discovered_at": discovered_ts,
                        },
                    )
                    storage.save_source(src)
                    hint, prio = topic_info(src.model_dump(mode="json"), corp)
                    by_site.setdefault(site_name, []).append({
                        "source_id": sid, "uri": target, "resource_type": kind,
                        "link_text": link.get("link_text", ""), "discovered_from": parent_url,
                        "topic_hint": hint, "topic_priority": prio,
                    })
                    new_pending_total += 1
        for site_name in by_site:
            by_site[site_name].sort(key=lambda x: (-x["topic_priority"], x["uri"]))

    return {
        "checked_sites": len(selected), "fetched_category_pages": len(fetched_pages),
        "new_pending_total": new_pending_total, "by_site": by_site, "errors": errors,
    }


def send_watch_notification(
    data_dir, watch_result: dict, token=None, chat_id=None, max_cards: int = 10, client: TelegramClient | None = None
) -> dict:
    cfg = load_telegram_config(data_dir, token, chat_id)
    target_chat = str(chat_id if chat_id is not None else cfg["chat_id"]).strip()
    if not target_chat:
        raise ValueError("Missing chat_id; run telegram --setup first")
    bot = client or TelegramClient(cfg["bot_token"])
    lines = [
        "<b>Bao cao Quet Danh muc Hang ngay</b>",
        f"Da kiem tra <b>{watch_result['fetched_category_pages']}</b> trang ({watch_result['checked_sites']} site) | Link Pending moi: <b>{watch_result['new_pending_total']}</b>",
    ]
    for site_name, items in watch_result["by_site"].items():
        lines.append(f"- <code>{escape(site_name)}</code>: +{len(items)} link moi")
    if watch_result["errors"]:
        lines.append(f"- Loi ket noi: {len(watch_result['errors'])} trang danh muc")
    if watch_result["new_pending_total"] == 0:
        lines.append("Khong co tai lieu moi hom nay.")
    bot.send_message(target_chat, "\n".join(lines))
    ranked = [(item["topic_priority"], site_name, item) for site_name, items in watch_result["by_site"].items() for item in items]
    ranked.sort(key=lambda x: (-x[0], x[1], x[2]["uri"]))
    sent_cards = 0
    for _, site_name, item in ranked[: max(0, int(max_cards))]:
        card_text, markup = _format_source_card(site_name, item)
        bot.send_message(target_chat, card_text, reply_markup=markup)
        sent_cards += 1
    return {"status": "notified", "new_pending_total": watch_result["new_pending_total"], "cards_sent": sent_cards}


def process_telegram_update(client: TelegramClient, update: dict, data_dir, allowed_chat_id: str, profile_dir=None) -> str | None:
    if cb := update.get("callback_query"):
        msg = cb.get("message") or {}
        chat_id = str((msg.get("chat") or {}).get("id") or "")
        if chat_id != str(allowed_chat_id):
            return None
        parsed = _parse_callback(cb.get("data", ""))
        if not parsed:
            client.answer_callback_query(cb["id"], "Du lieu nut khong hop le")
            return None
        site, sid, dec = parsed
        try:
            res = set_sources_decision(data_dir, site, [sid], dec, profile_dir=profile_dir)
            note = f"Da cap nhat {res['updated']} muc -> {dec.upper()}"
        except (ValueError, RuntimeError) as exc:
            client.answer_callback_query(cb["id"], f"Loi: {_clip(str(exc), 150)}")
            return "callback_error"
        client.answer_callback_query(cb["id"], note)
        prev = escape(msg.get("text") or f"[{site}] {sid}")
        client.edit_message_text(chat_id, msg["message_id"], f"{prev}\n\n<b>=> Trang thai moi: [{dec.upper()}]</b>", reply_markup={"inline_keyboard": []})
        return f"callback:{site}:{dec}"

    msg = update.get("message") or {}
    chat_id = str((msg.get("chat") or {}).get("id") or "")
    text = (msg.get("text") or "").strip()
    if chat_id != str(allowed_chat_id) or not text.startswith("/"):
        return None
    parts = text.split()
    cmd = parts[0].split("@", 1)[0].lower()
    args = parts[1:]
    if cmd in {"/start", "/help"}:
        client.send_message(chat_id, "<b>UET Ingestion Bot</b>\n/status - Thong ke 14 site\n/pending [SITE] [N] - Gui N link cho duyet (mac dinh UET 5)\n/watch [SITE...] - Quet danh muc tim link moi\n/ingest [SITE...] - Tai &amp; parse link Accepted Queued")
    elif cmd == "/status":
        client.send_message(chat_id, _format_status_html(get_all_sites_summary(data_dir, profile_dir)))
    elif cmd == "/pending":
        site = args[0].upper() if args and not args[0].isdigit() else "UET"
        count = max(1, min(15, int(args[1] if len(args) > 1 and args[1].isdigit() else (args[0] if args and args[0].isdigit() else 5))))
        try:
            data = list_site_sources(data_dir, site, status="pending", sort="priority", page=1, page_size=count, profile_dir=profile_dir)
        except (ValueError, KeyError) as exc:
            client.send_message(chat_id, f"Loi site <code>{escape(site)}</code>: {escape(str(exc))}")
            return "pending_error"
        if not data["items"]:
            client.send_message(chat_id, f"Site <code>{escape(site)}</code> khong con link Pending.")
        for item in data["items"]:
            card_text, markup = _format_source_card(site, item)
            client.send_message(chat_id, card_text, reply_markup=markup)
    elif cmd == "/watch":
        client.send_message(chat_id, "Dang quet danh muc tim link moi...")
        watch_res = run_daily_category_watch(data_dir, sites=[a.upper() for a in args] or None, profile_dir=profile_dir)
        send_watch_notification(data_dir, watch_res, chat_id=chat_id, client=client)
    elif cmd == "/ingest":
        try:
            report = run_incremental_ingest(data_dir, sites=[a.upper() for a in args] or None, profile_dir=profile_dir)
        except (ValueError, RuntimeError) as exc:
            client.send_message(chat_id, f"Khong the chay ingest: {escape(str(exc))}")
            return "ingest_error"
        tot = {k: sum(s.get(k, 0) for s in report["sites"].values()) for k in ("downloaded", "processed_saved", "skipped", "new_pending", "rejected_preserved", "errors")}
        client.send_message(chat_id, f"<b>Incremental Ingest ({escape(report['status'])})</b>\nDownloaded: {tot['downloaded']} | Processed: {tot['processed_saved']} | Skipped: {tot['skipped']}\nNew Pending: {tot['new_pending']} | Rejected giu nguyen: {tot['rejected_preserved']} | Errors: {tot['errors']}")
    return cmd


def run_telegram_polling(data_dir, token=None, chat_id=None, profile_dir=None) -> None:
    cfg = load_telegram_config(data_dir, token, chat_id)
    if not cfg["chat_id"]:
        raise ValueError("Missing chat_id; run telegram --setup first")
    client = TelegramClient(cfg["bot_token"])
    me = client.get_me()
    offset = None
    log.info("Telegram polling started for @%s (chat_id=%s)", me.get("username"), cfg["chat_id"])
    while True:
        try:
            for upd in client.get_updates(offset=offset, timeout=25):
                offset = upd["update_id"] + 1
                try:
                    process_telegram_update(client, upd, data_dir, cfg["chat_id"], profile_dir)
                except Exception as exc:
                    log.exception("Telegram update handling error: %s", exc)
        except KeyboardInterrupt:
            log.info("Telegram polling stopped.")
            break
        except (requests.RequestException, RuntimeError) as exc:
            log.warning("Telegram polling error: %s", exc)
            time.sleep(3)

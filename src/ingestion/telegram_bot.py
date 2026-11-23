"""Telegram Bot workflows for daily UET category discovery and two-way source curation."""

from __future__ import annotations

from html import escape
import logging
import re
import time
from urllib.parse import urlsplit
import requests
from src.base import DataSource
from .cloudflare_sync import load_cloudflare_config, pull_from_cloudflare, push_to_cloudflare
from .config.profiles import load_profiles
from .crawler.parser import parse_page
from .crawler.policy import in_discovery_scope, resource_type
from .curation import dedup_url_key, get_all_sites_summary, list_site_sources, run_incremental_ingest, set_sources_decision
from .curation_dedup import deduplicate_all_sites
from .curation_helpers import load_corpus_policies, topic_info
from .curation_yaml import ordered_site_names
from .downloader.http import HTMLFetcher
from .identity import normalize_site_url, source_id
from .storage import Storage, now, writer_lock
from .telegram_client import (
    BOT_HELP_HTML, TelegramClient, _clip, _format_dedup_html, _format_ingest_html, _format_pull_html, _format_push_html,
    _format_source_card, _format_status_html, _format_watch_html, _parse_callback, _parse_sync_args, _review_button_markup, load_telegram_config, save_telegram_config,
)
from .utils import completion_map

log = logging.getLogger(__name__)
__all__ = ["TelegramClient", "load_telegram_config", "save_telegram_config", "setup_telegram_bot", "send_test_status", "run_daily_category_watch", "send_watch_notification", "process_telegram_update", "run_telegram_polling"]
_BAD_LINK_CHARS = ('"', "'", "\u201d", "\u2019", "<", ">")
_BAD_LINK_TOKENS = ("%E2%80%9D", "%E2%80%99", "%22", "%3C", "%3E")
_PAGINATION_RE = re.compile(r"(?:^|/)page/\d+(?:/|$)|[?&](?:page|paged)=\d+", re.I)


def _register_menu(client: TelegramClient) -> None:
    try:
        client.set_my_commands()
    except Exception:
        pass


def _push_cf_safe(data_dir, *, sites, source_ids, force=False, profile_dir=None) -> dict | None:
    cf_cfg = load_cloudflare_config(data_dir)
    if not (cf_cfg["url"] and cf_cfg["sync_secret"]):
        return None
    try:
        res = push_to_cloudflare(data_dir, sites=sites, source_ids=source_ids, force=force, profile_dir=profile_dir)
        return {"status": "ok", "total_sent": res["total_sent"], "changed": res["changed"], "url": res["url"]}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "url": cf_cfg["url"]}


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
    _register_menu(client)
    client.send_message(target_chat, f"<b>Kết nối thành công với @{escape(bot_username)}</b>\nDùng /commands để xem danh sách lệnh.\n\n" + _format_status_html(get_all_sites_summary(data_dir, profile_dir)))
    return {"status": "ok", "bot_username": bot_username, "chat_id": target_chat, "owner_name": owner_name, "config_file": str(config_file)}


def send_test_status(data_dir, token=None, chat_id=None, profile_dir=None) -> dict:
    cfg = load_telegram_config(data_dir, token, chat_id)
    if not cfg["chat_id"]:
        raise ValueError("Missing chat_id; run telegram --setup first")
    res = TelegramClient(cfg["bot_token"]).send_message(cfg["chat_id"], _format_status_html(get_all_sites_summary(data_dir, profile_dir)))
    return {"status": "sent", "chat_id": cfg["chat_id"], "message_id": res.get("message_id")}


def run_daily_category_watch(
    data_dir, sites=None, profile_dir=None, max_entry_urls_per_site: int = 25, timeout: int = 15, sync_cloudflare: bool = True
) -> dict:
    profiles = load_profiles(profile_dir)
    selected = [s for s in ordered_site_names(profiles) if (not sites or s in sites) and s != "OTHER"]
    storage, policies, entry_jobs, entry_keys = Storage(data_dir), load_corpus_policies(profiles), [], set()
    for site in selected:
        profile = profiles[site]
        disc = profile.get("discovery", {})
        raw_entries = list(disc.get("watch_urls") or []) or (
            list(disc.get("start_urls", [])) + [r["uri"] for r in storage.records("sources", site).values() if r.get("metadata", {}).get("decision") == "listing"]
        )
        seen_entry, site_entries = set(), []
        for raw_u in raw_entries:
            try:
                norm_u = normalize_site_url(raw_u, profile)
            except ValueError:
                continue
            if norm_u in seen_entry or resource_type(norm_u) != "html" or urlsplit(norm_u).path.startswith("/offline/") or _PAGINATION_RE.search(norm_u):
                continue
            seen_entry.add(norm_u)
            entry_keys.add(dedup_url_key(norm_u, profile))
            site_entries.append((site, norm_u))
            if len(site_entries) >= max_entry_urls_per_site:
                break
        entry_jobs.extend(site_entries)

    fetcher, fetched_pages, errors = HTMLFetcher(timeout=timeout, retries=1, delay=0.1), [], []

    def _fetch_one(job):
        site_name, entry_url = job
        prof = profiles[site_name]
        try:
            return fetcher.fetch(entry_url, lambda u: in_discovery_scope(u, prof) and resource_type(u) == "html"), None
        except Exception as exc:
            return None, str(exc)

    try:
        if entry_jobs:
            for (site_name, entry_url), (page, err) in completion_map(_fetch_one, entry_jobs, min(12, len(entry_jobs))):
                if err:
                    errors.append({"site": site_name, "url": entry_url, "error": err})
                elif resource_type(page["final_url"]) == "html" and page.get("encoding") != "binary":
                    html = page["body"].decode(page.get("encoding") or "utf-8", errors="replace")
                    _, links = parse_page(html, page["final_url"], profiles[site_name])
                    fetched_pages.append((site_name, page["final_url"], links))
    finally:
        fetcher.close()

    by_site: dict[str, list[dict]] = {}
    new_pending_total = 0
    with writer_lock(storage.root):
        storage, discovered_ts = Storage(data_dir), now()
        global_known_ids = {sid for s in profiles for sid in storage.records("sources", s)}
        global_known_keys = set(entry_keys) | {
            dedup_url_key(r["uri"], profiles[s]) for s in profiles for r in storage.records("sources", s).values()
        }
        with storage.batched():
            for site_name, parent_url, links in fetched_pages:
                prof, corp = profiles[site_name], policies.get(site_name)
                global_known_keys.add(dedup_url_key(parent_url, prof))
                for link in links:
                    target = link["url"]
                    if any(ch in target for ch in _BAD_LINK_CHARS) or any(tok in target.upper() for tok in _BAD_LINK_TOKENS) or _PAGINATION_RE.search(target):
                        continue
                    kind = resource_type(target)
                    sid, target_key = source_id(target), dedup_url_key(target, prof)
                    if kind == "asset" or sid in global_known_ids or target_key in global_known_keys or not in_discovery_scope(target, prof):
                        continue
                    global_known_ids.add(sid)
                    global_known_keys.add(target_key)
                    meta = {
                        "site": site_name, "faculty": site_name, "resource_type": kind, "discovered_from": parent_url,
                        "link_text": link.get("link_text", ""), "crawl_depth": 1, "decision": "pending",
                        "decision_reason": "daily_watch_pending", "discovered_at": discovered_ts,
                    }
                    src = DataSource(source_id=sid, source_type="web", uri=target, metadata=meta)
                    storage.save_source(src)
                    hint, prio = topic_info(src.model_dump(mode="json"), corp)
                    by_site.setdefault(site_name, []).append({
                        "source_id": sid, "uri": target, "resource_type": kind, "link_text": link.get("link_text", ""),
                        "discovered_from": parent_url, "topic_hint": hint, "topic_priority": prio,
                    })
                    new_pending_total += 1
        for site_name in by_site:
            by_site[site_name].sort(key=lambda x: (-x["topic_priority"], x["uri"]))

    new_sids = [it["source_id"] for items in by_site.values() for it in items]
    cf_sync = _push_cf_safe(data_dir, sites=list(by_site.keys()), source_ids=new_sids, profile_dir=profile_dir) if sync_cloudflare and new_sids else None

    result = {"checked_sites": len(selected), "fetched_category_pages": len(fetched_pages), "new_pending_total": new_pending_total, "by_site": by_site, "errors": errors}
    if cf_sync is not None:
        result["cloudflare_sync"] = cf_sync
    return result


def send_watch_notification(data_dir, watch_result: dict, token=None, chat_id=None, max_cards: int = 10, client: TelegramClient | None = None) -> dict:
    cfg = load_telegram_config(data_dir, token, chat_id)
    target_chat = str(chat_id if chat_id is not None else cfg["chat_id"]).strip()
    if not target_chat:
        raise ValueError("Missing chat_id; run telegram --setup first")
    bot = client or TelegramClient(cfg["bot_token"])
    cf_url = (watch_result.get("cloudflare_sync") or {}).get("url") or load_cloudflare_config(data_dir).get("url")
    bot.send_message(target_chat, _format_watch_html(watch_result), reply_markup=_review_button_markup(cf_url))
    ranked = sorted(((it["topic_priority"], s, it) for s, items in watch_result["by_site"].items() for it in items), key=lambda x: (-x[0], x[1], x[2]["uri"]))
    cards = ranked[: max(0, int(max_cards))]
    for _, site_name, item in cards:
        bot.send_message(target_chat, *_format_source_card(site_name, item))
    return {"status": "notified", "new_pending_total": watch_result["new_pending_total"], "cards_sent": len(cards)}


def _run_cmd_and_reply(cmd: str, client: TelegramClient, chat_id: str, err_label: str, err_code: str, fn, formatter, with_review_btn: bool = False) -> str:
    try:
        res = fn()
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        client.send_message(chat_id, f"Lỗi {err_label}: {escape(str(exc))}")
        return err_code
    markup = _review_button_markup(res.get("url")) if with_review_btn else None
    client.send_message(chat_id, formatter(res), reply_markup=markup)
    return cmd


def process_telegram_update(client: TelegramClient, update: dict, data_dir, allowed_chat_id: str, profile_dir=None) -> str | None:
    if cb := update.get("callback_query"):
        msg = cb.get("message") or {}
        chat_id = str((msg.get("chat") or {}).get("id") or "")
        if chat_id != str(allowed_chat_id):
            return None
        parsed = _parse_callback(cb.get("data", ""))
        if not parsed:
            client.answer_callback_query(cb["id"], "Dữ liệu nút không hợp lệ")
            return None
        site, sid, dec = parsed
        try:
            res = set_sources_decision(data_dir, site, [sid], dec, profile_dir=profile_dir)
            note = f"Đã cập nhật {res['updated']} mục -> {dec.upper()}"
        except (ValueError, RuntimeError) as exc:
            client.answer_callback_query(cb["id"], f"Lỗi: {_clip(str(exc), 150)}")
            return "callback_error"
        client.answer_callback_query(cb["id"], note)
        prev = escape(msg.get("text") or f"[{site}] {sid}")
        client.edit_message_text(chat_id, msg["message_id"], f"{prev}\n\n<b>=> Trạng thái mới: [{dec.upper()}]</b>", reply_markup={"inline_keyboard": []})
        _push_cf_safe(data_dir, sites=[site], source_ids=[sid], force=True, profile_dir=profile_dir)
        return f"callback:{site}:{dec}"

    msg = update.get("message") or {}
    chat_id, text = str((msg.get("chat") or {}).get("id") or ""), (msg.get("text") or "").strip()
    if chat_id != str(allowed_chat_id) or not text.startswith("/"):
        return None
    parts = text.split()
    cmd, args = parts[0].split("@", 1)[0].lower(), parts[1:]
    if cmd in {"/start", "/help", "/commands", "/list", "/menu"}:
        client.send_message(chat_id, BOT_HELP_HTML)
    elif cmd == "/status":
        client.send_message(chat_id, _format_status_html(get_all_sites_summary(data_dir, profile_dir)))
    elif cmd == "/pending":
        site = args[0].upper() if args and not args[0].isdigit() else "UET"
        count = max(1, min(15, int(args[1] if len(args) > 1 and args[1].isdigit() else (args[0] if args and args[0].isdigit() else 5))))
        try:
            data = list_site_sources(data_dir, site, status="pending", sort="priority", page=1, page_size=count, profile_dir=profile_dir)
        except (ValueError, KeyError) as exc:
            client.send_message(chat_id, f"Lỗi site <code>{escape(site)}</code>: {escape(str(exc))}")
            return "pending_error"
        if not data["items"]:
            client.send_message(chat_id, f"Site <code>{escape(site)}</code> không còn link Pending.")
        for item in data["items"]:
            card_text, markup = _format_source_card(site, item)
            client.send_message(chat_id, card_text, reply_markup=markup)
    elif cmd == "/watch":
        client.send_message(chat_id, "Đang quét trang tin tức / thông báo (depth = 1)...")
        watch_res = run_daily_category_watch(data_dir, sites=[a.upper() for a in args] or None, profile_dir=profile_dir)
        send_watch_notification(data_dir, watch_res, chat_id=chat_id, client=client)
    elif cmd == "/pull":
        sites, status_filter, dry_run = _parse_sync_args(args)
        return _run_cmd_and_reply(cmd, client, chat_id, "/pull Cloudflare", "pull_error", lambda: pull_from_cloudflare(data_dir, sites=sites, status_filter=status_filter, dry_run=dry_run, profile_dir=profile_dir), _format_pull_html)
    elif cmd == "/push":
        sites, status_filter, _ = _parse_sync_args(args)
        return _run_cmd_and_reply(cmd, client, chat_id, "/push Cloudflare", "push_error", lambda: push_to_cloudflare(data_dir, sites=sites, status_filter=status_filter, profile_dir=profile_dir), _format_push_html, with_review_btn=True)
    elif cmd == "/dedup":
        cross_site = "--no-cross-site" not in [a.lower() for a in args]
        sites = [a.upper() for a in args if a.lower() != "--no-cross-site"] or None
        return _run_cmd_and_reply(cmd, client, chat_id, "/dedup", "dedup_error", lambda: deduplicate_all_sites(data_dir, sites=sites, cross_site=cross_site, profile_dir=profile_dir), _format_dedup_html)
    elif cmd == "/ingest":
        return _run_cmd_and_reply(cmd, client, chat_id, "ingest", "ingest_error", lambda: run_incremental_ingest(data_dir, sites=[a.upper() for a in args] or None, profile_dir=profile_dir), _format_ingest_html)
    else:
        client.send_message(chat_id, f"Lệnh không hợp lệ: <code>{escape(cmd)}</code>\n\n{BOT_HELP_HTML}")
    return cmd


def run_telegram_polling(data_dir, token=None, chat_id=None, profile_dir=None) -> None:
    cfg = load_telegram_config(data_dir, token, chat_id)
    if not cfg["chat_id"]:
        raise ValueError("Missing chat_id; run telegram --setup first")
    client = TelegramClient(cfg["bot_token"])
    me, offset = client.get_me(), None
    _register_menu(client)
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
"""Low-level Telegram Bot API client, configuration loader, and message formatters."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
from uuid import UUID
import requests
from .storage import now, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "contexts" / "telegram_config.json"


def _read_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.is_file():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip("'\"")
    return values


def _config_path(data_dir=None) -> Path:
    if data_dir is not None:
        resolved = Path(data_dir).resolve()
        if resolved != (REPO_ROOT / "data").resolve():
            return resolved / "telegram_config.json"
    return DEFAULT_CONFIG_PATH


def load_telegram_config(data_dir, token=None, chat_id=None) -> dict[str, str]:
    path = _config_path(data_dir)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    dotenv = _read_dotenv()
    tok = (token or "").strip() or str(saved.get("bot_token") or "").strip() or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or dotenv.get("TELEGRAM_BOT_TOKEN", "").strip()
    cid = str(chat_id).strip() if chat_id is not None and str(chat_id).strip() else str(saved.get("chat_id") or "").strip() or os.environ.get("TELEGRAM_CHAT_ID", "").strip() or dotenv.get("TELEGRAM_CHAT_ID", "").strip()
    return {"bot_token": tok, "chat_id": cid, "bot_username": str(saved.get("bot_username") or ""), "owner_name": str(saved.get("owner_name") or "")}


def save_telegram_config(data_dir, token: str, chat_id: str, bot_username: str = "", owner_name: str = "") -> Path:
    path = _config_path(data_dir)
    write_json(path, {"bot_token": token.strip(), "chat_id": str(chat_id).strip(), "bot_username": bot_username, "owner_name": owner_name, "updated_at": now()})
    return path


class TelegramClient:
    """Minimal Bot API wrapper over HTTP long polling and inline callbacks."""

    def __init__(self, token: str, timeout: float = 30.0):
        clean = (token or "").strip()
        if not clean:
            raise ValueError("Missing Telegram Bot Token (--token or TELEGRAM_BOT_TOKEN)")
        self.base_url = f"https://api.telegram.org/bot{clean}"
        self.timeout = timeout
        self.session = requests.Session()

    def _call(self, method: str, payload: dict | None = None, timeout: float | None = None):
        resp = self.session.post(
            f"{self.base_url}/{method}", json=payload or {}, timeout=timeout or self.timeout
        )
        try:
            data = resp.json()
        except ValueError as exc:
            resp.raise_for_status()
            raise RuntimeError(f"Invalid Telegram response for {method}") from exc
        if not resp.ok or not data.get("ok"):
            raise RuntimeError(f"Telegram API error ({method}): {data.get('description') or resp.status_code}")
        return data.get("result")

    def get_me(self) -> dict:
        return self._call("getMe")

    def get_updates(self, offset: int | None = None, timeout: int = 20) -> list[dict]:
        params = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", params, timeout=timeout + 10) or []

    def send_message(self, chat_id, text: str, reply_markup: dict | None = None, parse_mode: str = "HTML") -> dict:
        body = {"chat_id": str(chat_id), "text": text[:4096], "parse_mode": parse_mode, "disable_web_page_preview": True}
        if reply_markup is not None:
            body["reply_markup"] = reply_markup
        return self._call("sendMessage", body)

    def edit_message_text(self, chat_id, message_id: int, text: str, reply_markup: dict | None = None, parse_mode: str = "HTML") -> dict:
        body = {"chat_id": str(chat_id), "message_id": message_id, "text": text[:4096], "parse_mode": parse_mode, "disable_web_page_preview": True}
        if reply_markup is not None:
            body["reply_markup"] = reply_markup
        return self._call("editMessageText", body)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> bool:
        return bool(self._call("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text[:190]}))

    def set_my_commands(self, commands: list[dict] | None = None) -> bool:
        return bool(self._call("setMyCommands", {"commands": commands or BOT_COMMANDS}))


def _clip(value: str, limit: int = 120) -> str:
    clean = " ".join(str(value or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def _format_status_html(summary: dict) -> str:
    t = summary["totals"]
    lines = [
        "<b>TRẠNG THÁI DỮ LIỆU INGESTION</b>",
        f"• Tổng nguồn: <b>{t['total']:,}</b>",
        f"• Đã duyệt (Accepted): <b>{t['accepted_total']:,}</b> <i>(Đã nạp: {t['accepted_ingested']:,} • Chờ nạp: {t['accepted_queued']:,})</i>",
        f"• Chờ duyệt (Pending): <b>{t['pending']:,}</b>",
        f"• Trang danh mục (Listing): <b>{t['listing']:,}</b>",
        f"• Đã loại (Rejected): <b>{t['rejected_total']:,}</b>",
    ]
    if t.get("failed"):
        lines.append(f"• Lỗi (Failed): <b>{t['failed']:,}</b>")
    lines.extend(["", "<b>CHI TIẾT THEO ĐƠN VỊ</b>"])
    for s in summary["sites"]:
        if s["total"] or s["pending"]:
            lines.append(
                f"• <code>{escape(s['site'])}</code> — Chờ: <b>{s['pending']:,}</b> • "
                f"Duyệt: <b>{s['accepted_ingested']}/{s['accepted_total']}</b> • Loại: {s['rejected_total']}"
            )
    return "\n".join(lines)[:4096]


def _format_source_card(site: str, item: dict) -> tuple[str, dict]:
    sid_hex = UUID(item["source_id"]).hex
    hint = f" • <i>{escape(item['topic_hint'])} ({item.get('topic_priority', 0)})</i>" if item.get("topic_hint") else ""
    label = escape(_clip(item.get("link_text") or item.get("title") or item["uri"], 140))
    uri = escape(item["uri"])
    src_from = escape(_clip(item.get("discovered_from") or "-", 90))
    kind = escape((item.get("resource_type") or "html").upper())
    text = (
        f"<b>[{escape(site)}]</b> <code>{kind}</code>{hint}\n"
        f"• <b>Tiêu đề:</b> {label}\n"
        f"• <b>Đường dẫn:</b> <a href=\"{uri}\">{uri}</a>\n"
        f"• <b>Nguồn:</b> <code>{src_from}</code>"
    )
    row = [{"text": "Accept", "callback_data": f"d:{site}:{sid_hex}:a"}, {"text": "Reject", "callback_data": f"d:{site}:{sid_hex}:r"}]
    if (item.get("resource_type") or "html") == "html":
        row.insert(1, {"text": "Listing", "callback_data": f"d:{site}:{sid_hex}:l"})
    return text, {"inline_keyboard": [row]}


def _parse_callback(data: str) -> tuple[str, str, str] | None:
    parts = (data or "").split(":")
    if len(parts) != 4 or parts[0] not in {"d", "dec"}:
        return None
    dec = {"a": "accepted", "l": "listing", "r": "rejected", "p": "pending"}.get(parts[3], parts[3])
    if dec not in {"accepted", "listing", "pending", "rejected"}:
        return None
    try:
        sid = str(UUID(parts[2]))
    except ValueError:
        return None
    return parts[1].upper(), sid, dec


BOT_COMMANDS = [
    {"command": "commands", "description": "Xem danh sách toàn bộ câu lệnh"},
    {"command": "help", "description": "Hướng dẫn sử dụng và ví dụ lệnh"},
    {"command": "status", "description": "Thống kê nguồn dữ liệu 14 đơn vị"},
    {"command": "pending", "description": "Gửi link chờ duyệt (VD: /pending FIT 5)"},
    {"command": "watch", "description": "Quét tin tức mới (depth=1) & đẩy Cloudflare"},
    {"command": "pull", "description": "Kéo kết quả duyệt từ Cloudflare D1"},
    {"command": "push", "description": "Đẩy nguồn lên Cloudflare D1"},
    {"command": "dedup", "description": "Khử trùng URL nội site và xuyên site"},
    {"command": "ingest", "description": "Tải & trích xuất link Accepted Queued"},
]

BOT_HELP_HTML = (
    "<b>DANH SÁCH CÂU LỆNH INGESTION BOT</b>\n"
    "• <code>/commands</code>, <code>/list</code>, <code>/help</code> — Xem lại danh sách lệnh này\n"
    "• <code>/status</code> — Thống kê tổng hợp 14 đơn vị\n"
    "• <code>/pending [SITE] [N]</code> — Gửi N thẻ chờ duyệt <i>(VD: /pending FIT 5)</i>\n"
    "• <code>/watch [SITE...]</code> — Quét tin tức (depth=1) &amp; đồng bộ Cloudflare <i>(VD: /watch UET FIT)</i>\n"
    "• <code>/pull [SITE...] [STATUS] [dry]</code> — Kéo duyệt từ Cloudflare D1 <i>(VD: /pull UET accepted dry)</i>\n"
    "• <code>/push [SITE...] [STATUS]</code> — Đẩy nguồn lên Cloudflare D1 <i>(VD: /push FIT pending)</i>\n"
    "• <code>/dedup [SITE...] [--no-cross-site]</code> — Khử trùng URL nội &amp; xuyên site\n"
    "• <code>/ingest [SITE...]</code> — Tải &amp; trích xuất link Accepted Queued <i>(VD: /ingest UET)</i>"
)


def _review_button_markup(review_url: str | None) -> dict | None:
    clean = str(review_url or "").strip()
    if clean.startswith(("http://", "https://")):
        return {"inline_keyboard": [[{"text": "Mở Cloudflare Review", "url": clean}]]}
    return None


def _format_watch_html(watch_result: dict) -> str:
    lines = [
        "<b>BÁO CÁO QUÉT TIN TỨC HÀNG NGÀY</b>",
        f"• Đã kiểm tra: <b>{watch_result['fetched_category_pages']}</b> trang ({watch_result['checked_sites']} đơn vị)",
        f"• Link Pending mới: <b>{watch_result['new_pending_total']}</b>",
    ]
    for site_name, items in watch_result.get("by_site", {}).items():
        lines.append(f"  - <code>{escape(site_name)}</code>: +{len(items)} link mới")
    if watch_result.get("errors"):
        lines.append(f"• Lỗi kết nối: <b>{len(watch_result['errors'])}</b> trang danh mục")
    if watch_result.get("new_pending_total", 0) == 0:
        lines.append("• <i>Không có bài viết mới cần duyệt hôm nay.</i>")
    cf_sync = watch_result.get("cloudflare_sync")
    if cf_sync:
        if cf_sync.get("status") == "ok":
            lines.append(f"• Cloudflare D1: đã đồng bộ <b>{cf_sync.get('changed', 0)}/{cf_sync.get('total_sent', 0)}</b> link pending mới")
        else:
            lines.append(f"• Cloudflare D1: lỗi đồng bộ ({escape(_clip(cf_sync.get('error', ''), 120))})")
    return "\n".join(lines)[:4096]


def _format_pull_html(res: dict) -> str:
    mode = "CHẠY THỬ (DRY-RUN)" if res.get("dry_run") else "ĐÃ ÁP DỤNG"
    lines = [
        f"<b>ĐỒNG BỘ TỪ CLOUDFLARE ({mode})</b>",
        f"• Kéo về: <b>{res.get('pulled_total', 0)}</b> • Áp dụng: <b>{res.get('applied', 0)}</b> • Ack: <b>{res.get('acked', 0)}</b>",
    ]
    by_dec = res.get("by_decision") or {}
    if by_dec:
        lines.append("• Phân loại: " + " • ".join(f"{escape(k)}: <b>{v}</b>" for k, v in by_dec.items()))
    for site_name, counts in (res.get("by_site") or {}).items():
        summary = ", ".join(f"{escape(k)}: {v}" for k, v in counts.items())
        lines.append(f"  - <code>{escape(site_name)}</code>: {summary}")
    if res.get("skipped") or res.get("errors"):
        lines.append(f"• Bỏ qua: {len(res.get('skipped') or [])} • Lỗi: {len(res.get('errors') or [])}")
    return "\n".join(lines)[:4096]


def _format_push_html(res: dict) -> str:
    sites_str = ", ".join(escape(s) for s in (res.get("sites_pushed") or [])) or "-"
    lines = [
        "<b>ĐẨY DỮ LIỆU LÊN CLOUDFLARE</b>",
        f"• Đã gửi: <b>{res.get('total_sent', 0)}</b>",
        f"• Cập nhật trên D1: <b>{res.get('changed', 0)}</b> ({res.get('batches', 0)} lô)",
        f"• Đơn vị: <code>{sites_str}</code>",
    ]
    return "\n".join(lines)[:4096]


def _format_dedup_html(res: dict) -> str:
    restored = sum(int(s.get("restored_redirects") or 0) for s in (res.get("sites") or {}).values())
    lines = [
        "<b>KHỬ TRÙNG URL (DEDUPLICATION)</b>",
        f"• Tổng gộp: <b>{res.get('total_merged', 0)}</b> <i>(Nội site: {res.get('per_site_merged', 0)} • Xuyên site: {res.get('cross_site_merged', 0)})</i>",
        f"• Khôi phục redirect alias: <b>{restored}</b>",
        f"• Nguồn active còn lại: <b>{res.get('remaining_active', 0)}</b>",
    ]
    for site_name, info in (res.get("sites") or {}).items():
        merged, site_restored = int(info.get("duplicates_merged") or 0), int(info.get("restored_redirects") or 0)
        if merged or site_restored:
            lines.append(f"  - <code>{escape(site_name)}</code> — Gộp: {merged} • Khôi phục: {site_restored} • Còn: {info.get('remaining_active', 0)}")
    return "\n".join(lines)[:4096]


def _format_ingest_html(report: dict) -> str:
    keys = ("downloaded", "processed_saved", "skipped", "new_pending", "rejected_preserved", "errors")
    tot = {k: sum(s.get(k, 0) for s in report["sites"].values()) for k in keys}
    return (
        f"<b>TRÍCH XUẤT TĂNG DẦN ({escape(report['status'])})</b>\n"
        f"• Đã tải mới: <b>{tot['downloaded']}</b>\n"
        f"• Đã trích xuất: <b>{tot['processed_saved']}</b>\n"
        f"• Bỏ qua (đã có): <b>{tot['skipped']}</b>\n"
        f"• Link Pending mới: <b>{tot['new_pending']}</b>\n"
        f"• Rejected giữ nguyên: <b>{tot['rejected_preserved']}</b>\n"
        f"• Lỗi: <b>{tot['errors']}</b>"
    )


def _parse_sync_args(args: list[str]) -> tuple[list[str] | None, str | None, bool]:
    sites, status_filter, dry_run = [], None, False
    for raw in args:
        low = raw.lower()
        if low in {"dry", "dry-run", "--dry-run"}:
            dry_run = True
        elif low in {"pending", "accepted", "listing", "rejected"}:
            status_filter = low
        else:
            sites.append(raw.upper())
    return (sites or None), status_filter, dry_run

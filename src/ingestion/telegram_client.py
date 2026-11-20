"""Low-level Telegram Bot API client, configuration loader, and message formatters."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
from uuid import UUID
import requests
from .storage import now, write_json

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


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


def _config_path(data_dir) -> Path:
    return Path(data_dir).resolve() / "ingestion_state" / "telegram_config.json"


def load_telegram_config(data_dir, token=None, chat_id=None) -> dict[str, str]:
    path = _config_path(data_dir)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    dotenv = _read_dotenv()
    resolved_token = (
        (token or "").strip()
        or str(saved.get("bot_token") or "").strip()
        or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        or dotenv.get("TELEGRAM_BOT_TOKEN", "").strip()
    )
    resolved_chat = (
        str(chat_id).strip()
        if chat_id is not None and str(chat_id).strip()
        else str(saved.get("chat_id") or "").strip()
        or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        or dotenv.get("TELEGRAM_CHAT_ID", "").strip()
    )
    return {
        "bot_token": resolved_token,
        "chat_id": resolved_chat,
        "bot_username": str(saved.get("bot_username") or ""),
        "owner_name": str(saved.get("owner_name") or ""),
    }


def save_telegram_config(data_dir, token: str, chat_id: str, bot_username: str = "", owner_name: str = "") -> Path:
    path = _config_path(data_dir)
    write_json(
        path,
        {
            "bot_token": token.strip(),
            "chat_id": str(chat_id).strip(),
            "bot_username": bot_username,
            "owner_name": owner_name,
            "updated_at": now(),
        },
    )
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


def _clip(value: str, limit: int = 120) -> str:
    clean = " ".join(str(value or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def _format_status_html(summary: dict) -> str:
    t = summary["totals"]
    lines = [
        "<b>UET Ingestion Status</b>",
        f"Tong nguon: <b>{t['total']}</b> | Accepted: <b>{t['accepted_total']}</b> (Ingested: {t['accepted_ingested']}, Queued: {t['accepted_queued']})",
        f"Pending: <b>{t['pending']}</b> | Listing: {t['listing']} | Rejected: {t['rejected_total']} | Failed: {t['failed']}",
        "",
    ]
    for s in summary["sites"]:
        if s["total"] or s["pending"]:
            lines.append(f"- <code>{escape(s['site'])}</code>: pending <b>{s['pending']}</b> | acc {s['accepted_ingested']}/{s['accepted_total']} | rej {s['rejected_total']}")
    return "\n".join(lines)[:4096]


def _format_source_card(site: str, item: dict) -> tuple[str, dict]:
    sid_hex = UUID(item["source_id"]).hex
    hint = f"[{escape(item['topic_hint'])} ({item.get('topic_priority', 0)})] " if item.get("topic_hint") else ""
    label = escape(_clip(item.get("link_text") or item.get("title") or item["uri"], 140))
    uri = escape(item["uri"])
    src_from = escape(_clip(item.get("discovered_from") or "-", 90))
    kind = escape((item.get("resource_type") or "html").upper())
    text = (
        f"<b>[{escape(site)}]</b> {hint}({kind})\n"
        f"Tieu de: {label}\n"
        f"Link: <a href=\"{uri}\">{uri}</a>\n"
        f"Tu trang: {src_from}"
    )
    row = [
        {"text": "Accept", "callback_data": f"d:{site}:{sid_hex}:a"},
        {"text": "Reject", "callback_data": f"d:{site}:{sid_hex}:r"},
    ]
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

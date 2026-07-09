"""Zed AI thread transcript reader."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import zstandard as zstd

from .transcript import Transcript, UserMessage

_DB_CANDIDATES = [
    Path.home() / "Library/Application Support/Zed/threads/threads.db",
    Path.home() / ".local/share/zed/threads/threads.db",
    Path.home() / "AppData/Local/Zed/threads/threads.db",
]


def _find_db() -> Path:
    for p in _DB_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find Zed threads.db; tried: "
        + ", ".join(str(p) for p in _DB_CANDIDATES)
    )


def _decompress(data_type: str, raw: bytes) -> dict:
    if data_type == "zstd":
        raw = zstd.ZstdDecompressor().stream_reader(raw).read()
    return json.loads(raw)


def _extract_user_messages(thread_data: dict) -> list[UserMessage]:
    messages: list[UserMessage] = []
    for msg in thread_data.get("messages", []):
        if isinstance(msg, str):
            user = {"content": msg}
        elif isinstance(msg, dict):
            user = msg.get("User")
        else:
            continue
        if not user:
            continue
        content = user.get("content", [])
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "Text" in block:
                    parts.append(block["Text"].strip())
            text = "\n".join(parts).strip()
        else:
            text = str(content).strip()
        if text:
            messages.append(UserMessage(text=text))
    return messages


def _extract_models(thread_data: dict) -> tuple[list[str], dict[str, str]]:
    """Extract model identifiers and providers from a Zed thread.

    Returns (sorted_models, model_providers).
    """
    models: set[str] = set()
    model_providers: dict[str, str] = {}
    model_info = thread_data.get("model")
    if isinstance(model_info, dict):
        model = model_info.get("model", "")
        if model:
            models.add(model)
            provider = model_info.get("provider", "")
            if provider:
                model_providers[model] = provider
    elif isinstance(model_info, str) and model_info:
        models.add(model_info)
    return sorted(models), model_providers


def _parse_timestamp(ts: str | None) -> datetime:
    if not ts:
        return datetime.now(tz=timezone.utc)
    return datetime.fromisoformat(ts)


class ZedTranscript(Transcript):
    """A transcript from a Zed AI thread (threads.db)."""

    @classmethod
    def read(cls, name: str) -> ZedTranscript:
        db = _find_db()
        con = sqlite3.connect(str(db))
        row = con.execute(
            "SELECT id, summary, created_at, updated_at, data_type, data "
            "FROM threads WHERE id = ?",
            (name,),
        ).fetchone()
        con.close()
        if row is None:
            raise FileNotFoundError(f"No Zed thread found for {name!r}")
        return cls._from_row(row)

    @classmethod
    def readall(cls) -> list[ZedTranscript]:
        try:
            db = _find_db()
        except FileNotFoundError:
            return []
        con = sqlite3.connect(str(db))
        rows = con.execute(
            "SELECT id, summary, created_at, updated_at, data_type, data "
            "FROM threads ORDER BY updated_at DESC"
        ).fetchall()
        con.close()
        return [cls._from_row(r) for r in rows]

    @classmethod
    def _from_row(cls, row: tuple) -> ZedTranscript:
        thread_id, summary, created_at, updated_at, data_type, data = row
        thread_data = _decompress(data_type, data)
        user_messages = _extract_user_messages(thread_data)
        models, model_providers = _extract_models(thread_data)
        return cls(
            name=thread_id,
            summary=summary or "(no title)",
            created=_parse_timestamp(created_at),
            modified=_parse_timestamp(updated_at),
            models=models,
            model_providers=model_providers,
            user_messages=user_messages,
        )

"""Pi agent transcript reader."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .transcript import Transcript, UserMessage

_SESSIONS_DIR = Path.home() / ".pi" / "agent" / "sessions"


def _extract_text(content) -> str:
    """Extract plain text from message content (str or list-of-blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", "").strip())
        return " ".join(parts).strip()
    return str(content).strip()


def _parse_jsonl(
    path: Path,
) -> tuple[list[UserMessage], str, list[str], dict[str, str]]:
    """Return (user_messages, summary, models, model_providers) from a Pi JSONL session."""
    messages: list[UserMessage] = []
    summary = ""
    models: set[str] = set()
    model_providers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "message":
            continue
        msg = obj.get("message", {})

        # Extract model and provider from assistant messages
        if msg.get("role") == "assistant" and msg.get("model"):
            models.add(msg["model"])
            provider = msg.get("provider", "")
            if provider:
                model_providers.setdefault(msg["model"], provider)

        if msg.get("role") != "user":
            continue
        text = _extract_text(msg.get("content", ""))
        if text:
            messages.append(UserMessage(text=text))

    if messages:
        summary = messages[0].text[:50]

    return messages, summary, sorted(models), model_providers


class PiTranscript(Transcript):
    """A transcript from a Pi agent session (~/.pi/agent/sessions/)."""

    @classmethod
    def read(cls, name: str) -> PiTranscript:
        matches = list(_SESSIONS_DIR.glob(f"**/{name}.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No Pi session found for {name!r}")
        return cls._from_path(matches[0])

    @classmethod
    def readall(cls) -> list[PiTranscript]:
        if not _SESSIONS_DIR.exists():
            return []
        results = sorted(
            _SESSIONS_DIR.rglob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [cls._from_path(p) for p in results]

    @classmethod
    def _from_path(cls, path: Path) -> PiTranscript:
        stat = path.stat()
        user_messages, summary, models, model_providers = _parse_jsonl(path)
        return cls(
            name=path.stem,
            summary=summary,
            created=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            models=models,
            model_providers=model_providers,
            user_messages=user_messages,
        )

"""Claude Code transcript reader."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .transcript import Transcript, UserMessage

_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# XML-tag patterns to strip from Claude session messages
_XML_PATTERNS = [
    re.compile(r"<local-command-\w+>.*?</local-command-\w+>", re.DOTALL),
    re.compile(r"<command-\w+>.*?</command-\w+>", re.DOTALL),
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(r"<[^>]+>"),
]


def _clean_text(text: str) -> str:
    for pat in _XML_PATTERNS:
        text = pat.sub("", text)
    if "Base directory for this skill" in text:
        text = text.split("Base directory for this skill")[0]
    return text.strip()


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
    """Return (user_messages, summary, models, model_providers) from a Claude JSONL session."""
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

        if obj.get("type") == "summary" and obj.get("summary"):
            summary = obj["summary"]

        # Extract model from assistant messages; provider is always "claude"
        if obj.get("type") == "assistant":
            msg = obj.get("message", {})
            if isinstance(msg, dict) and msg.get("model"):
                model = msg["model"]
                if model != "<synthetic>":
                    models.add(model)
                    model_providers.setdefault(model, "claude")

        if obj.get("type") != "user":
            continue
        if obj.get("isMeta", False):
            continue

        msg = obj.get("message", {})
        text = _extract_text(msg.get("content", ""))
        text = _clean_text(text)
        if text:
            messages.append(UserMessage(text=text))

    if not summary and messages:
        summary = messages[0].text[:50]

    return messages, summary, sorted(models), model_providers


class ClaudeTranscript(Transcript):
    """A transcript from a Claude Code session (~/.claude/projects/)."""

    @classmethod
    def read(cls, name: str) -> ClaudeTranscript:
        matches = list(_PROJECTS_DIR.glob(f"**/{name}.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No Claude session found for {name!r}")
        return cls._from_path(matches[0])

    @classmethod
    def readall(cls) -> list[ClaudeTranscript]:
        if not _PROJECTS_DIR.exists():
            return []
        results = []
        for f in _PROJECTS_DIR.glob("**/*.jsonl"):
            if f.name.startswith("agent-"):
                continue
            if "/subagents/" in str(f):
                continue
            results.append(f)
        results.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [cls._from_path(p) for p in results]

    @classmethod
    def _from_path(cls, path: Path) -> ClaudeTranscript:
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

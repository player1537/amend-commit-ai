"""Base transcript model and formatting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserMessage:
    """A single user message extracted from a conversation."""

    text: str


def _wrap_markdown(text: str, width: int = 72) -> str:
    """Wrap text as markdown, preserving code blocks and structure.

    Falls back to the original text if mdformat fails.
    """
    try:
        import mdformat

        return mdformat.text(text, options={"wrap": width}).rstrip()
    except Exception:
        return text


@dataclass
class Transcript:
    """Base class for AI conversation transcripts."""

    name: str
    summary: str
    created: datetime
    modified: datetime
    models: list[str] = field(default_factory=list)
    model_providers: dict[str, str] = field(default_factory=dict)
    user_messages: list[UserMessage] = field(default_factory=list)

    @classmethod
    def read(cls, name: str) -> Transcript:
        """Read a single transcript by name/id."""
        raise NotImplementedError

    @classmethod
    def readall(cls) -> list[Transcript]:
        """Discover and return all available transcripts."""
        raise NotImplementedError

    def format_transcript(self) -> str:
        """Format user messages as a markdown transcript."""
        wrapped = [_wrap_markdown(m.text) for m in self.user_messages]
        return "\n\n---\n\n".join(wrapped)

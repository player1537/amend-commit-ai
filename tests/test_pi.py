"""Tests for PiTranscript."""

import json
import os
import tempfile
from datetime import timezone
from pathlib import Path
from unittest import mock

from amend_commit_ai.pi import PiTranscript, _extract_text, _parse_jsonl


def _write_jsonl(path: Path, entries: list[dict]):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


class TestExtractText:
    def test_string_content(self):
        assert _extract_text("  hello  ") == "hello"

    def test_list_content(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _extract_text(content) == "first second"

    def test_empty(self):
        assert _extract_text("") == ""


class TestParseJsonl:
    def test_basic_session(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entries = [
                {"type": "message", "message": {"role": "user", "content": "hello"}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "model": "pi-3",
                        "content": "hi there",
                    },
                },
                {"type": "message", "message": {"role": "user", "content": "thanks"}},
            ]
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            messages, summary, models, model_providers = _parse_jsonl(path)
            assert len(messages) == 2
            assert messages[0].text == "hello"
            assert messages[1].text == "thanks"
            assert summary == "hello"
            assert models == ["pi-3"]
            assert model_providers == {}
        finally:
            os.unlink(path)

    def test_skips_non_message_types(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entries = [
                {
                    "type": "system",
                    "message": {"role": "system", "content": "system prompt"},
                },
                {"type": "message", "message": {"role": "user", "content": "real"}},
            ]
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            messages, _, _, _ = _parse_jsonl(path)
            assert len(messages) == 1
            assert messages[0].text == "real"
        finally:
            os.unlink(path)

    def test_list_content_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entries = [
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "multi block"}],
                    },
                },
            ]
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            messages, _, _, _ = _parse_jsonl(path)
            assert len(messages) == 1
            assert messages[0].text == "multi block"
        finally:
            os.unlink(path)


class TestPiTranscriptRead:
    def test_read_not_found(self):
        with mock.patch("amend_commit_ai.pi._SESSIONS_DIR", Path("/nonexistent")):
            try:
                PiTranscript.read("nope")
                assert False, "should raise"
            except FileNotFoundError:
                pass

    def test_readall_missing_dir(self):
        with mock.patch("amend_commit_ai.pi._SESSIONS_DIR", Path("/nonexistent")):
            assert PiTranscript.readall() == []

    def test_read_and_readall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir)
            session = sessions_dir / "my-session.jsonl"
            entries = [
                {"type": "message", "message": {"role": "user", "content": "hello"}},
                {
                    "type": "message",
                    "message": {"role": "assistant", "model": "pi-3", "content": "hi"},
                },
            ]
            _write_jsonl(session, entries)

            with mock.patch("amend_commit_ai.pi._SESSIONS_DIR", sessions_dir):
                transcripts = PiTranscript.readall()
                assert len(transcripts) == 1
                t = transcripts[0]
                assert t.name == "my-session"
                assert len(t.user_messages) == 1
                assert t.user_messages[0].text == "hello"
                assert t.models == ["pi-3"]
                assert t.model_providers == {}
                assert t.created.tzinfo == timezone.utc

                t2 = PiTranscript.read("my-session")
                assert t2.name == "my-session"

"""Tests for ClaudeTranscript."""

import json
import os
import tempfile
from datetime import timezone
from pathlib import Path
from unittest import mock

from amend_commit_ai.claude import (
    ClaudeTranscript,
    _clean_text,
    _extract_text,
    _parse_jsonl,
)


def _write_jsonl(path: Path, entries: list[dict]):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


class TestCleanText:
    def test_strips_xml_tags(self):
        text = "hello <system-reminder>ignore</system-reminder> world"
        assert _clean_text(text) == "hello  world"

    def test_strips_command_tags(self):
        text = "before <command-foo>bar</command-foo> after"
        assert _clean_text(text) == "before  after"

    def test_strips_skill_directory_suffix(self):
        text = "real content\nBase directory for this skill is /foo"
        assert _clean_text(text) == "real content"

    def test_plain_text_unchanged(self):
        assert _clean_text("hello world") == "hello world"


class TestExtractText:
    def test_string_content(self):
        assert _extract_text("  hello  ") == "hello"

    def test_list_content(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "image", "url": "x"},
            {"type": "text", "text": "second"},
        ]
        assert _extract_text(content) == "first second"

    def test_empty_list(self):
        assert _extract_text([]) == ""


class TestParseJsonl:
    def test_basic_session(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entries = [
                {"type": "user", "message": {"content": "hello"}},
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                },
                {"type": "user", "message": {"content": "follow up"}},
            ]
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            messages, summary, models = _parse_jsonl(path)
            assert len(messages) == 2
            assert messages[0].text == "hello"
            assert messages[1].text == "follow up"
            assert "claude-sonnet-4-20250514" in models
            assert summary == "hello"
        finally:
            os.unlink(path)

    def test_skips_meta_messages(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entries = [
                {"type": "user", "isMeta": True, "message": {"content": "meta"}},
                {"type": "user", "message": {"content": "real"}},
            ]
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            messages, _, _ = _parse_jsonl(path)
            assert len(messages) == 1
            assert messages[0].text == "real"
        finally:
            os.unlink(path)

    def test_skips_synthetic_model(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entries = [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": "<synthetic>",
                        "content": "x",
                    },
                },
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": "x",
                    },
                },
                {"type": "user", "message": {"content": "hi"}},
            ]
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            _, _, models = _parse_jsonl(path)
            assert models == ["claude-sonnet-4-20250514"]
        finally:
            os.unlink(path)

    def test_uses_summary_entry(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entries = [
                {"type": "summary", "summary": "My session summary"},
                {"type": "user", "message": {"content": "hello"}},
            ]
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            _, summary, _ = _parse_jsonl(path)
            assert summary == "My session summary"
        finally:
            os.unlink(path)


class TestClaudeTranscriptRead:
    def test_read_not_found(self):
        with mock.patch("amend_commit_ai.claude._PROJECTS_DIR", Path("/nonexistent")):
            try:
                ClaudeTranscript.read("nope")
                assert False, "should raise"
            except FileNotFoundError:
                pass

    def test_readall_missing_dir(self):
        with mock.patch("amend_commit_ai.claude._PROJECTS_DIR", Path("/nonexistent")):
            assert ClaudeTranscript.readall() == []

    def test_read_and_readall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir)
            session = proj_dir / "test-session.jsonl"
            entries = [
                {"type": "user", "message": {"content": "hello"}},
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": "hi",
                    },
                },
            ]
            _write_jsonl(session, entries)

            with mock.patch("amend_commit_ai.claude._PROJECTS_DIR", proj_dir):
                # readall
                transcripts = ClaudeTranscript.readall()
                assert len(transcripts) == 1
                t = transcripts[0]
                assert t.name == "test-session"
                assert len(t.user_messages) == 1
                assert t.user_messages[0].text == "hello"
                assert "claude-sonnet-4-20250514" in t.models
                assert t.created.tzinfo == timezone.utc

                # read by name
                t2 = ClaudeTranscript.read("test-session")
                assert t2.name == "test-session"

    def test_readall_skips_agent_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir)
            normal = proj_dir / "session.jsonl"
            agent = proj_dir / "agent-abc.jsonl"
            sub_dir = proj_dir / "foo" / "subagents"
            sub_dir.mkdir(parents=True)
            subagent = sub_dir / "agent-xyz.jsonl"

            for p in [normal, agent, subagent]:
                _write_jsonl(p, [{"type": "user", "message": {"content": "hi"}}])

            with mock.patch("amend_commit_ai.claude._PROJECTS_DIR", proj_dir):
                transcripts = ClaudeTranscript.readall()
                names = [t.name for t in transcripts]
                assert "session" in names
                assert "agent-abc" not in names
                assert "agent-xyz" not in names

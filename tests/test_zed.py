"""Tests for ZedTranscript."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

from amend_commit_ai.transcript import UserMessage
from amend_commit_ai.zed import (
    ZedTranscript,
    _decompress,
    _extract_models,
    _extract_user_messages,
    _parse_timestamp,
)


def _make_thread_data(messages=None, model=None):
    data = {"messages": messages or [], "model": model}
    return data


def _create_test_db(path: Path, rows: list[tuple]):
    """Create a test threads.db with given rows.

    Each row: (id, summary, created_at, updated_at, data_type, data_bytes)
    """
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE threads ("
        "  id TEXT PRIMARY KEY,"
        "  summary TEXT NOT NULL,"
        "  updated_at TEXT NOT NULL,"
        "  data_type TEXT NOT NULL,"
        "  data BLOB NOT NULL,"
        "  parent_id TEXT,"
        "  folder_paths TEXT,"
        "  folder_paths_order TEXT,"
        "  created_at TEXT"
        ")"
    )
    for row_id, summary, created_at, updated_at, data_type, data in rows:
        con.execute(
            "INSERT INTO threads (id, summary, created_at, updated_at, data_type, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row_id, summary, created_at, updated_at, data_type, data),
        )
    con.commit()
    con.close()


class TestExtractUserMessages:
    def test_dict_user_messages(self):
        data = _make_thread_data(
            messages=[
                {"User": {"id": "1", "content": [{"Text": "hello"}]}},
                {"Agent": {"content": [{"Text": "response"}]}},
                {"User": {"id": "2", "content": [{"Text": "follow up"}]}},
            ]
        )
        msgs = _extract_user_messages(data)
        assert len(msgs) == 2
        assert msgs[0].text == "hello"
        assert msgs[1].text == "follow up"

    def test_string_messages(self):
        data = _make_thread_data(messages=["hello there"])
        msgs = _extract_user_messages(data)
        assert len(msgs) == 1
        assert msgs[0].text == "hello there"

    def test_empty_messages(self):
        data = _make_thread_data(messages=[])
        assert _extract_user_messages(data) == []

    def test_multiple_text_blocks(self):
        data = _make_thread_data(
            messages=[
                {
                    "User": {
                        "id": "1",
                        "content": [{"Text": "part one"}, {"Text": "part two"}],
                    }
                },
            ]
        )
        msgs = _extract_user_messages(data)
        assert len(msgs) == 1
        assert "part one" in msgs[0].text
        assert "part two" in msgs[0].text


class TestExtractModels:
    def test_dict_model(self):
        data = _make_thread_data(
            model={"provider": "openrouter", "model": "anthropic/claude-opus-4"}
        )
        models, model_providers = _extract_models(data)
        assert models == ["anthropic/claude-opus-4"]
        assert model_providers == {"anthropic/claude-opus-4": "openrouter"}

    def test_string_model(self):
        data = _make_thread_data(model="some-model")
        models, model_providers = _extract_models(data)
        assert models == ["some-model"]
        assert model_providers == {}

    def test_none_model(self):
        data = _make_thread_data(model=None)
        models, model_providers = _extract_models(data)
        assert models == []
        assert model_providers == {}


class TestDecompress:
    def test_plain_json(self):
        data = {"foo": "bar"}
        result = _decompress("json", json.dumps(data).encode())
        assert result == data

    def test_zstd_json(self):
        import zstandard as zstd

        data = {"messages": []}
        compressed = zstd.ZstdCompressor().compress(json.dumps(data).encode())
        result = _decompress("zstd", compressed)
        assert result == data


class TestParseTimestamp:
    def test_iso_format(self):
        dt = _parse_timestamp("2025-06-15T10:30:00+00:00")
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.day == 15

    def test_none_returns_now(self):
        dt = _parse_timestamp(None)
        assert dt.tzinfo is not None


class TestZedTranscriptRead:
    def test_read_not_found_db(self):
        with mock.patch("amend_commit_ai.zed._find_db", side_effect=FileNotFoundError):
            try:
                ZedTranscript.read("nope")
                assert False, "should raise"
            except FileNotFoundError:
                pass

    def test_readall_missing_db(self):
        with mock.patch("amend_commit_ai.zed._find_db", side_effect=FileNotFoundError):
            assert ZedTranscript.readall() == []

    def test_read_and_readall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "threads.db"
            thread_data = _make_thread_data(
                messages=[
                    {"User": {"id": "1", "content": [{"Text": "hello from zed"}]}},
                ],
                model={"provider": "test", "model": "test-model-1"},
            )
            raw = json.dumps(thread_data).encode()
            _create_test_db(
                db_path,
                [
                    (
                        "thread-001",
                        "Test thread",
                        "2025-01-01T00:00:00+00:00",
                        "2025-01-02T00:00:00+00:00",
                        "json",
                        raw,
                    ),
                ],
            )

            with mock.patch("amend_commit_ai.zed._find_db", return_value=db_path):
                # readall
                transcripts = ZedTranscript.readall()
                assert len(transcripts) == 1
                t = transcripts[0]
                assert t.name == "thread-001"
                assert t.summary == "Test thread"
                assert len(t.user_messages) == 1
                assert t.user_messages[0].text == "hello from zed"
                assert t.models == ["test-model-1"]
                assert t.model_providers == {"test-model-1": "test"}

                # read by name
                t2 = ZedTranscript.read("thread-001")
                assert t2.name == "thread-001"

    def test_read_nonexistent_thread(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "threads.db"
            _create_test_db(db_path, [])

            with mock.patch("amend_commit_ai.zed._find_db", return_value=db_path):
                try:
                    ZedTranscript.read("nonexistent")
                    assert False, "should raise"
                except FileNotFoundError:
                    pass

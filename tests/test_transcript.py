"""Tests for the base Transcript model and formatting."""

from datetime import datetime, timezone

from amend_commit_ai.transcript import Transcript, UserMessage, _wrap_markdown


def _make_transcript(**kwargs):
    defaults = dict(
        name="test",
        summary="test transcript",
        created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        modified=datetime(2025, 1, 2, tzinfo=timezone.utc),
        models=["test-model"],
        model_providers={"test-model": "test-provider"},
        user_messages=[],
    )
    defaults.update(kwargs)
    return Transcript(**defaults)


class TestUserMessage:
    def test_basic(self):
        m = UserMessage(text="hello")
        assert m.text == "hello"


class TestTranscript:
    def test_fields(self):
        t = _make_transcript()
        assert t.name == "test"
        assert t.summary == "test transcript"
        assert t.models == ["test-model"]
        assert t.model_providers == {"test-model": "test-provider"}
        assert t.user_messages == []

    def test_read_raises(self):
        try:
            Transcript.read("x")
            assert False, "should raise"
        except NotImplementedError:
            pass

    def test_readall_raises(self):
        try:
            Transcript.readall()
            assert False, "should raise"
        except NotImplementedError:
            pass

    def test_format_transcript_single_message(self):
        t = _make_transcript(user_messages=[UserMessage(text="hello world")])
        result = t.format_transcript()
        assert "hello world" in result

    def test_format_transcript_multiple_messages(self):
        t = _make_transcript(
            user_messages=[
                UserMessage(text="first"),
                UserMessage(text="second"),
            ]
        )
        result = t.format_transcript()
        assert "first" in result
        assert "---" in result
        assert "second" in result

    def test_format_transcript_preserves_code_block(self):
        code = "```python\ndef foo():\n    return 'long string that should not be wrapped at all by the formatter'\n```"
        t = _make_transcript(user_messages=[UserMessage(text=code)])
        result = t.format_transcript()
        # The code block content should not be line-wrapped
        assert "long string that should not be wrapped" in result
        assert "```python" in result

    def test_models_default_empty(self):
        t = Transcript(
            name="t",
            summary="s",
            created=datetime(2025, 1, 1, tzinfo=timezone.utc),
            modified=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert t.models == []
        assert t.model_providers == {}
        assert t.user_messages == []


class TestWrapMarkdown:
    def test_wraps_long_paragraph(self):
        long = "word " * 30
        result = _wrap_markdown(long.strip())
        lines = result.split("\n")
        assert all(len(line) <= 80 for line in lines)  # some slack for word boundaries

    def test_preserves_code_block(self):
        text = "```\nlong line that should not be wrapped at all even if it exceeds seventy two characters\n```"
        result = _wrap_markdown(text)
        assert "long line that should not be wrapped" in result

    def test_fallback_on_empty(self):
        assert _wrap_markdown("") == ""

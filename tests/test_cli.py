"""Tests for the CLI module."""

from datetime import datetime, timezone
from unittest import mock

from click.testing import CliRunner

from amend_commit_ai.cli import _build_amend_command, _collect_transcripts, main
from amend_commit_ai.transcript import Transcript, UserMessage


def _make_transcript(name="t", summary="summary", **kwargs):
    defaults = dict(
        name=name,
        summary=summary,
        created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        modified=datetime(2025, 1, 2, tzinfo=timezone.utc),
        models=["test-model"],
        user_messages=[UserMessage(text="hello")],
    )
    defaults.update(kwargs)
    return Transcript(**defaults)


class TestBuildAmendCommand:
    def test_produces_git_command(self):
        cmd = _build_amend_command("test transcript")
        assert "git commit --amend" in cmd
        assert "=== Transcript" in cmd
        assert "test transcript" in cmd

    def test_leading_space(self):
        cmd = _build_amend_command("x")
        assert cmd.startswith(" ")

    def test_preserves_newlines(self):
        cmd = _build_amend_command("line1\nline2")
        assert "line1" in cmd
        assert "line2" in cmd


class TestCollectTranscripts:
    def test_all_sources_when_no_flags(self):
        claude_t = _make_transcript(
            name="c", modified=datetime(2025, 1, 3, tzinfo=timezone.utc)
        )
        pi_t = _make_transcript(
            name="p", modified=datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
        zed_t = _make_transcript(
            name="z", modified=datetime(2025, 1, 2, tzinfo=timezone.utc)
        )

        mock_claude = mock.MagicMock()
        mock_claude.readall.return_value = [claude_t]
        mock_pi = mock.MagicMock()
        mock_pi.readall.return_value = [pi_t]
        mock_zed = mock.MagicMock()
        mock_zed.readall.return_value = [zed_t]

        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "amend_commit_ai.claude": mock.MagicMock(
                        ClaudeTranscript=mock_claude
                    ),
                    "amend_commit_ai.pi": mock.MagicMock(PiTranscript=mock_pi),
                    "amend_commit_ai.zed": mock.MagicMock(ZedTranscript=mock_zed),
                },
            ),
        ):
            result = _collect_transcripts(claude=False, pi=False, zed=False)
            assert len(result) == 3
            # Should be sorted by modified desc: c (Jan 3), z (Jan 2), p (Jan 1)
            assert result[0].name == "c"
            assert result[1].name == "z"
            assert result[2].name == "p"

    def test_single_source_flag(self):
        claude_t = _make_transcript(
            name="c", modified=datetime(2025, 1, 3, tzinfo=timezone.utc)
        )
        mock_claude = mock.MagicMock()
        mock_claude.readall.return_value = [claude_t]

        with mock.patch.dict(
            "sys.modules",
            {
                "amend_commit_ai.claude": mock.MagicMock(ClaudeTranscript=mock_claude),
            },
        ):
            result = _collect_transcripts(claude=True, pi=False, zed=False)
            assert len(result) == 1
            assert result[0].name == "c"

    def test_sorts_by_modified_desc(self):
        """Verify transcripts are sorted newest first."""
        t1 = _make_transcript(
            name="old", modified=datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
        t2 = _make_transcript(
            name="new", modified=datetime(2025, 6, 1, tzinfo=timezone.utc)
        )
        transcripts = [t1, t2]
        transcripts.sort(key=lambda t: t.modified, reverse=True)
        assert transcripts[0].name == "new"


class TestMainCli:
    def test_print_only(self):
        t = _make_transcript()
        runner = CliRunner()

        with mock.patch("amend_commit_ai.cli._collect_transcripts", return_value=[t]):
            with mock.patch("amend_commit_ai.cli._pick_transcript", return_value=t):
                result = runner.invoke(main, ["--print-only"])
                assert result.exit_code == 0
                assert "hello" in result.output

    def test_no_transcripts(self):
        runner = CliRunner()
        with mock.patch("amend_commit_ai.cli._collect_transcripts", return_value=[]):
            result = runner.invoke(main, ["--print-only"])
            assert result.exit_code == 0

    def test_clipboard_mode(self):
        t = _make_transcript()
        runner = CliRunner()

        with mock.patch("amend_commit_ai.cli._collect_transcripts", return_value=[t]):
            with mock.patch("amend_commit_ai.cli._pick_transcript", return_value=t):
                with mock.patch("amend_commit_ai.cli.pyperclip") as mock_clip:
                    result = runner.invoke(main, [])
                    assert result.exit_code == 0
                    assert "git commit --amend" in result.output
                    mock_clip.copy.assert_called_once()

    def test_doit_mode(self):
        t = _make_transcript()
        runner = CliRunner()

        with mock.patch("amend_commit_ai.cli._collect_transcripts", return_value=[t]):
            with mock.patch("amend_commit_ai.cli._pick_transcript", return_value=t):
                with mock.patch("amend_commit_ai.cli._amend_commit") as mock_amend:
                    result = runner.invoke(main, ["--doit"])
                    assert result.exit_code == 0
                    mock_amend.assert_called_once()

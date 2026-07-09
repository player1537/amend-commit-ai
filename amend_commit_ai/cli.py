"""CLI for amend-commit-ai."""

from __future__ import annotations

import shlex
import subprocess
import sys

import click
import pyperclip
import questionary

from .transcript import Transcript


def _collect_transcripts(
    claude: bool,
    pi: bool,
    zed: bool,
) -> list[Transcript]:
    """Gather transcripts from selected sources (all if none specified)."""
    use_all = not (claude or pi or zed)
    transcripts: list[Transcript] = []

    if use_all or claude:
        from .claude import ClaudeTranscript

        transcripts.extend(ClaudeTranscript.readall())

    if use_all or pi:
        from .pi import PiTranscript

        transcripts.extend(PiTranscript.readall())

    if use_all or zed:
        from .zed import ZedTranscript

        transcripts.extend(ZedTranscript.readall())

    # Sort all by modified time, most recent first
    transcripts.sort(key=lambda t: t.modified, reverse=True)
    return transcripts


def _pick_transcript(transcripts: list[Transcript]) -> Transcript | None:
    if not transcripts:
        click.echo("No transcripts found.", err=True)
        return None

    choices = [
        questionary.Choice(
            title=f"{t.modified:%Y-%m-%d %H:%M}  {t.summary[:50]}",
            value=t,
        )
        for t in transcripts
    ]
    return questionary.select("Select a transcript:", choices=choices).ask()


def _format_co_authored_by_trailers(model_providers: dict[str, str]) -> list[str]:
    """Generate Co-Authored-By trailer strings from model->provider mapping.

    Returns a list of trailer strings such as:
        Co-Authored-By: claude-fable-5 <noreply@claude.invalid>
        Co-Authored-By: kimi-k2.5 <noreply@nebula.invalid>
    """
    trailers: list[str] = []
    for model in sorted(model_providers):
        provider = model_providers[model]
        if provider:
            trailers.append(f"Co-Authored-By: {model} <noreply@{provider}.invalid>")
        else:
            trailers.append(f"Co-Authored-By: {model}")
    return trailers


_DEFAULT_TRAILER = "Co-Authored-By: AI Assistant"


_EMPTY_PROVIDERS: dict[str, str] = {}


def _build_amend_command(
    transcript_text: str, model_providers: dict[str, str] | None = None
) -> str:
    """Build a shell command string that amends the current commit."""
    suffix = shlex.quote(f"\n\n=== Transcript\n\n{transcript_text}")
    cmd = f' git commit --amend -m "$(git log -1 --pretty=%B)"{suffix}'
    trailers = _format_co_authored_by_trailers(model_providers or _EMPTY_PROVIDERS)
    if not trailers:
        trailers = [_DEFAULT_TRAILER]
    for trailer in trailers:
        cmd += f" --trailer {shlex.quote(trailer)}"
    # Leading space prevents bash history recording
    return cmd


def _amend_commit(
    transcript_text: str, model_providers: dict[str, str] | None = None
) -> None:
    """Directly amend the current git commit with the transcript."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%B"],
        capture_output=True,
        text=True,
        check=True,
    )
    current = result.stdout.rstrip()
    lines = current.split("\n")

    # Separate body from git trailers
    trailer_indices: list[int] = []
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if not line:
            break
        if ":" in line and not line.startswith((" ", "\t")):
            trailer_indices.append(i)
        else:
            break

    if trailer_indices:
        first = min(trailer_indices)
        body_end = first - 1 if first > 0 and not lines[first - 1] else first
        body = "\n".join(lines[:body_end]).rstrip()
        trailers = "\n".join(lines[first:])
    else:
        body = current
        trailers = ""

    new_message = f"{body}\n\n=== Transcript\n\n{transcript_text}"
    if trailers:
        new_message = f"{new_message}\n\n{trailers}"

    # Add Co-Authored-By trailers
    co_authored = _format_co_authored_by_trailers(model_providers or _EMPTY_PROVIDERS)
    if not co_authored:
        co_authored = [_DEFAULT_TRAILER]
    for trailer in co_authored:
        new_message = f"{new_message}\n{trailer}"

    subprocess.run(
        ["git", "commit", "--amend", "-F", "-"],
        input=new_message,
        capture_output=True,
        text=True,
        check=True,
    )
    click.echo("\u2713 Commit amended with transcript", err=True)


@click.command()
@click.option(
    "--claude", "use_claude", is_flag=True, help="Include Claude Code sessions."
)
@click.option("--pi", "use_pi", is_flag=True, help="Include Pi agent sessions.")
@click.option("--zed", "use_zed", is_flag=True, help="Include Zed AI threads.")
@click.option(
    "--doit",
    is_flag=True,
    help="Directly amend the commit instead of copying command to clipboard.",
)
@click.option("--print-only", is_flag=True, help="Print the transcript text and exit.")
def main(
    use_claude: bool,
    use_pi: bool,
    use_zed: bool,
    doit: bool,
    print_only: bool,
) -> None:
    """Amend git commits with AI conversation transcripts.

    By default, shows transcripts from all sources and copies an amend
    command to the clipboard.  Pass --claude, --pi, and/or --zed to
    limit sources.  Pass --doit to amend directly.
    """
    transcripts = _collect_transcripts(use_claude, use_pi, use_zed)
    selected = _pick_transcript(transcripts)
    if not selected:
        return

    transcript_text = selected.format_transcript()

    if print_only:
        click.echo(transcript_text)
        return

    if doit:
        _amend_commit(transcript_text, selected.model_providers)
    else:
        cmd = _build_amend_command(transcript_text, selected.model_providers)
        pyperclip.copy(cmd)
        click.echo(cmd)
        click.echo("\n(copied to clipboard)", err=True)

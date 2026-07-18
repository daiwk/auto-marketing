"""Bounded adapter for non-interactive reviews through the local Codex CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from subprocess import DEVNULL
from tempfile import TemporaryDirectory

from quant_trader.llm.base import MessageInput, canonical_messages


class CodexError(RuntimeError):
    """Sanitized local Codex process failure."""


class CodexReviewer:
    """Run one text review using the user's local Codex authentication."""

    def __init__(
        self,
        executable: str = "codex",
        timeout_seconds: float = 120,
        max_output_bytes: int = 65_536,
    ) -> None:
        if not executable.strip() or "\x00" in executable:
            raise ValueError("executable must be nonblank and contain no null byte")
        if timeout_seconds <= 0 or max_output_bytes < 1:
            raise ValueError("timeout and output bound must be positive")
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes

    def check_available(self) -> None:
        """Verify that the CLI can start and has active authentication."""
        self._invoke([self._executable, "login", "status"], input_bytes=None, timeout=5)

    def complete(self, messages: Sequence[MessageInput]) -> str:
        """Return only the final assistant message from an ephemeral Codex run."""
        canonical = canonical_messages(messages)
        prompt = "\n\n".join(
            f"[{message.role.upper()}]\n{message.content}" for message in canonical
        ).encode("utf-8")
        with TemporaryDirectory(prefix="quant-codex-") as directory:
            output_path = Path(directory) / "last-message.txt"
            self._invoke(
                [
                    self._executable,
                    "exec",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--output-last-message",
                    str(output_path),
                    "-",
                ],
                input_bytes=prompt,
                timeout=self._timeout_seconds,
            )
            try:
                with output_path.open("rb") as output_file:
                    raw = output_file.read(self._max_output_bytes + 1)
            except OSError:
                raise CodexError("Codex did not produce a final response") from None

        if len(raw) > self._max_output_bytes:
            raise CodexError("Codex response exceeded the allowed size")
        try:
            result = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise CodexError("Codex response was not valid UTF-8") from None
        if not result.strip():
            raise CodexError("Codex returned an empty response")
        return result

    def _invoke(self, args: list[str], *, input_bytes: bytes | None, timeout: float) -> None:
        try:
            completed = subprocess.run(
                args,
                check=False,
                input=input_bytes,
                shell=False,
                stderr=DEVNULL,
                stdin=DEVNULL if input_bytes is None else None,
                stdout=DEVNULL,
                timeout=timeout,
            )
        except OSError:
            raise CodexError(
                "Codex CLI is unavailable; repair or reinstall it and run codex login"
            ) from None
        except subprocess.TimeoutExpired:
            raise CodexError("Codex review timed out") from None
        if completed.returncode != 0:
            raise CodexError(
                "Codex CLI exited unsuccessfully; repair the CLI and run codex login"
            )

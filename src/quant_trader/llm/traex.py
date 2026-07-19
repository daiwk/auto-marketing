"""Bounded adapter for non-interactive reviews through the local Trae X CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from subprocess import DEVNULL
from tempfile import TemporaryDirectory

from quant_trader.llm.base import MessageInput, canonical_messages


class TraexError(RuntimeError):
    """Sanitized local Trae X process failure."""


class TraexReviewer:
    """Run one text review using the user's local Trae X authentication."""

    def __init__(
        self,
        executable: str = "traex",
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
        """Return only the final assistant message from an isolated Trae X run."""
        canonical = canonical_messages(messages)
        prompt = "\n\n".join(
            f"[{message.role.upper()}]\n{message.content}" for message in canonical
        ).encode("utf-8")
        with TemporaryDirectory(prefix="quant-traex-") as directory:
            output_path = Path(directory) / "last-message.txt"
            self._invoke(
                [
                    self._executable,
                    "exec",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "-C",
                    directory,
                    "--output-last-message",
                    str(output_path),
                    "-",
                ],
                input_bytes=prompt,
                timeout=self._timeout_seconds,
            )
            try:
                raw = output_path.read_bytes()[: self._max_output_bytes + 1]
            except OSError:
                raise TraexError("Trae X did not produce a final response") from None

        if len(raw) > self._max_output_bytes:
            raise TraexError("Trae X response exceeded the allowed size")
        try:
            result = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise TraexError("Trae X response was not valid UTF-8") from None
        if not result.strip():
            raise TraexError("Trae X returned an empty response")
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
            raise TraexError(
                "Trae X CLI is unavailable; repair or reinstall it and run traex login"
            ) from None
        except subprocess.TimeoutExpired:
            raise TraexError("Trae X review timed out") from None
        if completed.returncode != 0:
            raise TraexError(
                "Trae X CLI exited unsuccessfully; repair the CLI and run traex login"
            )

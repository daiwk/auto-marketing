"""Bounded adapter for non-interactive reviews through the local Trae X CLI."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from subprocess import DEVNULL
from tempfile import TemporaryDirectory, TemporaryFile

from quant_trader.llm.base import MessageInput, canonical_messages

_ERROR_ORIGIN = object()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SECRET = re.compile(
    r"(?i)\b(?:authorization|api[-_ ]?key|bearer)\b\s*[:=]?\s*(?:bearer\s+)?\S+"
)
_MAX_DIAGNOSTIC_CHARS = 500


class TraexError(RuntimeError):
    """Sanitized local Trae X process failure."""

    def __init__(self, message: str, *, _origin: object | None = None) -> None:
        safe = message if _origin is _ERROR_ORIGIN else "Trae X provider failed"
        super().__init__(safe)
        self.safe_message = safe


def _error(message: str) -> TraexError:
    return TraexError(message, _origin=_ERROR_ORIGIN)


def _diagnostic(raw: bytes, input_bytes: bytes | None) -> str:
    """Return a short CLI diagnostic without prompts, ANSI escapes, or obvious credentials."""
    text = raw.decode("utf-8", errors="replace")
    if input_bytes:
        prompt = input_bytes.decode("utf-8", errors="replace")
        text = text.replace(prompt, "[prompt redacted]")
    lines = [
        line
        for line in _ANSI_ESCAPE.sub("", text).splitlines()
        if "[SYSTEM]" not in line and "[USER]" not in line
    ]
    compact = " ".join(" ".join(lines).split())
    compact = _SECRET.sub("[credential redacted]", compact)
    home = str(Path.home())
    compact = compact.replace(home, "~")
    return compact[:_MAX_DIAGNOSTIC_CHARS]


class TraexReviewer:
    """Run one text review using the user's local Trae X authentication."""

    def __init__(
        self,
        executable: str = "traex",
        model: str = "gpt-5.5",
        timeout_seconds: float = 120,
        max_output_bytes: int = 65_536,
    ) -> None:
        if not executable.strip() or "\x00" in executable:
            raise ValueError("executable must be nonblank and contain no null byte")
        if not model.strip() or model != model.strip() or "\x00" in model:
            raise ValueError("model must be nonblank, trimmed, and contain no null byte")
        if timeout_seconds <= 0 or max_output_bytes < 1:
            raise ValueError("timeout and output bound must be positive")
        self._executable = executable
        self._model = model
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
                    "--model",
                    self._model,
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
                raise _error("Trae X did not produce a final response") from None

        if len(raw) > self._max_output_bytes:
            raise _error("Trae X response exceeded the allowed size")
        try:
            result = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise _error("Trae X response was not valid UTF-8") from None
        if not result.strip():
            raise _error("Trae X returned an empty response")
        return result

    def _invoke(self, args: list[str], *, input_bytes: bytes | None, timeout: float) -> None:
        with TemporaryFile() as error_file:
            try:
                completed = subprocess.run(
                    args,
                    check=False,
                    input=input_bytes,
                    shell=False,
                    stderr=error_file,
                    stdin=DEVNULL if input_bytes is None else None,
                    stdout=DEVNULL,
                    timeout=timeout,
                )
            except OSError:
                raise _error(
                    "Trae X CLI is unavailable; repair or reinstall it and run traex login"
                ) from None
            except subprocess.TimeoutExpired:
                error_file.seek(0)
                detail = _diagnostic(error_file.read(), input_bytes)
                suffix = f": {detail}" if detail else ""
                raise _error(
                    f"Trae X review timed out after {timeout:g} seconds{suffix}"
                ) from None
            if completed.returncode != 0:
                error_file.seek(0)
                detail = _diagnostic(error_file.read(), input_bytes)
                suffix = f": {detail}" if detail else " (no CLI diagnostic was emitted)"
                raise _error(f"Trae X CLI exited with code {completed.returncode}{suffix}")

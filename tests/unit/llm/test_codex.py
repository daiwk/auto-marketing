from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from quant_trader.llm.base import ChatMessage
from quant_trader.llm.codex import CodexError, CodexReviewer


def _success_response(args: list[str], content: bytes = b'{"action":"maintain"}') -> None:
    output_path = Path(args[args.index("--output-last-message") + 1])
    output_path.write_bytes(content)


def test_codex_uses_logged_in_read_only_ephemeral_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        if args != ["codex", "login", "status"]:
            _success_response(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("quant_trader.llm.codex.subprocess.run", run)
    reviewer = CodexReviewer(timeout_seconds=12)

    reviewer.check_available()
    result = reviewer.complete(
        (
            ChatMessage(role="system", content="Return JSON."),
            ChatMessage(role="user", content="Review this."),
        )
    )

    assert calls[0][0] == ["codex", "login", "status"]
    assert calls[0][1] == {
        "check": False,
        "input": None,
        "shell": False,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "timeout": 5,
    }
    exec_args, exec_kwargs = calls[1]
    assert exec_args[:7] == [
        "codex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--output-last-message",
    ]
    assert exec_args[-1] == "-"
    assert exec_kwargs["shell"] is False
    assert exec_kwargs["stdin"] is None
    assert exec_kwargs["timeout"] == 12
    assert exec_kwargs["input"] == b"[SYSTEM]\nReturn JSON.\n\n[USER]\nReview this."
    assert result == '{"action":"maintain"}'


@pytest.mark.parametrize(
    ("raised", "message"),
    [
        (FileNotFoundError(), "unavailable"),
        (subprocess.TimeoutExpired(["codex"], 1), "timed out"),
    ],
)
def test_codex_sanitizes_process_failures(
    monkeypatch: pytest.MonkeyPatch, raised: BaseException, message: str
) -> None:
    secret = "prompt-that-must-not-leak"

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise raised

    monkeypatch.setattr("quant_trader.llm.codex.subprocess.run", run)

    with pytest.raises(CodexError, match=message) as error:
        CodexReviewer().complete((ChatMessage(role="user", content=secret),))

    assert secret not in str(error.value)
    assert error.value.__cause__ is None


def test_codex_reports_broken_or_logged_out_cli_without_stderr_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ENOENT internal/provider/secret"

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 1, stderr=secret.encode())

    monkeypatch.setattr("quant_trader.llm.codex.subprocess.run", run)

    with pytest.raises(CodexError, match="repair.*codex login") as error:
        CodexReviewer().check_available()

    assert secret not in str(error.value)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "did not produce"),
        (b" \n", "empty response"),
        (b"x" * 65_537, "allowed size"),
        (b"\xff", "valid UTF-8"),
    ],
)
def test_codex_rejects_invalid_final_responses(
    monkeypatch: pytest.MonkeyPatch, content: bytes | None, message: str
) -> None:
    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if content is not None:
            _success_response(args, content)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("quant_trader.llm.codex.subprocess.run", run)

    with pytest.raises(CodexError, match=message):
        CodexReviewer().complete((ChatMessage(role="user", content="review"),))


def test_codex_validates_inputs_before_starting_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr("quant_trader.llm.codex.subprocess.run", run)

    with pytest.raises(ValueError, match="at least one"):
        CodexReviewer().complete(())
    with pytest.raises(ValueError, match="executable"):
        CodexReviewer(executable=" ")
    with pytest.raises(ValueError, match="positive"):
        CodexReviewer(timeout_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        CodexReviewer(max_output_bytes=0)

    assert calls == 0

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from quant_trader.llm.base import ChatMessage
from quant_trader.llm.traex import TraexError, TraexReviewer


def _write_response(args: list[str], content: bytes = b'{"action":"maintain"}') -> None:
    Path(args[args.index("--output-last-message") + 1]).write_bytes(content)


def test_traex_uses_logged_in_read_only_ephemeral_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        if args != ["traex", "login", "status"]:
            _write_response(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("quant_trader.llm.traex.subprocess.run", run)
    reviewer = TraexReviewer(timeout_seconds=12)
    reviewer.check_available()
    result = reviewer.complete((ChatMessage(role="user", content="Review this."),))

    assert calls[0][0] == ["traex", "login", "status"]
    args, kwargs = calls[1]
    assert args[:7] == [
        "traex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "-C",
    ]
    assert args[-1] == "-"
    assert kwargs["input"] == b"[USER]\nReview this."
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 12
    assert result == '{"action":"maintain"}'


@pytest.mark.parametrize(
    ("raised", "message"),
    [
        (FileNotFoundError(), "unavailable"),
        (subprocess.TimeoutExpired(["traex"], 1), "timed out"),
    ],
)
def test_traex_sanitizes_process_failures(
    monkeypatch: pytest.MonkeyPatch, raised: BaseException, message: str
) -> None:
    secret = "prompt-that-must-not-leak"

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise raised

    monkeypatch.setattr("quant_trader.llm.traex.subprocess.run", run)
    with pytest.raises(TraexError, match=message) as error:
        TraexReviewer().complete((ChatMessage(role="user", content=secret),))
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "did not produce"),
        (b" ", "empty response"),
        (b"x" * 65_537, "allowed size"),
        (b"\xff", "valid UTF-8"),
    ],
)
def test_traex_rejects_invalid_responses(
    monkeypatch: pytest.MonkeyPatch, content: bytes | None, message: str
) -> None:
    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if content is not None:
            _write_response(args, content)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("quant_trader.llm.traex.subprocess.run", run)
    with pytest.raises(TraexError, match=message):
        TraexReviewer().complete((ChatMessage(role="user", content="review"),))

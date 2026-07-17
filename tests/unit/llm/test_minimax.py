from __future__ import annotations

import json

import httpx
import pytest

from quant_trader.llm.base import ChatMessage
from quant_trader.llm.minimax import MiniMaxError, MiniMaxReviewer


def response(content: str = "{}") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_minimax_sends_exact_request_and_does_not_mutate_messages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response("review")

    messages = [{"role": "user", "content": "review"}]
    reviewer = MiniMaxReviewer(
        "secret-token",
        "https://api.minimax.io/v1",
        "MiniMax-M2.7",
        3,
        0,
        transport=httpx.MockTransport(handler),
    )

    assert reviewer.complete(messages) == "review"
    assert messages == [{"role": "user", "content": "review"}]
    assert str(seen[0].url) == "https://api.minimax.io/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer secret-token"
    assert seen[0].headers["content-type"] == "application/json"
    assert json.loads(seen[0].content) == {
        "model": "MiniMax-M2.7",
        "messages": messages,
        "temperature": 0.1,
        "max_completion_tokens": 1200,
        "stream": False,
    }
    reviewer.close()


@pytest.mark.parametrize("status", [429, 500])
def test_minimax_retries_retryable_http_statuses(status: int) -> None:
    calls = 0
    delays: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status) if calls == 1 else response()

    reviewer = MiniMaxReviewer(
        "secret",
        "https://api.minimax.io/v1",
        "model",
        10,
        1,
        transport=httpx.MockTransport(handler),
        sleeper=delays.append,
    )

    assert reviewer.complete([ChatMessage(role="user", content="x")]) == "{}"
    assert calls == 2
    assert delays == [1]


def test_minimax_retries_transport_errors_and_honors_capped_numeric_retry_after() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("network unavailable", request=request)
        if calls == 2:
            return httpx.Response(429, headers={"Retry-After": "99"})
        return response()

    reviewer = MiniMaxReviewer(
        "secret",
        "https://api.minimax.io/v1",
        "model",
        3,
        2,
        transport=httpx.MockTransport(handler),
        sleeper=delays.append,
    )

    assert reviewer.complete([{"role": "user", "content": "x"}]) == "{}"
    assert calls == 3
    assert delays == [1, 3]


def test_minimax_fails_without_retrying_terminal_or_malformed_responses() -> None:
    calls = 0
    secret = "only-for-test"
    body = "internal-body-that-must-not-leak"

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text=body)

    reviewer = MiniMaxReviewer(
        secret, "https://api.minimax.io/v1", "model", 1, 2, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(MiniMaxError) as error:
        reviewer.complete([{"role": "user", "content": "x"}])
    assert calls == 1
    assert error.value.__cause__ is not None
    assert secret not in str(error.value)
    assert body not in str(error.value)

    malformed = MiniMaxReviewer(
        secret,
        "https://api.minimax.io/v1",
        "model",
        1,
        2,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": []})),
    )
    with pytest.raises(MiniMaxError):
        malformed.complete([{"role": "user", "content": "x"}])


def test_minimax_terminal_exhaustion_and_client_ownership() -> None:
    delays: list[float] = []
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    reviewer = MiniMaxReviewer(
        "secret",
        "https://api.minimax.io/v1",
        "model",
        4,
        2,
        transport=httpx.MockTransport(handler),
        sleeper=delays.append,
    )
    with pytest.raises(MiniMaxError) as error:
        reviewer.complete([{"role": "user", "content": "x"}])
    assert error.value.status_code == 503
    assert error.value.attempts == 3
    assert attempts == 3
    assert delays == [1, 2]
    reviewer.close()
    assert reviewer.client.is_closed

    supplied = httpx.Client(transport=httpx.MockTransport(lambda _: response()))
    borrowed = MiniMaxReviewer("secret", client=supplied)
    borrowed.close()
    assert not supplied.is_closed
    supplied.close()


@pytest.mark.parametrize(
    "bad",
    [
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": 3}}]},
        {"choices": "not-a-list"},
    ],
)
def test_minimax_rejects_invalid_success_shape_without_response_leak(bad: object) -> None:
    reviewer = MiniMaxReviewer(
        "secret", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=bad))
    )
    with pytest.raises(MiniMaxError) as error:
        reviewer.complete([{"role": "user", "content": "x"}])
    assert "secret" not in str(error.value)


def test_minimax_retries_timeout_and_rejects_malformed_success_json() -> None:
    calls = 0
    delays: list[float] = []

    def timeout_then_success(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return response()

    reviewer = MiniMaxReviewer(
        "secret",
        max_retries=1,
        transport=httpx.MockTransport(timeout_then_success),
        sleeper=delays.append,
    )
    assert reviewer.complete([{"role": "user", "content": "x"}]) == "{}"
    assert delays == [1]

    malformed = MiniMaxReviewer(
        "secret", transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not json"))
    )
    with pytest.raises(MiniMaxError, match="valid JSON"):
        malformed.complete([{"role": "user", "content": "x"}])


def test_minimax_rejects_invalid_constructor_values() -> None:
    with pytest.raises((TypeError, ValueError)):
        MiniMaxReviewer("", timeout_seconds=1)
    with pytest.raises((TypeError, ValueError)):
        MiniMaxReviewer("key", timeout_seconds=True)  # type: ignore[arg-type]

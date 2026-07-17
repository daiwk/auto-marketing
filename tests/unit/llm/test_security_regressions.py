from __future__ import annotations

from collections.abc import Iterator, Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import httpx
import pytest

import quant_trader.llm.minimax as minimax_module
from quant_trader.llm.minimax import MAX_RESPONSE_BYTES, MiniMaxError, MiniMaxReviewer

_KEY = "regression-api-key-DO-NOT-EXPOSE"
_URL = "https://regression-user:regression-password@api.minimax.io/v1?token=unsafe"
_BODY = "regression-provider-body-DO-NOT-EXPOSE"
_AUTHORIZATION = f"Bearer {_KEY}"
_SOURCE_ROOT = Path(__file__).parents[3] / "src" / "quant_trader"
_MESSAGES = [{"role": "user", "content": "review"}]


class CountingStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], *, read_error: BaseException | None = None) -> None:
        self._chunks = chunks
        self._read_error = read_error
        self.reads = 0
        self.bytes_read = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self.reads += 1
            self.bytes_read += len(chunk)
            yield chunk
        if self._read_error is not None:
            raise self._read_error

    def close(self) -> None:
        self.closed = True


class RawOnlyResponse(httpx.Response):
    def iter_bytes(self, *args: object, **kwargs: object) -> Iterator[bytes]:
        raise AssertionError("decoded response streaming must not be used")


class SecretBearingOption:
    def __init__(self) -> None:
        self.secret = _BODY


class SecretBearingTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.secret = _BODY

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"transport must not be called: {request!r}")


def _success_bytes(content: str = "ok") -> bytes:
    return ('{"choices":[{"message":{"content":"' + content + '"}}]}').encode()


def _assert_sanitized_exception_graph(error: BaseException, *additional_forbidden: str) -> None:
    """Recursively inspect retained exception data and every library traceback local."""
    forbidden = (
        _KEY,
        _URL,
        _BODY,
        _AUTHORIZATION,
        "Authorization",
        *additional_forbidden,
    )
    seen_values: set[int] = set()
    seen_exceptions: set[int] = set()

    def inspect_value(value: object) -> None:
        if isinstance(value, str):
            assert all(token not in value for token in forbidden)
            return
        if isinstance(value, bytes):
            inspect_value(value.decode("utf-8", errors="replace"))
            return
        if isinstance(value, httpx.Request | httpx.Response | JSONDecodeError):
            pytest.fail(f"unsafe object retained: {type(value).__name__}")
        if value is None or isinstance(value, int | float | bool):
            return
        identity = id(value)
        if identity in seen_values:
            return
        seen_values.add(identity)
        if isinstance(value, Mapping):
            for key, item in value.items():
                inspect_value(key)
                inspect_value(item)
            return
        if isinstance(value, tuple | list | set | frozenset):
            for item in value:
                inspect_value(item)
            return
        try:
            attributes = vars(value)
        except TypeError:
            return
        inspect_value(attributes)

    def inspect_exception(current: BaseException | None) -> None:
        if current is None or id(current) in seen_exceptions:
            return
        seen_exceptions.add(id(current))
        assert not isinstance(current, JSONDecodeError)
        assert not hasattr(current, "request")
        assert not hasattr(current, "response")
        inspect_value(str(current))
        inspect_value(repr(current))
        inspect_value(current.args)
        inspect_value(vars(current))
        inspect_exception(current.__cause__)
        inspect_exception(current.__context__)

    def inspect_tracebacks(current: BaseException | None, visited: set[int]) -> None:
        if current is None or id(current) in visited:
            return
        visited.add(id(current))
        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            filename = Path(frame.f_code.co_filename)
            if _SOURCE_ROOT == filename.parent or _SOURCE_ROOT in filename.parents:
                inspect_value(frame.f_locals)
            traceback = traceback.tb_next
        inspect_tracebacks(current.__cause__, visited)
        inspect_tracebacks(current.__context__, visited)

    inspect_exception(error)
    inspect_tracebacks(error, set())


@pytest.mark.parametrize("invalid_option", ["sleeper", "client", "client_and_transport"])
def test_invalid_constructor_dependencies_do_not_survive_in_traceback_locals(
    invalid_option: str,
) -> None:
    client: httpx.Client | None = None
    option = SecretBearingOption()
    kwargs: dict[str, object]
    if invalid_option == "sleeper":
        kwargs = {"sleeper": option}
    elif invalid_option == "client":
        kwargs = {"client": option}
    else:
        client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        kwargs = {"client": client, "transport": SecretBearingTransport()}

    try:
        with pytest.raises(MiniMaxError) as raised:
            MiniMaxReviewer(_KEY, **kwargs)  # type: ignore[arg-type]
        _assert_sanitized_exception_graph(raised.value)
    finally:
        if client is not None:
            client.close()


@pytest.mark.parametrize(
    "unsafe_key",
    [
        " leading",
        "trailing ",
        "embedded\rreturn",
        "embedded\nnewline",
        "embedded\ttab",
        "embedded\x7fdelete",
        "non-ascii-\N{SNOWMAN}",
    ],
    ids=["leading-space", "trailing-space", "cr", "lf", "tab", "del", "non-ascii"],
)
def test_header_unsafe_api_keys_are_rejected_before_client_or_request(
    unsafe_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client_constructions = 0

    def unexpected_client(*args: object, **kwargs: object) -> None:
        nonlocal client_constructions
        client_constructions += 1

    monkeypatch.setattr(minimax_module.httpx, "Client", unexpected_client)
    with pytest.raises(MiniMaxError) as raised:
        MiniMaxReviewer(unsafe_key)
    assert client_constructions == 0
    _assert_sanitized_exception_graph(raised.value, unsafe_key)


def test_complete_after_owned_client_close_raises_only_a_sanitized_minimax_error() -> None:
    reviewer = MiniMaxReviewer(
        _KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, content=_success_bytes()))
    )
    reviewer.close()

    with pytest.raises(MiniMaxError) as raised:
        reviewer.complete(_MESSAGES)
    _assert_sanitized_exception_graph(raised.value)


@pytest.mark.parametrize("hook_name", ["request", "response"])
def test_ordinary_event_hook_runtime_errors_are_sanitized(hook_name: str) -> None:
    def fail_hook(_: httpx.Request | httpx.Response) -> None:
        raise RuntimeError(f"{_URL} {_BODY} {_AUTHORIZATION}")

    stream = CountingStream([_success_bytes()])
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)),
        event_hooks={hook_name: [fail_hook]},
    )
    reviewer = MiniMaxReviewer(_KEY, client=client, max_retries=0)
    try:
        with pytest.raises(MiniMaxError) as raised:
            reviewer.complete(_MESSAGES)
        _assert_sanitized_exception_graph(raised.value)
        if hook_name == "response":
            assert stream.closed
    finally:
        client.close()


def test_forged_hook_error_and_corrupted_retry_state_cannot_break_safe_metadata() -> None:
    forged = MiniMaxError(_BODY, status_code=500, attempts=1)

    def corrupt_then_fail(_: httpx.Request) -> None:
        reviewer._max_retries = SecretBearingOption()  # type: ignore[assignment]
        raise forged

    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=_success_bytes())),
        event_hooks={"request": [corrupt_then_fail]},
    )
    reviewer = MiniMaxReviewer(_KEY, client=client, max_retries=1)
    try:
        with pytest.raises(MiniMaxError) as raised:
            reviewer.complete(_MESSAGES)
        assert raised.value.status_code is None
        assert raised.value.attempts == 0
        _assert_sanitized_exception_graph(raised.value)
    finally:
        client.close()


def test_request_requires_exact_identity_accept_encoding() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_success_bytes())

    reviewer = MiniMaxReviewer(_KEY, transport=httpx.MockTransport(handler))
    try:
        assert reviewer.complete(_MESSAGES) == "ok"
        assert seen[0].headers["Accept-Encoding"] == "identity"
    finally:
        reviewer.close()


@pytest.mark.parametrize("encoding", ["gzip", "br", "brotli", "gzip, br"])
def test_encoded_responses_are_rejected_without_consuming_body_and_are_closed(
    encoding: str,
) -> None:
    stream = CountingStream([_success_bytes()])
    reviewer = MiniMaxReviewer(
        _KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"Content-Encoding": encoding}, stream=stream)
        ),
    )
    try:
        with pytest.raises(MiniMaxError) as raised:
            reviewer.complete(_MESSAGES)
        assert stream.reads == 0
        assert stream.closed
        _assert_sanitized_exception_graph(raised.value)
    finally:
        reviewer.close()


def test_oversized_declared_content_length_rejects_without_reading_and_closes() -> None:
    stream = CountingStream([_success_bytes()])
    reviewer = MiniMaxReviewer(
        _KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)},
                stream=stream,
            )
        ),
    )
    try:
        with pytest.raises(MiniMaxError) as raised:
            reviewer.complete(_MESSAGES)
        assert stream.reads == 0
        assert stream.closed
        _assert_sanitized_exception_graph(raised.value)
    finally:
        reviewer.close()


@pytest.mark.parametrize("declared_length", ["not-a-number", "-7"])
def test_malformed_or_negative_content_length_is_safely_bounded_and_closed(
    declared_length: str,
) -> None:
    stream = CountingStream([_success_bytes()])
    reviewer = MiniMaxReviewer(
        _KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"Content-Length": declared_length}, stream=stream
            )
        ),
    )
    try:
        assert reviewer.complete(_MESSAGES) == "ok"
        assert stream.bytes_read == len(_success_bytes())
        assert stream.closed
    finally:
        reviewer.close()


def test_dishonest_small_content_length_is_bounded_by_actual_raw_bytes_and_closed() -> None:
    chunk = b"x" * 8_192
    stream = CountingStream([chunk] * ((MAX_RESPONSE_BYTES // len(chunk)) + 2))
    reviewer = MiniMaxReviewer(
        _KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"Content-Length": "1"}, stream=stream)
        ),
    )
    try:
        with pytest.raises(MiniMaxError) as raised:
            reviewer.complete(_MESSAGES)
        assert MAX_RESPONSE_BYTES < stream.bytes_read <= MAX_RESPONSE_BYTES + len(chunk)
        assert stream.closed
        _assert_sanitized_exception_graph(raised.value)
    finally:
        reviewer.close()


def test_custom_byte_stream_uses_raw_not_decoded_streaming_with_bounded_consumption() -> None:
    stream = CountingStream([_success_bytes()])
    reviewer = MiniMaxReviewer(
        _KEY,
        transport=httpx.MockTransport(lambda _: RawOnlyResponse(200, stream=stream)),
    )
    try:
        assert reviewer.complete(_MESSAGES) == "ok"
        assert stream.reads == 1
        assert stream.bytes_read <= MAX_RESPONSE_BYTES
        assert stream.closed
    finally:
        reviewer.close()


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "retryable-status",
        "terminal-status",
        "malformed-json",
        "raw-read-error",
        "streamed-oversize",
        "encoded-response",
        "declared-oversize",
    ],
)
def test_every_response_lifecycle_closes_the_stream(scenario: str) -> None:
    streams: list[CountingStream] = []

    def make_response() -> httpx.Response:
        if scenario == "retryable-status" and not streams:
            stream = CountingStream([_BODY.encode()])
            response = httpx.Response(503, stream=stream)
        elif scenario == "terminal-status":
            stream = CountingStream([_BODY.encode()])
            response = httpx.Response(400, stream=stream)
        elif scenario == "malformed-json":
            stream = CountingStream([b"not-json"])
            response = httpx.Response(200, stream=stream)
        elif scenario == "raw-read-error":
            stream = CountingStream([], read_error=RuntimeError(_BODY))
            response = httpx.Response(200, stream=stream)
        elif scenario == "streamed-oversize":
            stream = CountingStream([b"x" * (MAX_RESPONSE_BYTES + 1)])
            response = httpx.Response(200, stream=stream)
        elif scenario == "encoded-response":
            stream = CountingStream([_success_bytes()])
            response = httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)
        elif scenario == "declared-oversize":
            stream = CountingStream([_success_bytes()])
            response = httpx.Response(
                200,
                headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)},
                stream=stream,
            )
        else:
            if scenario == "retryable-status":
                assert streams[0].closed, "retry response must close before the next request"
            stream = CountingStream([_success_bytes()])
            response = httpx.Response(200, stream=stream)
        streams.append(stream)
        return response

    reviewer = MiniMaxReviewer(
        _KEY,
        max_retries=1 if scenario == "retryable-status" else 0,
        transport=httpx.MockTransport(lambda _: make_response()),
        sleeper=lambda _: None,
    )
    try:
        if scenario in {"success", "retryable-status"}:
            assert reviewer.complete(_MESSAGES) == "ok"
        else:
            with pytest.raises(MiniMaxError) as raised:
                reviewer.complete(_MESSAGES)
            _assert_sanitized_exception_graph(raised.value)
        assert streams
        assert all(stream.closed for stream in streams)
    finally:
        reviewer.close()


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "retryable-status",
        "terminal-status",
        "malformed-json",
        "raw-read-error",
        "streamed-oversize",
        "encoded-response",
        "declared-oversize",
        "transport-error",
        "request-hook-error",
        "response-hook-error",
    ],
)
def test_borrowed_client_remains_open_after_success_and_each_failure_category(
    scenario: str,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if scenario == "transport-error":
            raise httpx.ConnectError(_BODY, request=request)
        if scenario == "retryable-status" and calls == 1:
            return httpx.Response(503, stream=CountingStream([_BODY.encode()]))
        if scenario == "terminal-status":
            return httpx.Response(400, stream=CountingStream([_BODY.encode()]))
        if scenario == "malformed-json":
            return httpx.Response(200, stream=CountingStream([b"not-json"]))
        if scenario == "raw-read-error":
            return httpx.Response(200, stream=CountingStream([], read_error=RuntimeError(_BODY)))
        if scenario == "streamed-oversize":
            return httpx.Response(200, stream=CountingStream([b"x" * (MAX_RESPONSE_BYTES + 1)]))
        if scenario == "encoded-response":
            return httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=CountingStream([_success_bytes()]),
            )
        if scenario == "declared-oversize":
            return httpx.Response(
                200,
                headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)},
                stream=CountingStream([_success_bytes()]),
            )
        return httpx.Response(200, stream=CountingStream([_success_bytes()]))

    def fail_hook(_: httpx.Request | httpx.Response) -> None:
        raise RuntimeError(_BODY)

    hooks: dict[str, list[Any]] = {}
    if scenario == "request-hook-error":
        hooks["request"] = [fail_hook]
    elif scenario == "response-hook-error":
        hooks["response"] = [fail_hook]
    client = httpx.Client(transport=httpx.MockTransport(handler), event_hooks=hooks)
    reviewer = MiniMaxReviewer(
        _KEY,
        max_retries=1 if scenario == "retryable-status" else 0,
        client=client,
        sleeper=lambda _: None,
    )
    try:
        if scenario in {"success", "retryable-status"}:
            assert reviewer.complete(_MESSAGES) == "ok"
        else:
            with pytest.raises(MiniMaxError) as raised:
                reviewer.complete(_MESSAGES)
            _assert_sanitized_exception_graph(raised.value)
        reviewer.close()
        assert not client.is_closed
    finally:
        client.close()

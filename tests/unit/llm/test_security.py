from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import httpx
import pytest

from quant_trader.llm.minimax import MiniMaxError, MiniMaxReviewer
from quant_trader.llm.parsing import LLMResponseError, parse_review

_KEY = "security-test-api-key-DO-NOT-EXPOSE"
_URL = "https://user:security-test-url-token@api.minimax.io/v1?key=security-test-url-token"
_BODY = "security-test-response-body-DO-NOT-EXPOSE"
_LIBRARY_ROOT = Path(__file__).parents[3] / "src" / "quant_trader"


def _assert_sanitized_exception_graph(error: BaseException, *forbidden: str) -> None:
    seen: set[int] = set()

    def inspect_value(value: Any) -> None:
        if isinstance(value, str):
            assert all(token not in value for token in forbidden)
        elif isinstance(value, httpx.Request | httpx.Response | JSONDecodeError):
            pytest.fail(f"unsafe object retained: {type(value).__name__}")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                inspect_value(key)
                inspect_value(item)
        elif isinstance(value, tuple | list | set):
            for item in value:
                inspect_value(item)

    def inspect_exception(current: BaseException | None) -> None:
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        assert not hasattr(current, "request")
        assert not hasattr(current, "response")
        assert not isinstance(current, JSONDecodeError)
        inspect_value(str(current))
        inspect_value(repr(current))
        inspect_value(current.args)
        inspect_value(vars(current))
        inspect_exception(current.__cause__)
        inspect_exception(current.__context__)

    def inspect_library_traceback(current: BaseException | None) -> None:
        if current is None:
            return
        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            filename = Path(frame.f_code.co_filename)
            if _LIBRARY_ROOT in filename.parents:
                inspect_value(frame.f_locals)
                for value in frame.f_locals.values():
                    try:
                        inspect_value(vars(value))
                    except TypeError:
                        pass
            traceback = traceback.tb_next
        inspect_library_traceback(current.__cause__)
        inspect_library_traceback(current.__context__)

    inspect_exception(error)
    inspect_library_traceback(error)


def test_constructor_validation_and_http_base_url_are_sanitized_and_fail_closed() -> None:
    with pytest.raises(MiniMaxError) as invalid_url:
        MiniMaxReviewer(_KEY, base_url=_URL)
    _assert_sanitized_exception_graph(invalid_url.value, _KEY, _URL, _BODY)

    with pytest.raises(MiniMaxError) as insecure_url:
        MiniMaxReviewer(_KEY, base_url="http://api.minimax.io/v1")
    _assert_sanitized_exception_graph(insecure_url.value, _KEY, _URL, _BODY)


@pytest.mark.parametrize("kind", ["transport", "json", "schema"])
def test_provider_and_parser_errors_have_no_raw_secret_exception_in_the_chain(kind: str) -> None:
    if kind == "transport":

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(_BODY, request=request)

        reviewer = MiniMaxReviewer(_KEY, max_retries=0, transport=httpx.MockTransport(handler))
        with pytest.raises(MiniMaxError) as error:
            reviewer.complete([{"role": "user", "content": "x"}])
    elif kind == "json":
        reviewer = MiniMaxReviewer(
            _KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, text=_BODY))
        )
        with pytest.raises(MiniMaxError) as error:
            reviewer.complete([{"role": "user", "content": "x"}])
    else:
        content = (
            '{"action":"invalid","weight_multiplier":0.5,"confidence":0.5,'
            f'"thesis":"{_BODY}","risks":[],"invalidation":"break","input_anomalies":[]}}'
        )
        with pytest.raises(LLMResponseError) as error:
            parse_review(content)
    _assert_sanitized_exception_graph(error.value, _KEY, _URL, _BODY)


def test_parse_review_sanitizes_json_decode_error_and_recursion_error() -> None:
    with pytest.raises(LLMResponseError) as malformed:
        parse_review('{"thesis":"' + _BODY)
    _assert_sanitized_exception_graph(malformed.value, _KEY, _URL, _BODY)

    deeply_nested = "{" * 1_100 + "0" + "}" * 1_100
    with pytest.raises(LLMResponseError) as recursive:
        parse_review(deeply_nested)
    _assert_sanitized_exception_graph(recursive.value, _KEY, _URL, _BODY)


def test_invalid_caller_message_does_not_chain_its_content() -> None:
    reviewer = MiniMaxReviewer(_KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    with pytest.raises(ValueError) as error:
        reviewer.complete([{"role": "user", "content": "x", "untrusted": _BODY}])
    _assert_sanitized_exception_graph(error.value, _KEY, _URL, _BODY)


def test_provider_failure_clears_library_traceback_locals() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(_BODY, request=request)

    reviewer = MiniMaxReviewer(_KEY, max_retries=0, transport=httpx.MockTransport(handler))
    with pytest.raises(MiniMaxError) as error:
        reviewer.complete([{"role": "user", "content": "x"}])
    _assert_sanitized_exception_graph(error.value, _KEY, _URL, _BODY, "Authorization")


@pytest.mark.parametrize("source", ["request", "response", "transport"])
def test_public_complete_never_copies_forged_provider_error_text(source: str) -> None:
    forged = MiniMaxError(_BODY, status_code=999, attempts=True)
    if source == "transport":

        def handler(_: httpx.Request) -> httpx.Response:
            raise forged

        client = httpx.Client(transport=httpx.MockTransport(handler))
    else:

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        event_hooks = {source: [lambda _: (_ for _ in ()).throw(forged)]}
        client = httpx.Client(transport=httpx.MockTransport(handler), event_hooks=event_hooks)
    reviewer = MiniMaxReviewer(_KEY, client=client, max_retries=0)
    with pytest.raises(MiniMaxError) as error:
        reviewer.complete([{"role": "user", "content": "x"}])
    _assert_sanitized_exception_graph(error.value, _KEY, _URL, _BODY)
    assert error.value.status_code is None
    assert error.value.attempts == 0
    client.close()


def test_mutated_retry_state_and_forged_error_stay_sanitized() -> None:
    forged = MiniMaxError(_BODY, status_code=500, attempts=1)

    def hook(_: httpx.Request) -> None:
        reviewer._max_retries = object()  # type: ignore[assignment]
        raise forged

    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
        event_hooks={"request": [hook]},
    )
    reviewer = MiniMaxReviewer(_KEY, client=client, max_retries=1)
    with pytest.raises(MiniMaxError) as error:
        reviewer.complete([{"role": "user", "content": "x"}])
    _assert_sanitized_exception_graph(error.value, _KEY, _URL, _BODY)
    client.close()

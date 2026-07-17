"""Bounded OpenAI-compatible MiniMax chat-completions client."""

from __future__ import annotations

import json
import math
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr, ValidationError

from quant_trader.config import LLMSettings
from quant_trader.llm.base import ChatMessage, MessageInput, SanitizedLLMCause, canonical_messages

_MAX_RETRY_AFTER_SECONDS = 60.0
MAX_RESPONSE_BYTES = 256 * 1024
MAX_COMPLETION_CHARS = 16 * 1024


class _Credential:
    """Keeps a credential out of ordinary instance dictionaries and repr output."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = value

    def reveal_for_request(self) -> str:
        return self.__value

    def __repr__(self) -> str:
        return "_Credential('**********')"

    __str__ = __repr__


class MiniMaxError(RuntimeError):
    """A safe provider failure that excludes response bodies and credentials."""

    def __init__(self, message: str, *, status_code: int | None, attempts: int) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


def _raise_minimax_error(
    message: str, category: str, *, status_code: int | None, attempts: int
) -> NoReturn:
    cause = SanitizedLLMCause(category, status_code=status_code, attempts=attempts)
    raise MiniMaxError(message, status_code=status_code, attempts=attempts) from cause


def _request_headers(credential: _Credential) -> dict[str, str]:
    api_key = credential.reveal_for_request()
    return {
        "Accept-Encoding": "identity",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _clear_traceback_frames(error: BaseException) -> None:
    seen: set[int] = set()

    def clear(current: BaseException | None) -> None:
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        clear(current.__cause__)
        clear(current.__context__)

    clear(error)


class MiniMaxReviewer:
    """Synchronous, retry-bounded text completion provider for constrained reviews."""

    def __init__(
        self,
        api_key: SecretStr | str,
        base_url: str = "https://api.minimax.io/v1",
        model: str = "MiniMax-M2.7",
        timeout_seconds: float = 30,
        max_retries: int = 2,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        key_error = False
        key_value = ""
        try:
            key_value = _api_key_value(api_key)
        except (TypeError, ValueError):
            key_error = True
        if key_error:
            del api_key, base_url, key_value
            _raise_minimax_error(
                "MiniMax connection settings are invalid",
                "invalid-api-key",
                status_code=None,
                attempts=0,
            )
        credential = _Credential(key_value)
        del api_key, key_value
        settings: LLMSettings | None = None
        settings_error = False
        try:
            settings = LLMSettings(
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        except ValidationError:
            settings_error = True
        if settings_error or settings is None:
            del base_url, credential
            _raise_minimax_error(
                "MiniMax connection settings are invalid",
                "invalid-connection-settings",
                status_code=None,
                attempts=0,
            )
        if urlsplit(settings.base_url).scheme != "https":
            del base_url, credential, settings
            _raise_minimax_error(
                "MiniMax requires an HTTPS base URL",
                "insecure-base-url",
                status_code=None,
                attempts=0,
            )
        if not callable(sleeper):
            _raise_minimax_error(
                "MiniMax connection settings are invalid",
                "invalid-sleeper",
                status_code=None,
                attempts=0,
            )
        if client is not None and transport is not None:
            _raise_minimax_error(
                "MiniMax connection settings are invalid",
                "conflicting-client-options",
                status_code=None,
                attempts=0,
            )
        if client is not None and not isinstance(client, httpx.Client):
            _raise_minimax_error(
                "MiniMax connection settings are invalid",
                "invalid-client",
                status_code=None,
                attempts=0,
            )

        self._credential = credential
        self.base_url = settings.base_url
        self.model = settings.model
        self.timeout_seconds = float(settings.timeout_seconds)
        self.max_retries = settings.max_retries
        self._sleeper = sleeper
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=self.timeout_seconds, transport=transport)
        self._endpoint = f"{self.base_url}/chat/completions"

    def __enter__(self) -> MiniMaxReviewer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close only the HTTP client this instance created."""
        if self._owns_client:
            self.client.close()

    def complete(self, messages: Sequence[MessageInput]) -> str:
        invalid_messages = False
        invalid_message_text = "messages are invalid"
        try:
            canonical = canonical_messages(messages)
        except (TypeError, ValueError) as error:
            invalid_message_text = str(error)
            _clear_traceback_frames(error)
            invalid_messages = True
        if invalid_messages:
            del messages
            raise ValueError(invalid_message_text) from SanitizedLLMCause("invalid-message")
        del messages
        failed_status: int | None = None
        failed_attempts = 0
        failed_message = "MiniMax request failed safely"
        try:
            return self._complete_request(canonical)
        except MiniMaxError as error:
            failed_status = error.status_code
            failed_attempts = error.attempts
            failed_message = str(error)
            _clear_traceback_frames(error)
        except Exception as error:
            _clear_traceback_frames(error)
            failed_message = "MiniMax request failed safely"
        _raise_minimax_error(
            failed_message,
            "provider-failure",
            status_code=failed_status,
            attempts=failed_attempts,
        )

    def _complete_request(self, canonical: tuple[ChatMessage, ...]) -> str:
        payload = {
            "max_completion_tokens": 1200,
            "messages": [message.model_dump(mode="json") for message in canonical],
            "model": self.model,
            "stream": False,
            "temperature": 0.1,
        }
        headers = _request_headers(self._credential)
        for attempt in range(1, self.max_retries + 2):
            transport_failure = False
            retry_delay: float | None = None
            try:
                with self.client.stream(
                    "POST",
                    self._endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    follow_redirects=False,
                ) as response:
                    if response.status_code == 429 or 500 <= response.status_code <= 599:
                        if self._can_retry(attempt):
                            retry_delay = self._retry_delay(attempt, response)
                        else:
                            status_message = (
                                "MiniMax request failed "
                                f"(status={response.status_code}, attempts={attempt})"
                            )
                            _raise_minimax_error(
                                status_message,
                                "retryable-http-status",
                                status_code=response.status_code,
                                attempts=attempt,
                            )
                    elif not 200 <= response.status_code < 300:
                        status_message = (
                            "MiniMax request failed "
                            f"(status={response.status_code}, attempts={attempt})"
                        )
                        _raise_minimax_error(
                            status_message,
                            "http-status",
                            status_code=response.status_code,
                            attempts=attempt,
                        )
                    else:
                        return self._response_content(response, attempt)
            except httpx.TransportError:
                transport_failure = True
            if transport_failure:
                if self._can_retry(attempt):
                    self._sleeper(self._retry_delay(attempt, None))
                    continue
                _raise_minimax_error(
                    "MiniMax network request failed",
                    "network-request",
                    status_code=None,
                    attempts=attempt,
                )
            if retry_delay is not None:
                self._sleeper(retry_delay)
                continue
        raise AssertionError("unreachable")

    def _can_retry(self, attempt: int) -> bool:
        return attempt <= self.max_retries

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        fallback = float(2 ** (attempt - 1))
        if response is None:
            return fallback
        raw = response.headers.get("Retry-After")
        if raw is None:
            return fallback
        try:
            retry_after = float(raw)
        except ValueError:
            return fallback
        if not math.isfinite(retry_after) or retry_after < 0:
            return fallback
        return min(retry_after, self.timeout_seconds, _MAX_RETRY_AFTER_SECONDS)

    @staticmethod
    def _response_content(response: httpx.Response, attempt: int) -> str:
        content_encoding = response.headers.get("Content-Encoding", "").strip().lower()
        if content_encoding not in {"", "identity"}:
            _raise_minimax_error(
                "MiniMax response uses an unsupported content encoding",
                "unsupported-content-encoding",
                status_code=response.status_code,
                attempts=attempt,
            )
        content_length = response.headers.get("Content-Length")
        try:
            declared_length = int(content_length) if content_length is not None else 0
        except ValueError:
            declared_length = 0
        if declared_length > MAX_RESPONSE_BYTES:
            _raise_minimax_error(
                "MiniMax response exceeds the allowed size",
                "response-too-large",
                status_code=response.status_code,
                attempts=attempt,
            )
        body_access_failed = False
        body = bytearray()
        try:
            # MockTransport may pre-buffer a response; real streamed responses use raw bytes.
            chunks = (
                (response.content,)
                if response.is_stream_consumed
                else response.iter_raw(chunk_size=8_192)
            )
            for chunk in chunks:
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    break
        except Exception:
            body_access_failed = True
        if body_access_failed:
            _raise_minimax_error(
                "MiniMax response could not be read",
                "response-read",
                status_code=response.status_code,
                attempts=attempt,
            )
        if len(body) > MAX_RESPONSE_BYTES:
            _raise_minimax_error(
                "MiniMax response exceeds the allowed size",
                "response-too-large",
                status_code=response.status_code,
                attempts=attempt,
            )
        invalid_json = False
        payload: Any = None
        try:
            payload = json.loads(bytes(body).decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError, RecursionError):
            invalid_json = True
        if invalid_json:
            _raise_minimax_error(
                "MiniMax response was not valid JSON",
                "invalid-response-json",
                status_code=response.status_code,
                attempts=attempt,
            )
        invalid_content = False
        content = ""
        try:
            if not isinstance(payload, Mapping):
                raise TypeError("response JSON root must be an object")
            choices = payload["choices"]
            if not isinstance(choices, list):
                raise TypeError("choices must be a list")
            first_choice = choices[0]
            if not isinstance(first_choice, Mapping):
                raise TypeError("choice must be an object")
            message = first_choice["message"]
            if not isinstance(message, Mapping):
                raise TypeError("message must be an object")
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("content must be a string")
            if not content.strip():
                raise ValueError("content must be nonblank")
        except (IndexError, KeyError, TypeError, ValueError):
            invalid_content = True
        if invalid_content:
            _raise_minimax_error(
                "MiniMax response did not contain valid completion content",
                "invalid-completion-content",
                status_code=response.status_code,
                attempts=attempt,
            )
        if len(content) > MAX_COMPLETION_CHARS:
            _raise_minimax_error(
                "MiniMax completion exceeds the allowed size",
                "completion-too-large",
                status_code=response.status_code,
                attempts=attempt,
            )
        return content


def _api_key_value(value: SecretStr | str) -> str:
    if isinstance(value, SecretStr):
        result = value.get_secret_value()
    elif isinstance(value, str):
        result = value
    else:
        raise TypeError("api_key must be a SecretStr or string")
    if (
        not result
        or result != result.strip()
        or not result.isascii()
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in result)
    ):
        raise ValueError("api_key must be a nonblank header-safe token")
    return result

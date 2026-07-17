"""Bounded OpenAI-compatible MiniMax chat-completions client."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx
from pydantic import SecretStr, ValidationError

from quant_trader.config import LLMSettings
from quant_trader.llm.base import MessageInput, canonical_messages

_MAX_RETRY_AFTER_SECONDS = 60.0


class MiniMaxError(RuntimeError):
    """A safe provider failure that excludes response bodies and credentials."""

    def __init__(self, message: str, *, status_code: int | None, attempts: int) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


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
        self._api_key = _api_key_value(api_key)
        try:
            settings = LLMSettings(
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        except ValidationError as error:
            raise ValueError("invalid MiniMax connection settings") from error
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        if client is not None and transport is not None:
            raise ValueError("client and transport cannot both be supplied")
        if client is not None and not isinstance(client, httpx.Client):
            raise TypeError("client must be an httpx.Client")

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
        canonical = canonical_messages(messages)
        payload = {
            "max_completion_tokens": 1200,
            "messages": [message.model_dump(mode="json") for message in canonical],
            "model": self.model,
            "stream": False,
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self.client.post(self._endpoint, json=payload, headers=headers)
            except httpx.TransportError as error:
                if self._can_retry(attempt):
                    self._sleeper(self._retry_delay(attempt, None))
                    continue
                raise MiniMaxError(
                    "MiniMax network request failed", status_code=None, attempts=attempt
                ) from error

            if response.status_code == 429 or response.status_code >= 500:
                status_error = RuntimeError(f"MiniMax returned HTTP {response.status_code}")
                if self._can_retry(attempt):
                    self._sleeper(self._retry_delay(attempt, response))
                    continue
                raise MiniMaxError(
                    f"MiniMax request failed (status={response.status_code}, attempts={attempt})",
                    status_code=response.status_code,
                    attempts=attempt,
                ) from status_error
            if not 200 <= response.status_code < 300:
                status_error = RuntimeError(f"MiniMax returned HTTP {response.status_code}")
                raise MiniMaxError(
                    f"MiniMax request failed (status={response.status_code}, attempts={attempt})",
                    status_code=response.status_code,
                    attempts=attempt,
                ) from status_error
            return self._response_content(response, attempt)
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
        try:
            payload: Any = response.json()
        except (ValueError, json.JSONDecodeError) as error:
            raise MiniMaxError(
                "MiniMax response was not valid JSON",
                status_code=response.status_code,
                attempts=attempt,
            ) from error
        try:
            choices = payload["choices"]
            first_choice = choices[0]
            content = first_choice["message"]["content"]
        except (IndexError, KeyError, TypeError) as error:
            raise MiniMaxError(
                "MiniMax response did not contain completion content",
                status_code=response.status_code,
                attempts=attempt,
            ) from error
        if not isinstance(choices, list) or not isinstance(first_choice, Mapping):
            raise MiniMaxError(
                "MiniMax response did not contain completion content",
                status_code=response.status_code,
                attempts=attempt,
            )
        if not isinstance(content, str) or not content.strip():
            raise MiniMaxError(
                "MiniMax response content must be a nonblank string",
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
    if not result.strip():
        raise ValueError("api_key must be nonblank")
    return result

"""Fail-closed parsing for one structured LLM review."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, NoReturn

from pydantic import ValidationError

from quant_trader.core.models import LLMReview
from quant_trader.llm.base import SanitizedLLMCause


class LLMResponseError(ValueError):
    """The provider response is not one valid, schema-conforming review."""


def _raise_response_error(message: str, category: str) -> NoReturn:
    raise LLMResponseError(message) from SanitizedLLMCause(category)


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON constants are not allowed")


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _remove_think_prefix(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("<think>"):
        return stripped
    closing = stripped.find("</think>", len("<think>"))
    if closing < 0:
        _raise_response_error("response has an unclosed reasoning prefix", "reasoning-prefix")
    remaining = stripped[closing + len("</think>") :].strip()
    if not remaining:
        _raise_response_error("response has no JSON review after reasoning", "missing-review")
    return remaining


def _unfence(content: str) -> str:
    if not content.startswith("```"):
        return content
    for prefix, suffix in (("```json\n", "\n```"), ("```json\r\n", "\r\n```")):
        if content.startswith(prefix):
            if not content.endswith(suffix):
                _raise_response_error("response must contain one closed JSON fence", "json-fence")
            return content[len(prefix) : -len(suffix)].strip()
    _raise_response_error("response fence must be labelled json", "json-fence")


def _contains_unsafe_text(value: Any) -> bool:
    if isinstance(value, str):
        return any(
            ord(character) < 0x20 or 0xD800 <= ord(character) <= 0xDFFF for character in value
        )
    if isinstance(value, dict):
        return any(
            _contains_unsafe_text(key) or _contains_unsafe_text(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_unsafe_text(item) for item in value)
    return False


def parse_review(content: str) -> LLMReview:
    """Parse exactly one JSON object, optionally after MiniMax reasoning or in a JSON fence."""
    if not isinstance(content, str) or not content.strip():
        _raise_response_error("response must be a nonblank string", "blank-response")
    candidate = _unfence(_remove_think_prefix(content))
    if not candidate:
        _raise_response_error("response has no JSON review", "missing-review")
    invalid_json = False
    parsed: object = None
    try:
        parsed = json.loads(
            candidate, parse_constant=_reject_json_constant, object_pairs_hook=_unique_object
        )
        if not isinstance(parsed, dict):
            invalid_json = True
        elif _contains_unsafe_text(parsed):
            invalid_json = True
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        invalid_json = True
    if invalid_json:
        _raise_response_error("response must contain one safe JSON object", "invalid-json")
    assert isinstance(parsed, dict)
    invalid_schema = False
    review: LLMReview | None = None
    try:
        review = LLMReview.model_validate(parsed)
    except (ValidationError, RecursionError):
        invalid_schema = True
    if invalid_schema:
        _raise_response_error("response JSON does not match the review schema", "invalid-schema")
    assert review is not None
    return review

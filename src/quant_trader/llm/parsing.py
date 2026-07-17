"""Fail-closed parsing for one structured LLM review."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from quant_trader.core.models import LLMReview


class LLMResponseError(ValueError):
    """The provider response is not one valid, schema-conforming review."""


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
        raise LLMResponseError("response has an unclosed reasoning prefix")
    remaining = stripped[closing + len("</think>") :].strip()
    if not remaining:
        raise LLMResponseError("response has no JSON review after reasoning")
    return remaining


def _unfence(content: str) -> str:
    if not content.startswith("```"):
        return content
    for prefix, suffix in (("```json\n", "\n```"), ("```json\r\n", "\r\n```")):
        if content.startswith(prefix):
            if not content.endswith(suffix):
                raise LLMResponseError("response must contain one closed JSON fence")
            return content[len(prefix) : -len(suffix)].strip()
    raise LLMResponseError("response fence must be labelled json")


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
        raise LLMResponseError("response must be a nonblank string")
    candidate = _unfence(_remove_think_prefix(content))
    if not candidate:
        raise LLMResponseError("response has no JSON review")
    try:
        parsed = json.loads(
            candidate, parse_constant=_reject_json_constant, object_pairs_hook=_unique_object
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise LLMResponseError("response must contain exactly one valid JSON object") from error
    if not isinstance(parsed, dict):
        raise LLMResponseError("response JSON must be an object")
    if _contains_unsafe_text(parsed):
        raise LLMResponseError("response JSON contains unsafe text")
    try:
        return LLMReview.model_validate(parsed)
    except ValidationError as error:
        raise LLMResponseError("response JSON does not match the review schema") from error

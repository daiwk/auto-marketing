"""Canonical, secret-free cache key construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from quant_trader.llm.base import MessageInput, canonical_messages


def _required_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonblank string without surrounding whitespace")
    return value


def review_cache_key(model: str, prompt_version: str, messages: Sequence[MessageInput]) -> str:
    """Hash the complete deterministic request identity, excluding credentials and endpoint."""
    canonical = canonical_messages(messages)
    payload = {
        "messages": [message.model_dump(mode="json") for message in canonical],
        "model": _required_label(model, "model"),
        "prompt_version": _required_label(prompt_version, "prompt_version"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

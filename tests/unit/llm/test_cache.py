from __future__ import annotations

import re

import pytest

from quant_trader.llm.base import ChatMessage
from quant_trader.llm.cache import review_cache_key


def test_cache_key_is_canonical_and_sensitive_to_semantic_inputs() -> None:
    messages = [
        ChatMessage(role="system", content="rules"),
        ChatMessage(role="user", content="München"),
    ]
    key = review_cache_key("MiniMax-M2.7", "v1", messages)

    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert key == review_cache_key(
        "MiniMax-M2.7",
        "v1",
        [{"content": "rules", "role": "system"}, {"role": "user", "content": "München"}],
    )
    assert key != review_cache_key("other", "v1", messages)
    assert key != review_cache_key("MiniMax-M2.7", "v2", messages)
    assert key != review_cache_key("MiniMax-M2.7", "v1", list(reversed(messages)))


@pytest.mark.parametrize(
    ("model", "prompt_version", "messages"),
    [
        ("", "v1", []),
        ("model", "", []),
        (True, "v1", []),
        ("model", "v1", [{"role": "tool", "content": "x"}]),
        ("model", "v1", [{"role": "user", "content": ""}]),
        ("model", "v1", [{"role": "user", "content": 1}]),
    ],
)
def test_cache_key_rejects_noncanonical_values(
    model: object, prompt_version: object, messages: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        review_cache_key(model, prompt_version, messages)  # type: ignore[arg-type]

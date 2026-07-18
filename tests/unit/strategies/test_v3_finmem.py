from __future__ import annotations

import json
from datetime import date

import pytest

from quant_trader.llm.parsing import parse_review
from quant_trader.strategies.v3_finmem.memory import MemoryBook, MemoryLayer, MemoryRecord
from quant_trader.strategies.v3_finmem.reviewer import FinMemReviewer


def record(
    identifier: str,
    *,
    layer: MemoryLayer = MemoryLayer.SHORT,
    available_date: date = date(2025, 1, 2),
    importance: float = 0.5,
) -> MemoryRecord:
    return MemoryRecord(
        id=identifier,
        event_date=date(2025, 1, 1),
        available_date=available_date,
        layer=layer,
        ticker="ABC",
        summary=f"memory {identifier}",
        importance=importance,
    )


def v1_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "V1 system"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "candidate": {"ticker": "ABC", "score": 1.2, "base_weight": 0.1},
                    "features": {"ticker": "ABC", "as_of": "2025-01-10"},
                    "portfolio": {"cash_weight": 0.8, "current_weight": 0, "drawdown": 0},
                }
            ),
        },
    ]


class ScriptedReviewer:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[object] = []

    def complete(self, messages: object) -> str:
        self.calls.append(messages)
        return self.response  # type: ignore[return-value]


def response(*, memory_ids: list[str], action: str = "reduce", multiplier: float = 0.5) -> str:
    return json.dumps(
        {
            "action": action,
            "weight_multiplier": multiplier,
            "confidence": 0.8,
            "thesis": "Memory supports a cautious position.",
            "risks": ["Volatility remains elevated."],
            "invalidation": "Price breaks the trend.",
            "input_anomalies": [],
            "memory_ids": memory_ids,
        }
    )


def test_memory_rejects_future_availability_and_evicts_stably() -> None:
    with pytest.raises(ValueError, match="available_date"):
        record("bad", available_date=date(2024, 12, 31))

    book = MemoryBook(capacities={MemoryLayer.SHORT: 2})
    book.add(record("b", importance=0.1))
    book.add(record("a", importance=0.1))
    book.add(record("c", importance=0.1))

    assert [item["id"] for item in book.snapshot()] == ["b", "c"]


def test_memory_excludes_future_records_and_ranks_by_score_then_id() -> None:
    book = MemoryBook()
    book.add(record("b", importance=0.5))
    book.add(record("a", importance=0.5))
    book.add(record("future", available_date=date(2025, 1, 11), importance=1))

    selected = book.retrieve("ABC", date(2025, 1, 10))

    assert [item.id for item in selected[MemoryLayer.SHORT]] == ["a", "b"]


def test_finmem_adds_bounded_profile_and_retrieved_memories_in_one_call() -> None:
    book = MemoryBook()
    book.add(record("m1", importance=0.9))
    provider = ScriptedReviewer(response(memory_ids=["m1"]))
    reviewer = FinMemReviewer(provider, book)

    actual = json.loads(reviewer.complete(v1_messages()))

    assert actual["action"] == "reduce"
    assert parse_review(json.dumps(actual)).action.value == "reduce"
    assert len(provider.calls) == 1
    prompt = provider.calls[0]
    assert isinstance(prompt, tuple)
    augmented = json.loads(prompt[-1].content)
    assert augmented["profile"] == {
        "risk_style": "conservative",
        "turnover": "low",
        "max_position_pct": 20,
    }
    assert augmented["memories"]["short"][0]["id"] == "m1"
    assert reviewer.last_decision == {
        "ticker": "ABC",
        "action": "reduce",
        "confidence": 0.8,
        "memory_ids": ["m1"],
        "reason": "accepted",
    }


@pytest.mark.parametrize(
    ("provider_response", "reason"),
    [
        (response(memory_ids=["unknown"]), "invalid_memory_ids"),
        (response(memory_ids=[], action="maintain", multiplier=0.5), "inconsistent_action"),
    ],
)
def test_finmem_fails_closed_without_repair_for_invalid_evidence_or_action(
    provider_response: str, reason: str
) -> None:
    provider = ScriptedReviewer(provider_response)
    reviewer = FinMemReviewer(provider, MemoryBook())

    actual = json.loads(reviewer.complete(v1_messages()))

    assert actual["action"] == "reject"
    assert actual["weight_multiplier"] == 0
    assert "memory_ids" not in actual
    assert len(provider.calls) == 1
    assert reviewer.last_decision["reason"] == reason

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from quant_trader.strategies.v2_multi_agent.context import (
    context_for,
    load_external_context,
    reject_future_context,
)


def _payload() -> dict[str, object]:
    return {
        "tickers": {
            "AAPL": {
                "news": [
                    {
                        "published_at": "2025-12-30",
                        "headline": "Guidance raised",
                        "summary": "Demand improved",
                    }
                ],
                "sentiment": [
                    {
                        "observed_at": "2025-12-30",
                        "source": "survey",
                        "text": "Sentiment improved",
                    }
                ],
                "fundamentals": {
                    "reported_at": "2025-10-31",
                    "metrics": {"revenue_growth": 0.08, "pe_ratio": 31.2},
                },
            }
        }
    }


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_context_filters_every_source_point_in_time(tmp_path: Path) -> None:
    context = load_external_context(_write(tmp_path / "context.json", _payload()))

    visible = context_for(context, "aapl", date(2025, 12, 29))

    assert visible.news == ()
    assert visible.sentiment == ()
    assert visible.fundamentals is not None
    assert visible.fundamentals.reported_at == date(2025, 10, 31)


def test_one_shot_rejects_future_context(tmp_path: Path) -> None:
    context = load_external_context(_write(tmp_path / "context.json", _payload()))

    with pytest.raises(ValueError, match="later than --as-of"):
        reject_future_context(context, "AAPL", date(2025, 12, 29))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload["tickers"].update(  # type: ignore[union-attr]
            {"aapl": payload["tickers"]["AAPL"]}  # type: ignore[index]
        ),
        lambda payload: payload["tickers"]["AAPL"]["fundamentals"]["metrics"].update(  # type: ignore[index,union-attr]
            {"bad": float("nan")}
        ),
        lambda payload: payload["tickers"]["AAPL"].update(  # type: ignore[index,union-attr]
            {"news": payload["tickers"]["AAPL"]["news"] * 21}  # type: ignore[index,operator]
        ),
        lambda payload: payload["tickers"]["AAPL"].update(  # type: ignore[index,union-attr]
            {
                "sentiment": [
                    {"observed_at": "2025-01-01", "source": "survey", "text": "x" * 2_001}
                ]
            }
        ),
    ],
)
def test_context_rejects_unbounded_or_ambiguous_payloads(
    tmp_path: Path, mutate: object
) -> None:
    payload = _payload()
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(ValueError):
        load_external_context(_write(tmp_path / "context.json", payload))


def test_context_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 65_537)

    with pytest.raises(ValueError, match="65536"):
        load_external_context(path)


def test_missing_context_is_empty() -> None:
    context = load_external_context(None)

    assert context.tickers == {}
    assert context_for(context, "MSFT", date(2025, 1, 1)).model_dump() == {
        "news": (),
        "sentiment": (),
        "fundamentals": None,
    }

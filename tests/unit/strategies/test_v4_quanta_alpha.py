import json

import pandas as pd
import pytest

from quant_trader.strategies.v4_quanta_alpha.dsl import DSLParseError, evaluate, parse_factor
from quant_trader.strategies.v4_quanta_alpha.miner import QuantaAlphaMiner, chronological_split


def _panel(days: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows: list[dict[str, object]] = []
    for day, stamp in enumerate(dates):
        for number, ticker in enumerate(("AAA", "BBB", "CCC")):
            close = 100.0 + day * (number + 1) + number
            rows.append(
                {
                    "date": stamp,
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000.0 + number,
                    "returns": 0.01 * (number + 1),
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


@pytest.mark.parametrize("expression", ["__import__('os')", "close.__class__", "foo(close)"])
def test_dsl_rejects_malicious_or_unknown_expressions(expression: str) -> None:
    with pytest.raises(DSLParseError):
        parse_factor(expression)


def test_dsl_canonicalizes_and_evaluates_deterministically() -> None:
    panel = _panel()
    factor = parse_factor(" add( delta(close, 1), div(volume, volume) ) ")

    assert factor.canonical == "add(delta(close,1),div(volume,volume))"
    first = evaluate(factor, panel)
    second = evaluate(parse_factor(factor.canonical), panel)
    pd.testing.assert_series_equal(first, second)
    assert first.index.equals(panel.index)
    assert first.iloc[0] != first.iloc[0]  # first delta is NaN


def test_chronological_split_preserves_order_and_minimum_dates() -> None:
    train, validation, test = chronological_split(_panel(30))

    assert tuple(
        frame.index.get_level_values("date").nunique() for frame in (train, validation, test)
    ) == (18, 6, 6)
    assert (
        train.index.get_level_values("date").max() < validation.index.get_level_values("date").min()
    )
    assert (
        validation.index.get_level_values("date").max() < test.index.get_level_values("date").min()
    )


class _Reviewer:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(json.loads(prompt))
        return json.dumps(self.responses.pop(0))


def test_miner_makes_at_most_two_calls_and_returns_frozen_lineage() -> None:
    reviewer = _Reviewer(
        [
            {"candidates": [{"expression": "delta(close,1)"}, {"expression": "rank(returns)"}]},
            {
                "candidates": [
                    {
                        "expression": "rolling_mean(close,3)",
                        "parent": "delta(close,1)",
                        "operation": "rolling_mean",
                    }
                ]
            },
        ]
    )
    result = QuantaAlphaMiner(reviewer).mine(_panel())

    assert len(reviewer.calls) == 2
    assert reviewer.calls[0]["stage"] == "seed"
    assert reviewer.calls[1]["stage"] == "descendant"
    assert result["status"] == "complete"
    assert result["champion"] is not None
    assert result["champion"]["frozen"] is True
    assert "test_ic" in result["champion"]
    assert result["edges"] == [
        {"parent": "delta(close,1)", "child": "rolling_mean(close,3)", "operation": "rolling_mean"}
    ]


def test_all_unsafe_seeds_stop_after_one_call_with_partial_result() -> None:
    reviewer = _Reviewer(
        [{"candidates": [{"expression": "__import__('os')"}, {"expression": "unknown(close)"}]}]
    )

    result = QuantaAlphaMiner(reviewer).mine(_panel())

    assert len(reviewer.calls) == 1
    assert result["status"] == "partial"
    assert result["champion"] is None
    assert all(candidate["rejection_reason"] for candidate in result["candidates"])

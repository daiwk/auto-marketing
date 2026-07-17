from __future__ import annotations

from datetime import date

import pytest

from quant_trader.features.snapshot import FeatureRow
from quant_trader.strategies.v1_rules_llm.rules import Candidate, rank_candidates


def row(ticker: str = "AAA", **changes: float | int) -> FeatureRow:
    values: dict[str, float | int | str | date] = dict(
        ticker=ticker,
        as_of=date(2025, 1, 2),
        observations=260,
        close=100.0,
        sma_200=90.0,
        return_20=0.02,
        return_60=0.04,
        return_120=0.06,
        volatility_20=0.20,
        atr_14=2.0,
        average_dollar_volume_20=30_000_000.0,
    )
    values.update(changes)
    return FeatureRow(**values)  # type: ignore[arg-type]


def test_ranks_scores_then_ticker_and_limits_count() -> None:
    result = rank_candidates([row("ZZZ"), row("AAA"), row("BBB", return_60=0.05)], max_candidates=2)
    assert [candidate.ticker for candidate in result] == ["BBB", "AAA"]
    assert result[0].score == pytest.approx((0.2 * 0.02 + 0.5 * 0.05 + 0.3 * 0.06) / 0.2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("observations", 251),
        ("close", 89.0),
        ("return_20", 0.0),
        ("return_60", 0.0),
        ("return_120", float("nan")),
        ("average_dollar_volume_20", 1.0),
        ("volatility_20", 0.0),
        ("atr_14", 0.0),
    ],
)
def test_excludes_each_ineligible_row(field: str, value: float | int) -> None:
    assert rank_candidates([row(**{field: value})]) == []


def test_inverse_vol_weights_scale_to_target_and_respect_caps() -> None:
    result = rank_candidates(
        [row("AAA", volatility_20=0.10), row("BBB", volatility_20=0.20)],
        target_volatility=0.06,
        max_position_weight=0.8,
        max_gross_exposure=0.8,
    )
    assert [candidate.base_weight for candidate in result] == pytest.approx(
        [0.4242640687, 0.2121320344]
    )
    assert sum(candidate.base_weight for candidate in result) == pytest.approx(0.6363961031)


def test_position_and_gross_caps_apply() -> None:
    result = rank_candidates(
        [row("AAA"), row("BBB"), row("CCC")],
        max_position_weight=0.15,
        max_gross_exposure=0.8,
        target_volatility=10,
    )
    assert all(candidate.base_weight <= 0.15 for candidate in result)
    assert sum(candidate.base_weight for candidate in result) <= 0.8


def test_empty_invalid_parameters_and_invalid_candidate_are_rejected() -> None:
    assert rank_candidates([]) == []
    with pytest.raises(ValueError):
        rank_candidates([row()], max_candidates=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        rank_candidates([row()], target_volatility=float("inf"))
    with pytest.raises(ValueError):
        Candidate("bad ticker ", 1, 0.2, 2, 100, 0.1)


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"max_candidates": 0},
        {"max_candidates": "4"},
        {"min_dollar_volume": True},
        {"min_dollar_volume": -1.0},
        {"target_volatility": 0.0},
        {"max_position_weight": 1.1},
        {"max_gross_exposure": float("nan")},
    ],
)
def test_rejects_each_invalid_sizing_parameter(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        rank_candidates([row()], **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_candidate_rejects_non_finite_or_non_numeric_values(value: object) -> None:
    with pytest.raises(ValueError):
        Candidate("AAA", value, 0.2, 2, 100, 0.1)

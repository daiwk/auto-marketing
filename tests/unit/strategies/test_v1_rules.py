from __future__ import annotations

from datetime import date
from inspect import signature

import pytest

from quant_trader.features.snapshot import FeatureRow
from quant_trader.strategies.base import Strategy
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


def test_rank_boundary_rejects_mixed_and_invalid_row_values() -> None:
    with pytest.raises(TypeError, match="FeatureRow"):
        rank_candidates([row(), {"ticker": "BBB"}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="FeatureRow"):
        rank_candidates([object()])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "field",
    [
        "close",
        "sma_200",
        "return_20",
        "return_60",
        "return_120",
        "volatility_20",
        "atr_14",
        "average_dollar_volume_20",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-number", True])
def test_feature_row_rejects_invalid_numeric_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        row(**{field: value})  # type: ignore[arg-type]


def test_rank_accepts_immutable_finite_but_ineligible_row() -> None:
    ineligible = row(return_20=0.0)
    assert rank_candidates((ineligible,)) == []


@pytest.mark.parametrize("observations", [0, -1, "260"])
def test_feature_row_rejects_invalid_observations(observations: object) -> None:
    with pytest.raises(ValueError):
        row(observations=observations)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), "0.1"])
def test_candidate_rejects_invalid_base_weight(weight: object) -> None:
    with pytest.raises(ValueError):
        Candidate("AAA", 1, 0.2, 2, 100, weight)  # type: ignore[arg-type]


def test_strategy_protocol_uses_plural_reviews_keyword() -> None:
    assert "reviews" in signature(Strategy.generate).parameters
    assert "review" not in signature(Strategy.generate).parameters

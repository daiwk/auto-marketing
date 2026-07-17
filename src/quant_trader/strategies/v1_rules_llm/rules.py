"""Long-only, deterministic candidate filters and sizing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot, isfinite

from quant_trader.data.validation import normalize_ticker
from quant_trader.features.snapshot import FeatureRow


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class Candidate:
    ticker: str
    score: float
    annualized_volatility: float
    atr_14: float
    close: float
    base_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", normalize_ticker(self.ticker))
        for name in ("score", "annualized_volatility", "atr_14", "close", "base_weight"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))
        if self.annualized_volatility <= 0 or self.atr_14 <= 0 or self.close <= 0:
            raise ValueError("volatility, ATR, and close must be positive")
        if not 0 <= self.base_weight <= 1:
            raise ValueError("base_weight must be in [0, 1]")


def _parameters(
    max_candidates: object,
    min_dollar_volume: object,
    target_volatility: object,
    max_position_weight: object,
    max_gross_exposure: object,
) -> tuple[int, float, float, float, float]:
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates <= 0
    ):
        raise ValueError("max_candidates must be a positive integer")
    adtv = _finite_number(min_dollar_volume, "min_dollar_volume")
    target = _finite_number(target_volatility, "target_volatility")
    position = _finite_number(max_position_weight, "max_position_weight")
    gross = _finite_number(max_gross_exposure, "max_gross_exposure")
    if adtv < 0 or target <= 0 or not 0 < position <= 1 or not 0 < gross <= 1:
        raise ValueError("invalid candidate sizing parameter range")
    return max_candidates, adtv, target, position, gross


def _eligible(row: FeatureRow, min_dollar_volume: float) -> bool:
    values = (
        row.close,
        row.sma_200,
        row.return_20,
        row.return_60,
        row.return_120,
        row.volatility_20,
        row.atr_14,
        row.average_dollar_volume_20,
    )
    return (
        row.observations >= 252
        and all(isfinite(value) for value in values)
        and row.close > row.sma_200
        and row.return_20 > 0
        and row.return_60 > 0
        and row.average_dollar_volume_20 >= min_dollar_volume
        and row.volatility_20 > 0
        and row.atr_14 > 0
    )


def _score(row: FeatureRow) -> float:
    score = (0.2 * row.return_20 + 0.5 * row.return_60 + 0.3 * row.return_120) / row.volatility_20
    if not isfinite(score):
        raise ValueError(f"{row.ticker}: derived score must be finite")
    return score


def rank_candidates(
    rows: Iterable[FeatureRow],
    *,
    max_candidates: int = 4,
    min_dollar_volume: float = 20_000_000,
    target_volatility: float = 0.10,
    max_position_weight: float = 0.15,
    max_gross_exposure: float = 0.80,
) -> list[Candidate]:
    """Filter, rank, and size candidates without correlation assumptions."""
    max_count, adtv, target, position_cap, gross_cap = _parameters(
        max_candidates,
        min_dollar_volume,
        target_volatility,
        max_position_weight,
        max_gross_exposure,
    )
    try:
        all_rows = tuple(rows)
    except TypeError as error:
        raise TypeError("rows must be an iterable of FeatureRow instances") from error
    for index, row in enumerate(all_rows):
        if not isinstance(row, FeatureRow):
            raise TypeError(f"rows[{index}] must be a FeatureRow")
    tickers = [row.ticker for row in all_rows]
    if len(set(tickers)) != len(tickers):
        raise ValueError("duplicate ticker in candidate rows")
    eligible = [row for row in all_rows if _eligible(row, adtv)]
    scored = sorted(
        ((row, _score(row)) for row in eligible),
        key=lambda item: (-item[1], item[0].ticker),
    )[:max_count]
    if not scored:
        return []
    reference_volatility = min(row.volatility_20 for row, _ in scored)
    relative_inverse_volatilities = [reference_volatility / row.volatility_20 for row, _ in scored]
    ratio_total = sum(relative_inverse_volatilities)
    if not isfinite(ratio_total) or ratio_total <= 0:
        raise ValueError("derived inverse-volatility ratios must be finite and positive")
    weights = [gross_cap * ratio / ratio_total for ratio in relative_inverse_volatilities]
    portfolio_volatility = hypot(
        *(weight * row.volatility_20 for weight, (row, _) in zip(weights, scored, strict=True))
    )
    if not isfinite(portfolio_volatility):
        raise ValueError("derived portfolio volatility must be finite")
    if portfolio_volatility > target:
        scale = target / portfolio_volatility
        weights = [weight * scale for weight in weights]
    weights = [min(weight, position_cap) for weight in weights]
    if not all(isfinite(weight) and 0 <= weight <= gross_cap for weight in weights):
        raise ValueError("derived candidate weights must be finite and within gross exposure")
    return [
        Candidate(row.ticker, score, row.volatility_20, row.atr_14, row.close, weight)
        for (row, score), weight in zip(scored, weights, strict=True)
    ]

"""Immutable point-in-time feature snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType

import pandas as pd

from quant_trader.data.validation import normalize_ticker, validate_ohlcv
from quant_trader.features.technical import technical_features


def _as_timestamp(value: date | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, datetime) and not isinstance(value, pd.Timestamp):
        raise ValueError("as_of must be a date or timezone-naive normalized Timestamp")
    if isinstance(value, pd.Timestamp):
        if value.tz is not None or value != value.normalize():
            raise ValueError("as_of Timestamp must be timezone-naive and normalized")
        return value
    if type(value) is date:
        return pd.Timestamp(value)
    raise ValueError("as_of must be a date or timezone-naive normalized Timestamp")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    ticker: str
    as_of: pd.Timestamp | date
    observations: int
    close: float
    sma_200: float
    return_20: float
    return_60: float
    return_120: float
    volatility_20: float
    atr_14: float
    average_dollar_volume_20: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", normalize_ticker(self.ticker))
        object.__setattr__(self, "as_of", _as_timestamp(self.as_of))
        if isinstance(self.observations, bool) or not isinstance(self.observations, int):
            raise ValueError("observations must be an integer")
        if self.observations < 1:
            raise ValueError("observations must be positive")
        for name in (
            "close",
            "sma_200",
            "return_20",
            "return_60",
            "return_120",
            "volatility_20",
            "atr_14",
            "average_dollar_volume_20",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    as_of: pd.Timestamp
    rows: Mapping[str, FeatureRow]
    skipped: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _as_timestamp(self.as_of))
        object.__setattr__(self, "rows", MappingProxyType(dict(self.rows)))
        object.__setattr__(self, "skipped", MappingProxyType(dict(self.skipped)))


def build_feature_snapshot(
    market_frames: Mapping[str, pd.DataFrame], as_of: date | pd.Timestamp
) -> FeatureSnapshot:
    """Build rows using bars through the requested date only, never a stale bar."""
    point_in_time = _as_timestamp(as_of)
    if not isinstance(market_frames, Mapping):
        raise ValueError("market_frames must be a mapping")
    rows: dict[str, FeatureRow] = {}
    skipped: dict[str, str] = {}
    for raw_ticker in sorted(market_frames):
        ticker = normalize_ticker(raw_ticker)
        frame = market_frames[raw_ticker]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{ticker}: OHLCV must be a DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex) or point_in_time not in frame.index:
            skipped[ticker] = "missing exact bar on as_of"
            continue
        canonical = validate_ohlcv(frame.loc[:point_in_time], ticker)
        features = technical_features(canonical.loc[:point_in_time], ticker)
        latest = features.loc[point_in_time]
        rows[ticker] = FeatureRow(
            ticker=ticker,
            as_of=point_in_time,
            observations=len(features),
            close=latest["close"],
            sma_200=latest["sma_200"],
            return_20=latest["return_20"],
            return_60=latest["return_60"],
            return_120=latest["return_120"],
            volatility_20=latest["volatility_20"],
            atr_14=latest["atr_14"],
            average_dollar_volume_20=latest["average_dollar_volume_20"],
        )
    return FeatureSnapshot(point_in_time, rows, skipped)

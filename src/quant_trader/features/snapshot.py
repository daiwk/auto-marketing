"""Immutable point-in-time feature snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
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
            if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.close <= 0 or self.sma_200 <= 0:
            raise ValueError("close and sma_200 must be positive")
        if any(getattr(self, name) <= -1 for name in ("return_20", "return_60", "return_120")):
            raise ValueError("returns must be greater than -1")
        if any(
            getattr(self, name) < 0
            for name in ("volatility_20", "atr_14", "average_dollar_volume_20")
        ):
            raise ValueError("volatility, ATR, and ADTV cannot be negative")


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
    normalized_frames: list[tuple[str, pd.DataFrame]] = []
    seen_tickers: set[str] = set()
    for raw_ticker, frame in market_frames.items():
        ticker = normalize_ticker(raw_ticker)
        if ticker in seen_tickers:
            raise ValueError(f"duplicate canonical ticker: {ticker}")
        seen_tickers.add(ticker)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{ticker}: OHLCV must be a DataFrame")
        normalized_frames.append((ticker, frame))

    rows: dict[str, FeatureRow] = {}
    skipped: dict[str, str] = {}
    for ticker, frame in sorted(normalized_frames):
        if not isinstance(frame.index, pd.DatetimeIndex) or point_in_time not in frame.index:
            skipped[ticker] = "missing exact bar on as_of"
            continue
        canonical = validate_ohlcv(frame.loc[:point_in_time], ticker)
        features = technical_features(canonical.loc[:point_in_time], ticker)
        latest = features.loc[point_in_time]
        try:
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
        except ValueError:
            skipped[ticker] = "incomplete feature history"
    return FeatureSnapshot(point_in_time, rows, skipped)

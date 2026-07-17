"""Validation for canonical daily OHLCV frames."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from pydantic import TypeAdapter, ValidationError

from quant_trader.validation import USEquityTicker

_TICKER_ADAPTER = TypeAdapter(USEquityTicker)
_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    """Raised when market data cannot safely be used."""


def normalize_ticker(ticker: Any) -> str:
    """Apply the application's shared US-equity ticker contract."""
    try:
        return _TICKER_ADAPTER.validate_python(ticker)
    except ValidationError as error:
        raise DataValidationError(f"invalid ticker: {ticker!r}") from error


def validate_ohlcv(
    frame: pd.DataFrame, ticker: str, *, max_close_ratio: float = 10.0
) -> pd.DataFrame:
    """Return a defensive canonical copy, rejecting unsafe daily OHLCV data."""
    normalized_ticker = normalize_ticker(ticker)
    if (
        isinstance(max_close_ratio, bool)
        or not isinstance(max_close_ratio, int | float)
        or not isfinite(max_close_ratio)
        or max_close_ratio <= 1
    ):
        raise DataValidationError(f"{normalized_ticker}: max_close_ratio must be a finite number above 1")
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(f"{normalized_ticker}: OHLCV must be a DataFrame")
    if frame.empty:
        raise DataValidationError(f"{normalized_ticker}: OHLCV frame is empty")
    if list(frame.columns) != _COLUMNS:
        raise DataValidationError(f"{normalized_ticker}: OHLCV columns must be exactly {_COLUMNS}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(f"{normalized_ticker}: OHLCV index must be a DatetimeIndex")
    if frame.index.tz is not None:
        raise DataValidationError(f"{normalized_ticker}: OHLCV dates must be timezone-naive")
    if not frame.index.is_unique:
        raise DataValidationError(f"{normalized_ticker}: OHLCV dates must be unique")
    if not frame.index.is_monotonic_increasing:
        raise DataValidationError(f"{normalized_ticker}: OHLCV dates must be ascending")
    if not (frame.index == frame.index.normalize()).all():
        raise DataValidationError(f"{normalized_ticker}: OHLCV index must contain normalized market dates")
    try:
        canonical = frame.astype(float, copy=True)
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"{normalized_ticker}: OHLCV values must be numeric") from error
    canonical.index = canonical.index.copy()
    canonical.index.name = "date"
    if not np.isfinite(canonical.to_numpy()).all():
        raise DataValidationError(f"{normalized_ticker}: OHLCV values must be finite")
    if (canonical[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError(f"{normalized_ticker}: OHLC prices must be positive")
    if (canonical["volume"] < 0).any():
        raise DataValidationError(f"{normalized_ticker}: volume cannot be negative")
    if (canonical["high"] < canonical[["open", "low", "close"]].max(axis=1)).any():
        raise DataValidationError(f"{normalized_ticker}: high is below another price")
    if (canonical["low"] > canonical[["open", "high", "close"]].min(axis=1)).any():
        raise DataValidationError(f"{normalized_ticker}: low is above another price")
    close_ratios = canonical["close"].div(canonical["close"].shift()).iloc[1:]
    if ((close_ratios > max_close_ratio) | (close_ratios < 1 / max_close_ratio)).any():
        raise DataValidationError(f"{normalized_ticker}: implausible one-day adjusted close jump")
    return canonical


def assert_fresh(frame: pd.DataFrame, expected_market_date: date, ticker: str) -> None:
    """Require that the newest validated bar is the expected completed market date."""
    canonical = validate_ohlcv(frame, ticker)
    latest = canonical.index[-1].date()
    if latest != expected_market_date:
        raise DataValidationError(
            f"{normalize_ticker(ticker)}: stale market data; expected {expected_market_date}, got {latest}"
        )

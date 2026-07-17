"""Deterministic daily technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trader.data.validation import validate_ohlcv


def technical_features(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate a canonical OHLCV frame and return a new frame with daily features."""
    result = validate_ohlcv(frame, ticker)
    close = result["close"]
    daily_returns = close.pct_change(fill_method=None)
    previous_close = close.shift()
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["sma_200"] = close.rolling(200, min_periods=200).mean()
    result["return_20"] = close.pct_change(20, fill_method=None)
    result["return_60"] = close.pct_change(60, fill_method=None)
    result["return_120"] = close.pct_change(120, fill_method=None)
    result["volatility_20"] = daily_returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    result["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    result["average_dollar_volume_20"] = (
        (close * result["volume"]).rolling(20, min_periods=20).mean()
    )
    return result

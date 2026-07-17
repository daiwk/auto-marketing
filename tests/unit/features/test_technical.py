from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trader.features.technical import technical_features


def bars(periods: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    close = pd.Series(np.arange(100, 100 + periods, dtype=float), index=index)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=index,
    )


def test_calculates_exact_features_without_mutating_input() -> None:
    frame = bars()
    original = frame.copy(deep=True)

    result = technical_features(frame, "abc")

    pd.testing.assert_frame_equal(frame, original)
    assert result is not frame
    assert list(result.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_200",
        "return_20",
        "return_60",
        "return_120",
        "volatility_20",
        "atr_14",
        "average_dollar_volume_20",
    ]
    expected_return = frame["close"].pct_change(20, fill_method=None)
    expected_volatility = frame["close"].pct_change(fill_method=None).rolling(
        20, min_periods=20
    ).std(ddof=1) * np.sqrt(252)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - frame["close"].shift()).abs(),
            (frame["low"] - frame["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    pd.testing.assert_series_equal(result["return_20"], expected_return, check_names=False)
    pd.testing.assert_series_equal(result["volatility_20"], expected_volatility, check_names=False)
    pd.testing.assert_series_equal(
        result["atr_14"], true_range.rolling(14, min_periods=14).mean(), check_names=False
    )
    assert result.loc[frame.index[199], "sma_200"] == pytest.approx(
        frame["close"].iloc[:200].mean()
    )
    assert np.isnan(result.loc[frame.index[18], "volatility_20"])


def test_future_rows_cannot_change_existing_features() -> None:
    frame = bars(240)
    baseline = technical_features(frame, "ABC")
    extended = pd.concat([frame, bars(5).set_axis(pd.bdate_range("2030-01-01", periods=5))])

    result = technical_features(extended, "ABC")

    earlier = result.reindex(frame.index)[baseline.columns]
    earlier.index.name = baseline.index.name
    pd.testing.assert_frame_equal(earlier, baseline)

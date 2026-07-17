from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_trader.data.validation import DataValidationError, assert_fresh, validate_ohlcv


def ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0], "close": [101.0, 102.0], "volume": [10.0, 20.0]},
        index=pd.DatetimeIndex(["2026-01-02", "2026-01-05"], name="date"),
    )


def test_validate_ohlcv_returns_defensive_canonical_copy() -> None:
    frame = ohlcv()
    result = validate_ohlcv(frame, "spy")
    result.loc[result.index[0], "close"] = 999.0
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert frame.iloc[0]["close"] == 101.0


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda f: f.iloc[0:0], "empty"),
        (lambda f: f.rename(columns={"close": "Close"}), "columns"),
        (lambda f: f.sort_index(ascending=False), "ascending"),
        (lambda f: pd.concat([f, f.iloc[[0]]]), "unique"),
        (lambda f: f.set_index(pd.DatetimeIndex(["2026-01-02 12:00", "2026-01-05"])), "market dates"),
        (lambda f: f.assign(close=np.nan), "finite"),
        (lambda f: f.assign(close=np.inf), "finite"),
        (lambda f: f.assign(open=0.0), "positive"),
        (lambda f: f.assign(volume=-1.0), "negative"),
        (lambda f: f.assign(high=98.0), "high"),
        (lambda f: f.assign(low=104.0), "low"),
    ],
)
def test_validate_ohlcv_rejects_invalid_data(mutate: object, match: str) -> None:
    with pytest.raises(DataValidationError, match=match):
        validate_ohlcv(mutate(ohlcv()), "SPY")  # type: ignore[operator]


def test_validate_ohlcv_rejects_large_adjusted_close_jump() -> None:
    frame = ohlcv()
    frame.loc[frame.index[1], ["open", "high", "low", "close"]] = [2000.0, 2100.0, 1900.0, 2000.0]
    with pytest.raises(DataValidationError, match="jump"):
        validate_ohlcv(frame, "SPY")


def test_validate_ohlcv_uses_shared_ticker_validation() -> None:
    with pytest.raises(DataValidationError, match="ticker"):
        validate_ohlcv(ohlcv(), "../SPY")


def test_assert_fresh_requires_exact_last_completed_market_date() -> None:
    assert_fresh(ohlcv(), date(2026, 1, 5), "SPY")
    with pytest.raises(DataValidationError, match="expected 2026-01-06"):
        assert_fresh(ohlcv(), date(2026, 1, 6), "SPY")

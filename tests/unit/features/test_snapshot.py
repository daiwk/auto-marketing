from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from quant_trader.features.snapshot import FeatureRow, build_feature_snapshot


def bars(periods: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    close = pd.Series(np.arange(100, 100 + periods, dtype=float), index=index)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000_000.0},
        index=index,
    )


def test_snapshot_uses_exact_date_and_is_immutable() -> None:
    frame = bars()
    as_of = frame.index[250]
    snapshot = build_feature_snapshot({"abc": frame, "MSFT": frame.drop(as_of)}, as_of.date())

    assert tuple(snapshot.rows) == ("ABC",)
    assert snapshot.skipped["MSFT"] == "missing exact bar on as_of"
    row = snapshot.rows["ABC"]
    assert row.as_of == as_of
    assert row.observations == 251
    with pytest.raises(TypeError):
        snapshot.rows["X"] = row  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        row.close = 1.0  # type: ignore[misc]


def test_snapshot_never_reads_future_prices() -> None:
    frame = bars()
    as_of = frame.index[220]
    altered = frame.copy()
    altered.loc[altered.index > as_of, "close"] = 9_999.0

    before = build_feature_snapshot({"ABC": frame}, as_of)
    after = build_feature_snapshot({"ABC": altered}, as_of)

    assert before.rows["ABC"] == after.rows["ABC"]


@pytest.mark.parametrize(
    "as_of", ["2024-01-02", pd.Timestamp("2024-01-02 12:00"), pd.Timestamp("2024-01-02", tz="UTC")]
)
def test_snapshot_rejects_invalid_as_of(as_of: object) -> None:
    with pytest.raises(ValueError):
        build_feature_snapshot({"ABC": bars()}, as_of)  # type: ignore[arg-type]


def test_feature_row_rejects_invalid_ticker_and_observations() -> None:
    with pytest.raises(ValueError):
        FeatureRow(
            " bad ",
            pd.Timestamp("2024-01-02"),
            1,
            1,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    with pytest.raises(ValueError):
        FeatureRow(
            "ABC",
            pd.Timestamp("2024-01-02"),
            True,
            1,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

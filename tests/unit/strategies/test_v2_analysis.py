from datetime import date

import numpy as np
import pandas as pd

from quant_trader.config import Settings
from quant_trader.strategies.v2_multi_agent.analysis import prepare_analysis


def _rising_frame() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=320)
    close = np.linspace(100.0, 180.0, len(index)) + np.sin(np.arange(len(index)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )


def test_prepare_analysis_builds_v1_review_messages_for_eligible_ticker() -> None:
    frame = _rising_frame()
    as_of = frame.index[-1].date()

    prepared = prepare_analysis({"aapl": frame}, Settings(), "AAPL", as_of)

    assert prepared.eligible is True
    assert prepared.ticker == "AAPL"
    assert prepared.as_of == as_of
    assert prepared.messages is not None
    assert '"ticker":"AAPL"' in prepared.messages[1].content


def test_prepare_analysis_skips_provider_when_exact_bar_is_missing() -> None:
    frame = _rising_frame()

    prepared = prepare_analysis(
        {"AAPL": frame}, Settings(), "AAPL", date(2023, 1, 3)
    )

    assert prepared.eligible is False
    assert prepared.messages is None
    assert "not eligible" in prepared.reason

from datetime import date, datetime

import pandas as pd
import pytest

from quant_trader.data.validation import DataValidationError
from quant_trader.data.yfinance_source import YFinanceSource


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [100, 101], "High": [102, 103], "Low": [99, 100], "Close": [101, 102], "Volume": [10, 20]},
        index=pd.DatetimeIndex(["2026-01-02 16:00", "2026-01-05 16:00"], tz="America/New_York"),
    )


def test_fetch_passes_end_unchanged_and_normalizes_output(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    def download(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return raw_frame()
    monkeypatch.setattr("yfinance.download", download)
    frame = YFinanceSource().fetch("spy", date(2026, 1, 2), date(2026, 1, 6))
    assert calls == [{"start": date(2026, 1, 2), "end": date(2026, 1, 6), "auto_adjust": True, "actions": False, "progress": False, "threads": False}]
    assert frame.index.equals(pd.DatetimeIndex(["2026-01-02", "2026-01-05"], name="date"))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_normalizes_single_ticker_multiindex(monkeypatch) -> None:
    frame = raw_frame()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["SPY"]])
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: frame)
    assert YFinanceSource().fetch("SPY", date(2026, 1, 2), date(2026, 1, 6)).shape == (2, 5)


@pytest.mark.parametrize("response", [pd.DataFrame(), pd.DataFrame({"Open": [1]})])
def test_fetch_rejects_empty_or_incomplete_responses(monkeypatch, response: pd.DataFrame) -> None:
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: response)
    with pytest.raises(DataValidationError, match="SPY"):
        YFinanceSource().fetch("SPY", date(2026, 1, 2), date(2026, 1, 6))


def test_fetch_rejects_multiple_ticker_output_and_invalid_range(monkeypatch) -> None:
    frame = raw_frame()
    frame = pd.concat([frame, frame], axis=1, keys=["SPY", "MSFT"])
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: frame)
    with pytest.raises(DataValidationError, match="multiple"):
        YFinanceSource().fetch("SPY", date(2026, 1, 2), date(2026, 1, 6))
    with pytest.raises(DataValidationError, match="start"):
        YFinanceSource().fetch("SPY", date(2026, 1, 6), date(2026, 1, 6))


@pytest.mark.parametrize(
    "start, end",
    [
        ("2026-01-02", date(2026, 1, 6)),
        (None, date(2026, 1, 6)),
        (1, date(2026, 1, 6)),
        (datetime(2026, 1, 2), date(2026, 1, 6)),
    ],
)
def test_fetch_rejects_non_date_inputs_without_calling_provider(monkeypatch, start: object, end: object) -> None:
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: pytest.fail("provider called"))
    with pytest.raises(DataValidationError, match="SPY.*date range"):
        YFinanceSource().fetch("SPY", start, end)  # type: ignore[arg-type]


def test_fetch_wraps_provider_exception_with_ticker_and_range(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("provider failure")
    monkeypatch.setattr("yfinance.download", fail)
    with pytest.raises(DataValidationError, match="SPY.*2026-01-02.*2026-01-06") as error:
        YFinanceSource().fetch("SPY", date(2026, 1, 2), date(2026, 1, 6))
    assert isinstance(error.value.__cause__, RuntimeError)


def test_fetch_rejects_unexpected_flat_columns(monkeypatch) -> None:
    frame = raw_frame().assign(Adj_Close=[101, 102])
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: frame)
    with pytest.raises(DataValidationError, match="SPY.*2026-01-02.*unexpected"):
        YFinanceSource().fetch("SPY", date(2026, 1, 2), date(2026, 1, 6))


def test_fetch_wraps_non_convertible_index(monkeypatch) -> None:
    frame = raw_frame().copy()
    frame.index = pd.Index(["not-a-date", "still-not-a-date"])
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: frame)
    with pytest.raises(DataValidationError, match="SPY.*invalid response.*2026-01-02"):
        YFinanceSource().fetch("SPY", date(2026, 1, 2), date(2026, 1, 6))

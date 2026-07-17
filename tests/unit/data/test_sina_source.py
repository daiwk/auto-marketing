import json
from datetime import date

import httpx
import pandas as pd
import pytest
import respx

from quant_trader.data.sina_source import DAILY_URL, FACTOR_URL, SinaSource
from quant_trader.data.validation import DataValidationError


def daily_text(symbol: str, rows: list[dict[str, str]]) -> str:
    return f"/* guard */\nvar _{symbol}=({json.dumps(rows)});"


def factor_text(symbol: str, rows: list[dict[str, str]]) -> str:
    payload = json.dumps({"total": len(rows), "data": rows})
    return f"var {symbol}_qfq= {payload}\n/* signature */"


@respx.mock
def test_fetch_applies_forward_adjustment_and_returns_half_open_frame() -> None:
    daily_route = respx.get(DAILY_URL.format(symbol="NVDA")).mock(
        return_value=httpx.Response(
            200,
            text=daily_text(
                "NVDA",
                [
                    {
                        "d": "2024-06-07",
                        "o": "1197.70",
                        "h": "1216.92",
                        "l": "1180.22",
                        "c": "1208.88",
                        "v": "412386000",
                        "a": "0",
                    },
                    {
                        "d": "2024-06-10",
                        "o": "120.37",
                        "h": "123.10",
                        "l": "117.01",
                        "c": "121.79",
                        "v": "314162700",
                        "a": "0",
                    },
                    {
                        "d": "2024-06-11",
                        "o": "121.77",
                        "h": "122.87",
                        "l": "118.74",
                        "c": "120.91",
                        "v": "222551200",
                        "a": "0",
                    },
                ],
            ),
        )
    )
    respx.get(FACTOR_URL.format(symbol="NVDA")).mock(
        return_value=httpx.Response(
            200,
            text=factor_text(
                "NVDA",
                [
                    {"d": "2024-06-11", "f": "1", "c": "-0.32"},
                    {"d": "2024-06-10", "f": "1", "c": "-0.33"},
                    {"d": "2024-03-05", "f": "0.1", "c": "-0.33"},
                    {"d": "1900-01-01", "f": "0.1", "c": "-0.34"},
                ],
            ),
        )
    )

    frame = SinaSource().fetch("nvda", date(2024, 6, 7), date(2024, 6, 11))

    assert daily_route.calls[0].request.url.params["symbol"] == "NVDA"
    assert daily_route.calls[0].request.url.params["___qn"] == "3"
    assert frame.index.equals(pd.DatetimeIndex(["2024-06-07", "2024-06-10"], name="date"))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.loc["2024-06-07", "close"] == pytest.approx(120.558)
    assert frame.loc["2024-06-10", "close"] == pytest.approx(121.46)
    assert frame.loc["2024-06-10", "volume"] == 314162700


@respx.mock
def test_fetch_rejects_unsupported_ticker_without_network() -> None:
    with pytest.raises(DataValidationError, match="TSLA.*not supported"):
        SinaSource().fetch("TSLA", date(2023, 1, 3), date(2023, 1, 5))
    assert not respx.calls


@pytest.mark.parametrize(
    "daily, factors, message",
    [
        ("not jsonp", factor_text("SPY", []), "invalid daily response"),
        (daily_text("SPY", []), factor_text("SPY", []), "empty daily response"),
        (daily_text("SPY", [{"d": "bad"}]), factor_text("SPY", []), "invalid daily response"),
        (
            daily_text(
                "SPY",
                [{"d": "2023-01-03", "o": "1", "h": "2", "l": "1", "c": "1", "v": "1", "a": "0"}],
            ),
            "not factors",
            "invalid factor response",
        ),
    ],
)
@respx.mock
def test_fetch_rejects_malformed_provider_data(daily: str, factors: str, message: str) -> None:
    respx.get(DAILY_URL.format(symbol="SPY")).mock(return_value=httpx.Response(200, text=daily))
    respx.get(FACTOR_URL.format(symbol="SPY")).mock(return_value=httpx.Response(200, text=factors))
    with pytest.raises(DataValidationError, match=f"SPY.*{message}"):
        SinaSource().fetch("SPY", date(2023, 1, 3), date(2023, 1, 5))


@respx.mock
def test_fetch_wraps_transport_errors() -> None:
    respx.get(DAILY_URL.format(symbol="SPY")).mock(side_effect=httpx.TimeoutException("timeout"))
    with pytest.raises(DataValidationError, match="SPY.*download failed"):
        SinaSource().fetch("SPY", date(2023, 1, 3), date(2023, 1, 5))

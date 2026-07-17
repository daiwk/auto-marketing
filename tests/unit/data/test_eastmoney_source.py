from datetime import date

import httpx
import pandas as pd
import pytest
import respx

from quant_trader.data.eastmoney_source import ENDPOINT, SECURITY_IDS, EastMoneySource
from quant_trader.data.validation import DataValidationError


def response_payload(*rows: str, code: str = "SPY") -> dict[str, object]:
    return {"rc": 0, "data": {"code": code, "klines": list(rows)}}


@respx.mock
def test_fetch_maps_ticker_and_returns_canonical_half_open_frame() -> None:
    route = respx.get(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=response_payload(
                "2023-01-03,384.370,380.820,386.430,377.831,74850731,28499102976.000",
                "2023-01-04,383.180,383.760,385.880,380.000,85934122,32910223360.000",
                "2023-01-05,383.000,379.380,383.840,378.760,76970531,29236241920.000",
            ),
        )
    )

    frame = EastMoneySource().fetch("spy", date(2023, 1, 3), date(2023, 1, 5))

    assert SECURITY_IDS == {
        "SPY": "107.SPY",
        "QQQ": "105.QQQ",
        "IWM": "105.IWM",
        "AAPL": "105.AAPL",
        "MSFT": "105.MSFT",
        "NVDA": "105.NVDA",
        "AMZN": "105.AMZN",
        "GOOGL": "105.GOOGL",
        "META": "105.META",
    }
    request = route.calls[0].request
    assert request.url.params["secid"] == "107.SPY"
    assert request.url.params["beg"] == "20230103"
    assert request.url.params["end"] == "20230105"
    assert request.url.params["klt"] == "101"
    assert request.url.params["fqt"] == "1"
    assert request.url.params["fields2"] == "f51,f52,f53,f54,f55,f56,f57"
    assert frame.index.equals(pd.DatetimeIndex(["2023-01-03", "2023-01-04"], name="date"))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.loc["2023-01-03"].to_dict() == {
        "open": 384.37,
        "high": 386.43,
        "low": 377.831,
        "close": 380.82,
        "volume": 74850731.0,
    }


@respx.mock
def test_fetch_rejects_unsupported_ticker_without_network() -> None:
    with pytest.raises(DataValidationError, match="TSLA.*not supported"):
        EastMoneySource().fetch("TSLA", date(2023, 1, 3), date(2023, 1, 5))
    assert not respx.calls


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"rc": 1, "data": None}, "provider error"),
        ({"rc": 0, "data": {"code": "SPY", "klines": []}}, "empty response"),
        (response_payload("bad,row"), "invalid response"),
        (response_payload("2023-01-03,x,1,1,1,1,1"), "invalid response"),
        (response_payload("2023-01-03,1,1,0.5,2,1,1"), "invalid response"),
    ],
)
@respx.mock
def test_fetch_rejects_malformed_provider_data(payload: dict[str, object], message: str) -> None:
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    with pytest.raises(DataValidationError, match=f"SPY.*{message}"):
        EastMoneySource().fetch("SPY", date(2023, 1, 3), date(2023, 1, 5))


@respx.mock
def test_fetch_wraps_transport_and_json_errors() -> None:
    respx.get(ENDPOINT).mock(side_effect=httpx.TimeoutException("timeout"))
    with pytest.raises(DataValidationError, match="SPY.*download failed"):
        EastMoneySource().fetch("SPY", date(2023, 1, 3), date(2023, 1, 5))

    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, content=b"not-json"))
    with pytest.raises(DataValidationError, match="SPY.*invalid response"):
        EastMoneySource().fetch("SPY", date(2023, 1, 3), date(2023, 1, 5))

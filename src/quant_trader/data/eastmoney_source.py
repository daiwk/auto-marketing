"""Eastmoney daily-kline adapter for the fixed V1 research universe."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pandas as pd

from quant_trader.data.validation import DataValidationError, normalize_ticker, validate_ohlcv

ENDPOINT = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SECURITY_IDS = {
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


class EastMoneySource:
    """Download forward-adjusted daily OHLCV bars for the V1 universe."""

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        normalized = normalize_ticker(ticker)
        if type(start) is not date or type(end) is not date or start >= end:
            raise DataValidationError(f"{normalized}: invalid date range {start!r} to {end!r}")
        security_id = SECURITY_IDS.get(normalized)
        if security_id is None:
            raise DataValidationError(f"{normalized}: ticker is not supported by Eastmoney")
        try:
            response = httpx.get(
                ENDPOINT,
                params={
                    "secid": security_id,
                    "beg": start.strftime("%Y%m%d"),
                    "end": end.strftime("%Y%m%d"),
                    "klt": "101",
                    "fqt": "1",
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57",
                },
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise DataValidationError(
                f"{normalized}: Eastmoney download failed for {start} to {end}"
            ) from error
        try:
            payload: Any = response.json()
        except ValueError as error:
            raise DataValidationError(f"{normalized}: invalid response from Eastmoney") from error
        if not isinstance(payload, dict) or payload.get("rc") != 0:
            raise DataValidationError(f"{normalized}: Eastmoney provider error")
        data = payload.get("data")
        if not isinstance(data, dict) or data.get("code") != normalized:
            raise DataValidationError(f"{normalized}: invalid response from Eastmoney")
        rows = data.get("klines")
        if not isinstance(rows, list) or not rows:
            raise DataValidationError(f"{normalized}: empty response from Eastmoney")
        try:
            records = [self._parse_row(row) for row in rows]
            frame = pd.DataFrame.from_records(records).set_index("date")
            frame.index = pd.DatetimeIndex(frame.index, name="date")
            frame = frame.loc[
                (frame.index.date >= start) & (frame.index.date < end),
                ["open", "high", "low", "close", "volume"],
            ]
            return validate_ohlcv(frame, normalized)
        except (DataValidationError, KeyError, TypeError, ValueError) as error:
            raise DataValidationError(
                f"{normalized}: invalid response from Eastmoney: {error}"
            ) from error

    @staticmethod
    def _parse_row(row: object) -> dict[str, object]:
        if not isinstance(row, str):
            raise ValueError("kline row must be text")
        fields = row.split(",")
        if len(fields) != 7:
            raise ValueError("kline row has unexpected fields")
        market_date, open_, close, high, low, volume, _amount = fields
        return {
            "date": date.fromisoformat(market_date),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }

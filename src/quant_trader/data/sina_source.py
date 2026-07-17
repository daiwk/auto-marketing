"""Sina Finance daily bars with forward-adjustment factors."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from datetime import date
from typing import Any

import httpx
import pandas as pd

from quant_trader.data.validation import DataValidationError, normalize_ticker, validate_ohlcv

DAILY_URL = (
    "https://stock.finance.sina.com.cn/usstock/api/jsonp.php/"
    "var%20_{symbol}=/US_MinKService.getDailyK"
)
FACTOR_URL = "https://finance.sina.com.cn/us_stock/company/reinstatement/{symbol}_qfq.js"
SUPPORTED_TICKERS = frozenset(
    {"SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"}
)


class SinaSource:
    """Download and validate forward-adjusted daily OHLCV for the V1 universe."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        normalized = normalize_ticker(ticker)
        if type(start) is not date or type(end) is not date or start >= end:
            raise DataValidationError(f"{normalized}: invalid date range {start!r} to {end!r}")
        if normalized not in SUPPORTED_TICKERS:
            raise DataValidationError(f"{normalized}: ticker is not supported by Sina")
        try:
            daily_response = httpx.get(
                DAILY_URL.format(symbol=normalized),
                params={"symbol": normalized, "___qn": "3"},
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            daily_response.raise_for_status()
            factor_response = httpx.get(
                FACTOR_URL.format(symbol=normalized),
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            factor_response.raise_for_status()
        except httpx.HTTPError as error:
            raise DataValidationError(
                f"{normalized}: Sina download failed for {start} to {end}"
            ) from error

        records = self._daily_records(daily_response.text, normalized)
        selected = [record for record in records if start <= record["date"] < end]
        if not selected:
            raise DataValidationError(f"{normalized}: empty daily response from Sina")
        factors = self._factor_records(factor_response.text, normalized)
        try:
            factor_dates = [factor[0] for factor in factors]
            adjusted: list[dict[str, object]] = []
            for record in selected:
                market_date = record["date"]
                factor_index = bisect_right(factor_dates, market_date) - 1
                if factor_index < 0:
                    raise ValueError("no adjustment factor for market date")
                _, multiplier, offset = factors[factor_index]
                adjusted.append(
                    {
                        "date": market_date,
                        "open": record["open"] * multiplier + offset,
                        "high": record["high"] * multiplier + offset,
                        "low": record["low"] * multiplier + offset,
                        "close": record["close"] * multiplier + offset,
                        "volume": record["volume"],
                    }
                )
            frame = pd.DataFrame.from_records(adjusted).set_index("date")
            frame.index = pd.DatetimeIndex(frame.index, name="date")
            return validate_ohlcv(frame, normalized)
        except (DataValidationError, KeyError, TypeError, ValueError) as error:
            raise DataValidationError(
                f"{normalized}: invalid adjusted response from Sina: {error}"
            ) from error

    @staticmethod
    def _daily_records(text: str, ticker: str) -> list[dict[str, Any]]:
        match = re.search(r"=\s*\((\[.*\])\);\s*$", text, re.DOTALL)
        try:
            payload = json.loads(match.group(1)) if match else None
            if not isinstance(payload, list):
                raise ValueError
            if not payload:
                raise DataValidationError(f"{ticker}: empty daily response from Sina")
            records = []
            for row in payload:
                if not isinstance(row, dict):
                    raise ValueError
                records.append(
                    {
                        "date": date.fromisoformat(row["d"]),
                        "open": float(row["o"]),
                        "high": float(row["h"]),
                        "low": float(row["l"]),
                        "close": float(row["c"]),
                        "volume": float(row["v"]),
                    }
                )
            return records
        except DataValidationError:
            raise
        except (AttributeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise DataValidationError(f"{ticker}: invalid daily response from Sina") from error

    @staticmethod
    def _factor_records(text: str, ticker: str) -> list[tuple[date, float, float]]:
        start = text.find("{")
        end = text.find("\n/*", start)
        if end < 0:
            end = text.rfind("}") + 1
        try:
            payload = json.loads(text[start:end])
            rows = payload["data"]
            if not isinstance(rows, list) or not rows:
                raise ValueError
            factors = [
                (date.fromisoformat(row["d"]), float(row["f"]), float(row["c"])) for row in rows
            ]
            factors.sort(key=lambda item: item[0])
            return factors
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise DataValidationError(f"{ticker}: invalid factor response from Sina") from error

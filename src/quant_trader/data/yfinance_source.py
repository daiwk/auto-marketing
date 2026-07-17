"""yfinance adapter isolated behind the market-data source boundary."""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import yfinance
from yfinance import exceptions as yf_exceptions
from yfinance import shared

from quant_trader.data.validation import DataValidationError, normalize_ticker, validate_ohlcv


class YFinanceSource:
    """Download one ticker's adjusted daily OHLCV bars for [start, end)."""

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        normalized_ticker = normalize_ticker(ticker)
        if type(start) is not date or type(end) is not date:
            raise DataValidationError(
                f"{normalized_ticker}: invalid date range {start!r} to {end!r}; dates are required"
            )
        if start >= end:
            raise DataValidationError(
                f"{normalized_ticker}: invalid date range {start} to {end}; "
                "start must be before end"
            )
        raw: pd.DataFrame | None = None
        for attempt in range(3):
            try:
                raw = yfinance.download(
                    normalized_ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=False,
                )
                rate_limited = self._was_rate_limited(normalized_ticker, raw)
            except yf_exceptions.YFRateLimitError:
                rate_limited = True
            except Exception as error:
                raise DataValidationError(
                    f"{normalized_ticker}: download failed for {start} to {end}"
                ) from error
            if not rate_limited:
                break
            if attempt == 2:
                raise DataValidationError(
                    f"{normalized_ticker}: Yahoo Finance rate limited; wait a few minutes and retry"
                )
            time.sleep(float(2**attempt))
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise DataValidationError(f"{normalized_ticker}: empty response for {start} to {end}")
        try:
            flat = self._flatten(raw, normalized_ticker)
            frame = self._canonical_columns(flat, normalized_ticker)
            index = pd.DatetimeIndex(frame.index)
            if index.tz is not None:
                index = index.tz_localize(None)
            frame.index = index.normalize()
            frame.index.name = "date"
            canonical = validate_ohlcv(frame, normalized_ticker)
            if any(
                market_date < start or market_date >= end for market_date in canonical.index.date
            ):
                raise DataValidationError(
                    f"{normalized_ticker}: bars outside requested interval [{start}, {end})"
                )
            return canonical
        except (DataValidationError, KeyError, TypeError, ValueError) as error:
            raise DataValidationError(
                f"{normalized_ticker}: invalid response for {start} to {end}: {error}"
            ) from error

    @staticmethod
    def _was_rate_limited(ticker: str, response: object) -> bool:
        if isinstance(response, pd.DataFrame) and not response.empty:
            return False
        provider_error = str(shared._ERRORS.get(ticker, ""))
        return "YFRateLimitError" in provider_error or "Too Many Requests" in provider_error

    @staticmethod
    def _canonical_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if isinstance(frame.columns, pd.MultiIndex):
            raise DataValidationError(f"{ticker}: unexpected multi-index response columns")
        required = ["open", "high", "low", "close", "volume"]
        labels = [str(column).lower() for column in frame.columns]
        if len(labels) != len(required) or set(labels) != set(required):
            raise DataValidationError(f"{ticker}: unexpected response columns")
        canonical = frame.copy()
        canonical.columns = labels
        return canonical[required]

    @staticmethod
    def _flatten(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if not isinstance(frame.columns, pd.MultiIndex):
            return frame
        matching_levels = [
            level
            for level in range(frame.columns.nlevels)
            if ticker in frame.columns.get_level_values(level)
        ]
        if len(matching_levels) != 1:
            raise DataValidationError(f"{ticker}: unexpected multi-ticker response")
        ticker_level = matching_levels[0]
        tickers = set(frame.columns.get_level_values(ticker_level))
        if tickers != {ticker}:
            raise DataValidationError(f"{ticker}: multiple ticker data returned")
        return frame.xs(ticker, level=ticker_level, axis=1, drop_level=True)

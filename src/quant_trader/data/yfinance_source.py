"""yfinance adapter isolated behind the market-data source boundary."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance

from quant_trader.data.validation import DataValidationError, normalize_ticker, validate_ohlcv


class YFinanceSource:
    """Download one ticker's adjusted daily OHLCV bars for [start, end)."""

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        normalized_ticker = normalize_ticker(ticker)
        if start >= end:
            raise DataValidationError(f"{normalized_ticker}: start must be before end")
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
        except Exception as error:
            raise DataValidationError(f"{normalized_ticker}: download failed for {start} to {end}") from error
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise DataValidationError(f"{normalized_ticker}: empty response for {start} to {end}")
        flat = self._flatten(raw, normalized_ticker)
        columns = {str(column).lower(): column for column in flat.columns}
        required = ["open", "high", "low", "close", "volume"]
        if not all(name in columns for name in required):
            raise DataValidationError(f"{normalized_ticker}: response lacks required OHLCV columns")
        frame = flat[[columns[name] for name in required]].copy()
        frame.columns = required
        index = pd.DatetimeIndex(frame.index)
        if index.tz is not None:
            index = index.tz_convert(None)
        frame.index = index.normalize()
        frame.index.name = "date"
        return validate_ohlcv(frame, normalized_ticker)

    @staticmethod
    def _flatten(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if not isinstance(frame.columns, pd.MultiIndex):
            return frame
        matching_levels = [
            level for level in range(frame.columns.nlevels) if ticker in frame.columns.get_level_values(level)
        ]
        if len(matching_levels) != 1:
            raise DataValidationError(f"{ticker}: unexpected multi-ticker response")
        ticker_level = matching_levels[0]
        tickers = set(frame.columns.get_level_values(ticker_level))
        if tickers != {ticker}:
            raise DataValidationError(f"{ticker}: multiple ticker data returned")
        return frame.xs(ticker, level=ticker_level, axis=1, drop_level=True)

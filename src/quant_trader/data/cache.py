"""Atomic local Parquet cache for validated daily market data."""

from __future__ import annotations

from datetime import UTC, date, datetime
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from quant_trader.data.validation import DataValidationError, normalize_ticker, validate_ohlcv


class CacheError(DataValidationError):
    """Raised when a local market-data cache entry is missing or corrupt."""


class ParquetMarketCache:
    """Store each ticker below ``root/market`` with a matching JSON sidecar."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write(
        self, ticker: str, frame: pd.DataFrame, retrieved_at: datetime | None = None
    ) -> None:
        normalized_ticker = self._ticker(ticker)
        try:
            canonical = validate_ohlcv(frame, normalized_ticker)
        except DataValidationError as error:
            raise CacheError(str(error)) from error
        timestamp = retrieved_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise CacheError("retrieved_at must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        data_path, metadata_path = self._paths(normalized_ticker)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "ticker": normalized_ticker,
            "retrieved_at": timestamp.isoformat(),
            "max_market_date": canonical.index[-1].date().isoformat(),
            "row_count": len(canonical),
            "schema_version": 1,
        }
        temp_data = data_path.with_name(f".{data_path.name}.{uuid4().hex}.tmp")
        temp_metadata = metadata_path.with_name(f".{metadata_path.name}.{uuid4().hex}.tmp")
        try:
            canonical.to_parquet(temp_data)
            temp_metadata.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            os.replace(temp_data, data_path)
            os.replace(temp_metadata, metadata_path)
        finally:
            temp_data.unlink(missing_ok=True)
            temp_metadata.unlink(missing_ok=True)

    def read(self, ticker: str) -> pd.DataFrame:
        normalized_ticker = self._ticker(ticker)
        data_path, _ = self._paths(normalized_ticker)
        metadata = self.read_metadata(normalized_ticker)
        if not data_path.is_file():
            raise CacheError(f"{normalized_ticker}: cache data is missing")
        self._validate_metadata(metadata, normalized_ticker)
        try:
            frame = pd.read_parquet(data_path)
            canonical = validate_ohlcv(frame, normalized_ticker)
        except (OSError, ValueError, DataValidationError) as error:
            raise CacheError(f"{normalized_ticker}: corrupt cache data") from error
        if len(canonical) != metadata["row_count"]:
            raise CacheError(f"{normalized_ticker}: cache row count does not match metadata")
        if canonical.index[-1].date().isoformat() != metadata["max_market_date"]:
            raise CacheError(f"{normalized_ticker}: cache max market date does not match metadata")
        return canonical.copy()

    def read_metadata(self, ticker: str) -> dict[str, Any]:
        normalized_ticker = self._ticker(ticker)
        _, metadata_path = self._paths(normalized_ticker)
        if not metadata_path.is_file():
            raise CacheError(f"{normalized_ticker}: cache metadata is missing")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CacheError(f"{normalized_ticker}: corrupt cache metadata") from error
        if not isinstance(metadata, dict):
            raise CacheError(f"{normalized_ticker}: corrupt cache metadata")
        self._validate_metadata(metadata, normalized_ticker)
        return metadata.copy()

    def _ticker(self, ticker: str) -> str:
        try:
            return normalize_ticker(ticker)
        except DataValidationError as error:
            raise CacheError(str(error)) from error

    def _paths(self, ticker: str) -> tuple[Path, Path]:
        directory = self.root / "market"
        return directory / f"{ticker}.parquet", directory / f"{ticker}.json"

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any], ticker: str) -> None:
        expected = {"ticker", "retrieved_at", "max_market_date", "row_count", "schema_version"}
        if set(metadata) != expected or type(metadata["ticker"]) is not str or metadata["ticker"] != ticker:
            raise CacheError(f"{ticker}: corrupt cache metadata")
        if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
            raise CacheError(f"{ticker}: unsupported cache schema version")
        if type(metadata["row_count"]) is not int or metadata["row_count"] <= 0:
            raise CacheError(f"{ticker}: corrupt cache metadata")
        try:
            retrieved_at = metadata["retrieved_at"]
            max_market_date = metadata["max_market_date"]
            if type(retrieved_at) is not str or type(max_market_date) is not str:
                raise ValueError
            timestamp = datetime.fromisoformat(retrieved_at)
            if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
                raise ValueError
            if timestamp.isoformat() != retrieved_at:
                raise ValueError
            parsed_date = date.fromisoformat(max_market_date)
            if parsed_date.isoformat() != max_market_date:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise CacheError(f"{ticker}: corrupt cache metadata") from error

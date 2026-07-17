"""Generation-atomic local Parquet cache for validated daily market data."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from quant_trader.data.validation import DataValidationError, normalize_ticker, validate_ohlcv


class CacheError(DataValidationError):
    """Raised when a local market-data cache entry is missing or corrupt."""


class ParquetMarketCache:
    """Store immutable data generations selected by an atomic JSON manifest."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write(self, ticker: str, frame: pd.DataFrame, retrieved_at: datetime | None = None) -> None:
        normalized_ticker = self._ticker(ticker)
        try:
            canonical = validate_ohlcv(frame, normalized_ticker)
        except DataValidationError as error:
            raise CacheError(str(error)) from error
        timestamp = retrieved_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise CacheError(f"{normalized_ticker}: retrieved_at must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        try:
            directory = self._ensure_directory()
        except OSError as error:
            raise CacheError(f"{normalized_ticker}: failed to create cache directory") from error
        generation = uuid4().hex
        data_name = f"{normalized_ticker}.{generation}.parquet"
        data_path = directory / data_name
        manifest_path = self.manifest_path_for(normalized_ticker)
        temp_data = directory / f".{data_name}.{uuid4().hex}.tmp"
        temp_manifest = directory / f".{manifest_path.name}.{uuid4().hex}.tmp"
        try:
            canonical.to_parquet(temp_data)
            self._fsync_file(temp_data)
            self._publish_generation(temp_data, data_path)
            self._fsync_directory(directory)
            manifest = {
                "ticker": normalized_ticker,
                "retrieved_at": timestamp.isoformat(),
                "max_market_date": canonical.index[-1].date().isoformat(),
                "row_count": len(canonical),
                "schema_version": 1,
                "generation": generation,
                "data_file": data_name,
                "sha256": self._digest(data_path),
            }
            self._write_json_fsynced(temp_manifest, manifest)
            self._publish_manifest(temp_manifest, manifest_path)
            self._fsync_directory(directory)
        except OSError as error:
            raise CacheError(f"{normalized_ticker}: failed to write cache generation") from error
        finally:
            temp_data.unlink(missing_ok=True)
            temp_manifest.unlink(missing_ok=True)

    def read(self, ticker: str) -> pd.DataFrame:
        normalized_ticker = self._ticker(ticker)
        metadata = self.read_metadata(normalized_ticker)
        data_path = self._resolve_data_path(metadata, normalized_ticker)
        if not data_path.is_file():
            raise CacheError(f"{normalized_ticker}: cache data is missing")
        try:
            if self._digest(data_path) != metadata["sha256"]:
                raise CacheError(f"{normalized_ticker}: cache data digest does not match manifest")
            frame = pd.read_parquet(data_path)
            canonical = validate_ohlcv(frame, normalized_ticker)
        except CacheError:
            raise
        except Exception as error:
            raise CacheError(f"{normalized_ticker}: corrupt cache data") from error
        if len(canonical) != metadata["row_count"]:
            raise CacheError(f"{normalized_ticker}: cache row count does not match metadata")
        if canonical.index[-1].date().isoformat() != metadata["max_market_date"]:
            raise CacheError(f"{normalized_ticker}: cache max market date does not match metadata")
        return canonical.copy()

    def read_metadata(self, ticker: str) -> dict[str, Any]:
        normalized_ticker = self._ticker(ticker)
        manifest_path = self.manifest_path_for(normalized_ticker)
        if not manifest_path.is_file():
            raise CacheError(f"{normalized_ticker}: cache metadata is missing")
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CacheError(f"{normalized_ticker}: corrupt cache metadata") from error
        if not isinstance(metadata, dict):
            raise CacheError(f"{normalized_ticker}: corrupt cache metadata")
        self._validate_metadata(metadata, normalized_ticker)
        return metadata.copy()

    def path_for(self, ticker: str) -> Path:
        """Return the immutable generation selected by the current manifest."""
        normalized_ticker = self._ticker(ticker)
        return self._resolve_data_path(self.read_metadata(normalized_ticker), normalized_ticker)

    def manifest_path_for(self, ticker: str) -> Path:
        """Return the stable atomic manifest path for a ticker."""
        return self._directory() / f"{self._ticker(ticker)}.json"

    def _ticker(self, ticker: str) -> str:
        try:
            return normalize_ticker(ticker)
        except DataValidationError as error:
            raise CacheError(str(error)) from error

    def _directory(self) -> Path:
        return self.root / "market"

    def _ensure_directory(self) -> Path:
        directory = self._directory()
        missing: list[Path] = []
        current = directory
        while not current.exists():
            missing.append(current)
            if current.parent == current:
                raise OSError(f"no existing ancestor for cache directory {directory}")
            current = current.parent
        for path in reversed(missing):
            try:
                path.mkdir()
            except FileExistsError:
                if not path.is_dir():
                    raise
            self._fsync_directory(path.parent)
        return directory

    def _resolve_data_path(self, metadata: dict[str, Any], ticker: str) -> Path:
        data_file = metadata["data_file"]
        directory = self._directory().resolve()
        candidate = directory / data_file
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as error:
            raise CacheError(f"{ticker}: invalid cache data file path") from error
        if candidate.name != data_file or resolved.parent != directory:
            raise CacheError(f"{ticker}: invalid cache data file path")
        return resolved

    @staticmethod
    def _digest(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as file:
            os.fsync(file.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _publish_generation(source: Path, target: Path) -> None:
        os.replace(source, target)

    @staticmethod
    def _publish_manifest(source: Path, target: Path) -> None:
        os.replace(source, target)

    @staticmethod
    def _write_json_fsynced(path: Path, manifest: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any], ticker: str) -> None:
        expected = {
            "ticker",
            "retrieved_at",
            "max_market_date",
            "row_count",
            "schema_version",
            "generation",
            "data_file",
            "sha256",
        }
        if (
            set(metadata) != expected
            or type(metadata["ticker"]) is not str
            or metadata["ticker"] != ticker
        ):
            raise CacheError(f"{ticker}: corrupt cache metadata")
        if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
            raise CacheError(f"{ticker}: unsupported cache schema version")
        if type(metadata["row_count"]) is not int or metadata["row_count"] <= 0:
            raise CacheError(f"{ticker}: corrupt cache metadata")
        generation, data_file, digest = (
            metadata["generation"],
            metadata["data_file"],
            metadata["sha256"],
        )
        if type(generation) is not str or type(data_file) is not str or type(digest) is not str:
            raise CacheError(f"{ticker}: corrupt cache metadata")
        try:
            if UUID(hex=generation).hex != generation:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise CacheError(f"{ticker}: corrupt cache metadata") from error
        if data_file != f"{ticker}.{generation}.parquet":
            raise CacheError(f"{ticker}: invalid cache data file path")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise CacheError(f"{ticker}: corrupt cache metadata")
        try:
            retrieved_at, max_market_date = metadata["retrieved_at"], metadata["max_market_date"]
            if type(retrieved_at) is not str or type(max_market_date) is not str:
                raise ValueError
            timestamp = datetime.fromisoformat(retrieved_at)
            if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
                raise ValueError
            if (
                timestamp.isoformat() != retrieved_at
                or date.fromisoformat(max_market_date).isoformat() != max_market_date
            ):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise CacheError(f"{ticker}: corrupt cache metadata") from error

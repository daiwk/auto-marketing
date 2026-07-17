from datetime import UTC, datetime
import json

import pandas as pd
import pytest

from quant_trader.data.cache import CacheError, ParquetMarketCache


def ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0], "close": [101.0, 102.0], "volume": [10.0, 20.0]},
        index=pd.DatetimeIndex(["2026-01-02", "2026-01-05"], name="date"),
    )


def test_cache_round_trip_metadata_and_defensive_reads(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    frame = ohlcv()
    cache.write("spy", frame, retrieved_at=datetime(2026, 1, 6, tzinfo=UTC))
    loaded = cache.read("SPY")
    metadata = cache.read_metadata("SPY")
    loaded.loc[loaded.index[0], "close"] = 999.0
    assert cache.read("SPY").iloc[0]["close"] == 101.0
    assert metadata == {"ticker": "SPY", "retrieved_at": "2026-01-06T00:00:00+00:00", "max_market_date": "2026-01-05", "row_count": 2, "schema_version": 1}
    assert not list(tmp_path.rglob("*.tmp"))


def test_cache_rejects_traversal_ticker(tmp_path) -> None:
    with pytest.raises(CacheError, match="ticker"):
        ParquetMarketCache(tmp_path).write("../SPY", ohlcv())


def test_cache_rejects_corrupt_metadata(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    (tmp_path / "market" / "SPY.json").write_text("not json")
    with pytest.raises(CacheError, match="metadata"):
        cache.read("SPY")


def test_cache_rejects_invalid_utf8_metadata(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    (tmp_path / "market" / "SPY.json").write_bytes(b"\xff\xfe")
    with pytest.raises(CacheError, match="SPY.*metadata") as error:
        cache.read("SPY")
    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_cache_requires_data_file_when_metadata_exists(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    (tmp_path / "market" / "SPY.parquet").unlink()
    with pytest.raises(CacheError, match="data is missing"):
        cache.read("SPY")


def test_cache_requires_metadata_file_when_data_exists(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    (tmp_path / "market" / "SPY.json").unlink()
    with pytest.raises(CacheError, match="metadata is missing"):
        cache.read("SPY")


def test_cache_rejects_corrupt_parquet(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    (tmp_path / "market" / "SPY.parquet").write_bytes(b"not parquet")
    with pytest.raises(CacheError, match="corrupt cache data"):
        cache.read("SPY")


def test_cache_rejects_non_utc_retrieval_metadata(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    path = tmp_path / "market" / "SPY.json"
    metadata = json.loads(path.read_text())
    metadata["retrieved_at"] = "2026-01-06T08:00:00+08:00"
    path.write_text(json.dumps(metadata))
    with pytest.raises(CacheError, match="metadata"):
        cache.read("SPY")


@pytest.mark.parametrize(
    "key, value",
    [
        ("schema_version", True),
        ("row_count", True),
        ("max_market_date", "2026-01-05T00:00:00"),
        ("retrieved_at", "2026-01-06T00:00:00"),
    ],
)
def test_cache_rejects_noncanonical_metadata_types(tmp_path, key: str, value: object) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    path = tmp_path / "market" / "SPY.json"
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(CacheError, match="metadata|schema"):
        cache.read("SPY")


@pytest.mark.parametrize(
    "key, value, match",
    [("row_count", 3, "row count"), ("max_market_date", "2026-01-06", "max market date")],
)
def test_cache_rejects_mismatched_metadata(tmp_path, key: str, value: object, match: str) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    path = tmp_path / "market" / "SPY.json"
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(CacheError, match=match):
        cache.read("SPY")

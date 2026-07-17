import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from quant_trader.data.cache import CacheError, ParquetMarketCache


def ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 20.0],
        },
        index=pd.DatetimeIndex(["2026-01-02", "2026-01-05"], name="date"),
    )


def metadata_path(tmp_path) -> object:
    return tmp_path / "market" / "SPY.json"


def test_cache_round_trip_uses_immutable_generation_and_defensive_reads(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("spy", ohlcv(), retrieved_at=datetime(2026, 1, 6, tzinfo=UTC))
    loaded = cache.read("SPY")
    metadata = cache.read_metadata("SPY")
    loaded.loc[loaded.index[0], "close"] = 999.0
    assert cache.read("SPY").iloc[0]["close"] == 101.0
    assert (
        metadata
        | {
            "generation": metadata["generation"],
            "data_file": metadata["data_file"],
            "sha256": metadata["sha256"],
        }
        == metadata
    )
    assert metadata["data_file"] == cache.path_for("SPY").name
    assert cache.path_for("SPY").name.startswith("SPY.")
    assert cache.path_for("SPY").suffix == ".parquet"
    assert not list(tmp_path.rglob("*.tmp"))


def test_cache_failed_generation_write_preserves_previous_manifest(tmp_path, monkeypatch) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    old_metadata = cache.read_metadata("SPY")
    old_data = cache.read("SPY")
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(CacheError, match="failed to write") as error:
        cache.write("SPY", ohlcv())
    assert isinstance(error.value.__cause__, OSError)
    assert cache.read_metadata("SPY") == old_metadata
    pd.testing.assert_frame_equal(cache.read("SPY"), old_data)
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("failure_point", ["generation", "manifest_write", "manifest_publish"])
def test_cache_pre_switch_failures_preserve_previous_generation(
    tmp_path, monkeypatch, failure_point: str
) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    old_manifest = metadata_path(tmp_path).read_bytes()
    old_metadata = cache.read_metadata("SPY")
    old_data = cache.read("SPY")

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError(failure_point)

    target = {
        "generation": "_publish_generation",
        "manifest_write": "_write_json_fsynced",
        "manifest_publish": "_publish_manifest",
    }[failure_point]
    monkeypatch.setattr(cache, target, fail)
    with pytest.raises(CacheError, match="failed to write") as error:
        cache.write("SPY", ohlcv())
    assert isinstance(error.value.__cause__, OSError)
    assert metadata_path(tmp_path).read_bytes() == old_manifest
    assert cache.read_metadata("SPY") == old_metadata
    pd.testing.assert_frame_equal(cache.read("SPY"), old_data)
    assert not list(tmp_path.rglob("*.tmp"))


def test_cache_fsyncs_market_directory_after_each_publish(tmp_path, monkeypatch) -> None:
    (tmp_path / "market").mkdir()
    cache = ParquetMarketCache(tmp_path)
    events: list[str] = []
    publish_generation = cache._publish_generation
    publish_manifest = cache._publish_manifest
    sync_directory = cache._fsync_directory

    def track_generation(*args: object) -> None:
        events.append("generation")
        publish_generation(*args)  # type: ignore[arg-type]

    def track_manifest(*args: object) -> None:
        events.append("manifest")
        publish_manifest(*args)  # type: ignore[arg-type]

    def track_directory(path: object) -> None:
        events.append("directory")
        sync_directory(path)  # type: ignore[arg-type]

    monkeypatch.setattr(cache, "_publish_generation", track_generation)
    monkeypatch.setattr(cache, "_publish_manifest", track_manifest)
    monkeypatch.setattr(cache, "_fsync_directory", track_directory)
    cache.write("SPY", ohlcv())
    assert events == ["generation", "directory", "manifest", "directory"]


def test_cache_fsyncs_parent_when_creating_market_directory(tmp_path, monkeypatch) -> None:
    cache = ParquetMarketCache(tmp_path)
    synced: list[object] = []
    monkeypatch.setattr(cache, "_fsync_directory", lambda path: synced.append(path))
    cache.write("SPY", ohlcv())
    assert synced == [tmp_path, tmp_path / "market", tmp_path / "market"]


def test_cache_fsyncs_each_parent_when_creating_nested_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "one" / "two"
    cache = ParquetMarketCache(root)
    synced: list[object] = []
    monkeypatch.setattr(cache, "_fsync_directory", lambda path: synced.append(path))
    cache.write("SPY", ohlcv())
    assert synced[:3] == [tmp_path, tmp_path / "one", tmp_path / "one" / "two"]


def test_cache_directory_creation_failure_leaves_no_selected_manifest(
    tmp_path, monkeypatch
) -> None:
    cache = ParquetMarketCache(tmp_path / "one" / "two")
    monkeypatch.setattr(
        cache, "_fsync_directory", lambda path: (_ for _ in ()).throw(OSError("fsync"))
    )
    with pytest.raises(CacheError, match="failed to create cache directory") as error:
        cache.write("SPY", ohlcv())
    assert isinstance(error.value.__cause__, OSError)
    assert not cache.manifest_path_for("SPY").exists()
    with pytest.raises(CacheError, match="metadata is missing"):
        cache.read("SPY")


def test_cache_rejects_digest_mismatch(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    cache.path_for("SPY").write_bytes(b"not parquet")
    with pytest.raises(CacheError, match="digest"):
        cache.read("SPY")


def test_cache_rejects_manifest_data_file_traversal(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    path = metadata_path(tmp_path)
    metadata = json.loads(path.read_text())
    metadata["data_file"] = "../outside.parquet"
    path.write_text(json.dumps(metadata))
    with pytest.raises(CacheError, match="data file"):
        cache.read("SPY")


def test_cache_rejects_same_shape_data_with_stale_metadata(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    frame = ohlcv()
    frame.index = pd.DatetimeIndex(["2026-01-05", "2026-01-06"], name="date")
    cache.write("SPY", frame)
    path = metadata_path(tmp_path)
    metadata = json.loads(path.read_text())
    metadata["max_market_date"] = "2026-01-05"
    path.write_text(json.dumps(metadata))
    with pytest.raises(CacheError, match="max market date"):
        cache.read("SPY")


def test_cache_rejects_traversal_ticker(tmp_path) -> None:
    with pytest.raises(CacheError, match="ticker"):
        ParquetMarketCache(tmp_path).write("../SPY", ohlcv())


def test_cache_rejects_corrupt_metadata(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    metadata_path(tmp_path).write_text("not json")
    with pytest.raises(CacheError, match="metadata"):
        cache.read("SPY")


def test_cache_rejects_invalid_utf8_metadata(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    metadata_path(tmp_path).write_bytes(b"\xff\xfe")
    with pytest.raises(CacheError, match="SPY.*metadata") as error:
        cache.read("SPY")
    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_cache_requires_data_file_when_manifest_exists(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    cache.path_for("SPY").unlink()
    with pytest.raises(CacheError, match="data is missing"):
        cache.read("SPY")


def test_cache_requires_manifest_when_generation_exists(tmp_path) -> None:
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", ohlcv())
    metadata_path(tmp_path).unlink()
    with pytest.raises(CacheError, match="metadata is missing"):
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
    path = metadata_path(tmp_path)
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(CacheError, match="metadata|schema"):
        cache.read("SPY")

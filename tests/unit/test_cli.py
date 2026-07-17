from typer.testing import CliRunner

from quant_trader.cli import app
from quant_trader.data.validation import DataValidationError


def test_cli_help_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0


def test_data_sync_prints_concise_provider_error(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise DataValidationError("SPY: Yahoo Finance rate limited; wait a few minutes")

    monkeypatch.setattr("quant_trader.cli.SinaSource.fetch", fail)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "sync",
            "--config",
            "configs/default.yaml",
            "--start",
            "2023-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    assert result.exit_code == 1
    assert "Yahoo Finance rate limited" in result.output
    assert "Traceback" not in result.output


def test_data_sync_uses_sina_by_default(monkeypatch) -> None:
    calls: list[str] = []

    def fetch(_source: object, ticker: str, *args: object) -> object:
        calls.append(ticker)
        return object()

    monkeypatch.setattr("quant_trader.cli.SinaSource.fetch", fetch)
    monkeypatch.setattr(
        "quant_trader.cli.YFinanceSource.fetch",
        lambda *args: (_ for _ in ()).throw(AssertionError("Yahoo should not be called")),
    )
    monkeypatch.setattr("quant_trader.cli.ParquetMarketCache.write", lambda *args: None)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "sync",
            "--config",
            "configs/default.yaml",
            "--start",
            "2023-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]


def test_data_sync_can_explicitly_use_yahoo(monkeypatch) -> None:
    calls: list[str] = []

    def fetch(_source: object, ticker: str, *args: object) -> object:
        calls.append(ticker)
        return object()

    monkeypatch.setattr("quant_trader.cli.YFinanceSource.fetch", fetch)
    monkeypatch.setattr("quant_trader.cli.ParquetMarketCache.write", lambda *args: None)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "sync",
            "--source",
            "yahoo",
            "--config",
            "configs/default.yaml",
            "--start",
            "2023-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 9

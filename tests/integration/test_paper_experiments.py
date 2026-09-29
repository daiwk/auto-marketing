from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_trader.backtest import BacktestResult
from quant_trader.cli import app
from quant_trader.config import Settings
from quant_trader.experiments.run import data_fingerprint


def _frames() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=30, name="date")
    result: dict[str, pd.DataFrame] = {}
    for offset, ticker in enumerate(("SPY", "QQQ")):
        close = 100 + offset + np.arange(len(dates)) * (0.2 + offset * 0.05)
        result[ticker] = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(len(dates), 1_000_000.0),
            },
            index=dates,
        )
    return result


def _config(path: Path) -> Path:
    path.write_text(
        "universe: [SPY, QQQ]\nstrategy:\n  min_average_dollar_volume: 0\n",
        encoding="utf-8",
    )
    return path


def _result() -> BacktestResult:
    dates = pd.bdate_range("2024-01-02", periods=2)
    return BacktestResult(
        pd.Series([100_000.0, 101_000.0], index=dates),
        pd.Series([0.0, 0.1], index=dates),
        (),
        2.0,
    )


class _Provider:
    def complete(self, messages: object) -> str:
        del messages
        return json.dumps(
            {
                "action": "maintain",
                "weight_multiplier": 1,
                "confidence": 0.8,
                "thesis": "bounded",
                "risks": [],
                "invalidation": "signal changes",
                "input_anomalies": [],
                "memory_ids": [],
            }
        )


def test_data_fingerprint_changes_when_market_values_change() -> None:
    frames = _frames()
    settings = Settings(universe=("SPY", "QQQ"))
    original = data_fingerprint(frames, settings)
    changed = {ticker: frame.copy() for ticker, frame in frames.items()}
    changed["SPY"].iloc[0, changed["SPY"].columns.get_loc("close")] += 1
    assert data_fingerprint(changed, settings) != original


def test_finmem_command_writes_bounded_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("quant_trader.cli._frames", lambda *args: _frames())
    monkeypatch.setattr(
        "quant_trader.cli._open_provider",
        lambda *args, **kwargs: (_Provider(), None, "Codex"),
    )
    monkeypatch.setattr("quant_trader.experiments.run.run_backtest", lambda *args, **kw: _result())
    output = tmp_path / "runs"

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "run",
            "finmem",
            "--config",
            str(_config(tmp_path / "config.yaml")),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--llm-provider",
            "codex",
        ],
    )

    assert result.exit_code == 0, result.output
    run = next(output.iterdir())
    assert json.loads((run / "finmem" / "memory.json").read_text()) == []
    decision = json.loads((run / "finmem" / "decisions.json").read_text())
    assert set(decision) == {"ticker", "action", "confidence", "memory_ids", "reason"}
    assert json.loads((run / "finmem" / "result.json").read_text())["metrics"]["costs"] == 2.0
    assert json.loads((run / "summary.json").read_text())["status"] == "completed"


def test_quanta_alpha_command_writes_result_from_date_ticker_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    class FakeMiner:
        def __init__(self, reviewer: object) -> None:
            observed["reviewer"] = reviewer

        def mine(self, panel: pd.DataFrame) -> dict[str, object]:
            observed["index"] = panel.index.names
            observed["columns"] = list(panel.columns)
            return {
                "status": "complete",
                "champion": {"expression": "close"},
                "candidates": [{"expression": "close", "rejection_reason": None}],
                "edges": [],
            }

    monkeypatch.setattr("quant_trader.cli._frames", lambda *args: _frames())
    monkeypatch.setattr(
        "quant_trader.cli._open_provider",
        lambda *args, **kwargs: (_Provider(), None, "Codex"),
    )
    monkeypatch.setattr("quant_trader.experiments.run.QuantaAlphaMiner", FakeMiner)
    output = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "run",
            "quanta-alpha",
            "--config",
            str(_config(tmp_path / "config.yaml")),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--llm-provider",
            "codex",
        ],
    )

    assert result.exit_code == 0, result.output
    run = next(output.iterdir())
    assert observed["index"] == ["date", "ticker"]
    assert observed["columns"] == ["open", "high", "low", "close", "volume", "returns"]
    assert json.loads((run / "quanta_alpha" / "result.json").read_text())["champion"]


@pytest.mark.parametrize(
    ("kind", "miner_name", "module_name"),
    [
        ("alpha-agent", "AlphaAgentMiner", "alpha_agent"),
        ("chain-of-alpha", "ChainOfAlphaMiner", "chain_of_alpha"),
    ],
)
def test_recent_paper_commands_write_bounded_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    miner_name: str,
    module_name: str,
) -> None:
    observed: dict[str, object] = {}

    class FakeMiner:
        def __init__(self, reviewer: object, **kwargs: object) -> None:
            observed["reviewer"] = reviewer
            observed["kwargs"] = kwargs

        def mine(self, panel: pd.DataFrame) -> dict[str, object]:
            observed["index"] = panel.index.names
            return {
                "status": "complete",
                "paper": {"arxiv_id": "test"},
                "champion": {"expression": "rank(returns)", "test_ic": 0.1},
                "candidates": [{"expression": "rank(returns)", "rejection_reason": None}],
            }

    monkeypatch.setattr("quant_trader.cli._frames", lambda *args: _frames())
    monkeypatch.setattr(
        "quant_trader.cli._open_provider",
        lambda *args, **kwargs: (_Provider(), None, "Codex"),
    )
    monkeypatch.setattr(f"quant_trader.experiments.run.{miner_name}", FakeMiner)
    output = tmp_path / "runs"

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "run",
            kind,
            "--config",
            str(_config(tmp_path / "config.yaml")),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--llm-provider",
            "codex",
            "--factor-candidate-limit",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    run = next(output.iterdir())
    payload = json.loads((run / module_name / "result.json").read_text())
    assert payload["champion"]["test_ic"] == 0.1
    assert observed["index"] == ["date", "ticker"]
    assert observed["kwargs"]["candidate_limit"] == 3
    assert json.loads((run / "summary.json").read_text())["metrics"]["provider_calls"] == 0


def test_alpha_arena_never_constructs_a_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("quant_trader.cli._frames", lambda *args: _frames())
    monkeypatch.setattr(
        "quant_trader.cli._open_provider",
        lambda *args: (_ for _ in ()).throw(AssertionError("provider must not open")),
    )
    contestant = tmp_path / "contestant"
    contestant.mkdir()
    output = tmp_path / "runs"

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "run",
            "alpha-arena",
            "--config",
            str(_config(tmp_path / "config.yaml")),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--contestant-run",
            str(contestant),
        ],
    )

    assert result.exit_code == 0, result.output
    run = next(output.iterdir())
    arena = json.loads((run / "alpha_arena" / "result.json").read_text())
    rows = {row["name"]: row for row in arena["leaderboard"]}
    assert rows["contestant"]["status"] == "failed"
    assert rows["finmem"]["status"] == "absent"


@pytest.mark.parametrize(
    ("kind", "module"),
    [("openpm", "openpm"), ("ktd-fin", "ktd_fin")],
)
def test_auditable_benchmark_commands_reuse_existing_backtest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    module: str,
) -> None:
    frames = _frames()
    monkeypatch.setattr("quant_trader.cli._frames", lambda *args: frames)
    monkeypatch.setattr(
        "quant_trader.cli._open_provider",
        lambda *args: (_ for _ in ()).throw(AssertionError("provider must not open")),
    )
    dates = frames["SPY"].index
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "runs": {
                    "rules_only": {
                        "equity": {
                            stamp.date().isoformat(): 100_000 + index * 100
                            for index, stamp in enumerate(dates)
                        },
                        "gross_exposure": {
                            stamp.date().isoformat(): 0.5 for stamp in dates
                        },
                        "fills": [
                            {
                                "decision_id": "candidate:20240105:digest",
                                "execution_date": "2024-01-08",
                                "shares": 10,
                                "price": 102,
                            }
                        ],
                        "metrics": {"costs": 1.5},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runs"

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "run",
            kind,
            "--config",
            str(_config(tmp_path / "config.yaml")),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--source-run",
            str(source),
        ],
    )

    assert result.exit_code == 0, result.output
    run = next(output.iterdir())
    details = json.loads((run / module / "result.json").read_text())
    assert details["paper"]["arxiv_id"] in {"2608.09988", "2605.28359"}
    summary = json.loads((run / "summary.json").read_text())
    assert summary["metrics"]["provider_calls"] == 0

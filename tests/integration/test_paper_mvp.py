from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_trader.backtest import MaintainReviewer, run_backtest
from quant_trader.cli import app
from quant_trader.config import Settings
from quant_trader.core.models import ApprovedOrder, SignalIntent, SignalSide
from quant_trader.execution import Account, HardRisk, Simulator
from quant_trader.paper import run_once
from quant_trader.state import PaperState


def _intent(identifier: str, ticker: str = "SPY", weight: float = 0.5) -> SignalIntent:
    signal = datetime(2025, 1, 2, 20, tzinfo=UTC)
    return SignalIntent(
        decision_id=identifier,
        ticker=ticker,
        side=SignalSide.BUY,
        proposed_weight=weight,
        signal_time=signal,
        earliest_execution_time=signal + timedelta(days=1),
        stop_price=90,
        invalidation="test",
        strategy_version="test",
        prompt_version="test",
        llm_cache_key=identifier,
    )


def _frames() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=280)
    result = {}
    for offset, ticker in enumerate(("SPY", "QQQ")):
        close = 100 + offset + np.arange(len(dates)) * (0.25 + offset * 0.02)
        result[ticker] = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(len(dates), 1_000_000.0),
            },
            index=dates.rename("date"),
        )
    return result


def _settings() -> Settings:
    return Settings.model_validate(
        {"universe": ["SPY", "QQQ"], "strategy": {"min_average_dollar_volume": 0}}
    )


def test_accounting_conservation_fractional_and_idempotent() -> None:
    account = Account(1_000, {}, 1_000)
    simulator = Simulator(account)
    order = ApprovedOrder(
        decision_id="one", ticker="SPY", target_weight=0.15, execution_date=date(2025, 1, 3)
    )
    fills = simulator.execute((order,), {"SPY": 100})
    assert fills[0].shares % 1 != 0
    assert account.equity({"SPY": fills[0].price}) == pytest.approx(1_000 - fills[0].commission)
    assert simulator.execute((order,), {"SPY": 100}) == ()


def test_hard_risk_caps_reduces_and_latches_halt() -> None:
    risk = HardRisk()
    intents = tuple(_intent(str(index), ticker, 0.9) for index, ticker in enumerate(("SPY", "QQQ")))
    reduced = risk.approve(intents, 0.10, date(2025, 1, 3))
    assert all(float(order.target_weight) <= 0.075 for order in reduced)
    halted = risk.approve(intents, 0.15, date(2025, 1, 3))
    assert risk.halted and all(order.target_weight == 0 for order in halted)
    assert all(order.target_weight == 0 for order in risk.approve(intents, 0, date(2025, 1, 3)))
    risk.reset()
    assert any(order.target_weight > 0 for order in risk.approve(intents, 0, date(2025, 1, 3)))


def test_offline_backtest_is_next_open_and_bounded() -> None:
    frames = _frames()
    result = run_backtest(frames, _settings(), reviewer=MaintainReviewer())
    assert result.fills
    for fill in result.fills:
        expected_open = frames[fill.ticker].loc[pd.Timestamp(fill.execution_date), "open"]
        direction = 1 if fill.shares > 0 else -1
        assert fill.price == pytest.approx(expected_open * (1 + direction * 0.001))
    assert result.gross_exposure.max() <= 0.80


def test_offline_paper_cycle_persists_and_rejects_duplicate(tmp_path) -> None:
    state = PaperState(tmp_path / "paper.db")
    frames = _frames()
    identifiers = run_once(state, frames, _settings(), reviewer=MaintainReviewer())
    assert identifiers
    assert state.status()["fill_count"] > 0
    with pytest.raises(ValueError, match="already processed"):
        run_once(state, frames, _settings(), reviewer=MaintainReviewer())
    state.close()


def test_cli_is_paper_only_and_requires_confirmation(tmp_path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "live" not in [
        line.strip().split()[0] for line in help_result.stdout.lower().splitlines() if line.strip()
    ]
    db = tmp_path / "paper.db"
    assert runner.invoke(app, ["paper", "init", "--db", str(db)]).exit_code == 0
    config = tmp_path / "config.yaml"
    config.write_text("universe: [SPY, QQQ]\n")
    result = runner.invoke(app, ["paper", "run", "--db", str(db), "--config", str(config)])
    assert result.exit_code != 0
    assert "--confirm" in result.output

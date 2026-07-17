"""One-cycle paper runner; there is deliberately no scheduler or broker adapter."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, time

import pandas as pd

from quant_trader.backtest import MaintainReviewer, _strategy
from quant_trader.config import Settings
from quant_trader.execution import CostModel, HardRisk, Simulator
from quant_trader.features.snapshot import build_feature_snapshot
from quant_trader.state import PaperState


def run_once(
    state: PaperState,
    frames: Mapping[str, pd.DataFrame],
    settings: Settings,
    *,
    reviewer: object | None = None,
) -> tuple[str, ...]:
    """Decide on the penultimate shared close and execute at the latest shared open."""
    if not frames:
        raise ValueError("market data is required")
    dates = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    if len(dates) < 2:
        raise ValueError("a complete snapshot and a later execution bar are required")
    decision_date, execution_date = dates[-2], dates[-1]
    if any(frame.index[-1] != execution_date for frame in frames.values()):
        raise ValueError("stale or misaligned market data")
    if state.cycle_processed(execution_date.date().isoformat()):
        raise ValueError("decision already processed")

    closes = {ticker: float(frame.loc[decision_date, "close"]) for ticker, frame in frames.items()}
    opens = {ticker: float(frame.loc[execution_date, "open"]) for ticker, frame in frames.items()}
    account = state.latest_account(settings.paper.initial_cash)
    equity = account.mark(closes)
    weights = account.weights(closes)
    strategy = _strategy(settings, reviewer or MaintainReviewer())
    snapshot = build_feature_snapshot(frames, decision_date)
    intents = strategy.generate(
        snapshot,
        signal_time=datetime.combine(decision_date.date(), time(20), UTC),
        earliest_execution_time=datetime.combine(execution_date.date(), time(14, 30), UTC),
        cash_weight=account.cash / equity,
        current_weights=weights,
        drawdown=account.drawdown(closes),
    )
    if not intents:
        raise ValueError("no actionable decisions")
    if any(state.processed(intent.decision_id) for intent in intents):
        raise ValueError("decision already processed")
    risk = HardRisk(
        settings.risk.max_position_weight,
        settings.risk.max_gross_exposure,
        settings.risk.reduce_drawdown,
        settings.risk.halt_drawdown,
    )
    orders = risk.approve(intents, account.drawdown(closes), execution_date.date())
    fills = Simulator(
        account, CostModel(settings.execution.slippage_bps, settings.execution.commission_bps)
    ).execute(orders, opens)
    records = [
        {
            "decision_id": order.decision_id,
            "ticker": order.ticker,
            "target_weight": float(order.target_weight),
            "execution_date": order.execution_date.isoformat(),
        }
        for order in orders
    ]
    account.mark(
        {ticker: float(frame.loc[execution_date, "close"]) for ticker, frame in frames.items()}
    )
    state.save_cycle(
        as_of=execution_date.date().isoformat(), decisions=records, fills=fills, account=account
    )
    return tuple(order.decision_id for order in orders)

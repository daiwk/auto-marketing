"""Chronological weekly-decision, next-open paper backtest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time
from math import sqrt

import numpy as np
import pandas as pd

from quant_trader.config import Settings
from quant_trader.core.models import ApprovedOrder
from quant_trader.execution import Account, CostModel, Fill, HardRisk, Simulator
from quant_trader.features.snapshot import build_feature_snapshot
from quant_trader.llm.base import MessageInput
from quant_trader.strategies.v1_rules_llm.strategy import V1RulesLLMStrategy, V1StrategyConfig


class MaintainReviewer:
    """Deterministic rules-only reviewer with the same constrained JSON contract."""

    def complete(self, messages: tuple[MessageInput, ...]) -> str:
        del messages
        return json.dumps(
            {
                "action": "maintain",
                "weight_multiplier": 1,
                "confidence": 1,
                "thesis": "Deterministic rules-only approval.",
                "risks": [],
                "invalidation": "Rules eligibility no longer holds.",
                "input_anomalies": [],
            }
        )


@dataclass(slots=True)
class BacktestResult:
    equity: pd.Series
    gross_exposure: pd.Series
    fills: tuple[Fill, ...]
    costs: float

    def metrics(self) -> dict[str, float | int]:
        returns = self.equity.pct_change().dropna()
        years = max(len(returns) / 252, 1 / 252)
        total = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        volatility = float(returns.std(ddof=1) * sqrt(252)) if len(returns) > 1 else 0.0
        annualized = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
        sharpe = float(returns.mean() / returns.std(ddof=1) * sqrt(252)) if volatility else 0.0
        drawdown = self.equity / self.equity.cummax() - 1
        return {
            "total_return": float(total),
            "annualized_return": annualized,
            "annualized_volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()),
            "trade_count": len(self.fills),
            "costs": self.costs,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics(),
            "equity": {str(index.date()): value for index, value in self.equity.items()},
            "gross_exposure": {
                str(index.date()): value for index, value in self.gross_exposure.items()
            },
            "fills": [
                {**asdict(fill), "execution_date": fill.execution_date.isoformat()}
                for fill in self.fills
            ],
        }


def _strategy(settings: Settings, reviewer: object) -> V1RulesLLMStrategy:
    return V1RulesLLMStrategy(
        reviewer,  # type: ignore[arg-type]
        model=settings.llm.model,
        prompt_version=settings.llm.prompt_version,
        config=V1StrategyConfig(
            max_candidates=settings.strategy.max_candidates,
            min_dollar_volume=settings.strategy.min_average_dollar_volume,
            target_volatility=settings.strategy.target_volatility,
            max_position_weight=settings.risk.max_position_weight,
            max_gross_exposure=settings.risk.max_gross_exposure,
            atr_multiple=settings.risk.atr_multiple,
        ),
    )


def run_backtest(
    frames: Mapping[str, pd.DataFrame], settings: Settings, *, reviewer: object | None = None
) -> BacktestResult:
    """Run only on cached frames; all decisions see bars through that day's close."""
    if not frames:
        raise ValueError("at least one market frame is required")
    dates = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    if len(dates) < 2:
        raise ValueError("at least two shared market dates are required")
    weekly_last: dict[tuple[int, int], pd.Timestamp] = {}
    for market_date in dates:
        iso = market_date.isocalendar()
        weekly_last[(iso.year, iso.week)] = market_date
    decision_dates = set(weekly_last.values())

    account = Account(settings.paper.initial_cash, {}, settings.paper.initial_cash)
    simulator = Simulator(
        account,
        CostModel(settings.execution.slippage_bps, settings.execution.commission_bps),
    )
    risk = HardRisk(
        settings.risk.max_position_weight,
        settings.risk.max_gross_exposure,
        settings.risk.reduce_drawdown,
        settings.risk.halt_drawdown,
    )
    strategy = _strategy(settings, reviewer or MaintainReviewer())
    pending: tuple[ApprovedOrder, ...] = ()
    fills: list[Fill] = []
    equity_values: list[float] = []
    gross_values: list[float] = []
    used_dates: list[pd.Timestamp] = []
    for index, market_date in enumerate(dates):
        opens = {ticker: float(frame.loc[market_date, "open"]) for ticker, frame in frames.items()}
        closes = {
            ticker: float(frame.loc[market_date, "close"]) for ticker, frame in frames.items()
        }
        if pending:
            fills.extend(simulator.execute(pending, opens))
            pending = ()
        equity = account.mark(closes)
        weights = account.weights(closes)
        used_dates.append(pd.Timestamp(market_date))
        equity_values.append(equity)
        gross_values.append(sum(weights.values()))
        if market_date in decision_dates and index + 1 < len(dates):
            snapshot = build_feature_snapshot(frames, market_date)
            next_date = dates[index + 1]
            intents = strategy.generate(
                snapshot,
                signal_time=datetime.combine(market_date.date(), time(20), UTC),
                earliest_execution_time=datetime.combine(next_date.date(), time(14, 30), UTC),
                cash_weight=account.cash / equity,
                current_weights=weights,
                drawdown=account.drawdown(closes),
            )
            pending = risk.approve(intents, account.drawdown(closes), next_date.date())
    return BacktestResult(
        pd.Series(equity_values, index=used_dates, name="equity"),
        pd.Series(gross_values, index=used_dates, name="gross_exposure"),
        tuple(fills),
        float(sum(fill.cost for fill in fills)),
    )


def buy_and_hold(frame: pd.DataFrame, initial_cash: float) -> BacktestResult:
    shares = initial_cash / float(frame.iloc[0]["open"])
    equity = frame["close"].astype(float) * shares
    gross = pd.Series(np.ones(len(frame)), index=frame.index, name="gross_exposure")
    return BacktestResult(equity.rename("equity"), gross, (), 0.0)

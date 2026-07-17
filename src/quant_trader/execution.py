"""Small, paper-only account, risk, and execution engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from math import fsum

from quant_trader.core.models import ApprovedOrder, SignalIntent


@dataclass(slots=True)
class Account:
    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    high_water_mark: float = 100_000.0

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + fsum(
            shares * prices[ticker] for ticker, shares in self.positions.items()
        )

    def mark(self, prices: Mapping[str, float]) -> float:
        value = self.equity(prices)
        self.high_water_mark = max(self.high_water_mark, value)
        return value

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        equity = self.equity(prices)
        return {
            ticker: shares * prices[ticker] / equity for ticker, shares in self.positions.items()
        }

    def drawdown(self, prices: Mapping[str, float]) -> float:
        return max(0.0, 1.0 - self.equity(prices) / self.high_water_mark)


@dataclass(frozen=True, slots=True)
class CostModel:
    slippage_bps: float = 10.0
    commission_bps: float = 1.0

    def fill_price(self, open_price: float, shares: float) -> float:
        direction = 1.0 if shares > 0 else -1.0
        return open_price * (1.0 + direction * self.slippage_bps / 10_000.0)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps / 10_000.0


@dataclass(frozen=True, slots=True)
class Fill:
    decision_id: str
    ticker: str
    execution_date: date
    shares: float
    price: float
    commission: float
    slippage: float = 0.0

    @property
    def cost(self) -> float:
        return self.commission + self.slippage


class HardRisk:
    """Non-overridable long-only portfolio limits with a latched drawdown halt."""

    def __init__(
        self,
        max_position: float = 0.15,
        max_gross: float = 0.80,
        reduce_drawdown: float = 0.10,
        halt_drawdown: float = 0.15,
    ) -> None:
        self.max_position = max_position
        self.max_gross = max_gross
        self.reduce_drawdown = reduce_drawdown
        self.halt_drawdown = halt_drawdown
        self.halted = False

    def reset(self) -> None:
        self.halted = False

    def approve(
        self, intents: tuple[SignalIntent, ...], drawdown: float, execution_date: date
    ) -> tuple[ApprovedOrder, ...]:
        if drawdown >= self.halt_drawdown:
            self.halted = True
        scale = 0.5 if drawdown >= self.reduce_drawdown else 1.0
        requested = {
            intent.ticker: 0.0
            if self.halted
            else min(float(intent.proposed_weight), self.max_position) * scale
            for intent in intents
        }
        gross = fsum(requested.values())
        if gross > self.max_gross:
            requested = {
                ticker: weight * self.max_gross / gross for ticker, weight in requested.items()
            }
        return tuple(
            ApprovedOrder(
                decision_id=intent.decision_id,
                ticker=intent.ticker,
                target_weight=requested[intent.ticker],
                execution_date=execution_date,
                reason_codes=(*intent.reason_codes, "hard_risk"),
            )
            for intent in intents
        )


class Simulator:
    """Rebalance fractional paper positions at supplied opens, once per decision."""

    def __init__(
        self, account: Account, costs: CostModel | None = None, min_cash_weight: float = 0.20
    ) -> None:
        self.account = account
        self.costs = costs or CostModel()
        self.min_cash_weight = min_cash_weight
        self.processed: set[str] = set()

    def execute(
        self, orders: tuple[ApprovedOrder, ...], open_prices: Mapping[str, float]
    ) -> tuple[Fill, ...]:
        fresh = [order for order in orders if order.decision_id not in self.processed]
        if not fresh:
            return ()
        equity = self.account.equity(open_prices)
        trades: list[tuple[ApprovedOrder, float]] = []
        for order in fresh:
            current = self.account.positions.get(order.ticker, 0.0)
            desired = equity * float(order.target_weight) / open_prices[order.ticker]
            trades.append((order, desired - current))
        trades.sort(key=lambda item: item[1])  # sells fund buys
        fills: list[Fill] = []
        for order, shares in trades:
            if abs(shares) < 1e-12:
                self.processed.add(order.decision_id)
                continue
            price = self.costs.fill_price(open_prices[order.ticker], shares)
            notional = shares * price
            commission = self.costs.commission(notional)
            spendable = max(0.0, self.account.cash - equity * self.min_cash_weight)
            if shares > 0 and notional + commission > spendable:
                shares = spendable / (price * (1 + self.costs.commission_bps / 10_000))
                notional = shares * price
                commission = self.costs.commission(notional)
            if abs(shares) < 1e-12:
                self.processed.add(order.decision_id)
                continue
            self.account.cash -= notional + commission
            updated = self.account.positions.get(order.ticker, 0.0) + shares
            if abs(updated) < 1e-12:
                self.account.positions.pop(order.ticker, None)
            else:
                self.account.positions[order.ticker] = updated
            self.processed.add(order.decision_id)
            slippage = abs(shares) * abs(price - open_prices[order.ticker])
            fills.append(
                Fill(
                    order.decision_id,
                    order.ticker,
                    order.execution_date,
                    shares,
                    price,
                    commission,
                    slippage,
                )
            )
        return tuple(fills)

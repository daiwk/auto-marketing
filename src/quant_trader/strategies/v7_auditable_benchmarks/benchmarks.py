"""Deterministic MVP adaptations of OpenPM and KTD-Fin."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

_OPENPM_PAPER = {
    "title": "OpenPM: Auditable Point-in-Time Evaluation for LLM Portfolio-Management Agents",
    "arxiv_id": "2608.09988",
    "url": "https://arxiv.org/abs/2608.09988",
}
_KTD_PAPER = {
    "title": "From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents",
    "arxiv_id": "2605.28359",
    "url": "https://arxiv.org/abs/2605.28359",
}
_SIGNAL_DATE = re.compile(r":(\d{8}):")


def _runs(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, Mapping) or not runs:
        raise ValueError("source run must contain non-empty runs")
    return runs


def _strategy(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    runs = _runs(payload)
    for name in ("llm", "rules_only", "result"):
        run = runs.get(name)
        if isinstance(run, Mapping):
            return name, run
    raise ValueError("source run has no supported strategy result")


def _series(raw: object, name: str) -> pd.Series:
    if not isinstance(raw, Mapping) or len(raw) < 2:
        raise ValueError(f"{name} must contain at least two dated values")
    values = pd.Series({pd.Timestamp(str(key)): float(value) for key, value in raw.items()})
    values = values.sort_index()
    if values.index.has_duplicates or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} must be finite with unique dates")
    return values.astype(float)


def _metrics(
    equity: pd.Series, fills: list[Mapping[str, Any]], costs: float
) -> dict[str, float | int]:
    returns = equity.pct_change().dropna()
    downside = returns[returns < 0]
    drawdown = equity / equity.cummax() - 1
    turnover = sum(
        abs(float(fill.get("shares", 0))) * float(fill.get("price", 0)) for fill in fills
    ) / float(equity.mean())
    volatility = float(returns.std(ddof=1) * sqrt(252)) if len(returns) > 1 else 0.0
    sortino = (
        float(returns.mean() / downside.std(ddof=1) * sqrt(252))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else 0.0
    )
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    return {
        "total_return": total_return,
        "annualized_volatility": volatility,
        "sharpe": (
            float(returns.mean() / returns.std(ddof=1) * sqrt(252))
            if volatility
            else 0.0
        ),
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
        "turnover": turnover,
        "trade_count": len(fills),
        "costs": costs,
        "cost_ratio": costs / float(equity.iloc[0]),
    }


class OpenPMAuditor:
    """Audit point-in-time execution, constraints, turnover, and cost sensitivity."""

    def __init__(
        self,
        *,
        max_gross_exposure: float,
        max_drawdown: float,
        max_turnover: float,
    ) -> None:
        if not 0 < max_gross_exposure <= 1:
            raise ValueError("max_gross_exposure must be in (0, 1]")
        if not 0 < max_drawdown <= 1:
            raise ValueError("max_drawdown must be in (0, 1]")
        if max_turnover <= 0:
            raise ValueError("max_turnover must be positive")
        self.max_gross_exposure = max_gross_exposure
        self.max_drawdown = max_drawdown
        self.max_turnover = max_turnover

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        name, run = _strategy(payload)
        equity = _series(run.get("equity"), "equity")
        gross = _series(run.get("gross_exposure"), "gross_exposure")
        raw_fills = run.get("fills", [])
        if not isinstance(raw_fills, list) or not all(
            isinstance(fill, Mapping) for fill in raw_fills
        ):
            raise ValueError("fills must be a list of objects")
        fills: list[Mapping[str, Any]] = list(raw_fills)
        raw_metrics = run.get("metrics")
        costs = float(raw_metrics.get("costs", 0)) if isinstance(raw_metrics, Mapping) else 0.0
        metrics = _metrics(equity, fills, costs)

        contamination: list[dict[str, str]] = []
        for fill in fills:
            try:
                execution = pd.Timestamp(str(fill.get("execution_date")))
            except (TypeError, ValueError):
                contamination.append(
                    {"execution_date": "未知", "reason": "成交时间无效"}
                )
                continue
            match = _SIGNAL_DATE.search(str(fill.get("decision_id", "")))
            if match is None:
                contamination.append(
                    {
                        "execution_date": str(execution.date()),
                        "reason": "决策编号缺少信号日期",
                    }
                )
                continue
            signal = pd.Timestamp(match.group(1))
            if signal >= execution:
                contamination.append(
                    {
                        "execution_date": str(execution.date()),
                        "reason": "信号日期没有早于成交日期",
                    }
                )
            if execution not in equity.index:
                contamination.append(
                    {
                        "execution_date": str(execution.date()),
                        "reason": "净值账本中不存在该成交日期",
                    }
                )

        observed_gross = float(gross.max())
        observed_drawdown = abs(float((equity / equity.cummax() - 1).min()))
        observed_turnover = float(metrics["turnover"])
        constraints = [
            {
                "name": "最大总仓位",
                "limit": self.max_gross_exposure,
                "observed": observed_gross,
                "passed": observed_gross <= self.max_gross_exposure + 1e-9,
            },
            {
                "name": "最大回撤",
                "limit": self.max_drawdown,
                "observed": observed_drawdown,
                "passed": observed_drawdown <= self.max_drawdown + 1e-9,
            },
            {
                "name": "最大换手率",
                "limit": self.max_turnover,
                "observed": observed_turnover,
                "passed": observed_turnover <= self.max_turnover + 1e-9,
            },
        ]
        traded_notional = observed_turnover * float(equity.mean())
        cost_curve = []
        for cost_bps in (0, 5, 10, 20, 50):
            extra_cost = traded_notional * cost_bps / 10_000
            terminal = float(equity.iloc[-1]) + costs - extra_cost
            cost_curve.append(
                {
                    "cost_bps": cost_bps,
                    "total_return": terminal / float(equity.iloc[0]) - 1,
                }
            )
        return {
            "paper": _OPENPM_PAPER,
            "source_strategy": name,
            "equity": {stamp.date().isoformat(): float(value) for stamp, value in equity.items()},
            "mandate": (
                f"仅做多；总仓位不超过 {self.max_gross_exposure:.0%}；最大回撤不超过 "
                f"{self.max_drawdown:.0%}；累计换手不超过 {self.max_turnover:.2f} 倍。"
            ),
            "metrics": metrics,
            "contamination_certificate": {
                "passed": not contamination,
                "checked_fills": len(fills),
                "violations": contamination,
            },
            "constraint_report": {
                "passed": all(item["passed"] for item in constraints),
                "constraints": constraints,
            },
            "cost_sensitivity": cost_curve,
            "note": "日线、无市场冲击的诊断性上界，不代表可部署收益。",
        }


class KTDFinBenchmark:
    """Export masked data views and attribute returns to observable style factors."""

    def run(
        self,
        payload: Mapping[str, Any],
        frames: Mapping[str, pd.DataFrame],
    ) -> dict[str, Any]:
        name, run = _strategy(payload)
        equity = _series(run.get("equity"), "equity")
        raw_fills = run.get("fills", [])
        if not isinstance(raw_fills, list) or not all(
            isinstance(fill, Mapping) for fill in raw_fills
        ):
            raise ValueError("fills must be a list of objects")
        fills: list[Mapping[str, Any]] = list(raw_fills)
        raw_metrics = run.get("metrics")
        costs = float(raw_metrics.get("costs", 0)) if isinstance(raw_metrics, Mapping) else 0.0
        metrics = _metrics(equity, fills, costs)

        aliases = {
            ticker: f"资产{index + 1:03d}" for index, ticker in enumerate(sorted(frames))
        }
        dates = equity.index
        masked_sample: list[dict[str, object]] = []
        for day_index, stamp in enumerate(dates[-5:]):
            for ticker in sorted(frames)[:3]:
                frame = frames[ticker]
                if stamp in frame.index:
                    masked_sample.append(
                        {
                            "day": f"D{len(dates) - 5 + day_index:04d}",
                            "asset": aliases[ticker],
                            "return_1d": float(
                                frame.loc[:stamp, "close"].pct_change().iloc[-1]
                            ),
                        }
                    )

        portfolio_returns = equity.pct_change().dropna().rename("portfolio")
        factors: dict[str, pd.Series] = {}
        if "SPY" in frames:
            factors["market"] = frames["SPY"]["close"].astype(float).pct_change()
        if "QQQ" in frames and "SPY" in frames:
            factors["growth"] = (
                frames["QQQ"]["close"].astype(float).pct_change() - factors["market"]
            )
        if "IWM" in frames and "SPY" in frames:
            factors["size"] = frames["IWM"]["close"].astype(float).pct_change() - factors["market"]
        if not factors:
            raise ValueError("KTD-Fin attribution requires SPY or style benchmark frames")
        joined = pd.concat([portfolio_returns, *factors.values()], axis=1, join="inner").dropna()
        factor_names = list(factors)
        joined.columns = ["portfolio", *factor_names]
        if len(joined) <= len(factor_names) + 1:
            raise ValueError("insufficient overlapping dates for attribution")
        x = np.column_stack([np.ones(len(joined)), joined[factor_names].to_numpy()])
        y = joined["portfolio"].to_numpy()
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        predicted = x @ coefficients
        residual = y - predicted
        denominator = float(np.square(y - y.mean()).sum())
        r_squared = 1 - float(np.square(residual).sum()) / denominator if denominator else 0.0
        exposures = {
            name_: float(value)
            for name_, value in zip(factor_names, coefficients[1:], strict=True)
        }
        contributions = {
            name_: float(exposures[name_] * joined[name_].sum()) for name_ in factor_names
        }
        daily_alpha = float(coefficients[0])
        return {
            "paper": _KTD_PAPER,
            "source_strategy": name,
            "equity": {stamp.date().isoformat(): float(value) for stamp, value in equity.items()},
            "masking_protocol": {
                "levels": ["明文", "日期匿名", "股票匿名", "完全匿名"],
                "ticker_aliases": len(aliases),
                "date_aliases": len(dates),
                "mapping_digest": hashlib.sha256(
                    repr(sorted(aliases.items())).encode()
                ).hexdigest(),
                "fully_masked_sample": masked_sample,
                "real_identifiers_exposed_in_sample": False,
            },
            "metrics": {**metrics, "attribution_r_squared": r_squared},
            "attribution": {
                "method": "轻量 Barra 风格时间序列归因（市场、成长、规模）",
                "annualized_selection_alpha": daily_alpha * 252,
                "factor_exposures": exposures,
                "cumulative_factor_contributions": contributions,
                "residual_volatility": float(pd.Series(residual).std(ddof=1) * sqrt(252)),
                "r_squared": r_squared,
            },
            "note": (
                "该 MVP 导出防记忆泄漏的数据视图并做风格归因，"
                "不等同于论文完整 CSI300/Barra 复现。"
            ),
        }

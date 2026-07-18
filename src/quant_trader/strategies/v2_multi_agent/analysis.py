"""Prepare one point-in-time TradingAgents analysis without calling an LLM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_trader.config import Settings
from quant_trader.data.validation import normalize_ticker
from quant_trader.features.snapshot import build_feature_snapshot
from quant_trader.llm.base import ChatMessage
from quant_trader.strategies.v1_rules_llm.prompt import render_review_prompt
from quant_trader.strategies.v1_rules_llm.rules import rank_candidates


@dataclass(frozen=True, slots=True)
class PreparedAnalysis:
    ticker: str
    as_of: date
    eligible: bool
    messages: tuple[ChatMessage, ...] | None
    reason: str


def prepare_analysis(
    frames: Mapping[str, pd.DataFrame],
    settings: Settings,
    ticker: str,
    as_of: date,
) -> PreparedAnalysis:
    """Build the same constrained candidate prompt used by the backtest."""
    normalized = normalize_ticker(ticker)
    snapshot = build_feature_snapshot(frames, as_of)
    candidates = rank_candidates(
        snapshot.rows.values(),
        max_candidates=settings.strategy.max_candidates,
        min_dollar_volume=settings.strategy.min_average_dollar_volume,
        target_volatility=settings.strategy.target_volatility,
        max_position_weight=settings.risk.max_position_weight,
        max_gross_exposure=settings.risk.max_gross_exposure,
    )
    candidate = next((item for item in candidates if item.ticker == normalized), None)
    feature = snapshot.rows.get(normalized)
    if candidate is None or feature is None:
        return PreparedAnalysis(
            ticker=normalized,
            as_of=as_of,
            eligible=False,
            messages=None,
            reason="ticker is not eligible under the deterministic rules at --as-of",
        )
    return PreparedAnalysis(
        ticker=normalized,
        as_of=as_of,
        eligible=True,
        messages=render_review_prompt(
            candidate,
            feature,
            cash_weight=1.0,
            current_weight=0.0,
            drawdown=0.0,
        ),
        reason="eligible",
    )

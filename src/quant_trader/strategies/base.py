"""Minimal strategy protocol for daily signal generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from quant_trader.features.snapshot import FeatureSnapshot

if TYPE_CHECKING:
    from quant_trader.strategies.v1_rules_llm.rules import Candidate
    from quant_trader.strategies.v1_rules_llm.strategy import StrategyDecision


class Strategy(Protocol):
    """A versioned strategy that returns immutable candidate-level decisions."""

    version: str

    def generate(
        self,
        snapshot: FeatureSnapshot,
        candidates: Sequence[Candidate],
        *,
        signal_time: datetime,
        earliest_execution_time: datetime,
        cash_weight: float,
        current_weights: Mapping[str, float],
        drawdown: float,
    ) -> tuple[StrategyDecision, ...]: ...

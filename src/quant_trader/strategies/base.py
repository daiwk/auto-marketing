"""Minimal strategy protocol for daily signal generation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from quant_trader.core.models import SignalIntent
from quant_trader.features.snapshot import FeatureSnapshot


class Strategy(Protocol):
    """A versioned strategy that returns only shared immutable signal intents."""

    version: str

    def generate(
        self,
        snapshot: FeatureSnapshot,
        *,
        signal_time: datetime,
        earliest_execution_time: datetime,
        cash_weight: float,
        current_weights: Mapping[str, float],
        drawdown: float,
    ) -> tuple[SignalIntent, ...]: ...

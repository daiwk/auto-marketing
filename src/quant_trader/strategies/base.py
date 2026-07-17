"""Minimal strategy protocol for daily signal generation."""

from __future__ import annotations

from typing import Protocol

from quant_trader.core.models import SignalIntent
from quant_trader.features.snapshot import FeatureSnapshot


class Strategy(Protocol):
    """A versioned strategy that may use a broad review dependency."""

    version: str

    def generate(
        self, snapshot: FeatureSnapshot, reviews: object | None = None
    ) -> list[SignalIntent]: ...

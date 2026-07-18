"""Strict, immutable contracts for durable paper-experiment artifacts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from quant_trader.validation import StrictNumber

RunIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]
BoundedLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]
EventStage = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
EventMessage = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StrictFrozenModel(BaseModel):
    """Base contract that prevents unreviewed artifact fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExperimentEvent(StrictFrozenModel):
    run_id: RunIdentifier
    sequence: int = Field(ge=1)
    at: datetime
    kind: BoundedLabel
    stage: EventStage
    message: EventMessage
    status: ExperimentStatus | None = None

    @field_validator("at")
    @classmethod
    def require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware")
        return value


class ExperimentManifest(StrictFrozenModel):
    """Reproducibility metadata deliberately excluding secrets and raw model data."""

    run_id: RunIdentifier
    experiment: BoundedLabel
    code_version: BoundedLabel
    data_fingerprint: BoundedLabel
    data_start: date
    data_end: date
    universe: tuple[BoundedLabel, ...] = Field(min_length=1, max_length=500)
    provider: BoundedLabel
    model: BoundedLabel
    attempt_limit: int = Field(ge=1, le=1_000)
    initial_cash: StrictNumber = Field(gt=0)
    commission_bps: StrictNumber = Field(ge=0, le=10_000)
    slippage_bps: StrictNumber = Field(ge=0, le=10_000)
    max_position_weight: StrictNumber = Field(gt=0, le=1)
    max_gross_exposure: StrictNumber = Field(gt=0, le=10)
    max_drawdown: StrictNumber = Field(gt=0, le=1)

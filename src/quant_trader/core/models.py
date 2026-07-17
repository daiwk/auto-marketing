"""Validated immutable workflow contracts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from quant_trader.validation import StrictNumber, USEquityTicker

NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ReviewAction(StrEnum):
    MAINTAIN = "maintain"
    REDUCE = "reduce"
    REJECT = "reject"


class SignalSide(StrEnum):
    """V1 is long-only, so every actionable intent must explicitly be a buy."""

    BUY = "buy"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMReview(_ImmutableModel):
    action: ReviewAction
    weight_multiplier: StrictNumber = Field(ge=0, le=1)
    confidence: StrictNumber = Field(ge=0, le=1)
    thesis: NonEmptyText
    risks: Annotated[tuple[NonEmptyText, ...], Field(max_length=20)] = ()
    invalidation: NonEmptyText
    input_anomalies: Annotated[tuple[NonEmptyText, ...], Field(max_length=20)] = ()


class SignalIntent(_ImmutableModel):
    decision_id: Identifier
    ticker: USEquityTicker
    side: SignalSide
    proposed_weight: StrictNumber = Field(ge=0, le=1)
    signal_time: datetime
    earliest_execution_time: datetime
    stop_price: StrictNumber = Field(gt=0)
    invalidation: NonEmptyText
    reason_codes: tuple[NonEmptyText, ...] = ()
    strategy_version: Identifier
    prompt_version: Identifier
    llm_cache_key: Identifier

    @field_validator("signal_time", "earliest_execution_time")
    @classmethod
    def require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_execution_after_signal(self) -> SignalIntent:
        if self.earliest_execution_time.date() <= self.signal_time.date():
            raise ValueError("earliest_execution_time must be on a later date than signal_time")
        return self


class ApprovedOrder(_ImmutableModel):
    decision_id: Identifier
    ticker: USEquityTicker
    target_weight: StrictNumber = Field(ge=0, le=1)
    execution_date: date
    reason_codes: tuple[NonEmptyText, ...] = ()

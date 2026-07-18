"""Strict immutable contracts for the TradingAgents MVP."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from quant_trader.core.models import LLMReview, ReviewAction
from quant_trader.data.validation import normalize_ticker
from quant_trader.validation import StrictNumber, USEquityTicker

BoundedText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
]
BoundedLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]
MetricLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class RoleName(StrEnum):
    MARKET = "market_analyst"
    SENTIMENT = "sentiment_analyst"
    NEWS = "news_analyst"
    FUNDAMENTALS = "fundamentals_analyst"
    BULL = "bull_researcher"
    BEAR = "bear_researcher"
    RESEARCH_MANAGER = "research_manager"
    TRADER = "trader"
    AGGRESSIVE_RISK = "aggressive_risk_analyst"
    NEUTRAL_RISK = "neutral_risk_analyst"
    CONSERVATIVE_RISK = "conservative_risk_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"


class ReportStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class Stance(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class NewsItem(StrictFrozenModel):
    published_at: date
    headline: BoundedText
    summary: BoundedText


class SentimentItem(StrictFrozenModel):
    observed_at: date
    source: BoundedLabel
    text: BoundedText


class FundamentalsContext(StrictFrozenModel):
    reported_at: date
    metrics: Annotated[dict[MetricLabel, StrictNumber], Field(max_length=50)]


class TickerContext(StrictFrozenModel):
    news: Annotated[tuple[NewsItem, ...], Field(max_length=20)] = ()
    sentiment: Annotated[tuple[SentimentItem, ...], Field(max_length=20)] = ()
    fundamentals: FundamentalsContext | None = None


class VisibleContext(StrictFrozenModel):
    news: tuple[NewsItem, ...] = ()
    sentiment: tuple[SentimentItem, ...] = ()
    fundamentals: FundamentalsContext | None = None


class ExternalContext(StrictFrozenModel):
    tickers: Annotated[dict[USEquityTicker, TickerContext], Field(max_length=100)] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_ticker_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw_tickers = value.get("tickers", {})
        if not isinstance(raw_tickers, dict):
            return value
        normalized: dict[str, object] = {}
        for raw_ticker, item in raw_tickers.items():
            ticker = normalize_ticker(raw_ticker)
            if ticker in normalized:
                raise ValueError("context contains a canonical ticker collision")
            normalized[ticker] = item
        return {**value, "tickers": normalized}


class RoleReport(StrictFrozenModel):
    role: RoleName
    status: ReportStatus
    stance: Stance
    confidence: StrictNumber = Field(ge=0, le=1)
    summary: BoundedText
    evidence: Annotated[tuple[BoundedText, ...], Field(max_length=10)] = ()
    risks: Annotated[tuple[BoundedText, ...], Field(max_length=10)] = ()
    input_anomalies: Annotated[tuple[BoundedText, ...], Field(max_length=10)] = ()


class TraderProposal(StrictFrozenModel):
    action: ReviewAction
    weight_multiplier: StrictNumber = Field(ge=0, le=1)
    confidence: StrictNumber = Field(ge=0, le=1)
    thesis: BoundedText
    risks: Annotated[tuple[BoundedText, ...], Field(max_length=10)] = ()
    invalidation: BoundedText


class DecisionTrace(StrictFrozenModel):
    ticker: USEquityTicker
    as_of: date
    provider: BoundedLabel
    provider_calls: int = Field(ge=0, le=12)
    reports: Annotated[tuple[RoleReport, ...], Field(max_length=11)]
    proposal: TraderProposal | None = None
    final_review: LLMReview
    failure_role: RoleName | None = None

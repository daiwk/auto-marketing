"""Sanitized immutable events for observing a TradingAgents workflow."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import model_validator

from quant_trader.core.models import LLMReview
from quant_trader.strategies.v2_multi_agent.models import (
    BoundedLabel,
    RoleName,
    RoleReport,
    StrictFrozenModel,
    TraderProposal,
)
from quant_trader.validation import USEquityTicker


class AgentEventKind(StrEnum):
    WORKFLOW_STARTED = "workflow_started"
    ROLE_STARTED = "role_started"
    ROLE_COMPLETED = "role_completed"
    ROLE_SKIPPED = "role_skipped"
    ROLE_FAILED = "role_failed"
    TRADER_COMPLETED = "trader_completed"
    FINAL_COMPLETED = "final_completed"
    WORKFLOW_COMPLETED = "workflow_completed"


class AgentEvent(StrictFrozenModel):
    kind: AgentEventKind
    ticker: USEquityTicker
    as_of: date
    provider: BoundedLabel
    role: RoleName | None = None
    report: RoleReport | None = None
    proposal: TraderProposal | None = None
    final_review: LLMReview | None = None

    @model_validator(mode="after")
    def require_matching_payload(self) -> AgentEvent:
        role_kinds = {
            AgentEventKind.ROLE_STARTED,
            AgentEventKind.ROLE_COMPLETED,
            AgentEventKind.ROLE_SKIPPED,
            AgentEventKind.ROLE_FAILED,
        }
        if (self.kind in role_kinds) != (self.role is not None):
            raise ValueError("role events require exactly one role")
        report_kinds = {
            AgentEventKind.ROLE_COMPLETED,
            AgentEventKind.ROLE_SKIPPED,
            AgentEventKind.ROLE_FAILED,
        }
        if (self.kind in report_kinds) != (self.report is not None):
            raise ValueError("completed role events require exactly one report")
        if self.report is not None and self.report.role is not self.role:
            raise ValueError("event report role must match event role")
        if (self.kind is AgentEventKind.TRADER_COMPLETED) != (self.proposal is not None):
            raise ValueError("trader completion requires exactly one proposal")
        decision_kinds = {
            AgentEventKind.FINAL_COMPLETED,
            AgentEventKind.WORKFLOW_COMPLETED,
        }
        if (self.kind in decision_kinds) != (self.final_review is not None):
            raise ValueError("decision completion requires exactly one final review")
        return self

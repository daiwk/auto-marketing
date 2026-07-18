"""Fixed, fail-closed TradingAgents role orchestration."""

from __future__ import annotations

import json
import traceback
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from quant_trader.core.models import LLMReview, ReviewAction
from quant_trader.llm.base import LLMReviewer, MessageInput, canonical_messages
from quant_trader.llm.parsing import parse_review
from quant_trader.strategies.v2_multi_agent.context import context_for
from quant_trader.strategies.v2_multi_agent.models import (
    DecisionTrace,
    ExternalContext,
    ReportStatus,
    RoleName,
    RoleReport,
    Stance,
    TraderProposal,
)
from quant_trader.strategies.v2_multi_agent.prompts import (
    render_portfolio_prompt,
    render_report_prompt,
    render_trader_prompt,
)

MAX_ROLE_OUTPUT = 16 * 1024


def _safe_json(raw: str) -> object:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("role response must be nonblank JSON")
    if len(raw) > MAX_ROLE_OUTPUT or len(raw.encode("utf-8")) > MAX_ROLE_OUTPUT:
        raise ValueError("role response exceeds the allowed size")
    return json.loads(raw)


def _synthetic_reject(role: RoleName) -> LLMReview:
    return LLMReview(
        action=ReviewAction.REJECT,
        weight_multiplier=0,
        confidence=0,
        thesis="The multi-agent workflow did not produce a valid decision.",
        risks=("multi_agent_failure",),
        invalidation="No position without a complete valid workflow.",
        input_anomalies=(f"failed_role:{role.value}",),
    )


def _unavailable(role: RoleName) -> RoleReport:
    return RoleReport(
        role=role,
        status=ReportStatus.UNAVAILABLE,
        stance=Stance.NEUTRAL,
        confidence=0,
        summary="No point-in-time external context was supplied for this role.",
        input_anomalies=("context_unavailable",),
    )


def _clear_error(error: BaseException) -> None:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        current.__traceback__ = None


def _request(messages: Sequence[MessageInput]) -> tuple[dict[str, object], str, date]:
    canonical = canonical_messages(messages)
    user_messages = [message.content for message in canonical if message.role == "user"]
    if not user_messages:
        raise ValueError("multi-agent workflow requires a V1 user payload")
    payload = _safe_json(user_messages[0])
    if not isinstance(payload, dict):
        raise ValueError("V1 user payload must be an object")
    candidate = payload.get("candidate")
    features = payload.get("features")
    portfolio = payload.get("portfolio")
    if (
        not isinstance(candidate, dict)
        or not isinstance(features, dict)
        or not isinstance(portfolio, dict)
    ):
        raise ValueError("V1 user payload is missing required objects")
    ticker = candidate.get("ticker")
    feature_ticker = features.get("ticker")
    as_of = features.get("as_of")
    if not isinstance(ticker, str) or ticker != feature_ticker or not isinstance(as_of, str):
        raise ValueError("V1 candidate and feature provenance do not match")
    try:
        point_in_time = date.fromisoformat(as_of)
    except ValueError:
        raise ValueError("V1 as_of must be an ISO date") from None
    return payload, ticker, point_in_time


class TradingAgentsReviewer:
    """Adapt a fixed TradingAgents workflow to the existing reviewer contract."""

    def __init__(
        self,
        provider: LLMReviewer,
        *,
        provider_name: str,
        external_context: ExternalContext | None = None,
    ) -> None:
        if not callable(getattr(provider, "complete", None)):
            raise TypeError("provider must implement complete(messages)")
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("provider_name must be nonblank")
        self._provider = provider
        self._provider_name = provider_name.strip()
        self._context = external_context or ExternalContext()
        self.traces: list[DecisionTrace] = []

    def _complete(self, messages: Sequence[MessageInput]) -> str:
        return self._provider.complete(messages)

    def _report(self, role: RoleName, payload: Mapping[str, object]) -> RoleReport:
        raw = self._complete(render_report_prompt(role, payload))
        parsed = RoleReport.model_validate(_safe_json(raw))
        if parsed.role is not role or parsed.status is not ReportStatus.AVAILABLE:
            raise ValueError("role response provenance does not match the requested role")
        return parsed

    def _trader(self, payload: Mapping[str, object]) -> TraderProposal:
        raw = self._complete(render_trader_prompt(payload))
        return TraderProposal.model_validate(_safe_json(raw))

    def _portfolio(self, payload: Mapping[str, object], proposal: TraderProposal) -> LLMReview:
        result = parse_review(self._complete(render_portfolio_prompt(payload)))
        if result.weight_multiplier > proposal.weight_multiplier:
            raise ValueError("portfolio manager cannot increase the trader multiplier")
        return result

    @staticmethod
    def _dump_reports(reports: Sequence[RoleReport]) -> list[dict[str, Any]]:
        return [report.model_dump(mode="json") for report in reports]

    def complete(self, messages: Sequence[MessageInput]) -> str:
        original, ticker, as_of = _request(messages)
        reports: list[RoleReport] = []
        proposal: TraderProposal | None = None
        provider_calls = 0
        current_role = RoleName.MARKET

        def report(role: RoleName, payload: Mapping[str, object]) -> RoleReport:
            nonlocal current_role, provider_calls
            current_role = role
            provider_calls += 1
            return self._report(role, payload)

        try:
            reports.append(report(RoleName.MARKET, {"request": original}))
            visible = context_for(self._context, ticker, as_of)
            optional = (
                (RoleName.SENTIMENT, visible.sentiment),
                (RoleName.NEWS, visible.news),
                (RoleName.FUNDAMENTALS, visible.fundamentals),
            )
            for role, value in optional:
                if not value:
                    reports.append(_unavailable(role))
                else:
                    context_payload = (
                        value.model_dump(mode="json")
                        if hasattr(value, "model_dump")
                        else [item.model_dump(mode="json") for item in value]
                    )
                    reports.append(report(role, {"context": context_payload, "ticker": ticker}))

            analyst_reports = self._dump_reports(reports)
            bull = report(RoleName.BULL, {"analyst_reports": analyst_reports})
            reports.append(bull)
            bear = report(RoleName.BEAR, {"analyst_reports": analyst_reports})
            reports.append(bear)
            manager = report(
                RoleName.RESEARCH_MANAGER,
                {"bull": bull.model_dump(mode="json"), "bear": bear.model_dump(mode="json")},
            )
            reports.append(manager)

            current_role = RoleName.TRADER
            provider_calls += 1
            proposal = self._trader(
                {"request": original, "reports": self._dump_reports(reports)}
            )

            risk_reports: list[RoleReport] = []
            for role in (
                RoleName.AGGRESSIVE_RISK,
                RoleName.NEUTRAL_RISK,
                RoleName.CONSERVATIVE_RISK,
            ):
                risk = report(
                    role,
                    {
                        "proposal": proposal.model_dump(mode="json"),
                        "research_manager": manager.model_dump(mode="json"),
                    },
                )
                risk_reports.append(risk)
                reports.append(risk)

            current_role = RoleName.PORTFOLIO_MANAGER
            provider_calls += 1
            final = self._portfolio(
                {
                    "proposal": proposal.model_dump(mode="json"),
                    "risk_reports": self._dump_reports(risk_reports),
                },
                proposal,
            )
            failure_role = None
        except Exception as error:
            _clear_error(error)
            final = _synthetic_reject(current_role)
            failure_role = current_role

        self.traces.append(
            DecisionTrace(
                ticker=ticker,
                as_of=as_of,
                provider=self._provider_name,
                provider_calls=provider_calls,
                reports=tuple(reports),
                proposal=proposal,
                final_review=final,
                failure_role=failure_role,
            )
        )
        return json.dumps(final.model_dump(mode="json"), sort_keys=True)

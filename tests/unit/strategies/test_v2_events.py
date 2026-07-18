from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date

from quant_trader.core.models import ReviewAction
from quant_trader.features.snapshot import FeatureRow
from quant_trader.llm.base import MessageInput
from quant_trader.llm.parsing import parse_review
from quant_trader.strategies.v1_rules_llm.prompt import render_review_prompt
from quant_trader.strategies.v1_rules_llm.rules import Candidate
from quant_trader.strategies.v2_multi_agent.events import AgentEvent, AgentEventKind
from quant_trader.strategies.v2_multi_agent.models import RoleName
from quant_trader.strategies.v2_multi_agent.orchestrator import TradingAgentsReviewer


def _messages() -> tuple[MessageInput, ...]:
    return render_review_prompt(
        Candidate("SPY", 1.0, 0.2, 3.0, 500.0, 0.1),
        FeatureRow(
            "SPY", date(2025, 12, 31), 300, 500, 450, 0.03, 0.08, 0.12, 0.2, 3, 1e9
        ),
        cash_weight=1,
        current_weight=0,
        drawdown=0,
    )


def _report(role: RoleName) -> str:
    return json.dumps(
        {
            "role": role.value,
            "status": "available",
            "stance": "neutral",
            "confidence": 0.5,
            "summary": f"Sanitized {role.value} report.",
            "evidence": [],
            "risks": [],
            "input_anomalies": [],
        }
    )


def _outputs() -> list[str]:
    return [
        _report(RoleName.MARKET),
        _report(RoleName.BULL),
        _report(RoleName.BEAR),
        _report(RoleName.RESEARCH_MANAGER),
        json.dumps(
            {
                "action": "reduce",
                "weight_multiplier": 0.5,
                "confidence": 0.5,
                "thesis": "Reduce exposure.",
                "risks": [],
                "invalidation": "Trend improves.",
            }
        ),
        _report(RoleName.AGGRESSIVE_RISK),
        _report(RoleName.NEUTRAL_RISK),
        _report(RoleName.CONSERVATIVE_RISK),
        json.dumps(
            {
                "action": "reduce",
                "weight_multiplier": 0.5,
                "confidence": 0.5,
                "thesis": "Risk review approved reduction.",
                "risks": [],
                "invalidation": "Trend improves.",
                "input_anomalies": [],
            }
        ),
    ]


class ScriptedReviewer:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def complete(self, messages: Sequence[MessageInput]) -> str:
        return self.outputs.pop(0)


def test_market_only_workflow_emits_sanitized_ordered_events() -> None:
    events: list[AgentEvent] = []
    reviewer = TradingAgentsReviewer(
        ScriptedReviewer(_outputs()), provider_name="MiniMax", on_event=events.append
    )

    result = parse_review(reviewer.complete(_messages()))

    assert result.action is ReviewAction.REDUCE
    assert events[0].kind is AgentEventKind.WORKFLOW_STARTED
    assert [event.kind for event in events if event.role is RoleName.SENTIMENT] == [
        AgentEventKind.ROLE_SKIPPED
    ]
    assert [event.kind for event in events if event.role is RoleName.TRADER] == [
        AgentEventKind.ROLE_STARTED
    ]
    assert [event.kind for event in events if event.role is RoleName.PORTFOLIO_MANAGER] == [
        AgentEventKind.ROLE_STARTED
    ]
    assert events[-1].kind is AgentEventKind.WORKFLOW_COMPLETED
    assert events[-1].final_review is not None
    serialized = "".join(event.model_dump_json() for event in events).lower()
    assert "raw_output" not in serialized
    assert "prompt" not in serialized


def test_broken_event_observer_cannot_change_the_decision() -> None:
    def fail(_event: AgentEvent) -> None:
        raise RuntimeError("dashboard failure must stay isolated")

    reviewer = TradingAgentsReviewer(
        ScriptedReviewer(_outputs()), provider_name="Codex", on_event=fail
    )

    result = parse_review(reviewer.complete(_messages()))

    assert result.action is ReviewAction.REDUCE
    assert result.weight_multiplier == 0.5
    assert reviewer.traces[0].failure_role is None


def test_failed_role_emits_sanitized_failure_before_completion() -> None:
    events: list[AgentEvent] = []
    reviewer = TradingAgentsReviewer(
        ScriptedReviewer(["not-json"]), provider_name="MiniMax", on_event=events.append
    )

    result = parse_review(reviewer.complete(_messages()))

    assert result.action is ReviewAction.REJECT
    assert [event.kind for event in events[-2:]] == [
        AgentEventKind.ROLE_FAILED,
        AgentEventKind.WORKFLOW_COMPLETED,
    ]
    assert events[-2].report is not None
    assert events[-2].report.summary == "该角色未能生成有效且满足边界要求的回复。"

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from quant_trader.core.models import ReviewAction
from quant_trader.features.snapshot import FeatureRow, FeatureSnapshot
from quant_trader.llm.minimax import MiniMaxError
from quant_trader.strategies.v1_rules_llm.rules import Candidate
from quant_trader.strategies.v1_rules_llm.strategy import V1RulesLLMStrategy, review_candidate


def feature(ticker: str = "AAA", **changes: object) -> FeatureRow:
    values: dict[str, object] = {
        "ticker": ticker,
        "as_of": date(2025, 1, 2),
        "observations": 260,
        "close": 100.0,
        "sma_200": 90.0,
        "return_20": 0.02,
        "return_60": 0.04,
        "return_120": 0.06,
        "volatility_20": 0.2,
        "atr_14": 2.0,
        "average_dollar_volume_20": 30_000_000.0,
    }
    values.update(changes)
    return FeatureRow(**values)  # type: ignore[arg-type]


def candidate(ticker: str = "AAA", **changes: object) -> Candidate:
    values: dict[str, object] = {
        "ticker": ticker,
        "score": 1.5,
        "annualized_volatility": 0.2,
        "atr_14": 2.0,
        "close": 100.0,
        "base_weight": 0.1,
    }
    values.update(changes)
    return Candidate(**values)  # type: ignore[arg-type]


def snapshot(*rows: FeatureRow) -> FeatureSnapshot:
    return FeatureSnapshot(pd.Timestamp(date(2025, 1, 2)), {row.ticker: row for row in rows}, {})


def response(action: str = "maintain", multiplier: float = 1.0) -> str:
    return json.dumps(
        {
            "action": action,
            "weight_multiplier": multiplier,
            "confidence": 0.8,
            "thesis": "Trend remains intact.",
            "risks": ["Normal volatility."],
            "invalidation": "Close below trend.",
            "input_anomalies": [],
        }
    )


class Reviewer:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.messages: list[tuple[object, ...]] = []

    def complete(self, messages: object) -> str:
        self.messages.append(tuple(messages))  # type: ignore[arg-type]
        outcome = self.outputs.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


def strategy(reviewer: Reviewer) -> V1RulesLLMStrategy:
    return V1RulesLLMStrategy(reviewer, model="model-a")


def times() -> tuple[datetime, datetime]:
    signal = datetime(2025, 1, 2, 16, tzinfo=UTC)
    return signal, signal + timedelta(hours=1)


def generated(reviewer: Reviewer, candidates: object = None, snap: FeatureSnapshot | None = None):
    source = snap or snapshot(feature())
    signal, execution = times()
    return strategy(reviewer).generate(
        source,
        (candidate(),) if candidates is None else candidates,
        signal_time=signal,
        earliest_execution_time=execution,
        cash_weight=0.8,
        current_weights={"AAA": 0.1},
        drawdown=0.02,
    )


def test_maintain_keeps_base_weight_and_audits_immutable_result() -> None:
    decisions = generated(Reviewer([response()]))

    assert isinstance(decisions, tuple)
    assert decisions[0].intent.proposed_weight == 0.1
    assert decisions[0].review_outcome.review.action is ReviewAction.MAINTAIN
    assert decisions[0].review_outcome.raw_outputs == (response(),)
    with pytest.raises((AttributeError, TypeError)):
        decisions[0].failure_reason = "nope"  # type: ignore[misc]


def test_reduce_never_increases_target_and_reject_has_zero_weight() -> None:
    reduced = generated(Reviewer([response("reduce", 0.4)]))[0]
    rejected = generated(Reviewer([response("reject", 0.0)]))[0]

    assert reduced.intent.proposed_weight == pytest.approx(0.04)
    assert reduced.intent.proposed_weight <= reduced.candidate.base_weight
    assert rejected.intent.proposed_weight == 0


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (response("maintain", 0.4), response()),
        ("not json", response()),
    ],
)
def test_invalid_first_output_gets_one_clean_repair(first: str, second: str) -> None:
    reviewer = Reviewer([first, second])
    decision = generated(reviewer)[0]

    assert len(reviewer.messages) == 2
    assert decision.review_outcome.repair_used is True
    assert decision.review_outcome.review.action is ReviewAction.MAINTAIN
    assert first not in reviewer.messages[1][-1].content
    assert "failed schema" in reviewer.messages[1][-1].content.lower()


def test_two_invalid_outputs_fail_closed_without_leaking_model_body() -> None:
    bad = "MODEL_BODY_SENTINEL"
    decision = generated(Reviewer([bad, bad]))[0]

    assert decision.intent.proposed_weight == 0
    assert decision.review_outcome.failure_reason == "invalid_review"
    assert decision.review_outcome.raw_outputs == (bad, bad)
    assert bad not in " ".join(decision.intent.reason_codes)


def test_provider_error_does_not_repair_or_retain_exception_text() -> None:
    error = MiniMaxError("EXCEPTION_SENTINEL", status_code=None, attempts=1)
    reviewer = Reviewer([error])
    decision = generated(reviewer)[0]

    assert len(reviewer.messages) == 1
    assert decision.intent.proposed_weight == 0
    assert decision.review_outcome.raw_outputs == ()
    assert "EXCEPTION_SENTINEL" not in str(decision)


def test_repair_provider_error_and_unexpected_reviewer_error_are_safe() -> None:
    repair_error = generated(Reviewer(["bad", MiniMaxError("BODY", status_code=None, attempts=1)]))[
        0
    ]
    unexpected = generated(Reviewer([RuntimeError("BODY")]))[0]

    assert repair_error.intent.proposed_weight == 0
    assert repair_error.review_outcome.failure_reason == "repair_provider_failure"
    assert unexpected.intent.proposed_weight == 0
    assert "BODY" not in str(unexpected)


def test_review_helper_leaves_unexpected_errors_to_the_public_strategy_boundary() -> None:
    with pytest.raises(RuntimeError, match="BODY"):
        review_candidate(
            Reviewer([RuntimeError("BODY")]),
            ({"role": "user", "content": "x"},),
            model="model",
            prompt_version="v1",
        )


def test_one_candidate_failure_does_not_stop_another_or_add_a_ticker() -> None:
    rows = (feature("AAA"), feature("BBB"))
    reviewer = Reviewer(["bad", "still bad", response()])
    decisions = generated(reviewer, (candidate("AAA"), candidate("BBB")), snapshot(*rows))

    assert [decision.intent.ticker for decision in decisions] == ["AAA", "BBB"]
    assert decisions[0].intent.proposed_weight == 0
    assert decisions[1].intent.proposed_weight == 0.1


def test_input_validation_ordering_ids_and_cache_are_deterministic() -> None:
    rows = (feature("AAA"), feature("BBB"))
    candidates = (candidate("BBB"), candidate("AAA"))
    first = generated(Reviewer([response(), response()]), candidates, snapshot(*rows))
    second = generated(
        Reviewer([response(), response()]), tuple(reversed(candidates)), snapshot(*rows)
    )

    assert [decision.candidate.ticker for decision in first] == ["AAA", "BBB"]
    assert [decision.intent.decision_id for decision in first] == [
        decision.intent.decision_id for decision in second
    ]
    assert [decision.intent.llm_cache_key for decision in first] == [
        decision.intent.llm_cache_key for decision in second
    ]
    assert first[0].intent.llm_cache_key[:24] in first[0].intent.decision_id
    with pytest.raises(ValueError, match="duplicate"):
        generated(Reviewer([]), (candidate("AAA"), candidate("AAA")))
    with pytest.raises(ValueError, match="missing"):
        generated(Reviewer([]), (candidate("ZZZ"),))
    with pytest.raises(ValueError, match="match"):
        generated(Reviewer([]), (candidate(close=99),))


def test_time_portfolio_and_stop_validation_are_fail_closed_per_candidate() -> None:
    reviewer = Reviewer([response()])
    source = snapshot(feature())
    signal, execution = times()
    with pytest.raises(ValueError):
        strategy(reviewer).generate(
            source,
            (candidate(),),
            signal_time=signal.replace(tzinfo=None),
            earliest_execution_time=execution,
            cash_weight=0.8,
            current_weights={},
            drawdown=0,
        )
    with pytest.raises(ValueError, match="unknown"):
        strategy(reviewer).generate(
            source,
            (candidate(),),
            signal_time=signal,
            earliest_execution_time=execution,
            cash_weight=0.8,
            current_weights={"ZZZ": 0.1},
            drawdown=0,
        )
    with pytest.raises(ValueError, match="collision"):
        strategy(reviewer).generate(
            source,
            (candidate(),),
            signal_time=signal,
            earliest_execution_time=execution,
            cash_weight=0.8,
            current_weights={"aaa": 0.1, "AAA": 0.1},
            drawdown=0,
        )
    invalid_stop = generated(
        Reviewer([response()]),
        (candidate(close=1, atr_14=2),),
        snapshot(feature(close=1, atr_14=2)),
    )[0]
    assert invalid_stop.intent.proposed_weight == 0
    assert invalid_stop.intent.stop_price > 0
    assert "invalid_stop" in invalid_stop.intent.reason_codes


def test_review_helper_bounds_raw_response_and_does_not_mutate_messages() -> None:
    original = ({"role": "user", "content": "x"},)
    outcome = review_candidate(
        Reviewer(["x" * 70_000, response()]), original, model="model", prompt_version="v1"
    )

    assert len(outcome.raw_outputs[0]) <= 16_384
    assert original == ({"role": "user", "content": "x"},)
    assert outcome.repair_used is True

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import UTC, date, datetime, timedelta, timezone
from inspect import signature

import pandas as pd
import pytest

from quant_trader.core.models import ReviewAction, SignalIntent
from quant_trader.features.snapshot import FeatureRow, FeatureSnapshot
from quant_trader.llm.cache import review_cache_key
from quant_trader.llm.minimax import MiniMaxError
from quant_trader.strategies.v1_rules_llm import strategy as strategy_module
from quant_trader.strategies.v1_rules_llm.prompt import render_review_prompt
from quant_trader.strategies.v1_rules_llm.rules import Candidate, rank_candidates
from quant_trader.strategies.v1_rules_llm.strategy import (
    MAX_AUDIT_RAW_BYTES,
    MAX_AUDIT_RAW_CHARS,
    V1RulesLLMStrategy,
    V1StrategyConfig,
    review_candidate,
)


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


def strategy(reviewer: Reviewer, config: V1StrategyConfig | None = None) -> V1RulesLLMStrategy:
    return V1RulesLLMStrategy(reviewer, model="model-a", config=config)


def times() -> tuple[datetime, datetime]:
    return (
        datetime(2025, 1, 2, 16, tzinfo=UTC),
        datetime(2025, 1, 3, 9, 30, tzinfo=UTC),
    )


def decide(
    reviewer: Reviewer,
    snap: FeatureSnapshot | None = None,
    *,
    config: V1StrategyConfig | None = None,
    cash_weight: float = 0.8,
    current_weights: dict[str, float] | None = None,
):
    signal, execution = times()
    return strategy(reviewer, config).decide(
        snap or snapshot(feature()),
        signal_time=signal,
        earliest_execution_time=execution,
        cash_weight=cash_weight,
        current_weights={"AAA": 0.1} if current_weights is None else current_weights,
        drawdown=0.02,
    )


def test_generate_returns_only_core_intents_and_decide_retains_immutable_audit() -> None:
    reviewer = Reviewer([response(), response()])
    subject = strategy(reviewer)
    signal, execution = times()
    kwargs = {
        "signal_time": signal,
        "earliest_execution_time": execution,
        "cash_weight": 0.8,
        "current_weights": {"AAA": 0.1},
        "drawdown": 0.02,
    }

    decisions = subject.decide(snapshot(feature()), **kwargs)
    intents = subject.generate(snapshot(feature()), **kwargs)

    assert all(isinstance(intent, SignalIntent) for intent in intents)
    assert intents[0] == decisions[0].intent
    assert decisions[0].intent.proposed_weight == decisions[0].candidate.base_weight  # type: ignore[union-attr]
    assert decisions[0].review_outcome.review.action is ReviewAction.MAINTAIN
    assert decisions[0].review_outcome.raw_outputs[0].text == response()
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        decisions[0].review_outcome.failure_reason = "nope"  # type: ignore[misc]


def test_reduce_never_increases_target_and_reject_has_zero_weight() -> None:
    reduced = decide(Reviewer([response("reduce", 0.4)]))[0]
    rejected = decide(Reviewer([response("reject", 0.0)]))[0]

    assert reduced.candidate is not None
    assert reduced.intent.proposed_weight == pytest.approx(reduced.candidate.base_weight * 0.4)
    assert reduced.intent.proposed_weight <= reduced.candidate.base_weight
    assert rejected.intent.proposed_weight == 0


@pytest.mark.parametrize(
    ("first", "second"),
    [(response("maintain", 0.4), response()), ("not json", response())],
)
def test_invalid_first_output_gets_one_clean_repair(first: str, second: str) -> None:
    reviewer = Reviewer([first, second])
    decision = decide(reviewer)[0]

    assert len(reviewer.messages) == 2
    assert decision.review_outcome.repair_used is True
    assert decision.review_outcome.review.action is ReviewAction.MAINTAIN
    assert first not in reviewer.messages[1][-1].content  # type: ignore[union-attr]
    assert "failed schema" in reviewer.messages[1][-1].content.lower()  # type: ignore[union-attr]


def test_two_invalid_outputs_have_one_authoritative_fail_closed_state() -> None:
    reviewer = Reviewer(["FIRST_INVALID", "SECOND_INVALID"])
    decision = decide(reviewer)[0]

    assert len(reviewer.messages) == 2
    assert decision.failure_reason == "invalid_review"
    assert decision.review_outcome.failure_reason == "invalid_review"
    assert decision.review_outcome.review.action is ReviewAction.REJECT
    assert tuple(output.text for output in decision.review_outcome.raw_outputs) == (
        "FIRST_INVALID",
        "SECOND_INVALID",
    )
    assert decision.intent.proposed_weight == 0
    assert "invalid_review" in decision.intent.reason_codes


def test_first_unexpected_provider_failure_preserves_exact_request_key_and_is_safe() -> None:
    sentinel = "FIRST_EXCEPTION_SENTINEL"
    reviewer = Reviewer([RuntimeError(sentinel)])
    row = feature()
    candidate = rank_candidates([row])[0]
    messages = render_review_prompt(
        candidate, row, cash_weight=0.8, current_weight=0.1, drawdown=0.02
    )
    expected_key = review_cache_key("model-a", "v1", messages)

    outcome = review_candidate(reviewer, messages, model="model-a", prompt_version="v1")

    assert len(reviewer.messages) == 1
    assert outcome.cache_key == expected_key
    assert outcome.failure_reason == "provider_failure"
    assert outcome.raw_outputs == ()
    assert sentinel not in repr(outcome)
    assert sentinel not in str(outcome.model_dump())


def test_repair_failures_preserve_first_raw_and_never_retain_exception_body() -> None:
    first = "FIRST_INVALID_BODY"
    for error in (
        MiniMaxError("MINIMAX_BODY", status_code=None, attempts=1),
        RuntimeError("UNEXPECTED_BODY"),
    ):
        reviewer = Reviewer([first, error])
        decision = decide(reviewer)[0]

        assert len(reviewer.messages) == 2
        assert decision.failure_reason == "repair_provider_failure"
        assert decision.review_outcome.raw_outputs[0].text == first
        assert decision.intent.llm_cache_key == decision.review_outcome.cache_key
        assert "BODY" not in repr(decision)


def test_provider_failure_is_isolated_per_candidate_in_deterministic_ticker_order() -> None:
    rows = (feature("BBB", return_60=0.05), feature("AAA"))
    reviewer = Reviewer([RuntimeError("BODY"), response()])

    decisions = decide(
        reviewer,
        snapshot(*rows),
        cash_weight=0.7,
        current_weights={"AAA": 0.1, "BBB": 0.1},
    )

    assert [decision.intent.ticker for decision in decisions] == ["AAA", "BBB"]
    assert decisions[0].intent.proposed_weight == 0
    assert decisions[0].failure_reason == "provider_failure"
    assert decisions[1].intent.proposed_weight > 0


def test_internal_programming_failure_surfaces_instead_of_becoming_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_prompt(*args: object, **kwargs: object) -> object:
        raise RuntimeError("programming defect")

    monkeypatch.setattr(strategy_module, "render_review_prompt", broken_prompt)

    with pytest.raises(RuntimeError, match="programming defect"):
        decide(Reviewer([]))


def test_dropped_holding_emits_zero_exit_without_reviewer_and_selected_holding_is_reviewed() -> (
    None
):
    reviewer = Reviewer([response()])
    rows = (feature("AAA"), feature("BBB", return_20=0.0, return_60=0.0))

    decisions = decide(
        reviewer,
        snapshot(*rows),
        cash_weight=0.8,
        current_weights={"AAA": 0.1, "BBB": 0.1},
    )

    assert [decision.intent.ticker for decision in decisions] == ["AAA", "BBB"]
    selected, dropped = decisions
    assert selected.candidate is not None
    assert json.loads(reviewer.messages[0][-1].content)["portfolio"]["current_weight"] == 0.1  # type: ignore[union-attr]
    assert dropped.candidate is None
    assert dropped.intent.proposed_weight == 0
    assert dropped.failure_reason == "ineligible_exit"
    assert dropped.review_outcome.review.action is ReviewAction.REJECT
    assert dropped.intent.reason_codes == ("ineligible_exit",)
    assert len(reviewer.messages) == 1


def test_empty_selection_without_holdings_is_empty_but_holding_requires_market_row() -> None:
    reviewer = Reviewer([])
    source = snapshot(feature(return_20=0.0, return_60=0.0))

    assert decide(reviewer, source, current_weights={}) == ()
    assert reviewer.messages == []
    with pytest.raises(ValueError, match="FeatureRow"):
        decide(reviewer, source, current_weights={"ZZZ": 0.1})


def test_zero_weight_does_not_create_an_exit() -> None:
    reviewer = Reviewer([])
    source = snapshot(feature(return_20=0.0, return_60=0.0))

    assert decide(reviewer, source, current_weights={"AAA": 0.0}) == ()


def test_exit_identity_includes_its_market_row_provenance() -> None:
    first = decide(
        Reviewer([]),
        snapshot(feature(close=100, return_20=0.0, return_60=0.0)),
        current_weights={"AAA": 0.1},
    )[0]
    second = decide(
        Reviewer([]),
        snapshot(feature(close=101, return_20=0.0, return_60=0.0)),
        current_weights={"AAA": 0.1},
    )[0]

    assert first.intent.decision_id != second.intent.decision_id


def test_same_day_execution_is_rejected_and_next_date_is_accepted() -> None:
    subject = strategy(Reviewer([response()]))
    source = snapshot(feature())
    signal = datetime(2025, 1, 2, 10, tzinfo=UTC)
    kwargs = {
        "cash_weight": 0.9,
        "current_weights": {"AAA": 0.1},
        "drawdown": 0,
    }

    with pytest.raises(ValueError, match="later date"):
        subject.decide(
            source,
            signal_time=signal,
            earliest_execution_time=signal + timedelta(hours=8),
            **kwargs,
        )
    result = subject.decide(
        source,
        signal_time=signal,
        earliest_execution_time=signal + timedelta(days=1),
        **kwargs,
    )
    assert result[0].intent.earliest_execution_time.date() > source.as_of.date()


def test_strategy_rejects_later_local_date_that_is_not_a_later_instant() -> None:
    subject = strategy(Reviewer([response()]))
    source = snapshot(feature())
    signal = datetime(2025, 1, 2, 23, tzinfo=UTC)
    plus_nine = timezone(timedelta(hours=9))
    kwargs = {
        "signal_time": signal,
        "cash_weight": 0.9,
        "current_weights": {"AAA": 0.1},
        "drawdown": 0,
    }

    with pytest.raises(ValueError, match="later than signal_time"):
        subject.decide(
            source,
            earliest_execution_time=datetime(2025, 1, 3, 0, tzinfo=plus_nine),
            **kwargs,
        )

    result = subject.decide(
        source,
        earliest_execution_time=datetime(2025, 1, 3, 9, tzinfo=plus_nine),
        **kwargs,
    )
    assert result[0].intent.earliest_execution_time > result[0].intent.signal_time


def test_config_is_frozen_strict_and_default_stop_multiple_is_exactly_two_point_five() -> None:
    config = V1StrategyConfig()

    assert config.atr_multiple == 2.5
    with pytest.raises(FrozenInstanceError):
        config.atr_multiple = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        strategy(Reviewer([])).config = replace(config, atr_multiple=3)  # type: ignore[misc]
    for kwargs in (
        {"max_candidates": True},
        {"min_dollar_volume": float("nan")},
        {"target_volatility": 0},
        {"max_position_weight": 1.01},
        {"max_gross_exposure": 0},
        {"atr_multiple": float("inf")},
        {"feature_version": " "},
    ):
        with pytest.raises(ValueError):
            V1StrategyConfig(**kwargs)  # type: ignore[arg-type]


def test_every_executable_config_change_changes_full_digest_identity() -> None:
    base = V1StrategyConfig()
    canonical_config = json.dumps(
        asdict(base), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    expected_config_digest = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
    variations = (
        replace(base, max_candidates=5),
        replace(base, min_dollar_volume=10_000_000),
        replace(base, target_volatility=0.11),
        replace(base, max_position_weight=0.16),
        replace(base, max_gross_exposure=0.81),
        replace(base, atr_multiple=2.6),
        replace(base, feature_version="technical-v2"),
    )
    source = snapshot(feature())
    ids = {
        decide(Reviewer([response()]), source, config=config)[0].intent.decision_id
        for config in (base, *variations)
    }

    assert len(ids) == 1 + len(variations)
    decision = decide(Reviewer([response()]), source, config=base)[0]
    parts = decision.intent.decision_id.split(":")
    assert len(parts[-2]) == 64
    assert len(parts[-1]) == 64
    assert parts[-2] == expected_config_digest
    assert strategy(Reviewer([]), base).config_digest == expected_config_digest
    assert parts[-1] == decision.review_outcome.cache_key
    assert len(decision.intent.decision_id) <= 200


def test_strategy_ranks_internally_and_revalidates_position_and_gross_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "candidates" not in signature(V1RulesLLMStrategy.decide).parameters
    assert "candidates" not in signature(V1RulesLLMStrategy.generate).parameters
    with pytest.raises(TypeError):
        strategy(Reviewer([])).decide(snapshot(feature()), ())  # type: ignore[misc]

    forged = Candidate("AAA", 1, 0.2, 2, 100, 0.9)
    monkeypatch.setattr(strategy_module, "rank_candidates", lambda *args, **kwargs: [forged])
    with pytest.raises(ValueError, match="position"):
        decide(Reviewer([]), current_weights={})

    forged_a = Candidate("AAA", 1, 0.2, 2, 100, 0.1)
    forged_b = Candidate("BBB", 1, 0.2, 2, 100, 0.1)
    monkeypatch.setattr(
        strategy_module, "rank_candidates", lambda *args, **kwargs: [forged_a, forged_b]
    )
    config = replace(V1StrategyConfig(), max_gross_exposure=0.15)
    with pytest.raises(ValueError, match="gross"):
        decide(
            Reviewer([]),
            snapshot(feature("AAA"), feature("BBB")),
            config=config,
            current_weights={},
        )


def test_invalid_stop_is_authoritative_across_outcome_decision_and_intent() -> None:
    reviewer = Reviewer([])
    decision = decide(reviewer, snapshot(feature(close=1, sma_200=0.5, atr_14=2)))[0]

    assert decision.failure_reason == "invalid_stop"
    assert decision.review_outcome.failure_reason == "invalid_stop"
    assert decision.review_outcome.review.action is ReviewAction.REJECT
    assert decision.intent.proposed_weight == 0
    assert decision.intent.stop_price == 1
    assert decision.intent.invalidation == decision.review_outcome.review.invalidation
    assert decision.intent.invalidation != "Close below trend."
    assert "invalid_stop" in decision.intent.reason_codes
    assert decision.review_outcome.raw_outputs == ()
    assert reviewer.messages == []


def test_raw_audit_is_bounded_with_complete_metadata_and_safe_normal_serialization() -> None:
    sentinel = "RAW_OUTPUT_SENTINEL"
    raw = sentinel + ("é" * 70_000)
    outcome = review_candidate(
        Reviewer([raw, response()]),
        ({"role": "user", "content": "x"},),
        model="model",
        prompt_version="v1",
    )
    audit = outcome.raw_outputs[0]

    assert audit.text.startswith(sentinel)
    assert len(audit.text) <= MAX_AUDIT_RAW_CHARS
    assert len(audit.text.encode("utf-8")) <= MAX_AUDIT_RAW_BYTES
    assert audit.original_char_length == len(raw)
    assert audit.original_utf8_byte_length == len(raw.encode("utf-8"))
    assert audit.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert audit.truncated is True
    assert audit.stored_char_length == len(audit.text)
    assert audit.stored_utf8_byte_length == len(audit.text.encode("utf-8"))
    assert sentinel not in repr(outcome)
    assert sentinel not in str(asdict(outcome))
    assert sentinel not in str(outcome.model_dump())


def test_portfolio_consistency_rejects_overallocation_collisions_and_unknown_tickers() -> None:
    source = snapshot(feature("AAA"), feature("BBB"))
    signal, execution = times()
    subject = strategy(Reviewer([]))
    kwargs = {
        "signal_time": signal,
        "earliest_execution_time": execution,
        "drawdown": 0,
    }

    with pytest.raises(ValueError, match="aggregate"):
        subject.decide(
            source,
            cash_weight=0.8,
            current_weights={"AAA": 0.200000000001},
            **kwargs,
        )
    with pytest.raises(ValueError, match="collision"):
        subject.decide(
            source,
            cash_weight=0.8,
            current_weights={"aaa": 0.1, "AAA": 0.1},
            **kwargs,
        )
    with pytest.raises(ValueError, match="FeatureRow"):
        subject.decide(
            source,
            cash_weight=0.8,
            current_weights={"ZZZ": 0.1},
            **kwargs,
        )
    assert decide(Reviewer([response(), response()]), source, cash_weight=0.8) != ()


def test_portfolio_consistency_accepts_machine_epsilon_rounding() -> None:
    weights = {
        "QQQ": 0.15083909894341532,
        "AAPL": 0.15044861450721747,
        "AMZN": 0.14782141588129044,
        "GOOGL": 0.11777914146360112,
    }
    cash = 0.43311172920447577
    assert cash + sum(weights.values()) == 1.0000000000000002

    result = decide(
        Reviewer([response(), response(), response(), response()]),
        snapshot(*(feature(ticker) for ticker in weights)),
        cash_weight=cash,
        current_weights=weights,
    )

    assert len(result) == 4

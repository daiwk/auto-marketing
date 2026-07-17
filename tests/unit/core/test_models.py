from datetime import UTC, date, datetime, timedelta
from math import inf, nan

import pytest
from pydantic import ValidationError

from quant_trader.core.models import (
    ApprovedOrder,
    LLMReview,
    ReviewAction,
    SignalIntent,
    SignalSide,
)


def test_llm_review_rejects_weight_multiplier_above_one() -> None:
    with pytest.raises(ValidationError):
        LLMReview(
            action=ReviewAction.MAINTAIN,
            weight_multiplier=1.01,
            confidence=0.8,
            thesis="A sufficiently specific thesis.",
            risks=["Market risk"],
            invalidation="The thesis no longer holds.",
            input_anomalies=[],
        )


@pytest.mark.parametrize("field_name", ["weight_multiplier", "confidence"])
def test_llm_review_rejects_boolean_and_numeric_string_scores(field_name: str) -> None:
    payload = {
        "action": ReviewAction.MAINTAIN,
        "weight_multiplier": 0.5,
        "confidence": 0.8,
        "thesis": "A sufficiently specific thesis.",
        "risks": ["Market risk"],
        "invalidation": "The thesis no longer holds.",
        "input_anomalies": [],
    }
    for malformed_value in (True, "0.5"):
        with pytest.raises(ValidationError):
            LLMReview(**(payload | {field_name: malformed_value}))


@pytest.mark.parametrize("field_name, value", [("weight_multiplier", inf), ("confidence", nan)])
def test_llm_review_rejects_non_finite_scores(field_name: str, value: float) -> None:
    payload = {
        "action": ReviewAction.MAINTAIN,
        "weight_multiplier": 0.5,
        "confidence": 0.8,
        "thesis": "A sufficiently specific thesis.",
        "risks": ["Market risk"],
        "invalidation": "The thesis no longer holds.",
        "input_anomalies": [],
    }

    with pytest.raises(ValidationError):
        LLMReview(**(payload | {field_name: value}))


def test_llm_review_accepts_input_anomalies_and_rejects_legacy_anomalies() -> None:
    review = LLMReview(
        action=ReviewAction.MAINTAIN,
        weight_multiplier=1,
        confidence=0.8,
        thesis="A sufficiently specific thesis.",
        risks=["Market risk"],
        invalidation="The thesis no longer holds.",
        input_anomalies=["Stale input"],
    )

    assert review.input_anomalies == ("Stale input",)
    with pytest.raises(ValidationError, match="anomalies"):
        LLMReview(
            action=ReviewAction.MAINTAIN,
            weight_multiplier=1,
            confidence=0.8,
            thesis="A sufficiently specific thesis.",
            risks=["Market risk"],
            invalidation="The thesis no longer holds.",
            anomalies=[],
        )


@pytest.mark.parametrize("field", ["risks", "input_anomalies"])
def test_llm_review_bounds_list_cardinality(field: str) -> None:
    payload = {
        "action": "reduce",
        "weight_multiplier": 0.5,
        "confidence": 0.5,
        "thesis": "trend",
        "risks": [],
        "invalidation": "break",
        "input_anomalies": [],
    }
    payload[field] = ["item"] * 21
    with pytest.raises(ValidationError):
        LLMReview(**payload)


def test_signal_intent_rejects_naive_datetimes() -> None:
    now = datetime(2026, 1, 2, 10, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        SignalIntent(
            decision_id="decision-1",
            ticker="spy",
            side=SignalSide.BUY,
            proposed_weight=0.1,
            signal_time=now,
            earliest_execution_time=now + timedelta(minutes=1),
            stop_price=100,
            invalidation="Price falls below support.",
            reason_codes=["momentum"],
            strategy_version="v1",
            prompt_version="v1",
            llm_cache_key="cache-1",
        )


def test_signal_intent_requires_execution_after_signal_and_normalizes_ticker() -> None:
    now = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    intent = SignalIntent(
        decision_id="decision-1",
        ticker="spy",
        side=SignalSide.BUY,
        proposed_weight=0.1,
        signal_time=now,
        earliest_execution_time=now + timedelta(minutes=1),
        stop_price=100,
        invalidation="Price falls below support.",
        reason_codes=["momentum"],
        strategy_version="v1",
        prompt_version="v1",
        llm_cache_key="cache-1",
    )
    assert intent.ticker == "SPY"

    with pytest.raises(ValidationError, match="later"):
        SignalIntent(**(intent.model_dump() | {"earliest_execution_time": now}))


def test_signal_intent_requires_a_valid_explicit_side() -> None:
    now = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    signal = {
        "decision_id": "decision-1",
        "ticker": "SPY",
        "side": "buy",
        "proposed_weight": 0.1,
        "signal_time": now,
        "earliest_execution_time": now + timedelta(minutes=1),
        "stop_price": 100,
        "invalidation": "Price falls below support.",
        "reason_codes": ["momentum"],
        "strategy_version": "v1",
        "prompt_version": "v1",
        "llm_cache_key": "cache-1",
    }

    assert SignalIntent(**signal).side is SignalSide.BUY
    with pytest.raises(ValidationError, match="side"):
        SignalIntent(**(signal | {"side": "sell"}))
    with pytest.raises(ValidationError, match="side"):
        SignalIntent(**{key: value for key, value in signal.items() if key != "side"})


@pytest.mark.parametrize("field_name", ["proposed_weight", "stop_price"])
def test_signal_intent_rejects_boolean_and_numeric_string_safety_values(field_name: str) -> None:
    now = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    payload = {
        "decision_id": "decision-1",
        "ticker": "SPY",
        "side": "buy",
        "proposed_weight": 0.1,
        "signal_time": now,
        "earliest_execution_time": now + timedelta(minutes=1),
        "stop_price": 100,
        "invalidation": "Price falls below support.",
        "reason_codes": ["momentum"],
        "strategy_version": "v1",
        "prompt_version": "v1",
        "llm_cache_key": "cache-1",
    }
    for malformed_value in (True, "0.1"):
        with pytest.raises(ValidationError):
            SignalIntent(**(payload | {field_name: malformed_value}))


@pytest.mark.parametrize("field_name, value", [("proposed_weight", nan), ("stop_price", inf)])
def test_signal_intent_rejects_non_finite_safety_values(field_name: str, value: float) -> None:
    now = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    payload = {
        "decision_id": "decision-1",
        "ticker": "SPY",
        "side": "buy",
        "proposed_weight": 0.1,
        "signal_time": now,
        "earliest_execution_time": now + timedelta(minutes=1),
        "stop_price": 100,
        "invalidation": "Price falls below support.",
        "reason_codes": ["momentum"],
        "strategy_version": "v1",
        "prompt_version": "v1",
        "llm_cache_key": "cache-1",
    }

    with pytest.raises(ValidationError):
        SignalIntent(**(payload | {field_name: value}))


@pytest.mark.parametrize("ticker", ["brk.b", "bf-b", "MSFT"])
def test_shared_ticker_contract_normalizes_valid_us_equity_symbols(ticker: str) -> None:
    order = ApprovedOrder(
        decision_id="decision-1",
        ticker=ticker,
        target_weight=0.1,
        execution_date=date(2026, 1, 2),
        reason_codes=["momentum"],
    )

    assert order.ticker == ticker.upper()


@pytest.mark.parametrize(
    "ticker", [" SPY", "SPY ", "../SPY", "SP/Y", "$SPY", ".", "BRK.", "-A", "ABCDEFGHIJK"]
)
def test_shared_ticker_contract_rejects_unsafe_symbols(ticker: str) -> None:
    with pytest.raises(ValidationError):
        ApprovedOrder(
            decision_id="decision-1",
            ticker=ticker,
            target_weight=0.1,
            execution_date=date(2026, 1, 2),
            reason_codes=["momentum"],
        )


def test_approved_order_rejects_boolean_and_numeric_string_target_weight() -> None:
    payload = {
        "decision_id": "decision-1",
        "ticker": "SPY",
        "target_weight": 0.1,
        "execution_date": date(2026, 1, 2),
        "reason_codes": ["momentum"],
    }
    for malformed_value in (True, "0.1"):
        with pytest.raises(ValidationError):
            ApprovedOrder(**(payload | {"target_weight": malformed_value}))


def test_approved_order_rejects_non_finite_target_weight() -> None:
    with pytest.raises(ValidationError):
        ApprovedOrder(
            decision_id="decision-1",
            ticker="SPY",
            target_weight=nan,
            execution_date=date(2026, 1, 2),
            reason_codes=["momentum"],
        )


def test_contracts_are_immutable() -> None:
    order = ApprovedOrder(
        decision_id="decision-1",
        ticker="spy",
        target_weight=0.1,
        execution_date=date(2026, 1, 2),
        reason_codes=["momentum"],
    )

    assert order.ticker == "SPY"
    with pytest.raises(ValidationError):
        order.target_weight = 0.2

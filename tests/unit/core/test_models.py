from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from quant_trader.core.models import ApprovedOrder, LLMReview, ReviewAction, SignalIntent


def test_llm_review_rejects_weight_multiplier_above_one() -> None:
    with pytest.raises(ValidationError):
        LLMReview(
            action=ReviewAction.MAINTAIN,
            weight_multiplier=1.01,
            confidence=0.8,
            thesis="A sufficiently specific thesis.",
            risks=["Market risk"],
            invalidation="The thesis no longer holds.",
            anomalies=[],
        )


def test_signal_intent_rejects_naive_datetimes() -> None:
    now = datetime(2026, 1, 2, 10, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        SignalIntent(
            decision_id="decision-1",
            ticker="spy",
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
        ticker=" spy ",
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

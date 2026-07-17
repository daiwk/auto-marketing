"""Fail-closed composition of deterministic V1 rules and constrained LLM reviews."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from sys import float_info
from typing import cast

from quant_trader.core.models import LLMReview, ReviewAction, SignalIntent, SignalSide
from quant_trader.data.validation import normalize_ticker
from quant_trader.features.snapshot import FeatureRow, FeatureSnapshot
from quant_trader.llm.base import (
    ChatMessage,
    LLMReviewer,
    MessageInput,
    SanitizedLLMCause,
    canonical_messages,
)
from quant_trader.llm.cache import review_cache_key
from quant_trader.llm.minimax import MiniMaxError
from quant_trader.llm.parsing import MAX_PARSE_CHARS, LLMResponseError, parse_review
from quant_trader.strategies.v1_rules_llm.prompt import PROMPT_VERSION, render_review_prompt
from quant_trader.strategies.v1_rules_llm.rules import Candidate

_MAX_AUDIT_RAW_CHARS = min(MAX_PARSE_CHARS, 16 * 1024)
_REPAIR_INSTRUCTION = (
    "The prior response failed schema validation. "
    "Return exactly one LLMReview JSON object and no other text. "
    "Do not repeat or discuss the prior response."
)
_GENERIC_THESIS = "No valid review was accepted."
_GENERIC_INVALIDATION = "No position without a valid review."


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """The complete bounded audit record for one LLM review attempt sequence."""

    candidate_ticker: str
    review: LLMReview
    raw_outputs: tuple[str, ...]
    cache_key: str
    repair_used: bool
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """One deterministic candidate paired with its review and executable signal intent."""

    candidate: Candidate
    review_outcome: ReviewOutcome
    intent: SignalIntent
    failure_reason: str | None = None


def _label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonblank string without surrounding whitespace")
    return value


def _unit_weight(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _synthetic_reject(reason: str) -> LLMReview:
    return LLMReview(
        action=ReviewAction.REJECT,
        weight_multiplier=0,
        confidence=0,
        thesis=_GENERIC_THESIS,
        risks=(reason,),
        invalidation=_GENERIC_INVALIDATION,
        input_anomalies=(),
    )


def _bounded_raw(value: object) -> tuple[str, ...]:
    return (value[:_MAX_AUDIT_RAW_CHARS],) if isinstance(value, str) else ()


def _prompt_ticker(messages: Sequence[ChatMessage]) -> str:
    """Best-effort audit label for direct helper calls; strategy replaces it authoritatively."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        try:
            payload = json.loads(message.content)
            ticker = payload["candidate"]["ticker"]
            return normalize_ticker(ticker)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            break
    return "UNKNOWN"


def _consistent(review: LLMReview) -> bool:
    if review.action is ReviewAction.MAINTAIN:
        return review.weight_multiplier == 1
    if review.action is ReviewAction.REDUCE:
        return 0 <= review.weight_multiplier < 1
    return review.action is ReviewAction.REJECT and review.weight_multiplier == 0


def _parsed_review(output: object) -> LLMReview | None:
    if not isinstance(output, str):
        return None
    try:
        review = parse_review(output)
    except LLMResponseError:
        return None
    return review if _consistent(review) else None


def _outcome(
    ticker: str,
    review: LLMReview,
    raw_outputs: tuple[str, ...],
    cache_key: str,
    repair_used: bool,
    failure_reason: str | None = None,
) -> ReviewOutcome:
    return ReviewOutcome(ticker, review, raw_outputs, cache_key, repair_used, failure_reason)


def review_candidate(
    reviewer: LLMReviewer,
    messages: Sequence[MessageInput],
    *,
    model: str,
    prompt_version: str,
) -> ReviewOutcome:
    """Review one trusted prompt, using exactly one clean repair only for invalid output."""
    canonical = canonical_messages(messages)
    model_name = _label(model, "model")
    version = _label(prompt_version, "prompt_version")
    cache_key = review_cache_key(model_name, version, canonical)
    ticker = _prompt_ticker(canonical)

    try:
        first = reviewer.complete(canonical)
    except (MiniMaxError, SanitizedLLMCause, ValueError):
        return _outcome(
            ticker, _synthetic_reject("provider_failure"), (), cache_key, False, "provider_failure"
        )

    first_raw = _bounded_raw(first)
    accepted = _parsed_review(first)
    if accepted is not None:
        return _outcome(ticker, accepted, first_raw, cache_key, False)

    repair_messages = (*canonical, ChatMessage(role="user", content=_REPAIR_INSTRUCTION))
    try:
        repaired = reviewer.complete(repair_messages)
    except (MiniMaxError, SanitizedLLMCause, ValueError):
        return _outcome(
            ticker,
            _synthetic_reject("repair_provider_failure"),
            first_raw,
            cache_key,
            True,
            "repair_provider_failure",
        )

    repair_raw = _bounded_raw(repaired)
    accepted = _parsed_review(repaired)
    if accepted is not None:
        return _outcome(ticker, accepted, first_raw + repair_raw, cache_key, True)
    return _outcome(
        ticker,
        _synthetic_reject("invalid_review"),
        first_raw + repair_raw,
        cache_key,
        True,
        "invalid_review",
    )


class V1RulesLLMStrategy:
    """Generate only long-only, deterministically bounded V1 review decisions."""

    version = "v1_rules_llm"

    def __init__(
        self,
        reviewer: LLMReviewer,
        *,
        model: str,
        prompt_version: str = PROMPT_VERSION,
        atr_multiple: float = 2.0,
    ) -> None:
        if not callable(getattr(reviewer, "complete", None)):
            raise TypeError("reviewer must implement complete(messages)")
        self.reviewer = reviewer
        self.model = _label(model, "model")
        self.prompt_version = _label(prompt_version, "prompt_version")
        if (
            isinstance(atr_multiple, bool)
            or not isinstance(atr_multiple, int | float)
            or not isfinite(atr_multiple)
            or atr_multiple <= 0
        ):
            raise ValueError("atr_multiple must be a positive finite number")
        self.atr_multiple = float(atr_multiple)

    def _validated_candidates(
        self, snapshot: FeatureSnapshot, candidates: Sequence[Candidate]
    ) -> tuple[Candidate, ...]:
        if isinstance(candidates, str) or not isinstance(candidates, Sequence):
            raise TypeError("candidates must be a sequence of Candidate values")
        canonical = tuple(candidates)
        seen: set[str] = set()
        for index, candidate in enumerate(canonical):
            if not isinstance(candidate, Candidate):
                raise TypeError(f"candidates[{index}] must be a Candidate")
            if candidate.ticker in seen:
                raise ValueError("duplicate candidate ticker")
            seen.add(candidate.ticker)
            row = snapshot.rows.get(candidate.ticker)
            if row is None:
                raise ValueError(f"missing snapshot FeatureRow for {candidate.ticker}")
            if (
                not isinstance(row, FeatureRow)
                or row.ticker != candidate.ticker
                or row.as_of != snapshot.as_of
            ):
                raise ValueError("candidate does not match snapshot FeatureRow/as_of")
            if (
                candidate.close != row.close
                or candidate.atr_14 != row.atr_14
                or candidate.annualized_volatility != row.volatility_20
            ):
                raise ValueError("candidate values do not match snapshot FeatureRow")
        return tuple(sorted(canonical, key=lambda candidate: candidate.ticker))

    @staticmethod
    def _validated_current_weights(
        current_weights: Mapping[str, float], snapshot: FeatureSnapshot
    ) -> dict[str, float]:
        if not isinstance(current_weights, Mapping):
            raise TypeError("current_weights must be a mapping")
        canonical: dict[str, float] = {}
        for ticker, weight in current_weights.items():
            normalized = normalize_ticker(ticker)
            if normalized in canonical:
                raise ValueError("current_weights canonical ticker collision")
            if normalized not in snapshot.rows:
                raise ValueError(f"unknown current_weights ticker: {normalized}")
            canonical[normalized] = _unit_weight(weight, f"current_weights[{normalized}]")
        return canonical

    @staticmethod
    def _validated_times(
        snapshot: FeatureSnapshot, signal_time: datetime, execution_time: datetime
    ) -> None:
        time_values = (("signal_time", signal_time), ("earliest_execution_time", execution_time))
        for name, value in time_values:
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware datetime")
        if execution_time <= signal_time:
            raise ValueError("earliest_execution_time must be later than signal_time")
        if signal_time.date() != snapshot.as_of.date():
            raise ValueError("signal_time date must match snapshot as_of")

    def _stop_price(self, candidate: Candidate) -> tuple[float, bool]:
        stop = candidate.close - self.atr_multiple * candidate.atr_14
        if isfinite(stop) and stop > 0:
            return stop, False
        return max(float_info.min, candidate.close * 0.5), True

    def _decision_id(self, ticker: str, snapshot: FeatureSnapshot, cache_key: str) -> str:
        return (
            f"{self.version}:{ticker}:{snapshot.as_of:%Y%m%d}:"
            f"{self.prompt_version}:{cache_key[:24]}"
        )

    def _failed_decision(
        self,
        candidate: Candidate,
        snapshot: FeatureSnapshot,
        signal_time: datetime,
        earliest_execution_time: datetime,
        reason: str,
    ) -> StrategyDecision:
        cache_key = hashlib.sha256(
            f"{self.version}|{candidate.ticker}|{snapshot.as_of.date().isoformat()}|{self.prompt_version}".encode()
        ).hexdigest()
        outcome = _outcome(
            candidate.ticker, _synthetic_reject(reason), (), cache_key, False, reason
        )
        stop, _ = self._stop_price(candidate)
        intent = SignalIntent(
            decision_id=self._decision_id(candidate.ticker, snapshot, cache_key),
            ticker=candidate.ticker,
            side=SignalSide.BUY,
            proposed_weight=0,
            signal_time=signal_time,
            earliest_execution_time=earliest_execution_time,
            stop_price=stop,
            invalidation=_GENERIC_INVALIDATION,
            reason_codes=("rules_candidate", "review_reject", reason),
            strategy_version=self.version,
            prompt_version=self.prompt_version,
            llm_cache_key=cache_key,
        )
        return StrategyDecision(candidate, outcome, intent, reason)

    def generate(
        self,
        snapshot: FeatureSnapshot,
        candidates: Sequence[Candidate],
        *,
        signal_time: datetime,
        earliest_execution_time: datetime,
        cash_weight: float,
        current_weights: Mapping[str, float],
        drawdown: float,
    ) -> tuple[StrategyDecision, ...]:
        if not isinstance(snapshot, FeatureSnapshot):
            raise TypeError("snapshot must be a FeatureSnapshot")
        self._validated_times(snapshot, signal_time, earliest_execution_time)
        cash = _unit_weight(cash_weight, "cash_weight")
        drawdown_value = _unit_weight(drawdown, "drawdown")
        weights = self._validated_current_weights(current_weights, snapshot)
        canonical_candidates = self._validated_candidates(snapshot, candidates)
        decisions: list[StrategyDecision] = []
        for candidate in canonical_candidates:
            row = cast(FeatureRow, snapshot.rows[candidate.ticker])
            try:
                messages = render_review_prompt(
                    candidate,
                    row,
                    cash_weight=cash,
                    current_weight=weights.get(candidate.ticker, 0.0),
                    drawdown=drawdown_value,
                )
                outcome = review_candidate(
                    self.reviewer, messages, model=self.model, prompt_version=self.prompt_version
                )
                outcome = replace(outcome, candidate_ticker=candidate.ticker)
                stop, invalid_stop = self._stop_price(candidate)
                failure_reason = outcome.failure_reason
                proposed_weight = min(
                    candidate.base_weight,
                    candidate.base_weight * outcome.review.weight_multiplier,
                )
                if outcome.review.action is ReviewAction.REJECT or failure_reason is not None:
                    proposed_weight = 0.0
                if not isfinite(proposed_weight) or proposed_weight < 0:
                    proposed_weight = 0.0
                    failure_reason = "invalid_weight"
                if invalid_stop:
                    proposed_weight = 0.0
                    failure_reason = "invalid_stop"
                reason_codes = ["rules_candidate", f"review_{outcome.review.action.value}"]
                if failure_reason is not None:
                    reason_codes.append(failure_reason)
                intent = SignalIntent(
                    decision_id=self._decision_id(candidate.ticker, snapshot, outcome.cache_key),
                    ticker=candidate.ticker,
                    side=SignalSide.BUY,
                    proposed_weight=proposed_weight,
                    signal_time=signal_time,
                    earliest_execution_time=earliest_execution_time,
                    stop_price=stop,
                    invalidation=(
                        _GENERIC_INVALIDATION
                        if outcome.failure_reason is not None
                        else outcome.review.invalidation
                    ),
                    reason_codes=tuple(reason_codes),
                    strategy_version=self.version,
                    prompt_version=self.prompt_version,
                    llm_cache_key=outcome.cache_key,
                )
                decisions.append(StrategyDecision(candidate, outcome, intent, failure_reason))
            except Exception:
                decisions.append(
                    self._failed_decision(
                        candidate,
                        snapshot,
                        signal_time,
                        earliest_execution_time,
                        "candidate_failure",
                    )
                )
        return tuple(decisions)

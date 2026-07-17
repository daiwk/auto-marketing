"""Fail-closed composition of deterministic V1 rules and constrained LLM reviews."""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from math import fsum, isclose, isfinite
from typing import Any

from quant_trader.core.models import LLMReview, ReviewAction, SignalIntent, SignalSide
from quant_trader.data.validation import normalize_ticker
from quant_trader.features.snapshot import FeatureRow, FeatureSnapshot
from quant_trader.llm.base import ChatMessage, LLMReviewer, MessageInput, canonical_messages
from quant_trader.llm.cache import review_cache_key
from quant_trader.llm.parsing import LLMResponseError, parse_review
from quant_trader.strategies.v1_rules_llm.prompt import PROMPT_VERSION, render_review_prompt
from quant_trader.strategies.v1_rules_llm.rules import Candidate, rank_candidates

MAX_AUDIT_RAW_CHARS = 16 * 1024
MAX_AUDIT_RAW_BYTES = 16 * 1024
_REPAIR_INSTRUCTION = (
    "The prior response failed schema validation. "
    "Return exactly one LLMReview JSON object and no other text. "
    "Do not repeat or discuss the prior response."
)
_GENERIC_THESIS = "No valid review was accepted."
_GENERIC_INVALIDATION = "No position without a valid review."


def _label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be a nonblank string of at most 200 characters")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and (result < minimum or (result == minimum and not minimum_inclusive)):
        operator = "greater than" if not minimum_inclusive else "at least"
        raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _unit_weight(value: object, name: str) -> float:
    return _finite_number(value, name, minimum=0, maximum=1)


@dataclass(frozen=True, slots=True)
class V1StrategyConfig:
    """Immutable executable configuration for candidate selection, sizing, and stops."""

    max_candidates: int = 4
    min_dollar_volume: float = 20_000_000.0
    target_volatility: float = 0.10
    max_position_weight: float = 0.15
    max_gross_exposure: float = 0.80
    atr_multiple: float = 2.5
    feature_version: str = "technical-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or self.max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")
        object.__setattr__(
            self,
            "min_dollar_volume",
            _finite_number(self.min_dollar_volume, "min_dollar_volume", minimum=0),
        )
        for name in ("target_volatility", "max_position_weight", "max_gross_exposure"):
            object.__setattr__(
                self,
                name,
                _finite_number(
                    getattr(self, name), name, minimum=0, maximum=1, minimum_inclusive=False
                ),
            )
        object.__setattr__(
            self,
            "atr_multiple",
            _finite_number(self.atr_multiple, "atr_multiple", minimum=0, minimum_inclusive=False),
        )
        object.__setattr__(self, "feature_version", _label(self.feature_version, "feature_version"))
        if self.max_position_weight > self.max_gross_exposure:
            raise ValueError("max_position_weight must not exceed max_gross_exposure")


class RawOutputAudit:
    """Explicitly accessed bounded raw text with safe default representation/serialization."""

    _text: str
    original_char_length: int
    original_utf8_byte_length: int
    sha256: str
    stored_char_length: int
    stored_utf8_byte_length: int
    truncated: bool

    __slots__ = (
        "_text",
        "original_char_length",
        "original_utf8_byte_length",
        "sha256",
        "stored_char_length",
        "stored_utf8_byte_length",
        "truncated",
    )

    def __init__(self, value: str) -> None:
        original_bytes = value.encode("utf-8", errors="replace")
        char_bounded = value[:MAX_AUDIT_RAW_CHARS]
        bounded_bytes = char_bounded.encode("utf-8", errors="replace")[:MAX_AUDIT_RAW_BYTES]
        bounded_text = bounded_bytes.decode("utf-8", errors="ignore")
        object.__setattr__(self, "_text", bounded_text)
        object.__setattr__(self, "original_char_length", len(value))
        object.__setattr__(self, "original_utf8_byte_length", len(original_bytes))
        object.__setattr__(self, "sha256", hashlib.sha256(original_bytes).hexdigest())
        object.__setattr__(self, "stored_char_length", len(bounded_text))
        object.__setattr__(
            self, "stored_utf8_byte_length", len(bounded_text.encode("utf-8", errors="replace"))
        )
        object.__setattr__(self, "truncated", bounded_text != value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RawOutputAudit is immutable")

    @property
    def text(self) -> str:
        """Return bounded raw text only through this explicit audit accessor."""
        return self._text

    def model_dump(self) -> dict[str, int | str | bool]:
        return {
            "original_char_length": self.original_char_length,
            "original_utf8_byte_length": self.original_utf8_byte_length,
            "sha256": self.sha256,
            "stored_char_length": self.stored_char_length,
            "stored_utf8_byte_length": self.stored_utf8_byte_length,
            "truncated": self.truncated,
        }

    def __repr__(self) -> str:
        metadata = self.model_dump()
        fields = ", ".join(f"{name}={value!r}" for name, value in metadata.items())
        return f"RawOutputAudit({fields})"

    def __deepcopy__(self, memo: dict[int, object]) -> RawOutputAudit:
        return self


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """The complete bounded and safely serializable audit for one review sequence."""

    candidate_ticker: str
    review: LLMReview
    raw_outputs: tuple[RawOutputAudit, ...]
    cache_key: str
    repair_used: bool
    failure_reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "candidate_ticker": self.candidate_ticker,
            "review": self.review.model_dump(mode="json"),
            "raw_outputs": tuple(output.model_dump() for output in self.raw_outputs),
            "cache_key": self.cache_key,
            "repair_used": self.repair_used,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """One ranked candidate or required exit, its audit, and its shared signal intent."""

    candidate: Candidate | None
    review_outcome: ReviewOutcome
    intent: SignalIntent

    @property
    def failure_reason(self) -> str | None:
        """Expose the outcome's single authoritative failure state."""
        return self.review_outcome.failure_reason


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


def _bounded_raw(value: object) -> tuple[RawOutputAudit, ...]:
    return (RawOutputAudit(value),) if isinstance(value, str) else ()


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
    raw_outputs: tuple[RawOutputAudit, ...],
    cache_key: str,
    repair_used: bool,
    failure_reason: str | None = None,
) -> ReviewOutcome:
    return ReviewOutcome(ticker, review, raw_outputs, cache_key, repair_used, failure_reason)


def _clear_exception_graph(error: Exception) -> None:
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
    except Exception as error:
        _clear_exception_graph(error)
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
    except Exception as error:
        _clear_exception_graph(error)
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
    """Generate long-only V1 intents from internally ranked point-in-time features."""

    __slots__ = ("_config", "_config_digest", "_model", "_prompt_version", "_reviewer")
    version = "v1_rules_llm"

    def __init__(
        self,
        reviewer: LLMReviewer,
        *,
        model: str,
        prompt_version: str = PROMPT_VERSION,
        config: V1StrategyConfig | None = None,
    ) -> None:
        if not callable(getattr(reviewer, "complete", None)):
            raise TypeError("reviewer must implement complete(messages)")
        if config is not None and not isinstance(config, V1StrategyConfig):
            raise TypeError("config must be a V1StrategyConfig")
        self._reviewer = reviewer
        self._model = _label(model, "model")
        self._prompt_version = _label(prompt_version, "prompt_version")
        self._config = V1StrategyConfig(**asdict(config or V1StrategyConfig()))
        canonical = json.dumps(
            asdict(self._config), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._config_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def config(self) -> V1StrategyConfig:
        return self._config

    @property
    def config_digest(self) -> str:
        return self._config_digest

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @staticmethod
    def _validated_snapshot_rows(snapshot: FeatureSnapshot) -> tuple[FeatureRow, ...]:
        if not isinstance(snapshot, FeatureSnapshot):
            raise TypeError("snapshot must be a FeatureSnapshot")
        canonical: dict[str, FeatureRow] = {}
        for raw_ticker, row in snapshot.rows.items():
            ticker = normalize_ticker(raw_ticker)
            if ticker in canonical:
                raise ValueError("snapshot rows contain a canonical ticker collision")
            if not isinstance(row, FeatureRow):
                raise TypeError(f"snapshot row for {ticker} must be a FeatureRow")
            if row.ticker != ticker or row.as_of != snapshot.as_of:
                raise ValueError(f"snapshot FeatureRow for {ticker} does not match key/as_of")
            canonical[ticker] = row
        return tuple(canonical[ticker] for ticker in sorted(canonical))

    def _ranked_candidates(self, rows: tuple[FeatureRow, ...]) -> tuple[Candidate, ...]:
        candidates = tuple(
            rank_candidates(
                rows,
                max_candidates=self.config.max_candidates,
                min_dollar_volume=self.config.min_dollar_volume,
                target_volatility=self.config.target_volatility,
                max_position_weight=self.config.max_position_weight,
                max_gross_exposure=self.config.max_gross_exposure,
            )
        )
        if len(candidates) > self.config.max_candidates:
            raise ValueError("ranked candidates exceed max_candidates")
        rows_by_ticker = {row.ticker: row for row in rows}
        seen: set[str] = set()
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Candidate):
                raise TypeError(f"ranked candidates[{index}] must be a Candidate")
            if candidate.ticker in seen:
                raise ValueError("ranked candidates contain a duplicate ticker")
            seen.add(candidate.ticker)
            row = rows_by_ticker.get(candidate.ticker)
            if row is None:
                raise ValueError("ranked candidate has no matching FeatureRow")
            if (
                candidate.close != row.close
                or candidate.atr_14 != row.atr_14
                or candidate.annualized_volatility != row.volatility_20
            ):
                raise ValueError("ranked candidate provenance does not match its FeatureRow")
            if candidate.base_weight > self.config.max_position_weight:
                raise ValueError("ranked candidate exceeds max position weight")
        gross = fsum(candidate.base_weight for candidate in candidates)
        if gross > self.config.max_gross_exposure:
            raise ValueError("ranked candidates exceed max gross exposure")
        return tuple(sorted(candidates, key=lambda candidate: candidate.ticker))

    @staticmethod
    def _validated_current_weights(
        current_weights: Mapping[str, float], rows: tuple[FeatureRow, ...]
    ) -> dict[str, float]:
        if not isinstance(current_weights, Mapping):
            raise TypeError("current_weights must be a mapping")
        row_tickers = {row.ticker for row in rows}
        canonical: dict[str, float] = {}
        for ticker, weight in current_weights.items():
            normalized = normalize_ticker(ticker)
            if normalized in canonical:
                raise ValueError("current_weights canonical ticker collision")
            if normalized not in row_tickers:
                raise ValueError(
                    f"current_weights ticker {normalized} requires a snapshot FeatureRow"
                )
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
        if signal_time.date() != snapshot.as_of.date():
            raise ValueError("signal_time date must match snapshot as_of")
        if execution_time <= signal_time:
            raise ValueError("earliest_execution_time must be later than signal_time")
        if execution_time.date() <= snapshot.as_of.date():
            raise ValueError("earliest_execution_time must be on a later date than snapshot as_of")

    def _stop_price(self, candidate: Candidate) -> tuple[float, bool]:
        stop = candidate.close - self.config.atr_multiple * candidate.atr_14
        if isfinite(stop) and stop > 0:
            return stop, False
        return candidate.close, True

    def _decision_id(self, ticker: str, snapshot: FeatureSnapshot, cache_key: str) -> str:
        identifier = (
            f"{self.version}:{ticker}:{snapshot.as_of:%Y%m%d}:{self.config_digest}:{cache_key}"
        )
        if len(identifier) > 200:
            raise ValueError("decision identity exceeds the shared Identifier bound")
        return identifier

    def _synthetic_cache_key(self, row: FeatureRow, snapshot: FeatureSnapshot, reason: str) -> str:
        row_payload = asdict(row)
        row_payload["as_of"] = snapshot.as_of.date().isoformat()
        payload = {
            "as_of": snapshot.as_of.date().isoformat(),
            "config_digest": self.config_digest,
            "feature_row": row_payload,
            "prompt_version": self.prompt_version,
            "reason": reason,
            "strategy_version": self.version,
            "ticker": row.ticker,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _exit_decision(
        self,
        row: FeatureRow,
        snapshot: FeatureSnapshot,
        signal_time: datetime,
        earliest_execution_time: datetime,
    ) -> StrategyDecision:
        reason = "ineligible_exit"
        cache_key = self._synthetic_cache_key(row, snapshot, reason)
        outcome = _outcome(row.ticker, _synthetic_reject(reason), (), cache_key, False, reason)
        intent = SignalIntent(
            decision_id=self._decision_id(row.ticker, snapshot, cache_key),
            ticker=row.ticker,
            side=SignalSide.BUY,
            proposed_weight=0,
            signal_time=signal_time,
            earliest_execution_time=earliest_execution_time,
            stop_price=row.close,
            invalidation=outcome.review.invalidation,
            reason_codes=(reason,),
            strategy_version=self.version,
            prompt_version=self.prompt_version,
            llm_cache_key=cache_key,
        )
        return StrategyDecision(None, outcome, intent)

    def _reviewed_decision(
        self,
        candidate: Candidate,
        row: FeatureRow,
        snapshot: FeatureSnapshot,
        signal_time: datetime,
        earliest_execution_time: datetime,
        cash_weight: float,
        current_weight: float,
        drawdown: float,
    ) -> StrategyDecision:
        stop, invalid_stop = self._stop_price(candidate)
        if invalid_stop:
            cache_key = self._synthetic_cache_key(row, snapshot, "invalid_stop")
            outcome = _outcome(
                candidate.ticker,
                _synthetic_reject("invalid_stop"),
                (),
                cache_key,
                False,
                "invalid_stop",
            )
        else:
            messages = render_review_prompt(
                candidate,
                row,
                cash_weight=cash_weight,
                current_weight=current_weight,
                drawdown=drawdown,
            )
            outcome = review_candidate(
                self._reviewer, messages, model=self.model, prompt_version=self.prompt_version
            )
            outcome = replace(outcome, candidate_ticker=candidate.ticker)
        proposed_weight = min(
            candidate.base_weight,
            candidate.base_weight * outcome.review.weight_multiplier,
        )
        if not isfinite(proposed_weight) or proposed_weight < 0:
            raise ValueError("derived proposed weight violates an internal invariant")
        if outcome.review.action is ReviewAction.REJECT or outcome.failure_reason is not None:
            proposed_weight = 0.0
        reason_codes = ["rules_candidate", f"review_{outcome.review.action.value}"]
        if outcome.failure_reason is not None:
            reason_codes.append(outcome.failure_reason)
        intent = SignalIntent(
            decision_id=self._decision_id(candidate.ticker, snapshot, outcome.cache_key),
            ticker=candidate.ticker,
            side=SignalSide.BUY,
            proposed_weight=proposed_weight,
            signal_time=signal_time,
            earliest_execution_time=earliest_execution_time,
            stop_price=stop,
            invalidation=outcome.review.invalidation,
            reason_codes=tuple(reason_codes),
            strategy_version=self.version,
            prompt_version=self.prompt_version,
            llm_cache_key=outcome.cache_key,
        )
        return StrategyDecision(candidate, outcome, intent)

    def decide(
        self,
        snapshot: FeatureSnapshot,
        *,
        signal_time: datetime,
        earliest_execution_time: datetime,
        cash_weight: float,
        current_weights: Mapping[str, float],
        drawdown: float,
    ) -> tuple[StrategyDecision, ...]:
        rows = self._validated_snapshot_rows(snapshot)
        self._validated_times(snapshot, signal_time, earliest_execution_time)
        cash = _unit_weight(cash_weight, "cash_weight")
        drawdown_value = _unit_weight(drawdown, "drawdown")
        weights = self._validated_current_weights(current_weights, rows)
        occupied_weight = fsum([cash, *(weights[ticker] for ticker in sorted(weights))])
        if occupied_weight > 1.0 and not isclose(
            occupied_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("portfolio aggregate weight must not exceed 1")
        candidates = self._ranked_candidates(rows)
        candidates_by_ticker = {candidate.ticker: candidate for candidate in candidates}
        rows_by_ticker = {row.ticker: row for row in rows}
        tickers = sorted(
            set(candidates_by_ticker) | {ticker for ticker, weight in weights.items() if weight > 0}
        )
        decisions: list[StrategyDecision] = []
        for ticker in tickers:
            candidate = candidates_by_ticker.get(ticker)
            row = rows_by_ticker[ticker]
            if candidate is None:
                decisions.append(
                    self._exit_decision(row, snapshot, signal_time, earliest_execution_time)
                )
                continue
            decisions.append(
                self._reviewed_decision(
                    candidate,
                    row,
                    snapshot,
                    signal_time,
                    earliest_execution_time,
                    cash,
                    weights.get(ticker, 0.0),
                    drawdown_value,
                )
            )
        return tuple(decisions)

    def generate(
        self,
        snapshot: FeatureSnapshot,
        *,
        signal_time: datetime,
        earliest_execution_time: datetime,
        cash_weight: float,
        current_weights: Mapping[str, float],
        drawdown: float,
    ) -> tuple[SignalIntent, ...]:
        return tuple(
            decision.intent
            for decision in self.decide(
                snapshot,
                signal_time=signal_time,
                earliest_execution_time=earliest_execution_time,
                cash_weight=cash_weight,
                current_weights=current_weights,
                drawdown=drawdown,
            )
        )

"""Fail-closed one-call FinMem adapter for the existing V1 reviewer contract."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from quant_trader.core.models import LLMReview, ReviewAction
from quant_trader.llm.base import ChatMessage, LLMReviewer, MessageInput, canonical_messages
from quant_trader.strategies.v3_finmem.memory import MemoryBook, MemoryLayer, MemoryRecord
from quant_trader.validation import StrictNumber

_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)]
_MemoryId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
_MAX_RESPONSE_BYTES = 64 * 1024


class ReviewProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    risk_style: str = "conservative"
    turnover: str = "low"
    max_position_pct: int = 20


class FinMemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: Annotated[ReviewAction, Field(strict=False)]
    weight_multiplier: StrictNumber = Field(ge=0, le=1)
    confidence: StrictNumber = Field(ge=0, le=1)
    thesis: _Text
    risks: Annotated[tuple[_Text, ...], Field(max_length=20, strict=False)] = ()
    invalidation: _Text
    input_anomalies: Annotated[tuple[_Text, ...], Field(max_length=20, strict=False)] = ()
    memory_ids: Annotated[tuple[_MemoryId, ...], Field(max_length=9, strict=False)] = ()

    def review(self) -> LLMReview:
        return LLMReview.model_validate(self.model_dump(exclude={"memory_ids"}))


def _consistent(value: FinMemResponse) -> bool:
    if value.action is ReviewAction.MAINTAIN:
        return value.weight_multiplier == 1
    if value.action is ReviewAction.REDUCE:
        return 0 <= value.weight_multiplier < 1
    return value.action is ReviewAction.REJECT and value.weight_multiplier == 0


def _reject() -> FinMemResponse:
    return FinMemResponse(
        action=ReviewAction.REJECT,
        weight_multiplier=0,
        confidence=0,
        thesis="No valid memory-aware review was accepted.",
        risks=("finmem_review_failure",),
        invalidation="No position without a valid memory-aware review.",
        input_anomalies=(),
        memory_ids=(),
    )


class FinMemReviewer:
    """Augment a valid V1 request with eligible memories, then validate citations."""

    def __init__(self, provider: LLMReviewer, memory: MemoryBook) -> None:
        if not callable(getattr(provider, "complete", None)):
            raise TypeError("provider must implement complete(messages)")
        if not isinstance(memory, MemoryBook):
            raise TypeError("memory must be a MemoryBook")
        self._provider = provider
        self._memory = memory
        self._profile = ReviewProfile()
        self.last_decision: dict[str, Any] = {
            "ticker": "UNKNOWN",
            "action": "reject",
            "confidence": 0,
            "memory_ids": [],
            "reason": "not_run",
        }

    @staticmethod
    def _request(
        messages: Sequence[MessageInput],
    ) -> tuple[tuple[ChatMessage, ...], dict[str, object], str, date]:
        canonical = canonical_messages(messages)
        users = [message for message in canonical if message.role == "user"]
        if not users:
            raise ValueError("V1 request requires a user payload")
        payload = json.loads(users[-1].content)
        if not isinstance(payload, dict):
            raise ValueError("V1 user payload must be an object")
        candidate, features = payload.get("candidate"), payload.get("features")
        if not isinstance(candidate, dict) or not isinstance(features, dict):
            raise ValueError("V1 request is missing candidate or features")
        ticker = candidate.get("ticker")
        feature_ticker = features.get("ticker")
        as_of = features.get("as_of")
        if not isinstance(ticker, str) or ticker != feature_ticker or not isinstance(as_of, str):
            raise ValueError("V1 candidate and feature provenance do not match")
        return canonical, payload, ticker, date.fromisoformat(as_of)

    @staticmethod
    def _memory_payload(
        memories: dict[MemoryLayer, tuple[MemoryRecord, ...]],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            layer.value: [item.model_dump() for item in memories[layer]] for layer in MemoryLayer
        }

    @staticmethod
    def _parse(raw: object) -> FinMemResponse | None:
        if (
            not isinstance(raw, str)
            or len(raw.encode("utf-8", errors="replace")) > _MAX_RESPONSE_BYTES
        ):
            return None
        try:
            parsed = json.loads(raw)
            return FinMemResponse.model_validate(parsed)
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError, RecursionError):
            return None

    def _finish(self, ticker: str, result: FinMemResponse, reason: str) -> str:
        self.last_decision = {
            "ticker": ticker,
            "action": result.action.value,
            "confidence": result.confidence,
            "memory_ids": list(result.memory_ids),
            "reason": reason,
        }
        return json.dumps(
            result.review().model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def complete(self, messages: Sequence[MessageInput]) -> str:
        ticker = "UNKNOWN"
        try:
            canonical, candidate, ticker, as_of = self._request(messages)
            memories = self._memory.retrieve(ticker, as_of)
            available_ids = {item.id for values in memories.values() for item in values}
            augmented = json.dumps(
                {
                    "v1_candidate": candidate,
                    "profile": self._profile.model_dump(),
                    "memories": self._memory_payload(memories),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            prompt = (*canonical, ChatMessage(role="user", content=augmented))
            result = self._parse(self._provider.complete(prompt))
            if result is None:
                return self._finish(ticker, _reject(), "invalid_response")
            if not set(result.memory_ids).issubset(available_ids):
                return self._finish(ticker, _reject(), "invalid_memory_ids")
            if not _consistent(result):
                return self._finish(ticker, _reject(), "inconsistent_action")
            return self._finish(ticker, result, "accepted")
        except Exception:
            return self._finish(ticker, _reject(), "request_or_provider_failure")

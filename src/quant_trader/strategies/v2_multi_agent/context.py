"""Bounded external context loading with point-in-time filtering."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from quant_trader.data.validation import normalize_ticker
from quant_trader.strategies.v2_multi_agent.models import (
    ExternalContext,
    TickerContext,
    VisibleContext,
)

MAX_CONTEXT_BYTES = 65_536


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def load_external_context(path: Path | None) -> ExternalContext:
    if path is None:
        return ExternalContext()
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_CONTEXT_BYTES + 1)
    except OSError as error:
        raise ValueError("context file could not be read") from error
    if len(raw) > MAX_CONTEXT_BYTES:
        raise ValueError("context file exceeds 65536 bytes")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError):
        raise ValueError("context file must be valid UTF-8 JSON") from None
    try:
        return ExternalContext.model_validate(payload)
    except ValidationError as error:
        raise ValueError("context file failed schema validation") from error


def context_for(context: ExternalContext, ticker: str, as_of: date) -> VisibleContext:
    item = context.tickers.get(normalize_ticker(ticker), TickerContext())
    return VisibleContext(
        news=tuple(entry for entry in item.news if entry.published_at <= as_of),
        sentiment=tuple(entry for entry in item.sentiment if entry.observed_at <= as_of),
        fundamentals=(
            item.fundamentals
            if item.fundamentals is not None and item.fundamentals.reported_at <= as_of
            else None
        ),
    )


def reject_future_context(context: ExternalContext, ticker: str, as_of: date) -> None:
    item = context.tickers.get(normalize_ticker(ticker), TickerContext())
    has_future = (
        any(entry.published_at > as_of for entry in item.news)
        or any(entry.observed_at > as_of for entry in item.sentiment)
        or (item.fundamentals is not None and item.fundamentals.reported_at > as_of)
    )
    if has_future:
        raise ValueError("context contains data later than --as-of")

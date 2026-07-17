"""Shared validation primitives for configuration and immutable contracts."""

from __future__ import annotations

from math import isfinite
from typing import Annotated, Any

from pydantic import BeforeValidator, StringConstraints


def _require_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("value must be a numeric JSON/YAML scalar, not a boolean or string")
    if not isfinite(value):
        raise ValueError("value must be finite")
    return value


def _require_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("value must be an integer, not a boolean or string")
    return value


def normalize_us_equity_ticker(value: Any) -> str:
    """Uppercase a conservative US-equity ticker without accepting surrounding whitespace."""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("ticker must be a non-whitespace string")
    return value.upper()


StrictNumber = Annotated[float, BeforeValidator(_require_number)]
StrictInteger = Annotated[int, BeforeValidator(_require_integer)]
USEquityTicker = Annotated[
    str,
    BeforeValidator(normalize_us_equity_ticker),
    StringConstraints(pattern=r"^[A-Z0-9]+(?:[.-][A-Z0-9]+)?$", min_length=1, max_length=10),
]

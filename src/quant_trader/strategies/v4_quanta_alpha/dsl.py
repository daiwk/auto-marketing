"""A deliberately tiny factor language; parsing never executes input."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_FIELDS = frozenset({"open", "high", "low", "close", "volume", "returns"})
_FUNCTIONS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "delay",
        "delta",
        "rank",
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "zscore",
    }
)
_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "delay": 2,
    "delta": 2,
    "rank": 1,
    "rolling_mean": 2,
    "rolling_std": 2,
    "rolling_min": 2,
    "rolling_max": 2,
    "zscore": 2,
}
_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|[0-9]+|[(),])")


class DSLParseError(ValueError):
    """Raised when a factor is outside the safe DSL."""


@dataclass(frozen=True)
class Node:
    name: str
    args: tuple[Node | int, ...] = ()

    @property
    def canonical(self) -> str:
        return (
            self.name
            if not self.args
            else f"{self.name}({','.join(_render(arg) for arg in self.args)})"
        )


@dataclass(frozen=True)
class Factor:
    root: Node
    nodes: int

    @property
    def canonical(self) -> str:
        return self.root.canonical


def _render(value: Node | int) -> str:
    return value.canonical if isinstance(value, Node) else str(value)


class _Parser:
    def __init__(self, expression: str) -> None:
        if not expression or len(expression) > 1000:
            raise DSLParseError("expression must be non-empty and bounded")
        self.tokens: list[str] = []
        position = 0
        while position < len(expression):
            match = _TOKEN.match(expression, position)
            if not match:
                if expression[position:].strip() == "":
                    break
                raise DSLParseError("invalid token")
            self.tokens.append(match.group(1))
            position = match.end()
        self.position = 0
        self.nodes = 0

    def parse(self) -> Factor:
        root = self.node(1)
        if self.position != len(self.tokens):
            raise DSLParseError("unexpected trailing token")
        return Factor(root, self.nodes)

    def take(self) -> str:
        if self.position >= len(self.tokens):
            raise DSLParseError("unexpected end of expression")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def node(self, depth: int) -> Node:
        if depth > 8:
            raise DSLParseError("maximum depth is 8")
        name = self.take()
        if name not in _FIELDS and name not in _FUNCTIONS:
            raise DSLParseError(f"unknown field or function: {name}")
        self.nodes += 1
        if self.nodes > 40:
            raise DSLParseError("maximum node count is 40")
        if name in _FIELDS:
            return Node(name)
        if self.take() != "(":
            raise DSLParseError("function requires parentheses")
        args: list[Node | int] = []
        for index in range(_ARITY[name]):
            if index:
                if self.take() != ",":
                    raise DSLParseError("arguments require commas")
            if (
                name
                in {
                    "delay",
                    "delta",
                    "rolling_mean",
                    "rolling_std",
                    "rolling_min",
                    "rolling_max",
                    "zscore",
                }
                and index == 1
            ):
                value = self.take()
                if not value.isdigit() or not 1 <= int(value) <= 252:
                    raise DSLParseError("window must be an integer from 1 to 252")
                args.append(int(value))
            else:
                args.append(self.node(depth + 1))
        if self.take() != ")":
            raise DSLParseError("missing closing parenthesis")
        return Node(name, tuple(args))


def parse_factor(expression: str) -> Factor:
    """Parse ``expression`` without using Python evaluation facilities."""
    return _Parser(expression).parse()


def evaluate(factor: Factor, panel: pd.DataFrame) -> pd.Series:
    """Evaluate a parsed factor over a ``date,ticker`` indexed panel."""
    if not isinstance(panel.index, pd.MultiIndex) or list(panel.index.names) != ["date", "ticker"]:
        raise ValueError("panel index must have levels date,ticker")
    missing = _FIELDS - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing fields: {sorted(missing)}")

    def visit(node: Node) -> pd.Series:
        if node.name in _FIELDS:
            return panel[node.name].astype(float)
        values = [visit(arg) for arg in node.args if isinstance(arg, Node)]
        if node.name == "add":
            return values[0] + values[1]
        if node.name == "sub":
            return values[0] - values[1]
        if node.name == "mul":
            return values[0] * values[1]
        if node.name == "div":
            return values[0] / values[1].replace(0, np.nan)
        if node.name == "rank":
            return values[0].groupby(level="date").rank(pct=True)
        window_value = node.args[1]
        if not isinstance(window_value, int):
            raise AssertionError("windowed functions require an integer window")
        window = window_value
        grouped = values[0].groupby(level="ticker")
        if node.name == "delay":
            return grouped.shift(window)
        if node.name == "delta":
            return values[0] - grouped.shift(window)
        rolling = grouped.rolling(window, min_periods=window)
        if node.name == "rolling_mean":
            return rolling.mean().droplevel(0).reindex(panel.index)
        if node.name == "rolling_std":
            return rolling.std().droplevel(0).reindex(panel.index)
        if node.name == "rolling_min":
            return rolling.min().droplevel(0).reindex(panel.index)
        if node.name == "rolling_max":
            return rolling.max().droplevel(0).reindex(panel.index)
        if node.name == "zscore":
            mean = rolling.mean().droplevel(0).reindex(panel.index)
            std = rolling.std().droplevel(0).reindex(panel.index).replace(0, np.nan)
            return (values[0] - mean) / std
        raise AssertionError("unreachable DSL node")

    return visit(factor.root).replace([np.inf, -np.inf], np.nan)

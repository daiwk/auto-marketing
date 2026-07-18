"""Two-call, validation-only factor selection for a QuantaAlpha-style MVP."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .dsl import DSLParseError, evaluate, parse_factor

Reviewer = Callable[[str], str]


def chronological_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return non-overlapping 60/20/20 date splits, requiring five dates each."""
    dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    if len(dates) < 15:
        raise ValueError("at least 15 dates are required")
    train_end = int(len(dates) * 0.6)
    validation_end = int(len(dates) * 0.8)
    if min(train_end, validation_end - train_end, len(dates) - validation_end) < 5:
        raise ValueError("each split requires at least 5 dates")
    level = panel.index.get_level_values("date")
    return tuple(
        panel[level.isin(part)]
        for part in (dates[:train_end], dates[train_end:validation_end], dates[validation_end:])
    )  # type: ignore[return-value]


def _daily_ic(factor: pd.Series, panel: pd.DataFrame) -> float:
    target = panel["returns"].groupby(level="ticker").shift(-1)
    joined = pd.DataFrame({"factor": factor, "target": target}).dropna()

    def spearman(frame: pd.DataFrame) -> float:
        ranks = frame.rank(method="average")
        return float(ranks["factor"].corr(ranks["target"]))

    correlations = joined.groupby(level="date").apply(spearman)
    return float(correlations.mean()) if not correlations.empty else float("nan")


class QuantaAlphaMiner:
    """A bounded reviewer loop: seeds, optional descendants, then frozen testing."""

    def __init__(self, reviewer: Reviewer, node_penalty: float = 0.001) -> None:
        self.reviewer = reviewer
        self.node_penalty = node_penalty

    def _ask(self, stage: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = json.dumps(
            {"stage": stage, "limit": 4, "candidates": candidates}, separators=(",", ":")
        )
        try:
            answer = json.loads(self.reviewer(prompt))
        except (json.JSONDecodeError, TypeError, ValueError):
            return [{"expression": "", "rejection_reason": "reviewer returned invalid JSON"}]
        raw = answer.get("candidates") if isinstance(answer, dict) else None
        if not isinstance(raw, list):
            return [{"expression": "", "rejection_reason": "reviewer response lacks candidates"}]
        return [
            item
            if isinstance(item, dict)
            else {"expression": "", "rejection_reason": "candidate must be an object"}
            for item in raw[:4]
        ]

    def _score(self, raw: dict[str, Any], validation: pd.DataFrame) -> dict[str, Any]:
        expression = raw.get("expression")
        record: dict[str, Any] = {
            "expression": expression if isinstance(expression, str) else "",
            "parent": raw.get("parent"),
            "operation": raw.get("operation"),
            "rejection_reason": None,
        }
        if not isinstance(expression, str):
            record["rejection_reason"] = "expression must be a string"
            return record
        try:
            factor = parse_factor(expression)
            ic = _daily_ic(evaluate(factor, validation), validation)
        except (DSLParseError, ValueError) as exc:
            record["rejection_reason"] = str(exc)
            return record
        if not np.isfinite(ic):
            record["rejection_reason"] = "validation IC is not finite"
            return record
        record.update(
            expression=factor.canonical,
            nodes=factor.nodes,
            validation_ic=ic,
            score=ic - self.node_penalty * factor.nodes,
        )
        return record

    def mine(self, panel: pd.DataFrame) -> dict[str, Any]:
        _, validation, test = chronological_split(panel)
        candidates = [self._score(raw, validation) for raw in self._ask("seed", [])]
        valid = [item for item in candidates if item["rejection_reason"] is None]
        if valid:
            parents = [
                {"expression": item["expression"], "score": item["score"]} for item in valid[:4]
            ]
            descendants = [self._score(raw, validation) for raw in self._ask("descendant", parents)]
            candidates.extend(descendants)
            valid.extend(item for item in descendants if item["rejection_reason"] is None)
        edges = [
            {"parent": item["parent"], "child": item["expression"], "operation": item["operation"]}
            for item in candidates
            if item.get("parent") and item.get("operation") and item["rejection_reason"] is None
        ]
        champion = max(valid, key=lambda item: float(item["score"])) if valid else None
        if champion is not None:
            frozen = dict(champion)
            factor = parse_factor(str(frozen["expression"]))
            frozen["test_ic"] = _daily_ic(evaluate(factor, test), test)
            frozen["frozen"] = True
            champion = frozen
        return {
            "status": "complete" if champion else "partial",
            "champion": champion,
            "candidates": candidates,
            "edges": edges,
        }

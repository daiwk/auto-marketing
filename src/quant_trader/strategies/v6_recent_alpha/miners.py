"""Safe MVP adaptations of AlphaAgent and Chain-of-Alpha."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from quant_trader.strategies.v4_quanta_alpha.dsl import (
    DSLParseError,
    evaluate,
    parse_factor,
)
from quant_trader.strategies.v4_quanta_alpha.miner import chronological_split

Reviewer = Callable[[str], str]
Progress = Callable[[str, str], None]
_AST_TOKENS = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[(),]")
_PAPER_ALPHA_AGENT = {
    "title": "AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration",
    "arxiv_id": "2502.16789",
    "url": "https://arxiv.org/abs/2502.16789",
}
_PAPER_CHAIN = {
    "title": "Chain-of-Alpha: Unleashing the Power of LLMs for Alpha Mining",
    "arxiv_id": "2508.06312",
    "url": "https://arxiv.org/abs/2508.06312",
}


def _daily_ic(factor: pd.Series, panel: pd.DataFrame) -> float:
    target = panel["returns"].groupby(level="ticker").shift(-1)
    joined = pd.DataFrame({"factor": factor, "target": target}).dropna()

    def correlation(frame: pd.DataFrame) -> float:
        ranked = frame.rank(method="average")
        return float(ranked["factor"].corr(ranked["target"]))

    values = joined.groupby(level="date").apply(correlation)
    return float(values.mean()) if not values.empty else float("nan")


def _halve(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    midpoint = max(1, len(dates) // 2)
    level = panel.index.get_level_values("date")
    return panel[level.isin(dates[:midpoint])], panel[level.isin(dates[midpoint:])]


def _similarity(left: str, right: str) -> float:
    left_tokens = Counter(_AST_TOKENS.findall(left))
    right_tokens = Counter(_AST_TOKENS.findall(right))
    intersection = sum((left_tokens & right_tokens).values())
    union = sum((left_tokens | right_tokens).values())
    return intersection / union if union else 1.0


class _BoundedMiner:
    def __init__(
        self,
        reviewer: Reviewer,
        candidate_limit: int,
        progress: Progress | None = None,
    ) -> None:
        if not 1 <= candidate_limit <= 8:
            raise ValueError("candidate_limit must be from 1 to 8")
        self.reviewer = reviewer
        self.candidate_limit = candidate_limit
        self.progress = progress or (lambda _stage, _message: None)

    def _ask(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        request = {
            **payload,
            "candidate_limit": self.candidate_limit,
            "dsl": (
                "fields=open,high,low,close,volume,returns; functions=add,sub,mul,div,"
                "delay,delta,rank,rolling_mean,rolling_std,rolling_min,rolling_max,zscore"
            ),
            "response_schema": {
                "candidates": [{"hypothesis": "string", "expression": "safe DSL string"}]
            },
        }
        try:
            answer = json.loads(
                self.reviewer(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return [{"rejection_reason": "模型返回的不是有效 JSON"}]
        candidates = answer.get("candidates") if isinstance(answer, dict) else None
        if not isinstance(candidates, list):
            return [{"rejection_reason": "模型结果缺少 candidates"}]
        return [
            candidate
            if isinstance(candidate, dict)
            else {"rejection_reason": "候选因子必须是对象"}
            for candidate in candidates[: self.candidate_limit]
        ]

    @staticmethod
    def _base_record(raw: dict[str, Any], round_number: int) -> dict[str, Any]:
        expression, hypothesis = raw.get("expression"), raw.get("hypothesis")
        return {
            "round": round_number,
            "hypothesis": hypothesis if isinstance(hypothesis, str) else "",
            "expression": expression if isinstance(expression, str) else "",
            "rejection_reason": raw.get("rejection_reason"),
        }


class AlphaAgentMiner(_BoundedMiner):
    """Regularize novelty, complexity and temporal decay before frozen testing."""

    def __init__(
        self,
        reviewer: Reviewer,
        *,
        candidate_limit: int = 4,
        complexity_penalty: float = 0.001,
        novelty_penalty: float = 0.10,
        decay_penalty: float = 0.50,
        progress: Progress | None = None,
    ) -> None:
        super().__init__(reviewer, candidate_limit, progress)
        for name, value in (
            ("complexity_penalty", complexity_penalty),
            ("novelty_penalty", novelty_penalty),
            ("decay_penalty", decay_penalty),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.complexity_penalty = complexity_penalty
        self.novelty_penalty = novelty_penalty
        self.decay_penalty = decay_penalty

    def _score(
        self,
        raw: dict[str, Any],
        validation: pd.DataFrame,
        prior_expressions: list[str],
    ) -> dict[str, Any]:
        record = self._base_record(raw, 1)
        if record["rejection_reason"]:
            return record
        if not record["hypothesis"].strip():
            record["rejection_reason"] = "缺少可审计的市场假设"
            return record
        try:
            factor = parse_factor(record["expression"])
            early, late = _halve(validation)
            early_ic = _daily_ic(evaluate(factor, early), early)
            late_ic = _daily_ic(evaluate(factor, late), late)
        except (DSLParseError, ValueError) as error:
            record["rejection_reason"] = str(error)
            return record
        if not np.isfinite(early_ic) or not np.isfinite(late_ic):
            record["rejection_reason"] = "分段验证 IC 无效"
            return record
        similarity = max(
            (_similarity(factor.canonical, expression) for expression in prior_expressions),
            default=0.0,
        )
        mean_ic = (early_ic + late_ic) / 2
        decay = abs(early_ic - late_ic)
        score = (
            mean_ic
            - self.complexity_penalty * factor.nodes
            - self.novelty_penalty * similarity
            - self.decay_penalty * decay
        )
        record.update(
            expression=factor.canonical,
            nodes=factor.nodes,
            early_validation_ic=early_ic,
            late_validation_ic=late_ic,
            validation_ic=mean_ic,
            temporal_decay=decay,
            max_similarity=similarity,
            originality=1 - similarity,
            score=score,
        )
        return record

    def mine(self, panel: pd.DataFrame) -> dict[str, Any]:
        _, validation, test = chronological_split(panel)
        self.progress("generate", "AlphaAgent 正在生成市场假设和候选因子")
        raw_candidates = self._ask(
            {
                "paper": "AlphaAgent",
                "stage": "hypothesis_and_factor",
                "objective": "提出有金融逻辑、结构新颖、不过度复杂且跨时期稳定的因子",
            }
        )
        accepted: list[str] = []
        candidates: list[dict[str, Any]] = []
        for raw in raw_candidates:
            record = self._score(raw, validation, accepted)
            candidates.append(record)
            if record["rejection_reason"] is None:
                accepted.append(str(record["expression"]))
        self.progress(
            "regularize",
            f"完成 {len(candidates)} 个候选的复杂度、新颖性和时序衰减评分",
        )
        valid = [candidate for candidate in candidates if candidate["rejection_reason"] is None]
        champion = max(valid, key=lambda item: float(item["score"])) if valid else None
        if champion is not None:
            champion = dict(champion)
            factor = parse_factor(str(champion["expression"]))
            champion["test_ic"] = _daily_ic(evaluate(factor, test), test)
            champion["frozen"] = True
        return {
            "status": "complete" if champion else "partial",
            "paper": _PAPER_ALPHA_AGENT,
            "mechanisms": ["AST 复杂度约束", "结构相似度新颖性惩罚", "分段 IC 衰减惩罚"],
            "champion": champion,
            "candidates": candidates,
        }


class ChainOfAlphaMiner(_BoundedMiner):
    """Run a bounded generation chain followed by feedback-driven optimization."""

    def __init__(
        self,
        reviewer: Reviewer,
        *,
        candidate_limit: int = 4,
        optimization_rounds: int = 2,
        complexity_penalty: float = 0.001,
        progress: Progress | None = None,
    ) -> None:
        super().__init__(reviewer, candidate_limit, progress)
        if not 1 <= optimization_rounds <= 3:
            raise ValueError("optimization_rounds must be from 1 to 3")
        if not np.isfinite(complexity_penalty) or complexity_penalty < 0:
            raise ValueError("complexity_penalty must be finite and non-negative")
        self.optimization_rounds = optimization_rounds
        self.complexity_penalty = complexity_penalty

    def _score(
        self, raw: dict[str, Any], validation: pd.DataFrame, round_number: int
    ) -> dict[str, Any]:
        record = self._base_record(raw, round_number)
        if record["rejection_reason"]:
            return record
        if not record["hypothesis"].strip():
            record["rejection_reason"] = "缺少可审计的市场假设"
            return record
        try:
            factor = parse_factor(record["expression"])
            validation_ic = _daily_ic(evaluate(factor, validation), validation)
        except (DSLParseError, ValueError) as error:
            record["rejection_reason"] = str(error)
            return record
        if not np.isfinite(validation_ic):
            record["rejection_reason"] = "验证 IC 无效"
            return record
        record.update(
            expression=factor.canonical,
            nodes=factor.nodes,
            validation_ic=validation_ic,
            score=validation_ic - self.complexity_penalty * factor.nodes,
        )
        return record

    def mine(self, panel: pd.DataFrame) -> dict[str, Any]:
        _, validation, test = chronological_split(panel)
        rounds: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        self.progress("generate", "Chain-of-Alpha 生成链正在提出初始因子")
        raw = self._ask(
            {
                "paper": "Chain-of-Alpha",
                "chain": "generation",
                "objective": "只根据市场数据提出可解释的初始因子",
            }
        )
        for round_number in range(self.optimization_rounds + 1):
            scored = [self._score(item, validation, round_number) for item in raw]
            candidates.extend(scored)
            valid = [item for item in scored if item["rejection_reason"] is None]
            rounds.append(
                {
                    "round": round_number,
                    "chain": "generation" if round_number == 0 else "optimization",
                    "candidate_count": len(scored),
                    "best_score": max(
                        (float(item["score"]) for item in valid), default=None
                    ),
                }
            )
            self.progress(
                "evaluate",
                f"第 {round_number} 轮完成：{len(scored)} 个候选，"
                f"{len(valid)} 个通过安全校验",
            )
            if round_number == self.optimization_rounds or not valid:
                break
            feedback = [
                {
                    "hypothesis": item["hypothesis"],
                    "expression": item["expression"],
                    "validation_ic": item["validation_ic"],
                    "nodes": item["nodes"],
                    "score": item["score"],
                }
                for item in sorted(valid, key=lambda item: float(item["score"]), reverse=True)
            ]
            self.progress(
                "optimize", f"优化链正在读取第 {round_number} 轮验证反馈"
            )
            raw = self._ask(
                {
                    "paper": "Chain-of-Alpha",
                    "chain": "optimization",
                    "round": round_number + 1,
                    "backtest_feedback": feedback,
                    "objective": "根据验证反馈修复或组合因子，不得直接查看测试集",
                }
            )
        valid_all = [
            candidate for candidate in candidates if candidate["rejection_reason"] is None
        ]
        champion = (
            max(valid_all, key=lambda item: float(item["score"])) if valid_all else None
        )
        if champion is not None:
            champion = dict(champion)
            factor = parse_factor(str(champion["expression"]))
            champion["test_ic"] = _daily_ic(evaluate(factor, test), test)
            champion["frozen"] = True
        return {
            "status": "complete" if champion else "partial",
            "paper": _PAPER_CHAIN,
            "mechanisms": ["因子生成链", "回测反馈", "因子优化链", "冻结测试集"],
            "champion": champion,
            "candidates": candidates,
            "rounds": rounds,
        }

"""Deterministic, artifact-only leaderboard for a small Alpha Arena MVP."""

from __future__ import annotations

import json
import math
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONTESTANTS = ("rules", "trading-agents", "finmem", "quanta-alpha")
MAX_ACTIONS = 1_000


class ContestantStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    ABSENT = "absent"


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class ArenaConfig(BaseModel):
    """The shared run settings required for comparable contestant artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str = Field(min_length=1, max_length=200)
    universe: tuple[str, ...] = Field(min_length=1, max_length=500)
    start_date: date
    end_date: date
    initial_cash: FiniteFloat = Field(gt=0)
    cost_bps: FiniteFloat = Field(ge=0)


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    date: date
    ticker: str = Field(min_length=1, max_length=20)
    action: Literal["buy", "sell", "hold"]
    confidence: FiniteFloat = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=2_000)


class ContestantResult(BaseModel):
    """A bounded, JSON-ready contestant result independent of the strategy runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=100)
    status: ContestantStatus
    equity: dict[date, FiniteFloat] = Field(default_factory=dict, max_length=2_000)
    total_return: FiniteFloat = 0.0
    max_drawdown: FiniteFloat = 0.0
    sharpe: FiniteFloat = 0.0
    costs: FiniteFloat = Field(default=0.0, ge=0)
    risk_violations: int = Field(default=0, ge=0, le=10_000)
    error_category: str | None = Field(default=None, max_length=100)
    actions: tuple[ActionRecord, ...] = Field(default=(), max_length=MAX_ACTIONS)

    @field_validator("equity")
    @classmethod
    def require_equity_for_completed(cls, value: dict[date, float], info: Any) -> dict[date, float]:
        if info.data.get("status") == ContestantStatus.COMPLETED and not value:
            raise ValueError("completed contestants require equity")
        return value


class ArtifactLoader:
    """Loads only the declared manifest, summary, and explicit JSON result artifact."""

    def __init__(self, expected_config: ArenaConfig) -> None:
        self._expected_config = expected_config

    def load(self, name: str, run_path: Path) -> ContestantResult:
        try:
            manifest = self._read_json(run_path / "manifest.json")
            self._validate_manifest(manifest)
            if isinstance(manifest, dict) and "experiment" in manifest:
                return self._load_experiment(name, run_path, manifest)
            result_name = manifest.get("result_file", "summary.json")
            if not isinstance(result_name, str) or not self._safe_json_name(result_name):
                return self._failed(name, "invalid_artifact")
            summary = self._read_json(run_path / "summary.json")
            result = (
                summary
                if result_name == "summary.json"
                else self._read_json(run_path / result_name)
            )
            if not isinstance(result, dict):
                return self._failed(name, "invalid_artifact")
            if not self._all_finite(result):
                return self._failed(name, "invalid_metrics")
            return ContestantResult.model_validate({"name": name, **result})
        except json.JSONDecodeError:
            return self._failed(name, "invalid_json")
        except ValueError as error:
            return self._failed(name, str(error))
        except (OSError, TypeError, ValidationError):
            return self._failed(name, "invalid_artifact")

    def _validate_manifest(self, manifest: Any) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("invalid_artifact")
        expected = self._expected_config.model_dump(mode="json")
        if "experiment" in manifest:
            observed = {
                "fingerprint": manifest.get("data_fingerprint"),
                "universe": manifest.get("universe"),
                "start_date": manifest.get("data_start"),
                "end_date": manifest.get("data_end"),
                "initial_cash": manifest.get("initial_cash"),
                "cost_bps": (
                    manifest.get("commission_bps", 0) + manifest.get("slippage_bps", 0)
                    if isinstance(manifest.get("commission_bps"), int | float)
                    and isinstance(manifest.get("slippage_bps"), int | float)
                    else None
                ),
            }
            if observed != expected:
                raise ValueError("config_mismatch")
            return
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError("config_mismatch")

    def _load_experiment(
        self, name: str, run_path: Path, manifest: dict[str, Any]
    ) -> ContestantResult:
        experiment = manifest.get("experiment")
        result_paths = {
            "finmem": Path("finmem/result.json"),
            "quanta-alpha": Path("quanta_alpha/result.json"),
        }
        if not isinstance(experiment, str):
            return self._failed(name, "invalid_artifact")
        result_path = result_paths.get(experiment)
        if result_path is None:
            return self._failed(name, "invalid_artifact")
        result = self._read_json(run_path / result_path)
        if not isinstance(result, dict) or not self._all_finite(result):
            return self._failed(name, "invalid_metrics")
        if experiment == "finmem":
            metrics = result.get("metrics")
            equity = result.get("equity")
            if not isinstance(metrics, dict) or not isinstance(equity, dict):
                return self._failed(name, "invalid_artifact")
            return ContestantResult.model_validate(
                {
                    "name": name,
                    "status": "completed",
                    "equity": equity,
                    "total_return": metrics.get("total_return", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "sharpe": metrics.get("sharpe", 0),
                    "costs": metrics.get("costs", 0),
                }
            )
        return ContestantResult(name=name, status=ContestantStatus.PARTIAL)

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _safe_json_name(name: str) -> bool:
        return Path(name).name == name and name.endswith(".json")

    @staticmethod
    def _all_finite(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(ArtifactLoader._all_finite(item) for item in value.values())
        if isinstance(value, list):
            return all(ArtifactLoader._all_finite(item) for item in value)
        return True

    @staticmethod
    def _failed(name: str, category: str) -> ContestantResult:
        return ContestantResult(name=name, status=ContestantStatus.FAILED, error_category=category)


class AlphaArena:
    """Build an arena report solely from previously written strategy artifacts."""

    def __init__(
        self,
        config: ArenaConfig,
        default_contestants: tuple[str, ...] = DEFAULT_CONTESTANTS,
    ) -> None:
        self._loader = ArtifactLoader(config)
        self._default_contestants = default_contestants

    def run(self, runs: dict[str, Path]) -> dict[str, Any]:
        results = [self._loader.load(name, path) for name, path in runs.items()]
        provided = set(runs)
        results.extend(
            ContestantResult(name=name, status=ContestantStatus.ABSENT)
            for name in self._default_contestants
            if name not in provided
        )
        ranked = self._rank(results)
        completed = [item for _, item in ranked if item.status == ContestantStatus.COMPLETED]
        return {
            "leaderboard": [self._row(item, rank) for rank, item in ranked],
            "equity": {item.name: self._json_equity(item) for item in completed},
            "action_distribution": {
                action: sum(
                    record.action == action for item in completed for record in item.actions
                )
                for action in ("buy", "sell", "hold")
            },
            "costs": {item.name: item.costs for item in completed},
            "risk_markers": [
                {
                    "name": item.name,
                    "risk_violations": item.risk_violations,
                    "max_drawdown": item.max_drawdown,
                }
                for _, item in ranked
            ],
        }

    @staticmethod
    def _rank(results: list[ContestantResult]) -> list[tuple[int | None, ContestantResult]]:
        completed = sorted(
            (item for item in results if item.status == ContestantStatus.COMPLETED),
            key=lambda item: (
                item.risk_violations,
                abs(item.max_drawdown),
                -item.total_return,
                item.name,
            ),
        )
        other = sorted(
            (item for item in results if item.status != ContestantStatus.COMPLETED),
            key=lambda item: item.name,
        )
        return [(index, item) for index, item in enumerate(completed, start=1)] + [
            (None, item) for item in other
        ]

    @staticmethod
    def _row(item: ContestantResult, rank: int | None) -> dict[str, Any]:
        row = item.model_dump(mode="json")
        row["rank"] = rank
        return row

    @staticmethod
    def _json_equity(item: ContestantResult) -> dict[str, float]:
        return {point.isoformat(): value for point, value in item.equity.items()}

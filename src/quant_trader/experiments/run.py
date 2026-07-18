"""Thin runners that connect the paper strategy cores to durable artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from quant_trader.backtest import run_backtest
from quant_trader.config import Settings
from quant_trader.experiments.models import ExperimentManifest, ExperimentStatus
from quant_trader.experiments.store import ArtifactStore
from quant_trader.llm.base import ChatMessage, LLMReviewer
from quant_trader.strategies.v3_finmem import FinMemReviewer, MemoryBook
from quant_trader.strategies.v4_quanta_alpha import QuantaAlphaMiner
from quant_trader.strategies.v5_alpha_arena import AlphaArena, ArenaConfig

DashboardUpdate = Callable[[str, str, dict[str, object]], None]
ProgressFactory = Callable[[object], object]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _shared_dates(frames: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    if not frames:
        raise ValueError("at least one cached frame is required")
    dates = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    if not dates:
        raise ValueError("cached frames do not share any market dates")
    return pd.DatetimeIndex(dates)


def data_fingerprint(frames: Mapping[str, pd.DataFrame], settings: Settings) -> str:
    dates = _shared_dates(frames)
    payload = {
        "universe": list(settings.universe),
        "start": dates[0].date().isoformat(),
        "end": dates[-1].date().isoformat(),
        "rows": {ticker: len(frames[ticker]) for ticker in settings.universe},
        "costs": [settings.execution.commission_bps, settings.execution.slippage_bps],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _create_store(
    kind: str,
    settings: Settings,
    frames: Mapping[str, pd.DataFrame],
    output_dir: Path,
    provider: str,
    model: str,
    attempt_limit: int,
) -> ArtifactStore:
    run_id = f"{kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    store = ArtifactStore.create(output_dir, kind, run_id)
    dates = _shared_dates(frames)
    store.write_manifest(
        ExperimentManifest(
            run_id=run_id,
            experiment=kind,
            code_version="0.1.0",
            data_fingerprint=data_fingerprint(frames, settings),
            data_start=dates[0].date(),
            data_end=dates[-1].date(),
            universe=settings.universe,
            provider=provider,
            model=model,
            attempt_limit=attempt_limit,
            initial_cash=settings.paper.initial_cash,
            commission_bps=settings.execution.commission_bps,
            slippage_bps=settings.execution.slippage_bps,
            max_position_weight=settings.risk.max_position_weight,
            max_gross_exposure=settings.risk.max_gross_exposure,
            max_drawdown=settings.risk.halt_drawdown,
        )
    )
    store.append_event("stage", "prepare", "Validated cached data and settings.")
    return store


def run_finmem(
    settings: Settings,
    frames: Mapping[str, pd.DataFrame],
    output_dir: Path,
    provider: LLMReviewer,
    provider_name: str,
    model: str,
    progress_factory: ProgressFactory,
    dashboard: DashboardUpdate | None = None,
) -> Path:
    store = _create_store("finmem", settings, frames, output_dir, provider_name, model, 1)
    if dashboard is not None:
        dashboard("prepare", "preparing", {"run_id": store.root.name, "calls": 0})
    memory = MemoryBook()
    finmem = FinMemReviewer(provider, memory)
    reviewer = progress_factory(finmem)
    store.append_event(
        "stage", "backtest", "Started bounded FinMem backtest.", ExperimentStatus.RUNNING
    )
    if dashboard is not None:
        dashboard(
            "backtest",
            "running",
            {"calls": 0, "memory": {"short": [], "mid": [], "long": []}},
        )
    result = run_backtest(frames, settings, reviewer=reviewer)
    decision = finmem.last_decision
    _write_json(store.root / "finmem" / "memory.json", memory.snapshot())
    _write_json(store.root / "finmem" / "decisions.json", decision)
    result_payload = result.to_dict()
    _write_json(store.root / "finmem" / "result.json", result_payload)
    calls = int(getattr(reviewer, "real_calls", 0))
    store.append_event("result", "persist", "Wrote validated FinMem artifacts.")
    store.write_summary(
        ExperimentStatus.COMPLETED,
        {"provider_calls": calls, **result.metrics()},
    )
    if dashboard is not None:
        dashboard(
            "complete",
            "completed",
            {
                "calls": calls,
                "memory": {"short": [], "mid": [], "long": []},
                "decision": decision,
                "metrics": result.metrics(),
            },
        )
    return store.root


def _panel(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume"]
    pieces: list[pd.DataFrame] = []
    for ticker in sorted(frames):
        frame = frames[ticker].loc[:, columns].copy()
        frame["returns"] = frame["close"].astype(float).pct_change()
        frame["ticker"] = ticker
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        pieces.append(frame.reset_index().set_index(["date", "ticker"]))
    return pd.concat(pieces).sort_index().loc[:, [*columns, "returns"]]


def run_quanta_alpha(
    settings: Settings,
    frames: Mapping[str, pd.DataFrame],
    output_dir: Path,
    provider: LLMReviewer,
    provider_name: str,
    model: str,
    dashboard: DashboardUpdate | None = None,
) -> Path:
    store = _create_store("quanta-alpha", settings, frames, output_dir, provider_name, model, 2)
    if dashboard is not None:
        dashboard("prepare", "preparing", {"run_id": store.root.name, "calls": 0})
    calls = 0

    def review(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if len(prompt.encode("utf-8")) > 16_384:
            raise ValueError("factor review request is too large")
        return provider.complete(
            (
                ChatMessage(role="system", content="Return only bounded QuantaAlpha DSL JSON."),
                ChatMessage(role="user", content=prompt),
            )
        )

    store.append_event("stage", "mine", "Started bounded factor mining.", ExperimentStatus.RUNNING)
    if dashboard is not None:
        dashboard("mine", "running", {"calls": 0, "candidates": [], "edges": []})
    result = _json_safe(QuantaAlphaMiner(review).mine(_panel(frames)))
    assert isinstance(result, dict)
    _write_json(store.root / "quanta_alpha" / "result.json", result)
    candidates = result.get("candidates")
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    champion = result.get("champion") is not None
    status = ExperimentStatus.COMPLETED if champion else ExperimentStatus.PARTIAL
    store.append_event("result", "persist", "Wrote validated factor artifacts.", status)
    store.write_summary(
        status,
        {"provider_calls": calls, "candidate_count": candidate_count, "champion": champion},
    )
    if dashboard is not None:
        dashboard("complete", status.value, {"calls": calls, **result})
    return store.root


def run_alpha_arena(
    settings: Settings,
    frames: Mapping[str, pd.DataFrame],
    output_dir: Path,
    contestant_runs: tuple[Path, ...],
    dashboard: DashboardUpdate | None = None,
) -> Path:
    store = _create_store("alpha-arena", settings, frames, output_dir, "none", "none", 1)
    if dashboard is not None:
        dashboard("prepare", "preparing", {"run_id": store.root.name, "calls": 0})
    dates = _shared_dates(frames)
    config = ArenaConfig(
        fingerprint=data_fingerprint(frames, settings),
        universe=settings.universe,
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
        initial_cash=settings.paper.initial_cash,
        cost_bps=settings.execution.commission_bps + settings.execution.slippage_bps,
    )
    runs: dict[str, Path] = {}
    for path in contestant_runs:
        name = path.resolve().name
        if name in runs:
            raise ValueError(f"duplicate contestant run name: {name}")
        runs[name] = path
    store.append_event("stage", "compare", "Loaded existing contestant artifacts only.")
    if dashboard is not None:
        dashboard("compare", "running", {"calls": 0, "leaderboard": []})
    result = AlphaArena(config).run(runs)
    _write_json(store.root / "alpha_arena" / "result.json", result)
    rows = result["leaderboard"]
    completed = sum(row["status"] == "completed" for row in rows)
    store.append_event("result", "persist", "Wrote deterministic arena leaderboard.")
    store.write_summary(
        ExperimentStatus.COMPLETED,
        {"provider_calls": 0, "contestant_count": len(runs), "completed_count": completed},
    )
    if dashboard is not None:
        dashboard("complete", "completed", {"calls": 0, **result})
    return store.root

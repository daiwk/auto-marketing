import json
from datetime import date
from pathlib import Path

from quant_trader.experiments.models import ExperimentManifest, ExperimentStatus
from quant_trader.experiments.store import ArtifactStore


def fixed_manifest(*, run_id: str = "fixed-run", experiment: str = "finmem") -> ExperimentManifest:
    return ExperimentManifest(
        run_id=run_id,
        experiment=experiment,
        code_version="test-version",
        data_fingerprint="test-data",
        data_start=date(2025, 1, 1),
        data_end=date(2025, 12, 31),
        universe=("AAPL",),
        provider="test-provider",
        model="test-model",
        attempt_limit=2,
        initial_cash=100_000,
        commission_bps=1,
        slippage_bps=2,
        max_position_weight=0.1,
        max_gross_exposure=1.0,
        max_drawdown=0.2,
    )


def test_store_writes_manifest_event_and_summary_atomically(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, "finmem", "fixed-run")
    store.write_manifest(fixed_manifest(run_id="fixed-run", experiment="finmem"))
    store.append_event("stage_started", "load_data", "Loading cached bars.")
    store.write_summary(ExperimentStatus.COMPLETED, {"calls": 0})

    assert json.loads((store.root / "manifest.json").read_text())["experiment"] == "finmem"
    assert json.loads((store.root / "events.jsonl").read_text().splitlines()[0])["sequence"] == 1
    assert json.loads((store.root / "summary.json").read_text())["status"] == "completed"

"""Atomic, local persistence for experiment artifacts."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from quant_trader.experiments.models import ExperimentEvent, ExperimentManifest, ExperimentStatus

type JSONPrimitive = str | int | float | bool | None


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _validated_metrics(metrics: dict[str, JSONPrimitive]) -> dict[str, JSONPrimitive]:
    validated: dict[str, JSONPrimitive] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric names must be non-empty strings")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metric values must be finite")
        if value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError("metric values must be JSON primitive values")
        validated[key] = value
    return validated


class ArtifactStore:
    """Owns one newly-created run directory and its ordered event stream."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self._run_id = run_id
        self._sequence = 0
        self._lock = Lock()

    @classmethod
    def create(cls, output_dir: Path, experiment: str, run_id: str) -> ArtifactStore:
        # The manifest owns the experiment label; no module directory is created here.
        del experiment
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single directory name")
        root = (output_dir / run_id).resolve()
        output_root = output_dir.resolve()
        if root.parent != output_root:
            raise ValueError("run_id must resolve directly beneath output_dir")
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"run directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        return cls(root, run_id)

    def write_manifest(self, manifest: ExperimentManifest) -> None:
        if manifest.run_id != self._run_id:
            raise ValueError("manifest run_id does not match artifact store")
        _atomic_json(self.root / "manifest.json", manifest.model_dump(mode="json"))

    def append_event(
        self,
        kind: str,
        stage: str,
        message: str,
        status: ExperimentStatus | None = None,
    ) -> ExperimentEvent:
        with self._lock:
            self._sequence += 1
            event = ExperimentEvent(
                run_id=self._run_id,
                sequence=self._sequence,
                at=datetime.now(UTC),
                kind=kind,
                stage=stage,
                message=message,
                status=status,
            )
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as destination:
                destination.write(event.model_dump_json() + "\n")
            return event

    def write_summary(self, status: ExperimentStatus, metrics: dict[str, JSONPrimitive]) -> None:
        if not isinstance(status, ExperimentStatus):
            raise TypeError("status must be an ExperimentStatus")
        _atomic_json(
            self.root / "summary.json",
            {"status": status.value, "metrics": _validated_metrics(metrics)},
        )

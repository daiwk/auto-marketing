from datetime import UTC, datetime

import pytest

from quant_trader.experiments.models import ExperimentEvent, ExperimentManifest, ExperimentStatus


def test_event_is_bounded_and_strict() -> None:
    event = ExperimentEvent(
        run_id="run-1",
        sequence=1,
        at=datetime(2026, 7, 18, tzinfo=UTC),
        kind="stage_started",
        stage="load_data",
        message="Loading cached bars.",
    )

    assert event.status is None

    with pytest.raises(ValueError):
        ExperimentEvent(
            run_id="run-1",
            sequence=2,
            at=datetime.now(UTC),
            kind="stage_started",
            stage="x" * 81,
            message="bad",
        )


def test_status_has_terminal_partial_state() -> None:
    assert ExperimentStatus.PARTIAL.value == "partial"


def test_manifest_schema_cannot_store_keys_or_prompts() -> None:
    with pytest.raises(ValueError):
        ExperimentManifest.model_validate(
            {
                "run_id": "run-1",
                "experiment": "finmem",
                "code_version": "abc",
                "data_fingerprint": "def",
                "universe": ["AAPL"],
                "attempt_limit": 2,
                "api_key": "secret",
            }
        )

from datetime import UTC, date, datetime

import pytest

from quant_trader.experiments.models import ExperimentEvent, ExperimentManifest, ExperimentStatus


def _manifest_payload() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "experiment": "finmem",
        "code_version": "abc",
        "data_fingerprint": "def",
        "data_start": date(2026, 1, 1),
        "data_end": date(2026, 1, 2),
        "universe": ("AAPL",),
        "provider": "test-provider",
        "model": "test-model",
        "attempt_limit": 2,
        "initial_cash": 100_000,
        "commission_bps": 1,
        "slippage_bps": 2,
        "max_position_weight": 0.1,
        "max_gross_exposure": 1.0,
        "max_drawdown": 0.2,
    }


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
    payload = _manifest_payload()
    payload["api_key"] = "secret"

    with pytest.raises(ValueError):
        ExperimentManifest.model_validate(payload)


def test_event_rejects_coerced_sequence_timestamp_and_status() -> None:
    event = {
        "run_id": "run-1",
        "sequence": 1,
        "at": datetime(2026, 7, 18, tzinfo=UTC),
        "kind": "stage_started",
        "stage": "load_data",
        "message": "Loading cached bars.",
    }

    for field, value in {
        "sequence": "1",
        "at": "2026-07-18T00:00:00+00:00",
        "status": "completed",
    }.items():
        invalid = {**event, field: value}
        with pytest.raises(ValueError):
            ExperimentEvent.model_validate(invalid)


def test_manifest_rejects_coerced_dates_and_attempt_limit() -> None:
    for field, value in {"data_start": "2026-01-01", "attempt_limit": "2"}.items():
        payload = {**_manifest_payload(), field: value}
        with pytest.raises(ValueError):
            ExperimentManifest.model_validate(payload)

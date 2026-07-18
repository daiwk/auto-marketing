# Shared Experiment Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the typed experiment lifecycle, durable artifacts, provider-attempt budget, CLI entry point, and generic live dashboard needed by all paper-inspired strategies.

**Architecture:** Preserve the existing TradingAgents dashboard API while adding an experiment event projection beside it. A small runner owns status, cancellation and artifact writes; provider attempt hooks make retries visible and budgeted without changing strategy prompts.

**Tech Stack:** Python 3.12, Pydantic 2, Typer, standard-library HTTP/threading, pytest

---

**Dependency:** Start from commit `d0d7731` on `codex/paper-strategies-mvp`.

### Task 1: Typed lifecycle and atomic artifact store

**Files:**
- Create: `src/quant_trader/experiments/__init__.py`
- Create: `src/quant_trader/experiments/models.py`
- Create: `src/quant_trader/experiments/store.py`
- Create: `tests/unit/experiments/test_models.py`
- Create: `tests/unit/experiments/test_store.py`

- [ ] **Step 1: Write failing model and store tests**

```python
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
        ExperimentManifest.model_validate({
            "run_id": "run-1", "experiment": "finmem", "code_version": "abc",
            "data_fingerprint": "def", "universe": ["AAPL"], "attempt_limit": 2,
            "api_key": "secret",
        })
```

```python
import json

from quant_trader.experiments.models import ExperimentStatus
from quant_trader.experiments.store import ArtifactStore


def test_store_writes_manifest_event_and_summary_atomically(tmp_path) -> None:
    store = ArtifactStore.create(tmp_path, "finmem", "fixed-run")
    store.write_manifest(fixed_manifest(run_id="fixed-run", experiment="finmem"))
    store.append_event("stage_started", "load_data", "Loading cached bars.")
    store.write_summary(ExperimentStatus.COMPLETED, {"calls": 0})

    assert json.loads((store.root / "manifest.json").read_text())["experiment"] == "finmem"
    assert json.loads((store.root / "events.jsonl").read_text().splitlines()[0])["sequence"] == 1
    assert json.loads((store.root / "summary.json").read_text())["status"] == "completed"
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run: `pytest tests/unit/experiments/test_models.py tests/unit/experiments/test_store.py -q`

Expected: FAIL because `quant_trader.experiments` does not exist.

- [ ] **Step 3: Implement strict events and atomic JSON writes**

Implement `ExperimentStatus` with `pending`, `running`, `partial`, `completed`, `failed`, and
`cancelled`. Implement frozen `ExperimentEvent` fields exactly as used above, with `run_id` capped
at 100 characters, `stage` at 80, and `message` at 500. Add a strict, extra-forbidden
`ExperimentManifest` containing only run/experiment IDs, code version, data fingerprint and dates,
universe, provider/model labels, attempt limit, initial cash, costs and risk bounds. Add the
`fixed_manifest` test helper in `tests/unit/experiments/test_store.py`; it must return a complete
`ExperimentManifest` and accept only `run_id` and `experiment` overrides. `ArtifactStore.create()` must resolve
`<output-dir>/<run-id>`, reject existing non-empty run directories, and keep a monotonically
increasing sequence. Write JSON through a sibling `.tmp` file followed by `Path.replace()`; append
events as one compact JSON object per line under a lock. Export these public types from `__init__.py`.

```python
def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
```

- [ ] **Step 4: Run tests and static checks**

Run: `pytest tests/unit/experiments/test_models.py tests/unit/experiments/test_store.py -q && ruff check src/quant_trader/experiments tests/unit/experiments`

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/experiments tests/unit/experiments
git commit -m "feat: add durable experiment lifecycle"
```

### Task 2: Provider attempt budget and visible deadlines

**Files:**
- Modify: `src/quant_trader/llm/base.py`
- Modify: `src/quant_trader/llm/minimax.py`
- Modify: `src/quant_trader/llm/codex.py`
- Create: `src/quant_trader/experiments/budget.py`
- Create: `tests/unit/experiments/test_budget.py`
- Modify: `tests/unit/llm/test_minimax.py`
- Modify: `tests/unit/llm/test_codex.py`

- [ ] **Step 1: Write failing attempt-budget tests**

```python
import pytest

from quant_trader.experiments.budget import AttemptBudget, BudgetExceeded


def test_every_retry_consumes_budget() -> None:
    budget = AttemptBudget(limit=2)
    assert budget.consume("minimax") == 1
    assert budget.consume("minimax") == 2
    with pytest.raises(BudgetExceeded, match="attempt budget exhausted"):
        budget.consume("minimax")
```

Add provider tests that inject `before_attempt=budget.consume`; a MiniMax 500 followed by success
must consume two attempts, while one Codex subprocess invocation consumes one.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit/experiments/test_budget.py tests/unit/llm/test_minimax.py tests/unit/llm/test_codex.py -q`

Expected: FAIL because attempt hooks and `AttemptBudget` are absent.

- [ ] **Step 3: Implement the hook without changing the reviewer protocol**

Add this public callback type in `llm/base.py`:

```python
type BeforeAttempt = Callable[[str], int]
```

Add optional `before_attempt: BeforeAttempt | None = None` constructor arguments to both providers.
MiniMax invokes it immediately before every HTTP attempt inside its existing retry loop; Codex
invokes it immediately before spawning the command. `AttemptBudget.consume()` increments under a
lock, emits a supplied `Callable[[int, int, str], None]`, and raises before an over-budget attempt.
The exception contains no prompt, response, key or command content.

- [ ] **Step 4: Run provider regressions**

Run: `pytest tests/unit/experiments/test_budget.py tests/unit/llm -q && mypy src/quant_trader/llm src/quant_trader/experiments/budget.py`

Expected: all tests and mypy pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/llm src/quant_trader/experiments/budget.py tests/unit/llm tests/unit/experiments/test_budget.py
git commit -m "feat: budget every provider attempt"
```

### Task 3: Runner, cancellation, and generic dashboard projection

**Files:**
- Create: `src/quant_trader/experiments/runner.py`
- Modify: `src/quant_trader/dashboard.py`
- Modify: `src/quant_trader/dashboard_template.py`
- Create: `tests/unit/experiments/test_runner.py`
- Modify: `tests/unit/test_dashboard.py`

- [ ] **Step 1: Write failing runner and dashboard tests**

```python
from quant_trader.experiments.models import ExperimentStatus
from quant_trader.experiments.runner import CancellationToken, ExperimentRunner
from quant_trader.experiments.store import ArtifactStore


def test_cancelled_runner_keeps_partial_artifacts(tmp_path) -> None:
    store = ArtifactStore.create(tmp_path, "finmem", "run-1")
    token = CancellationToken()
    token.cancel()
    result = ExperimentRunner(store, token=token).run(lambda context: context.check_cancelled())
    assert result.status is ExperimentStatus.CANCELLED
    assert (store.root / "summary.json").exists()
```

Extend dashboard HTTP tests to publish an `ExperimentEvent`, fetch its state, POST the tokenized
cancel route twice, assert both responses are 202, and assert an unknown POST route is 404.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit/experiments/test_runner.py tests/unit/test_dashboard.py -q`

Expected: FAIL because the runner and experiment dashboard mode are missing.

- [ ] **Step 3: Implement compatibility-preserving experiment mode**

`CancellationToken` wraps `threading.Event`; `check_cancelled()` raises a sanitized
`ExperimentCancelled`. `ExperimentRunner.run()` writes running and terminal events/summaries and
maps domain failures to `partial` while mapping precondition failures to `failed`.

Keep `DashboardState.publish(AgentEvent)` unchanged. Add `publish_experiment(ExperimentEvent)` and
an `experiment` snapshot branch containing current stage, status, started/deadline timestamps,
attempt counts and bounded module data. Add optional `cancel: Callable[[], None]` to
`DashboardServer`; only when supplied, expose `POST /<token>/cancel`, require an empty body, reply
202, and invoke it idempotently. Update the existing fixed template to switch on snapshot mode and
render dynamic strings only through `textContent`.

- [ ] **Step 4: Run dashboard and runner regressions**

Run: `pytest tests/unit/experiments/test_runner.py tests/unit/test_dashboard.py tests/unit/strategies/test_v2_events.py -q`

Expected: all tests pass, including existing TradingAgents dashboard behavior.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/experiments/runner.py src/quant_trader/dashboard.py src/quant_trader/dashboard_template.py tests/unit/experiments/test_runner.py tests/unit/test_dashboard.py
git commit -m "feat: run and visualize bounded experiments"
```

### Task 4: CLI skeleton and full kernel verification

**Files:**
- Modify: `src/quant_trader/cli.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_experiment_rejects_unknown_kind_without_opening_provider(runner, tmp_path) -> None:
    result = runner.invoke(
        app,
        ["experiment", "run", "unknown", "--config", "configs/default.yaml", "--data-root", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_experiment_finmem_reports_not_installed_yet(runner, cached_data) -> None:
    result = runner.invoke(app, ["experiment", "run", "finmem", *cached_data])
    assert result.exit_code == 1
    assert "finmem experiment is not available in this build" in result.output
```

- [ ] **Step 2: Run the CLI tests and verify failure**

Run: `pytest tests/unit/test_cli.py -q`

Expected: FAIL because the `experiment` command group is absent.

- [ ] **Step 3: Add the stable command surface**

Add `ExperimentKind(StrEnum)` values `finmem`, `quanta-alpha`, and `alpha-arena`; add the
`experiment` Typer group and `experiment run` options from the design. Dispatch through a registry
whose absent entries fail with the exact sanitized message tested above. Provider construction must
occur after config, cache and output preconditions. Implement `_open_experiment_provider` separately
from the existing commands: MiniMax receives `timeout_seconds=120`, `max_retries=1` and the shared
attempt hook; Codex receives `timeout_seconds=120` and the same hook. Add a short README section stating that the
three strategies arrive in the next sequential plans.

- [ ] **Step 4: Run the complete project checks**

Run: `pytest -q && ruff check . && mypy src`

Expected: the full suite passes with no lint or type errors.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/cli.py tests/unit/test_cli.py README.md
git commit -m "feat: expose paper experiment command"
```

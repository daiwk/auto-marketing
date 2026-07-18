# Alpha Arena MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local paper-trading arena that compares four strategies on identical cached data and costs, isolates failures, and visualizes a replayable risk-first leaderboard.

**Architecture:** Arena consumes normalized contestant artifacts rather than model prose. Adapter functions either load prior runs or explicitly launch missing strategies in sequence; comparison and ranking are deterministic and make no LLM calls of their own.

**Tech Stack:** Python 3.12, Pydantic 2, pandas/numpy, Plotly already present for reports, pytest

---

**Dependency:** Complete the shared kernel, FinMem and QuantaAlpha plans first.

### Task 1: Contestant artifact contract and risk-first ranking

**Files:**
- Create: `src/quant_trader/strategies/v5_alpha_arena/__init__.py`
- Create: `src/quant_trader/strategies/v5_alpha_arena/models.py`
- Create: `src/quant_trader/strategies/v5_alpha_arena/ranking.py`
- Create: `tests/unit/strategies/test_v5_alpha_arena_ranking.py`

- [ ] **Step 1: Write failing ranking and isolation tests**

```python
from quant_trader.strategies.v5_alpha_arena.models import ContestantResult, ContestantStatus
from quant_trader.strategies.v5_alpha_arena.ranking import rank_contestants


def result(name, violations, drawdown, total_return):
    return ContestantResult(
        name=name, status=ContestantStatus.COMPLETED, equity={"2026-01-02": 100_000.0},
        total_return=total_return, max_drawdown=drawdown, sharpe=0.0, turnover=0.0,
        costs=0.0, risk_violations=violations, actions=(), artifact_path=None,
        error_category=None,
    )


def test_ranking_is_risk_first_then_return() -> None:
    ranked = rank_contestants([
        result("high-return-risky", 1, -0.01, 0.50),
        result("safe-deep-dd", 0, -0.20, 0.30),
        result("safe-shallow-dd", 0, -0.05, 0.10),
    ])
    assert [item.name for item in ranked] == ["safe-shallow-dd", "safe-deep-dd", "high-return-risky"]
```

Add a failed contestant and assert it remains in output with no rank while completed contestants
still rank.

- [ ] **Step 2: Run and verify import failure**

Run: `pytest tests/unit/strategies/test_v5_alpha_arena_ranking.py -q`

Expected: FAIL because `v5_alpha_arena` is absent.

- [ ] **Step 3: Implement strict normalized results**

Add a strict `ActionRecord` containing date, ticker, `buy|sell|hold`, confidence in `[0,1]`, and a
bounded reason. Bound names and artifact paths, require finite metrics, ordered ISO equity dates, and require an
error category only for failed/absent results. Sort completed entries by `(risk_violations,
abs(max_drawdown), -total_return, name)`; append partial, failed and absent entries without rank.
Do not silently coerce NaN or infinity—convert the contestant to failed with `invalid_metrics`.

- [ ] **Step 4: Run tests and type checks**

Run: `pytest tests/unit/strategies/test_v5_alpha_arena_ranking.py -q && mypy src/quant_trader/strategies/v5_alpha_arena`

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v5_alpha_arena tests/unit/strategies/test_v5_alpha_arena_ranking.py
git commit -m "feat: add risk-first arena ranking"
```

### Task 2: Frozen QuantaAlpha factor contestant

**Files:**
- Create: `src/quant_trader/strategies/v4_quanta_alpha/strategy.py`
- Create: `tests/unit/strategies/test_v4_quanta_alpha_strategy.py`
- Modify: `src/quant_trader/backtest.py`
- Create: `tests/unit/test_backtest.py`

- [ ] **Step 1: Write failing factor-strategy boundary tests**

```python
from quant_trader.strategies.v4_quanta_alpha.dsl import parse_factor
from quant_trader.strategies.v4_quanta_alpha.strategy import FrozenFactorStrategy


def test_frozen_factor_emits_only_shared_signal_intents(feature_snapshot, signal_times) -> None:
    strategy = FrozenFactorStrategy(parse_factor("delta(close,20)"), top_count=1)
    intents = strategy.generate(feature_snapshot, **signal_times)
    assert len(intents) <= 1
    assert all(intent.target_weight <= 0.15 for intent in intents)
```

Add a backtest test that supplies an arbitrary `Strategy` instance through a new
`run_strategy_backtest` function and proves the existing `run_backtest` results remain unchanged.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/unit/strategies/test_v4_quanta_alpha_strategy.py tests/unit/test_backtest.py -q`

Expected: FAIL because frozen factor execution and generic strategy backtesting are absent.

- [ ] **Step 3: Refactor without changing execution semantics**

Extract the current chronological loop into `run_strategy_backtest(frames, settings, strategy)`;
keep `run_backtest()` as a compatibility wrapper that constructs V1 and delegates. The frozen
factor strategy evaluates only the supplied snapshot history, ranks the latest cross-section,
emits long-only intents for positive top scores, and caps each target at the existing configured
position limit. HardRisk remains authoritative and next-open execution remains unchanged.

- [ ] **Step 4: Run backtest regressions**

Run: `pytest tests/unit/test_backtest.py tests/integration/test_paper_mvp.py tests/unit/strategies/test_v4_quanta_alpha_strategy.py -q`

Expected: all old and new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/backtest.py src/quant_trader/strategies/v4_quanta_alpha/strategy.py tests/unit/test_backtest.py tests/unit/strategies/test_v4_quanta_alpha_strategy.py
git commit -m "feat: execute frozen QuantaAlpha factors"
```

### Task 3: Artifact loading and optional sequential contestant runs

**Files:**
- Create: `src/quant_trader/strategies/v5_alpha_arena/artifacts.py`
- Create: `src/quant_trader/strategies/v5_alpha_arena/experiment.py`
- Create: `tests/unit/strategies/test_v5_alpha_arena_artifacts.py`
- Create: `tests/integration/test_alpha_arena_experiment.py`

- [ ] **Step 1: Write failing artifact and fault-isolation tests**

```python
def test_arena_keeps_running_after_one_bad_artifact(tmp_path, completed_run) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "summary.json").write_text("not-json")
    result = run_alpha_arena(
        contestant_runs={"finmem": completed_run, "quanta-alpha": bad},
        run_missing=False,
    )
    assert result.by_name("finmem").status.value == "completed"
    assert result.by_name("quanta-alpha").error_category == "invalid_artifact"
    assert result.by_name("trading-agents").status.value == "absent"
```

Add an integration test with fixed cached bars that runs the rules baseline plus prepared FinMem and
QuantaAlpha artifacts, then asserts every completed contestant shares identical start/end dates and
initial equity.

- [ ] **Step 2: Run and verify missing loader**

Run: `pytest tests/unit/strategies/test_v5_alpha_arena_artifacts.py tests/integration/test_alpha_arena_experiment.py -q`

Expected: FAIL because artifact loading and arena orchestration are absent.

- [ ] **Step 3: Implement adapters and explicit call behavior**

Validate each manifest data fingerprint, universe, date range, initial cash, slippage, commission and
risk settings against the arena manifest before accepting it. Convert module results into
`ContestantResult`, including only validated action/confidence/reason records; never read raw prompts
or provider responses. Rules-only actions receive confidence `1.0` and a fixed deterministic reason.

Always run the rules baseline locally. With default `run_missing=False`, mark other missing entries
absent and consume zero provider attempts. With explicit `--run-missing`, run TradingAgents,
FinMem, then the frozen factor sequentially using their existing handlers and caps; each writes a
child run referenced by the arena manifest. Catch each sanitized failure separately. Arena code
must not call `reviewer.complete()` itself.

- [ ] **Step 4: Run integration and budget tests**

Run: `pytest tests/unit/strategies/test_v5_alpha_arena_artifacts.py tests/integration/test_alpha_arena_experiment.py tests/unit/experiments/test_budget.py -q`

Expected: tests pass; the default arena case records zero provider attempts.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v5_alpha_arena tests/unit/strategies/test_v5_alpha_arena_artifacts.py tests/integration/test_alpha_arena_experiment.py
git commit -m "feat: orchestrate isolated arena contestants"
```

### Task 4: CLI, live leaderboard, replay, and Chinese guide

**Files:**
- Modify: `src/quant_trader/experiments/models.py`
- Modify: `src/quant_trader/dashboard.py`
- Modify: `src/quant_trader/dashboard_template.py`
- Modify: `src/quant_trader/cli.py`
- Modify: `README.md`
- Create: `src/quant_trader/strategies/v5_alpha_arena/README.md`
- Modify: `tests/unit/test_dashboard.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/integration/test_alpha_arena_experiment.py`

- [ ] **Step 1: Write failing CLI and dashboard tests**

```python
def test_arena_default_does_not_start_llm_contestants(runner, cached_data, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("quant_trader.cli._open_provider", lambda *args: (_ for _ in ()).throw(AssertionError()))
    result = runner.invoke(app, [
        "experiment", "run", "alpha-arena", *cached_data,
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "rules" in result.output
    assert "absent" in result.output
```

Add dashboard assertions for ordered rank, equity series, action/confidence distributions, cost drag,
risk markers and a failed contestant card. Add a replay test that rebuilds the final dashboard state
from `events.jsonl` without any provider.

- [ ] **Step 2: Run and verify missing registration**

Run: `pytest tests/unit/test_cli.py tests/unit/test_dashboard.py tests/integration/test_alpha_arena_experiment.py -q`

Expected: FAIL because Arena is not registered or visualized.

- [ ] **Step 3: Register commands and render the arena view**

Add repeatable `--contestant-run PATH`, opt-in `--run-missing`, and existing provider/review-cap
options. Emit contestant-started/completed/failed and leaderboard-updated typed events. Atomically
write `alpha_arena/contestants.json`, `leaderboard.json`, and `equity.json`.

Render a sortable-looking but deterministic risk-first table, CSS/SVG-free equity polylines,
distribution bars, cost totals and risk markers with safe text APIs. Document in Chinese the default
zero-call replay workflow, the explicit `--run-missing` cost implication, metric definitions,
failure isolation, and why this is not a reproduction of crypto leverage or a profit claim.

- [ ] **Step 4: Run all checks and one local smoke test**

Run: `pytest -q && ruff check . && mypy src`

Expected: all project checks pass.

Run: `arena_smoke_dir=$(mktemp -d /tmp/quant-trader-arena-smoke.XXXXXX)`

Expected: prints no output and creates one new temporary directory.

Run: `quant-trader experiment run alpha-arena --config configs/default.yaml --data-root data --output-dir "$arena_smoke_dir"`

Expected: rules completes from cached data, missing LLM contestants are shown as absent, no API key
is required, and the command exits successfully.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/experiments/models.py src/quant_trader/dashboard.py src/quant_trader/dashboard_template.py src/quant_trader/cli.py src/quant_trader/strategies/v5_alpha_arena README.md tests
git commit -m "feat: add visual Alpha Arena benchmark"
```

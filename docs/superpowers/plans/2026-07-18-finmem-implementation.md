# FinMem MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable FinMem-inspired paper strategy with deterministic three-layer memory, bounded LLM decisions/reflections, durable audit artifacts, and a live memory-lane dashboard.

**Architecture:** A `FinMemReviewer` implements the existing `LLMReviewer` contract, enriches the existing V1 candidate review with retrieved memories, and returns the same constrained `LLMReview` JSON. Memory ranking, promotion, eviction and trigger rules stay deterministic; the shared experiment runner owns provider attempts and artifacts.

**Tech Stack:** Python 3.12, Pydantic 2, existing V1 strategy/backtester, pytest

---

**Dependency:** Complete `2026-07-18-experiment-kernel-implementation.md` first.

### Task 1: Layered memory model and deterministic retrieval

**Files:**
- Create: `src/quant_trader/strategies/v3_finmem/__init__.py`
- Create: `src/quant_trader/strategies/v3_finmem/memory.py`
- Create: `tests/unit/strategies/test_v3_finmem_memory.py`

- [ ] **Step 1: Write failing memory tests**

```python
from datetime import date

from quant_trader.strategies.v3_finmem.memory import FinMemory, MemoryBook, MemoryLayer


def test_retrieval_never_sees_future_memory() -> None:
    book = MemoryBook()
    book.add(FinMemory("past", date(2026, 1, 1), date(2026, 1, 2), MemoryLayer.SHORT, "AAPL", "past", 0.8))
    book.add(FinMemory("future", date(2026, 2, 1), date(2026, 2, 2), MemoryLayer.SHORT, "AAPL", "future", 1.0))
    found = book.retrieve("AAPL", date(2026, 1, 15), per_layer=3)
    assert [item.memory_id for item in found] == ["past"]


def test_eviction_is_stable_for_equal_scores() -> None:
    book = MemoryBook(short_capacity=1)
    book.add(FinMemory("b", date(2026, 1, 1), date(2026, 1, 1), MemoryLayer.SHORT, "AAPL", "b", 0.5))
    book.add(FinMemory("a", date(2026, 1, 1), date(2026, 1, 1), MemoryLayer.SHORT, "AAPL", "a", 0.5))
    assert [item.memory_id for item in book.items()] == ["a"]
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `pytest tests/unit/strategies/test_v3_finmem_memory.py -q`

Expected: FAIL because `v3_finmem` is absent.

- [ ] **Step 3: Implement the immutable memory book**

Use a frozen Pydantic or dataclass record with bounded IDs/summary, importance in `[0, 1]`, and
`available_date >= event_date`. Score only eligible same-ticker memories with
`importance * exp(-age_days / half_life[layer])`; sort by negative score then stable ID. Defaults:
short/mid/long capacities `24/16/8`, half-lives `7/30/120` days, per-layer Top-K `3`. Promotion is
deterministic after the same source is retrieved on 3 and 6 distinct decision dates; promoted
copies retain their source ID. Export snapshots as JSON-ready values.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/strategies/test_v3_finmem_memory.py -q && ruff check src/quant_trader/strategies/v3_finmem tests/unit/strategies/test_v3_finmem_memory.py`

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v3_finmem tests/unit/strategies/test_v3_finmem_memory.py
git commit -m "feat: add deterministic FinMem memory"
```

### Task 2: Bounded decision and reflection reviewer

**Files:**
- Create: `src/quant_trader/strategies/v3_finmem/models.py`
- Create: `src/quant_trader/strategies/v3_finmem/prompts.py`
- Create: `src/quant_trader/strategies/v3_finmem/reviewer.py`
- Create: `tests/unit/strategies/test_v3_finmem_reviewer.py`

- [ ] **Step 1: Write failing reviewer tests**

```python
import json

from quant_trader.strategies.v3_finmem.memory import MemoryBook
from quant_trader.strategies.v3_finmem.models import FinMemProfile
from quant_trader.strategies.v3_finmem.reviewer import FinMemReviewer


class StubReviewer:
    def complete(self, messages):
        payload = json.loads(messages[-1].content)
        assert payload["retrieved_memories"][0]["memory_id"] == "m1"
        return json.dumps({
            "action": "maintain", "weight_multiplier": 1, "confidence": 0.7,
            "thesis": "Momentum remains valid.", "risks": [],
            "invalidation": "Trend breaks.", "input_anomalies": [], "memory_ids": ["m1"],
        })


def test_reviewer_rejects_unknown_memory_evidence(seed_memory, candidate_messages) -> None:
    profile = FinMemProfile(name="conservative", risk_appetite="low", horizon_days=20)
    reviewer = FinMemReviewer(StubReviewer(), MemoryBook([seed_memory]), profile=profile)
    output = json.loads(reviewer.complete(candidate_messages))
    assert output["action"] == "maintain"
    assert reviewer.last_decision.memory_ids == ("m1",)
```

Add cases for an unknown evidence ID, invalid JSON, and an attempted multiplier above 1; all must
return the existing fail-closed reject review without a repair call.

- [ ] **Step 2: Run the test and verify failure**

Run: `pytest tests/unit/strategies/test_v3_finmem_reviewer.py -q`

Expected: FAIL because models, prompts and reviewer are absent.

- [ ] **Step 3: Implement one-call decisions and trigger-only reflection**

Parse the existing final user message as JSON, extract ticker/date/current weight/drawdown, retrieve
memory, and render one new bounded JSON prompt. Add a strict `FinMemProfile` with a bounded name,
`risk_appetite` in `low|medium|high`, and `horizon_days` in `1..252`; the experiment default is
`conservative/low/20`. Include it in every decision prompt, while hard position and drawdown limits
remain authoritative. Parse one response into `FinMemDecision`; validate
that every `memory_id` belongs to the retrieved set, then strip `memory_ids` and return a canonical
existing `LLMReview` JSON to V1. Never issue a repair call.

Track the prior accepted decision per ticker. `reflect_if_needed()` calls the same underlying
reviewer only when current weight moves from positive to zero or drawdown crosses the configured
risk threshold. Parse a `Reflection` containing a summary and importance, then add it to short-term
memory with `available_date` equal to the current decision date. A failed reflection records an
event but cannot change the trading decision.

- [ ] **Step 4: Run reviewer and V1 regressions**

Run: `pytest tests/unit/strategies/test_v3_finmem_reviewer.py tests/unit/strategies/test_v1_strategy.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v3_finmem tests/unit/strategies/test_v3_finmem_reviewer.py
git commit -m "feat: add bounded FinMem reviewer"
```

### Task 3: FinMem experiment, artifacts, and dashboard events

**Files:**
- Create: `src/quant_trader/strategies/v3_finmem/experiment.py`
- Modify: `src/quant_trader/experiments/models.py`
- Modify: `src/quant_trader/dashboard.py`
- Modify: `src/quant_trader/dashboard_template.py`
- Modify: `src/quant_trader/cli.py`
- Create: `tests/integration/test_finmem_experiment.py`
- Modify: `tests/unit/test_dashboard.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing end-to-end and projection tests**

```python
def test_finmem_experiment_writes_replayable_memory_artifacts(
    runner, fixed_config, cached_data, stub_minimax, tmp_path
) -> None:
    result = runner.invoke(app, [
        "experiment", "run", "finmem", "--config", str(fixed_config),
        "--data-root", str(cached_data), "--output-dir", str(tmp_path),
        "--llm-provider", "minimax", "--llm-max-reviews", "1",
    ])
    assert result.exit_code == 0
    run_dir = next(tmp_path.iterdir())
    assert (run_dir / "finmem" / "memory.json").exists()
    assert "completed" in (run_dir / "summary.json").read_text()
```

Add a dashboard test that publishes memory-retrieved, decision and reflection events and asserts the
snapshot exposes three lanes, selected evidence IDs, action, confidence and risk result.

- [ ] **Step 2: Run tests and verify missing registration**

Run: `pytest tests/integration/test_finmem_experiment.py tests/unit/test_dashboard.py tests/unit/test_cli.py -q`

Expected: FAIL because FinMem is not registered.

- [ ] **Step 3: Implement the experiment handler**

Register `run_finmem_experiment` in the CLI registry. Build the provider with the shared attempt
budget, wrap it in `FinMemReviewer`, and call the existing chronological `run_backtest`. Emit typed
memory-retrieved, decision, risk and reflection events; after every accepted state change atomically
write `finmem/memory.json` and `finmem/decisions.json`. Write the standard backtest result under the
module directory and metrics into `summary.json`.

Extend the dashboard projection and fixed template with three compact memory lanes. Highlight only
IDs in the latest decision and connect them to the latest action using CSS borders; do not add a
front-end dependency or render raw prompts.

- [ ] **Step 4: Run FinMem and full regressions**

Run: `pytest tests/integration/test_finmem_experiment.py tests/unit/strategies/test_v3_finmem_memory.py tests/unit/strategies/test_v3_finmem_reviewer.py tests/unit/test_dashboard.py tests/unit/test_cli.py -q && ruff check . && mypy src`

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v3_finmem src/quant_trader/experiments/models.py src/quant_trader/dashboard.py src/quant_trader/dashboard_template.py src/quant_trader/cli.py tests
git commit -m "feat: run and visualize FinMem experiments"
```

### Task 4: Chinese usage documentation and full verification

**Files:**
- Modify: `README.md`
- Create: `src/quant_trader/strategies/v3_finmem/README.md`

- [ ] **Step 1: Add a documentation assertion**

Extend the existing README CLI test to assert the checked-in README contains `experiment run
finmem`, `--llm-max-reviews`, the 120-second deadline explanation, and the memory-lane legend.

- [ ] **Step 2: Run it and verify failure**

Run: `pytest tests/unit/test_cli.py -q`

Expected: FAIL until the Chinese instructions are present.

- [ ] **Step 3: Document a copy-paste MiniMax and Codex run**

Add Chinese examples for both providers, explain the output tree, decision/reflection budget,
Dashboard states, paper-only limitation, and how to distinguish a slow provider from a timed-out
call. The module README briefly maps each implemented mechanism to the FinMem paper and lists the
deliberate MVP omissions.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q && ruff check . && mypy src`

Expected: all project checks pass.

- [ ] **Step 5: Commit**

```bash
git add README.md src/quant_trader/strategies/v3_finmem/README.md tests/unit/test_cli.py
git commit -m "docs: explain FinMem experiment workflow"
```

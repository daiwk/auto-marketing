# QuantaAlpha MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe two-call QuantaAlpha-inspired experiment that generates, validates, evolves, freezes and visualizes a small family of factor DSL expressions.

**Architecture:** LLM output is data, never executable Python. A hand-written parser builds a bounded AST, an allowlisted evaluator operates on pandas frames, and a chronological evaluator selects only on train/validation data before one frozen test evaluation.

**Tech Stack:** Python 3.12, Pydantic 2, pandas/numpy, pytest

---

**Dependency:** Complete the shared experiment kernel plan first. FinMem is not required.

### Task 1: Safe factor DSL parser and canonical AST

**Files:**
- Move: `src/quant_trader/strategies/v3_factor_mining/README.md` → `src/quant_trader/strategies/v4_quanta_alpha/README.md`
- Create: `src/quant_trader/strategies/v4_quanta_alpha/__init__.py`
- Create: `src/quant_trader/strategies/v4_quanta_alpha/dsl.py`
- Create: `tests/unit/strategies/test_v4_quanta_alpha_dsl.py`

- [ ] **Step 1: Write failing allowlist tests**

```python
import pytest

from quant_trader.strategies.v4_quanta_alpha.dsl import FactorSyntaxError, parse_factor


def test_factor_has_stable_canonical_form() -> None:
    factor = parse_factor("zscore(delta(close, 5))")
    assert factor.canonical == "zscore(delta(close,5))"
    assert factor.depth == 3


@pytest.mark.parametrize("expression", [
    "__import__('os').system('id')",
    "close.__class__",
    "rolling_mean(close, 9999)",
    "unknown(close)",
])
def test_factor_rejects_code_and_unbounded_windows(expression: str) -> None:
    with pytest.raises(FactorSyntaxError):
        parse_factor(expression)
```

- [ ] **Step 2: Run and verify import failure**

Run: `pytest tests/unit/strategies/test_v4_quanta_alpha_dsl.py -q`

Expected: FAIL because `v4_quanta_alpha` is absent.

- [ ] **Step 3: Implement a parser without `eval` or Python AST execution**

Tokenize identifiers, finite numeric literals, commas and parentheses. Parse with recursive descent
into immutable `Field`, `Number`, `Unary`, `Binary`, and `Call` nodes. Allow fields `open`, `high`,
`low`, `close`, `volume`, `returns`; arithmetic `add`, `sub`, `mul`, `div`; and calls `delay`,
`delta`, `rank`, `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, `zscore`. Window
arguments are integers in `1..252`; total nodes are at most 40 and depth at most 8. Canonical output
removes whitespace and formats numbers deterministically. Do not import `ast`, call `eval`, or accept
attribute/subscript syntax.

- [ ] **Step 4: Run parser security checks**

Run: `pytest tests/unit/strategies/test_v4_quanta_alpha_dsl.py -q && rg -n "eval\(|exec\(|import ast" src/quant_trader/strategies/v4_quanta_alpha`

Expected: tests pass and `rg` returns no matches.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v3_factor_mining src/quant_trader/strategies/v4_quanta_alpha tests/unit/strategies/test_v4_quanta_alpha_dsl.py
git commit -m "feat: add safe QuantaAlpha factor DSL"
```

### Task 2: Allowlisted evaluator and chronological scoring

**Files:**
- Create: `src/quant_trader/strategies/v4_quanta_alpha/evaluator.py`
- Create: `src/quant_trader/strategies/v4_quanta_alpha/scoring.py`
- Create: `tests/unit/strategies/test_v4_quanta_alpha_evaluator.py`
- Create: `tests/unit/strategies/test_v4_quanta_alpha_scoring.py`

- [ ] **Step 1: Write failing evaluator and split tests**

```python
import numpy as np

from quant_trader.strategies.v4_quanta_alpha.dsl import parse_factor
from quant_trader.strategies.v4_quanta_alpha.evaluator import evaluate_factor


def test_delta_uses_only_prior_rows(panel) -> None:
    values = evaluate_factor(parse_factor("delta(close,2)"), panel)
    expected = panel["close"] - panel.groupby(level="ticker")["close"].shift(2)
    np.testing.assert_allclose(values.dropna(), expected.dropna())
```

```python
from quant_trader.strategies.v4_quanta_alpha.scoring import chronological_splits


def test_test_dates_are_never_in_train_or_validation(shared_dates) -> None:
    split = chronological_splits(shared_dates, train_fraction=0.6, validation_fraction=0.2)
    assert max(split.train) < min(split.validation)
    assert max(split.validation) < min(split.test)
```

- [ ] **Step 2: Run and verify missing evaluator**

Run: `pytest tests/unit/strategies/test_v4_quanta_alpha_evaluator.py tests/unit/strategies/test_v4_quanta_alpha_scoring.py -q`

Expected: FAIL because evaluator and scoring modules are absent.

- [ ] **Step 3: Implement recursive evaluation and frozen scoring**

Evaluate each node by explicit `isinstance` dispatch. Rolling operations group by ticker and use
`min_periods=window`; `rank` ranks cross-sectionally by date; division replaces zero denominators
with NaN; all infinities become NaN. Reject coverage below 60%.

Split unique sorted dates 60/20/20 with at least 20 dates per partition. Compute next-day returns,
daily Spearman IC, long-top-quartile/short-bottom-quartile returns, turnover, Sharpe and maximum
drawdown. Validation score is `mean_ic - 0.05 * turnover - 0.001 * node_count`. Deduplicate by
canonical expression and reject a later candidate when absolute validation correlation with an
accepted candidate exceeds `0.95`. The test partition API accepts only a `FrozenFactor` produced by
validation selection.

- [ ] **Step 4: Run deterministic evaluator tests**

Run: `pytest tests/unit/strategies/test_v4_quanta_alpha_evaluator.py tests/unit/strategies/test_v4_quanta_alpha_scoring.py -q && mypy src/quant_trader/strategies/v4_quanta_alpha`

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v4_quanta_alpha tests/unit/strategies/test_v4_quanta_alpha_evaluator.py tests/unit/strategies/test_v4_quanta_alpha_scoring.py
git commit -m "feat: evaluate QuantaAlpha factors chronologically"
```

### Task 3: Two-call generation and one-generation evolution

**Files:**
- Create: `src/quant_trader/strategies/v4_quanta_alpha/models.py`
- Create: `src/quant_trader/strategies/v4_quanta_alpha/prompts.py`
- Create: `src/quant_trader/strategies/v4_quanta_alpha/miner.py`
- Create: `tests/unit/strategies/test_v4_quanta_alpha_miner.py`

- [ ] **Step 1: Write failing call-cap and rejection tests**

```python
import json

from quant_trader.strategies.v4_quanta_alpha.miner import QuantaAlphaMiner


class QueueReviewer:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0
    def complete(self, messages):
        self.calls += 1
        return next(self.outputs)


def test_miner_uses_two_batched_calls_at_most(training_panel) -> None:
    reviewer = QueueReviewer([
        json.dumps({"candidates": [{"name": "mom", "hypothesis": "trend", "expression": "delta(close,20)"}]}),
        json.dumps({"candidates": [{"name": "mom2", "hypothesis": "short trend", "expression": "delta(close,10)", "parents": ["mom"], "operation": "mutation"}]}),
    ])
    result = QuantaAlphaMiner(reviewer).mine(training_panel)
    assert reviewer.calls == 2
    assert result.frozen.expression in {"delta(close,20)", "delta(close,10)"}
```

Add a case where every seed is unsafe; assert one call, partial status, no frozen factor and explicit
rejection reasons.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/unit/strategies/test_v4_quanta_alpha_miner.py -q`

Expected: FAIL because mining types and prompts are absent.

- [ ] **Step 3: Implement strict batch contracts**

Use Pydantic models capped at four candidates, 80-character names, 500-character hypotheses and
500-character expressions. The seed prompt includes only DSL grammar and data schema. The evolution
prompt includes candidate IDs, canonical expressions, bounded validation metrics and rejection
stage; it never includes test metrics. Parse each response once with no repair call. Validate every
candidate independently, emit rejection records, run one mutation/crossover generation, select on
validation, freeze one winner and evaluate the test partition exactly once.

- [ ] **Step 4: Run miner and security tests**

Run: `pytest tests/unit/strategies/test_v4_quanta_alpha_miner.py tests/unit/strategies/test_v4_quanta_alpha_dsl.py tests/unit/strategies/test_v4_quanta_alpha_scoring.py -q`

Expected: all tests pass and no test metric appears in the evolution prompt assertion.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v4_quanta_alpha tests/unit/strategies/test_v4_quanta_alpha_miner.py
git commit -m "feat: evolve one bounded QuantaAlpha generation"
```

### Task 4: Experiment artifacts, factor tree, and CLI documentation

**Files:**
- Create: `src/quant_trader/strategies/v4_quanta_alpha/experiment.py`
- Modify: `src/quant_trader/experiments/models.py`
- Modify: `src/quant_trader/dashboard.py`
- Modify: `src/quant_trader/dashboard_template.py`
- Modify: `src/quant_trader/cli.py`
- Modify: `README.md`
- Create: `tests/integration/test_quanta_alpha_experiment.py`
- Modify: `tests/unit/test_dashboard.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing integration and dashboard tests**

```python
def test_quanta_alpha_freezes_champion_and_records_tree(
    runner, fixed_config, cached_data, stub_minimax, tmp_path
) -> None:
    result = runner.invoke(app, [
        "experiment", "run", "quanta-alpha", "--config", str(fixed_config),
        "--data-root", str(cached_data), "--output-dir", str(tmp_path),
        "--llm-provider", "minimax",
    ])
    assert result.exit_code == 0
    module = next(tmp_path.iterdir()) / "quanta_alpha"
    assert (module / "factor_tree.json").exists()
    assert '"frozen": true' in (module / "champion.json").read_text()
```

Add a projection test for parent edges, gate states, rejection labels and separated train,
validation and test cards.

- [ ] **Step 2: Run and verify missing registration**

Run: `pytest tests/integration/test_quanta_alpha_experiment.py tests/unit/test_dashboard.py tests/unit/test_cli.py -q`

Expected: FAIL because QuantaAlpha is not registered.

- [ ] **Step 3: Register and visualize the experiment**

Build the cross-sectional panel only from cached frames, run the miner with an attempt limit of two
by default, and atomically write `candidates.json`, `factor_tree.json`, `rejections.json`, and
`champion.json`. Emit typed candidate-created, gate-result, lineage and champion-frozen events.
Render a compact SVG-free HTML/CSS tree with selectable nodes; the detail card shows expression,
complexity, gates and partitioned metrics through safe text rendering. Add Chinese copy-paste CLI,
budget, DSL and test-freeze documentation to README and the module README.

- [ ] **Step 4: Run the full project checks**

Run: `pytest -q && ruff check . && mypy src`

Expected: all project checks pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/strategies/v4_quanta_alpha src/quant_trader/experiments/models.py src/quant_trader/dashboard.py src/quant_trader/dashboard_template.py src/quant_trader/cli.py README.md tests
git commit -m "feat: run and visualize QuantaAlpha experiments"
```

# Paper Strategies MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver independently runnable and strongly visual FinMem, QuantaAlpha and Alpha Arena MVPs without making the first release a single oversized change.

**Architecture:** Execute four dependency-ordered plans. Each phase ends in working software and a green full suite; later phases consume only stable public artifacts and interfaces from earlier phases.

**Tech Stack:** Python 3.12, Pydantic 2, Typer, pandas/numpy, standard-library local dashboard, pytest

---

### Task 1: Shared experiment kernel

**Files:**
- Follow: `docs/superpowers/plans/2026-07-18-experiment-kernel-implementation.md`

- [ ] Execute every checkbox in the shared-kernel plan.
- [ ] Verify `pytest -q && ruff check . && mypy src` passes.
- [ ] Confirm `quant-trader experiment run finmem ...` fails quickly with the intentional
  not-available message and never opens a provider.

### Task 2: FinMem

**Files:**
- Follow: `docs/superpowers/plans/2026-07-18-finmem-implementation.md`

- [ ] Execute every checkbox in the FinMem plan.
- [ ] Verify one mocked-provider FinMem run produces memory, decisions, events and a completed
  dashboard projection.
- [ ] Verify `pytest -q && ruff check . && mypy src` passes.

### Task 3: QuantaAlpha

**Files:**
- Follow: `docs/superpowers/plans/2026-07-18-quanta-alpha-implementation.md`

- [ ] Execute every checkbox in the QuantaAlpha plan.
- [ ] Verify the security test suite proves no generated Python can execute and the champion is
  frozen before the test partition is evaluated.
- [ ] Verify `pytest -q && ruff check . && mypy src` passes.

### Task 4: Alpha Arena and release verification

**Files:**
- Follow: `docs/superpowers/plans/2026-07-18-alpha-arena-implementation.md`

- [ ] Execute every checkbox in the Alpha Arena plan.
- [ ] Verify a zero-provider-call default run ranks the rules contestant and marks missing strategies
  absent without failing.
- [ ] Verify a replay from `events.jsonl` rebuilds the final dashboard state.
- [ ] Run `pytest -q && ruff check . && mypy src` one final time.
- [ ] Review the diff for credentials, generated run artifacts, raw model output and unrelated files;
  none may be committed.

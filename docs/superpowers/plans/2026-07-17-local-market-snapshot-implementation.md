# Local Market Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit a validated 2023–2025 Sina OHLCV snapshot for the nine-symbol V1 universe and make offline data the default quickstart.

**Architecture:** Add one `MarketDataSource` adapter for Sina daily bars plus forward-adjustment factors and select it by default in the existing sync command. Reuse `ParquetMarketCache` for all persisted files so checked-in data follows the same validation and integrity path as runtime data.

**Tech Stack:** Python 3.12, httpx, pandas, PyArrow/Parquet, Typer, pytest/respx.

---

## File map

- Create `src/quant_trader/data/sina_source.py`: Sina daily/factor requests and canonical response conversion.
- Create `tests/unit/data/test_sina_source.py`: request, adjustment, conversion, and failure tests.
- Modify `src/quant_trader/data/__init__.py`: export the adapter.
- Modify `src/quant_trader/cli.py`: default sync to Sina with explicit Yahoo fallback.
- Modify `tests/unit/test_cli.py`: source selection coverage.
- Modify `.gitignore`: allow only the committed snapshot and provenance beneath `data/`.
- Create `data/SOURCES.md` and `data/market/*`: provenance plus validated cache generations.
- Modify `README.md`: offline-first quickstart and optional refresh command.

### Task 1: Sina source adapter

**Files:**
- Create: `src/quant_trader/data/sina_source.py`
- Create: `tests/unit/data/test_sina_source.py`
- Modify: `src/quant_trader/data/__init__.py`

- [ ] **Step 1: Write failing adapter tests**

Cover the two symbol-specific URLs, half-open range filtering, factor-date forward filling,
`raw * qfq_factor + adjust`, canonical column order, and malformed/out-of-range data.

```python
payload = {
    "rc": 0,
    "data": {
        "code": "SPY",
        "klines": [
            "2023-01-03,384.370,380.820,386.430,377.831,74850731,28499102976.000"
        ],
    },
}
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/unit/data/test_sina_source.py`
Expected: collection fails because `SinaSource` does not exist.

- [ ] **Step 3: Implement the minimal adapter**

Call Sina's `US_MinKService.getDailyK` JSONP endpoint and the corresponding
`reinstatement/{symbol}_qfq.js` endpoint. Parse without evaluating JavaScript, apply the most recent
factor whose date is not after the market date, enforce `[start, end)`, and finish with
`validate_ohlcv`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest -q tests/unit/data/test_sina_source.py tests/unit/data/test_validation.py`
Expected: all pass.

Commit: `feat: add Sina market data source`

### Task 2: Source selection and offline defaults

**Files:**
- Modify: `src/quant_trader/cli.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write a failing CLI test**

Invoke `data sync` without `--source`, patch `SinaSource.fetch`, and assert it is used for all
configured tickers. Add a second assertion that `--source yahoo` selects `YFinanceSource`.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/unit/test_cli.py`
Expected: failure because `--source` and the Sina default do not exist.

- [ ] **Step 3: Add minimal source selection**

Add a `MarketSource` string enum with `sina` and `yahoo`, default the option to Sina, and
instantiate exactly one matching adapter. Preserve the existing concise `DataValidationError`
handling.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest -q tests/unit/test_cli.py`
Expected: all pass.

Commit: `feat: default data sync to Sina`

### Task 3: Generate, commit, and verify the snapshot

**Files:**
- Modify: `.gitignore`
- Create: `data/SOURCES.md`
- Create: `data/market/*.parquet`
- Create: `data/market/*.json`
- Modify: `README.md`

- [ ] **Step 1: Download through production boundaries**

Run:

```bash
uv run quant-trader data sync --source sina --config configs/default.yaml \
  --start 2023-01-01 --end 2026-01-01 --data-root data
```

Expected: nine `cached TICKER` lines and no partial invalid generation.

- [ ] **Step 2: Record provenance and allowlist snapshot files**

Change `/data/` ignore behavior so only `data/SOURCES.md` and `data/market/**` are tracked. Document
both endpoints, the forward-adjustment formula, retrieval timestamp, range, symbols, row counts, hashes,
and research-only limitation in `data/SOURCES.md`.

- [ ] **Step 3: Verify every committed cache entry**

Run a short Python check that loads `configs/default.yaml`, reads all symbols via
`ParquetMarketCache`, asserts first date `2023-01-03`, last date `2025-12-31`, validates each frame,
and prints row count plus manifest SHA-256.

- [ ] **Step 4: Verify an offline backtest and documentation**

Update README so the first command is:

```bash
quant-trader backtest --config configs/default.yaml --data-root data --output run.json
```

Run that command followed by
`quant-trader report --run-json run.json --output report.html`; both must exit zero without a data
download.

- [ ] **Step 5: Full verification and commit**

Run:

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
git diff --check
```

Expected: all commands exit zero.

Commit only `.gitignore`, `README.md`, `data/`, source/tests, and plan/spec files; do not add
`llm_quant_papers_summary.md`.

Commit: `data: add validated offline market snapshot`

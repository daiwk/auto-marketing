# Local Market Snapshot Design

## Goal

Make the V1 backtest runnable immediately without Yahoo Finance. Commit a validated, fixed
research snapshot for the configured nine-symbol universe and make future refreshes reproducible.

## Chosen approach

Use Sina Finance's public US daily-kline and reinstatement-factor endpoints as the snapshot source. Fetch forward-adjusted daily
OHLC and raw volume for `SPY`, `QQQ`, `IWM`, `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, and `META`
over the half-open interval `[2023-01-01, 2026-01-01)`. The endpoint has already been checked with
SPY and returned 752 rows through 2025-12-31.

The snapshot is for reproducible research, not a promise that the upstream endpoints will remain
available. Source URLs, adjustment mode, retrieval time, covered dates, row counts, and file
SHA-256 values will be recorded alongside the data.

## Architecture and data flow

1. `SinaSource` downloads raw daily klines and per-symbol forward-adjustment factors with bounded
   timeouts. All nine application tickers use their existing symbols.
2. It applies Sina's documented formula `adjusted = raw * qfq_factor + adjust`, converts the result
   to the canonical, timezone-naive `open/high/low/close/volume` frame, and calls `validate_ohlcv`.
3. `ParquetMarketCache` writes each validated frame and its atomic manifest under `data/market/`.
4. Those Parquet generations and manifests are committed. Backtest and paper commands continue to
   read through the existing cache API, so no strategy or execution code changes.
5. `quant-trader data sync` uses Sina by default; Yahoo remains an explicit fallback source.
   The README quickstart starts with the checked-in snapshot and does not require a sync.

## Repository contents

- Nine Parquet files plus their existing cache manifests under `data/market/`.
- `data/SOURCES.md` with provenance, field mapping, adjustment semantics, date coverage, and a
  research-only disclaimer.
- A small Sina source adapter and focused tests.
- `.gitignore` exceptions only for the committed snapshot and provenance file; other generated
  contents under `data/` stay ignored.

## Error handling

Reject missing symbols, malformed JSON, upstream error codes, empty responses, non-numeric values,
duplicate or out-of-range dates, and invalid OHLCV. A failed refresh never replaces an existing
valid cache generation. CLI errors remain concise and do not write partial invalid data.

## Verification

- Unit tests cover ticker mapping, request parameters, canonical conversion, and malformed data.
- Every committed symbol is loaded through `ParquetMarketCache.read` and revalidated.
- Verify exact symbol coverage, 2023-01-03 first bar, 2025-12-31 last bar, row counts, checksums,
  and cross-symbol aligned latest dates.
- Run the full test suite, Ruff, formatting, mypy, and one rules-only backtest using only committed
  files.

## Deliberate limits

No scheduler, real-time feed, database server, or automatic refresh in CI. The snapshot is frozen
through 2025-12-31 and is suitable only for research and paper simulation.

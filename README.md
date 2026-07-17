# quant-trader

A deliberately small, long-only US-equity research and **paper-trading** tool. It downloads
validated daily bars, runs weekly rules decisions at the close, fills target weights at the next
available open with costs, and persists one confirmed paper cycle to SQLite.

> **Disclaimer:** Research and paper simulation only. This is not investment advice. The package
> has no live broker integration and cannot place live orders. Simulated results do not guarantee
> future performance.

## Quickstart

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

quant-trader data sync --config configs/default.yaml --start 2023-01-01 --end 2026-01-01
quant-trader backtest --config configs/default.yaml --data-root data --output run.json
quant-trader report --run-json run.json --output report.html

quant-trader paper init --db paper.db
quant-trader paper status --db paper.db
quant-trader paper run --db paper.db --config configs/default.yaml --confirm
```

Rules-only is the offline/default mode. `backtest --use-llm` is optional and requires
`MINIMAX_API_KEY`; the LLM can only reduce or reject rules-selected targets. Hard limits remain
15% per position, 80% gross, long-only/no leverage, with drawdown reduction and a latched halt.

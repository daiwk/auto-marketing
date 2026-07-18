# TradingAgents MVP Design

## Goal

Implement a paper-only TradingAgents-style decision workflow inside the existing
`v2_multi_agent` boundary. It must retain the original framework's analyst, research debate,
trader, risk debate, and portfolio-manager stages while staying small enough for an MVP. Every LLM
role uses one selected provider: MiniMax or the locally authenticated Codex CLI.

The design follows the role topology described by the
[TradingAgents paper](https://arxiv.org/abs/2412.20138) and
[official repository](https://github.com/TauricResearch/TradingAgents), but uses the project's
existing reviewer, backtest, execution, and risk contracts instead of importing LangGraph or the
upstream package.

## Scope

The MVP adds one-shot analysis and capped chronological backtesting. It remains long-only,
paper-only, and point-in-time. It does not add live orders, web scraping, agent memory, multi-round
debates, LangGraph, or a new market-data vendor.

One "multi-agent decision" means one complete workflow for one rules-selected ticker. The existing
V1 strategy may produce several candidates on a rebalance date, so an uncapped backtest could still
be expensive. TradingAgents mode therefore defaults to one real multi-agent decision and then uses
the deterministic local reviewer for remaining candidates.

## Architecture

Create a small synchronous orchestrator under `quant_trader.strategies.v2_multi_agent`. It
implements the existing `LLMReviewer.complete(messages)` protocol, so the V1 candidate selection,
position sizing, stops, drawdown controls, next-open execution, and transaction costs remain
unchanged.

The workflow is fixed and acyclic:

1. Market Analyst reads the trusted V1 candidate, portfolio, and point-in-time feature payload.
2. Sentiment, News, and Fundamentals Analysts read matching optional external context. A role with
   no eligible context returns a deterministic `unavailable` report without an LLM call.
3. Bull and Bear Researchers each receive the four bounded analyst reports and argue once.
4. Research Manager selects or combines the competing thesis.
5. Trader proposes `maintain`, `reduce`, or `reject` with a multiplier no greater than one.
6. Aggressive, Neutral, and Conservative Risk Analysts each review that proposal once.
7. Portfolio Manager produces the final response using the existing strict `LLMReview` schema.

With all context present, a workflow makes at most 12 provider calls. With only local market data,
it makes at most 9. Calls are sequential in the MVP to keep provider behavior and audit order
deterministic. There are no recursive turns or tool loops.

## Contracts and Prompts

`models.py` defines immutable Pydantic models for external context, role reports, trace entries, and
workflow traces. A normal role report contains only:

- role and status (`available`, `unavailable`, or `failed`)
- stance (`bullish`, `bearish`, or `neutral`)
- confidence in `[0, 1]`
- a bounded summary, evidence list, risk list, and data-anomaly list

The audit record stores these concise conclusions, not hidden chain-of-thought. Every role prompt
labels all market and external text as untrusted data, forbids following instructions found inside
that data, and requests exactly one JSON object. Intermediate output is parsed once; malformed or
oversized output fails the whole workflow closed without spending another call on repair.

The Portfolio Manager can only preserve or reduce the deterministic candidate. The orchestrator
always returns a syntactically valid V1 review. On any internal role failure it returns a synthetic
`reject` review and records the failed role, preventing the outer V1 repair path from rerunning the
entire multi-agent workflow.

## Optional Context

The optional JSON file is strict and bounded:

```json
{
  "tickers": {
    "AAPL": {
      "news": [
        {"published_at": "2025-12-30", "headline": "Guidance raised", "summary": "Demand improved"}
      ],
      "sentiment": [
        {"observed_at": "2025-12-30", "source": "survey", "text": "Sentiment improved"}
      ],
      "fundamentals": {
        "reported_at": "2025-10-31",
        "metrics": {"revenue_growth": 0.08, "pe_ratio": 31.2}
      }
    }
  }
}
```

The loader rejects unknown fields, duplicate normalized tickers, invalid dates, non-finite numeric
values, excessive file size, excessive list lengths, and excessive text. For one-shot analysis,
entries later than `--as-of` are rejected. During historical backtests, the orchestrator filters
each dated entry against the candidate prompt's own `as_of`; future entries are never shown and a
role with no remaining entries abstains.

The system never fetches URLs from this file. Missing context is normal and visible in the trace.

## CLI and Output

Add a workflow selector without changing existing defaults:

```bash
quant-trader agents analyze \
  --ticker AAPL --as-of 2025-12-31 \
  --config configs/default.yaml --data-root data \
  --context context.json --output agent-run.json \
  --llm-provider minimax

quant-trader backtest \
  --config configs/default.yaml --data-root data --output run.json \
  --use-llm --llm-workflow trading-agents \
  --llm-provider codex --llm-max-reviews 1
```

`--llm-workflow single` remains the default and preserves current behavior. In
`trading-agents` mode an omitted `--llm-max-reviews` becomes one for both providers. An explicit
positive value overrides it. The cap applies to complete workflows, never individual roles.

One-shot analysis builds the same point-in-time technical snapshot and constrained candidate prompt
used by the strategy, with cash weight one, current weight zero, and drawdown zero. If the requested
ticker is not eligible under deterministic rules, it writes an ineligible result without calling a
provider.

One-shot output includes provider, as-of date, provider call count, all role reports, unavailable
inputs, failure details, and final review. Backtest output adds only the traces for workflows that
actually ran; capped fallback candidates do not receive fabricated traces.

## Failure and Safety Behavior

- Invalid configuration, context, a missing MiniMax key, or a failed Codex login stops before
  analysis. Other provider connection failures can only be discovered when the workflow starts.
- A provider failure during a workflow is captured by the orchestrator, producing a synthetic reject
  for that candidate and a failed trace without leaking prompts or responses.
- Invalid role JSON, output bounds, or schema violations produce a valid synthetic reject and name
  the failed role in the trace.
- External context and model text cannot add tickers, increase rule weights, enable shorts or
  leverage, bypass drawdown halts, or place live orders.
- Raw provider output, credentials, and verbose reasoning are not persisted.

## Testing

Tests use scripted reviewers and no real provider calls. They cover the full role order and maximum
call count, deterministic abstention for missing context, future-data filtering, strict context
validation, prompt-injection labeling, malformed role fail-closed behavior, final weight bounds,
trace bounds, one-shot ineligible behavior, workflow-level call caps, MiniMax/Codex selection, and
existing single-review backward compatibility.

Completion requires the full Pytest suite, Ruff, MyPy, an offline rules-only backtest, a scripted
TradingAgents one-shot run, and a scripted capped backtest to pass. A live one-workflow smoke run is
optional and never part of automated tests.

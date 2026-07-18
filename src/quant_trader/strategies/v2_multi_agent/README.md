# V2 TradingAgents MVP

This is a native, synchronous TradingAgents-style reviewer built on the validated V1 contracts. It
has no recursive tool loop and no live-broker path.

The fixed role order is market, sentiment, news, fundamentals, bull, bear, research manager, trader,
aggressive risk, neutral risk, conservative risk, and portfolio manager. Missing optional context
causes the corresponding analyst to abstain without an LLM call. Any invalid response or provider
failure rejects the candidate and records the failed role.

The final portfolio-manager response still passes through the V1 parser and may only maintain,
reduce, or reject the deterministic candidate. It cannot increase the trader multiplier, and neither
agent layer can override long-only, exposure, position, drawdown, or paper-only controls.

`context.py` accepts a strict JSON document with this shape:

```json
{
  "tickers": {
    "SPY": {
      "news": [
        {
          "published_at": "2025-12-30",
          "headline": "Example headline",
          "summary": "Concise point-in-time summary"
        }
      ],
      "sentiment": [
        {
          "observed_at": "2025-12-30",
          "source": "survey",
          "text": "Concise observation"
        }
      ],
      "fundamentals": {
        "reported_at": "2025-10-31",
        "metrics": {"revenue_growth": 0.08}
      }
    }
  }
}
```

The file is limited to 64 KiB, unknown fields are rejected, and future information is never exposed
to a historical role prompt. Output traces contain parsed reports and decisions, not raw provider
responses or hidden reasoning.

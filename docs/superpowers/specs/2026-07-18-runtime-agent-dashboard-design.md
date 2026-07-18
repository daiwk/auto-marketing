# Runtime TradingAgents Dashboard Design

## Goal

Add an opt-in local browser dashboard that explains the multi-agent decision process while
`agents analyze` or a TradingAgents backtest is running. The dashboard is observational only: it
must never change candidate selection, model responses, risk checks, or paper-trading decisions.

## User interface

Both supported commands gain an explicit `--dashboard` flag:

```bash
quant-trader agents analyze ... --dashboard
quant-trader backtest ... --use-llm --llm-workflow trading-agents --dashboard
```

The flag starts a temporary HTTP server bound to `127.0.0.1` on an available port and opens the
system browser. The CLI also prints the complete local URL as a fallback.

The page contains:

1. A header with ticker, analysis date, provider, workflow count, and overall status.
2. A twelve-node decision flow. Each node is visibly waiting, running, completed, skipped, or
   failed and shows stance/confidence when available.
3. A detail panel for the selected or currently active node, showing its sanitized summary,
   evidence, risks, and input anomalies.
4. Trader proposal and final portfolio decision cards once those stages complete.
5. A permanent safety panel explaining the deterministic candidate boundary, no-weight-increase
   rule, fail-closed behavior, and paper-only execution.

The page selects the active node automatically. A user may click a completed node to inspect it
without interrupting the workflow. For backtests, the header shows the current candidate workflow
and how many complete workflows have finished; the page retains the latest workflow's details.

## Architecture

Create a dependency-light `quant_trader.dashboard` package with three boundaries:

- `DashboardState` is a thread-safe in-memory projection of sanitized workflow events. It owns a
  monotonically increasing version and produces JSON-ready immutable snapshots.
- `DashboardServer` owns a standard-library `ThreadingHTTPServer`, a background server thread,
  browser opening, final-update delivery, and shutdown. It exposes only a fixed HTML page and a
  read-only JSON state endpoint.
- The fixed HTML template polls the state endpoint approximately twice per second and renders all
  dynamic values using DOM `textContent`, never `innerHTML`.

The server uses a random capability token in the URL and endpoint path in addition to loopback-only
binding. It has no write endpoint and serves no repository files.

## Event model and data flow

Extend `TradingAgentsReviewer` with an optional structured observer callback. The reviewer emits
immutable, bounded events for:

- workflow started;
- each role started, completed, skipped, or failed;
- trader proposal completed;
- portfolio decision completed;
- workflow completed.

Completed events carry only existing validated models (`RoleReport`, `TraderProposal`, and
`LLMReview`) or their JSON projections. They never carry prompts, raw provider output, exception
messages, credentials, or hidden reasoning. Missing sentiment, news, or fundamentals emits a
`skipped` event without consuming a provider call.

The CLI creates the dashboard before constructing the external LLM provider, passes the state
observer into each `TradingAgentsReviewer`, and publishes preparation and command-completion states.
Ineligible one-shot analyses finish with zero provider calls and a visible explanation. Rules-only
backtests and single-review LLM workflows reject `--dashboard` with a concise usage error because
they do not produce the required multi-agent events.

## Lifecycle and failure handling

Dashboard startup is explicit. If the local server cannot bind or initialize, the command exits
before any provider call with a concise error. Failure to open a browser does not stop the command;
the printed URL remains usable.

After startup, dashboard observation is best-effort. Observer exceptions are contained and cannot
propagate into the trading workflow or turn a valid result into a failure. Browser disconnection and
poll errors have no effect on the command.

At command completion, the CLI publishes the final status and gives a connected browser a bounded
opportunity to fetch that version before shutting down the server thread. The already-loaded page
keeps the rendered final state after shutdown, although refreshing it is not supported. Keyboard
interrupts and ordinary errors publish a stopped/failed status before cleanup when possible.

## Security and privacy

- Bind only to `127.0.0.1` and choose an ephemeral port.
- Require an unguessable per-run URL token.
- Serve fixed resources only; never accept paths or file names from requests.
- Apply strict JSON bounds already enforced by the multi-agent models.
- Use safe DOM text rendering and a restrictive Content Security Policy.
- Do not persist event payloads or raw LLM material. Existing JSON run output remains the durable
  audit artifact.

## Testing

Unit tests cover event ordering, skipped and failed roles, bounded sanitized payloads, snapshot
versioning, and observer failures that leave decisions unchanged. HTTP tests verify token handling,
fixed routes, state JSON, security headers, and safe template behavior.

CLI tests cover both supported `--dashboard` commands, invalid workflow combinations, startup
failure before provider construction, ineligible one-shot completion, browser-open failure, and
clean shutdown. Existing multi-agent, CLI, type, lint, and offline backtest suites remain green.

## Non-goals

- No remote hosting, authentication accounts, or network exposure.
- No WebSocket, SSE, FastAPI, React, or separate dashboard process.
- No live broker integration or order controls in the page.
- No storage of dashboard sessions and no multi-run historical explorer.
- No visualization for the V1 single-review workflow in this MVP.

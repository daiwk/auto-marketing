# Codex Reviewer Design

## Goal

Add an optional backtest reviewer that reuses the user's local Codex login and quota through
`codex exec`. Keep MiniMax working unchanged and keep the first version small, observable, and
paper-only.

## User Interface

The existing `--use-llm` switch remains the opt-in safety gate. A new option selects the provider:

```bash
quant-trader backtest --config configs/default.yaml --data-root data --output run.json \
  --use-llm --llm-provider codex --llm-max-reviews 3
```

`--llm-provider` accepts `minimax` or `codex` and defaults to `minimax` for backward
compatibility. Codex requires no API key. When Codex is selected and the user omits
`--llm-max-reviews`, the CLI defaults the cap to three real reviews; an explicit value overrides
the cap.

## Components and Data Flow

Add `CodexReviewer` under `quant_trader.llm`. It implements the existing `LLMReviewer.complete`
contract and has one responsibility: convert canonical chat messages into a bounded prompt, invoke
one non-interactive local Codex process, and return its final text response.

The reviewer launches `codex exec` with an argument list and `shell=False`, sends the prompt on
standard input, disables repository mutation with Codex's read-only sandbox, and captures only the
final response through `--output-last-message`. It uses `--ephemeral` so Codex does not persist a
session, applies a configurable timeout, and bounds the response read. The short-lived response file
lives in an OS temporary directory and is removed immediately; no market data, prompt, credentials,
or model response is written into the project.

The existing progress/cap wrapper remains provider-independent. The CLI constructs either a
`MiniMaxReviewer` or `CodexReviewer`, prints provider-specific progress, and falls back to the
deterministic local reviewer after the cap. The strategy, JSON parsing, portfolio rules, and trading
simulation remain unchanged.

The output note records the selected provider, number of real reviews, and whether the run was
truncated. This prevents a three-review smoke test from being mistaken for a full LLM backtest.

## Failure Behavior

Before starting the LLM simulation, Codex mode verifies that the configured executable can start.
Missing or broken CLI installation, missing login, timeout, non-zero exit, empty output, or excessive
output produces a short actionable error and a non-zero command exit. Errors must not silently turn
an intended Codex review into a trading decision. The existing post-cap local fallback is the only
fallback and is clearly reported.

The subprocess environment inherits the user's Codex authentication but the program never reads,
logs, or stores those credentials. Error messages exclude prompt and response contents.

## Testing

Unit tests replace the subprocess runner and cover prompt construction, safe arguments, successful
output, timeout, missing/broken executable, non-zero exit, empty output, and output bounds. CLI tests
cover provider selection, MiniMax API-key compatibility, Codex's default three-review cap, explicit
cap overrides, progress text, and output notes. The full existing suite, Ruff, MyPy, and a local
rules-only backtest must continue to pass.

No live Codex request is part of automated tests. A manual one-review smoke command is optional once
the user's local Codex CLI installation is functional.

## Out of Scope

- Calling the Codex desktop UI directly
- Installing, repairing, or logging in to Codex CLI automatically
- OpenAI API integration or separate API billing
- Parallel or batched chronological reviews
- Live brokerage execution

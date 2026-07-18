# Runtime TradingAgents Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in local browser dashboard that displays sanitized TradingAgents decisions in real time for one-shot analysis and multi-agent backtests.

**Architecture:** The existing reviewer emits bounded immutable events to a best-effort observer. A thread-safe in-memory projection feeds a loopback-only standard-library HTTP server; a fixed JavaScript page polls that state and renders the twelve-role flow. CLI commands own server startup and shutdown, so the dashboard remains an observational side channel.

**Tech Stack:** Python 3.12, Pydantic, `http.server`, threads/conditions, vanilla HTML/CSS/JavaScript, Typer, pytest.

---

## File structure

- Create `src/quant_trader/strategies/v2_multi_agent/events.py`: strict event kinds and payload contract.
- Modify `src/quant_trader/strategies/v2_multi_agent/orchestrator.py`: emit sanitized events without allowing observer failures to alter decisions.
- Modify `src/quant_trader/strategies/v2_multi_agent/__init__.py`: export the event contract.
- Create `src/quant_trader/dashboard.py`: state projection, loopback HTTP server, lifecycle, and fixed routes.
- Create `src/quant_trader/dashboard_template.py`: fixed dependency-free browser UI.
- Modify `src/quant_trader/cli.py`: add and validate `--dashboard`, connect events, and clean up.
- Create `tests/unit/strategies/test_v2_events.py`: event ordering, skipped roles, sanitized payload, and observer isolation.
- Create `tests/unit/test_dashboard.py`: state and HTTP/security behavior.
- Modify `tests/unit/test_cli.py`: both supported commands, invalid combinations, and lifecycle.
- Modify `README.md`: commands, lifecycle, privacy, and limitations.

### Task 1: Emit bounded workflow events

**Files:**
- Create: `src/quant_trader/strategies/v2_multi_agent/events.py`
- Modify: `src/quant_trader/strategies/v2_multi_agent/orchestrator.py`
- Modify: `src/quant_trader/strategies/v2_multi_agent/__init__.py`
- Create: `tests/unit/strategies/test_v2_events.py`

- [ ] **Step 1: Write failing event-sequence tests**

Build the existing market-only scripted workflow and collect events:

```python
def test_market_only_workflow_emits_sanitized_ordered_events() -> None:
    events: list[AgentEvent] = []
    reviewer = TradingAgentsReviewer(
        ScriptedReviewer(market_only_outputs()),
        provider_name="MiniMax",
        on_event=events.append,
    )

    reviewer.complete(review_messages())

    assert events[0].kind is AgentEventKind.WORKFLOW_STARTED
    assert [event.kind for event in events if event.role is RoleName.SENTIMENT] == [
        AgentEventKind.ROLE_SKIPPED
    ]
    assert events[-1].kind is AgentEventKind.WORKFLOW_COMPLETED
    assert events[-1].final_review is not None
    assert "raw" not in "".join(event.model_dump_json() for event in events).lower()
```

Add a second test whose observer raises on every event and assert the returned review and trace are
identical to a run without the observer. Add a failure test asserting `ROLE_FAILED` precedes
`WORKFLOW_COMPLETED` and carries only the sanitized failed `RoleReport`.

- [ ] **Step 2: Run the focused tests and confirm the missing contract**

Run:

```bash
.venv/bin/pytest -q tests/unit/strategies/test_v2_events.py
```

Expected: collection fails because `AgentEvent` and `AgentEventKind` do not exist.

- [ ] **Step 3: Add the strict event contract**

Implement `events.py` with these exact public fields:

```python
from datetime import date
from enum import StrEnum

from pydantic import Field, model_validator

from quant_trader.core.models import LLMReview
from quant_trader.strategies.v2_multi_agent.models import (
    RoleName,
    RoleReport,
    StrictFrozenModel,
    TraderProposal,
)
from quant_trader.validation import USEquityTicker


class AgentEventKind(StrEnum):
    WORKFLOW_STARTED = "workflow_started"
    ROLE_STARTED = "role_started"
    ROLE_COMPLETED = "role_completed"
    ROLE_SKIPPED = "role_skipped"
    ROLE_FAILED = "role_failed"
    TRADER_COMPLETED = "trader_completed"
    FINAL_COMPLETED = "final_completed"
    WORKFLOW_COMPLETED = "workflow_completed"


class AgentEvent(StrictFrozenModel):
    kind: AgentEventKind
    ticker: USEquityTicker
    as_of: date
    provider: str = Field(min_length=1, max_length=100)
    role: RoleName | None = None
    report: RoleReport | None = None
    proposal: TraderProposal | None = None
    final_review: LLMReview | None = None

    @model_validator(mode="after")
    def require_matching_payload(self) -> "AgentEvent":
        role_kinds = {
            AgentEventKind.ROLE_STARTED,
            AgentEventKind.ROLE_COMPLETED,
            AgentEventKind.ROLE_SKIPPED,
            AgentEventKind.ROLE_FAILED,
        }
        if (self.kind in role_kinds) != (self.role is not None):
            raise ValueError("role events require exactly one role")
        if self.report is not None and self.report.role is not self.role:
            raise ValueError("event report role must match event role")
        return self
```

The implementation may strengthen payload-to-kind validation, but must keep these fields and remain
backward-compatible with callers that do not provide an observer.

- [ ] **Step 4: Emit events at the orchestration boundary**

Add `on_event: Callable[[AgentEvent], None] | None = None` to `TradingAgentsReviewer`. Add a private
emitter that catches and clears observer exceptions:

```python
def _emit(self, event: AgentEvent) -> None:
    if self._on_event is None:
        return
    try:
        self._on_event(event)
    except Exception as error:
        _clear_error(error)
```

After parsing the trusted V1 request, emit `WORKFLOW_STARTED`. Emit role start/completion around real
provider calls, `ROLE_SKIPPED` for unavailable optional roles, `TRADER_COMPLETED` with the validated
proposal, `FINAL_COMPLETED` with the consistent final review, and exactly one
`WORKFLOW_COMPLETED`. On failure, emit `ROLE_FAILED` with `_failed(current_role)` before the final
workflow event. Never add prompts, raw output, exception text, or credentials to an event.

- [ ] **Step 5: Run event and orchestrator tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/strategies/test_v2_events.py tests/unit/strategies/test_v2_orchestrator.py
.venv/bin/ruff check src/quant_trader/strategies/v2_multi_agent tests/unit/strategies
.venv/bin/mypy src/quant_trader/strategies/v2_multi_agent
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit the event boundary**

```bash
git add src/quant_trader/strategies/v2_multi_agent tests/unit/strategies
git commit -m "feat: emit sanitized trading agent events"
```

### Task 2: Build the local dashboard server and page

**Files:**
- Create: `src/quant_trader/dashboard.py`
- Create: `src/quant_trader/dashboard_template.py`
- Create: `tests/unit/test_dashboard.py`

- [ ] **Step 1: Write failing state and server tests**

Test a state projection with a start event, completed market report, skipped sentiment report, and
final review. Assert monotonically increasing versions and JSON-ready snapshots. Then start the
server with `browser_open=lambda _: False` and verify:

```python
def test_dashboard_server_exposes_only_tokenized_fixed_routes() -> None:
    state = DashboardState()
    server = DashboardServer(state, browser_open=lambda _: False)
    url = server.start()
    try:
        page = httpx.get(url)
        snapshot = httpx.get(f"{url}state")
        forbidden = httpx.get(url.replace(server.token, "wrong-token"))
        traversal = httpx.get(f"{url}../../README.md")
    finally:
        server.stop()

    assert page.status_code == 200
    assert snapshot.status_code == 200
    assert forbidden.status_code == 404
    assert traversal.status_code == 404
    assert page.headers["Content-Security-Policy"].startswith("default-src 'none'")
```

Also assert the template contains no dynamic `innerHTML`, remote script, or remote stylesheet. Test
that `wait_until_seen(version, timeout_seconds=0.1)` becomes true after the state endpoint returns
that version.

- [ ] **Step 2: Run dashboard tests and confirm the module is missing**

```bash
.venv/bin/pytest -q tests/unit/test_dashboard.py
```

Expected: collection fails because `quant_trader.dashboard` does not exist.

- [ ] **Step 3: Implement the thread-safe state projection**

Use a lock and condition, not global mutable state:

```python
class DashboardState:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0
        self._seen_version = 0
        self._snapshot: dict[str, object] = {
            "version": 0,
            "command_status": "preparing",
            "workflow_count": 0,
            "workflow": None,
        }

    def publish(self, event: AgentEvent) -> None:
        with self._condition:
            self._project(event)
            self._version += 1
            self._snapshot["version"] = self._version
            self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return copy.deepcopy(self._snapshot)
```

`_project` resets twelve waiting role nodes on `WORKFLOW_STARTED`, stores sanitized reports by role,
stores proposal/final decision cards, increments `workflow_count` on `WORKFLOW_COMPLETED`, and never
stores an `AgentEvent` object or raw model response. Add `set_command_status`, `mark_seen`, and
`wait_until_seen` methods under the same condition.

- [ ] **Step 4: Implement the fixed polling page**

Put one `DASHBOARD_HTML` constant in `dashboard_template.py`. It must render the approved layout:
header, twelve ordered role buttons, node detail panel, proposal/final cards, and safety panel. Poll
`state` every 500 milliseconds. Build dynamic elements with `document.createElement` and assign
untrusted values only through `textContent`. Use relative `fetch('state', {cache: 'no-store'})`; do
not embed the token or any run data into JavaScript.

- [ ] **Step 5: Implement bounded loopback lifecycle**

`DashboardServer.start()` must bind `ThreadingHTTPServer(("127.0.0.1", 0), handler)`, generate a
`secrets.token_urlsafe(32)` path, start one daemon thread, call the injected browser opener, and
return `http://127.0.0.1:<port>/<token>/`. The handler serves only the exact page and state paths,
returns JSON with `ensure_ascii=False`, records the observed version, suppresses request logs, and
adds:

```text
Content-Security-Policy: default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'
X-Content-Type-Options: nosniff
Cache-Control: no-store
```

`stop()` calls `shutdown`, `server_close`, and joins with a bounded timeout. Convert bind/start
`OSError` into a sanitized `DashboardError("local dashboard could not start")`.

- [ ] **Step 6: Run focused dashboard checks**

```bash
.venv/bin/pytest -q tests/unit/test_dashboard.py
.venv/bin/ruff check src/quant_trader/dashboard.py src/quant_trader/dashboard_template.py tests/unit/test_dashboard.py
.venv/bin/mypy src/quant_trader/dashboard.py src/quant_trader/dashboard_template.py
```

Expected: all commands exit zero.

- [ ] **Step 7: Commit the dashboard core**

```bash
git add src/quant_trader/dashboard.py src/quant_trader/dashboard_template.py tests/unit/test_dashboard.py
git commit -m "feat: serve local agent dashboard"
```

### Task 3: Integrate both CLI entry points

**Files:**
- Modify: `src/quant_trader/cli.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Patch `DashboardServer` with a fake that records `start`, published events, final status, wait, and
stop calls. Add these cases:

```python
class FakeDashboardRun:
    instances: list["FakeDashboardRun"] = []

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.started = False
        self.stopped = False
        self.statuses: list[str] = []
        self.state = SimpleNamespace(publish=lambda event: None)
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def finish(self, status: str) -> None:
        self.statuses.append(status)

    def close(self) -> None:
        self.stopped = True


def test_agents_analyze_dashboard_runs_full_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCodexReviewer:
        def check_available(self) -> None:
            return None

        def complete(self, messages: tuple[ChatMessage, ...]) -> str:
            return MaintainReviewer().complete(messages)

    monkeypatch.setattr("quant_trader.cli._DashboardRun", FakeDashboardRun)
    monkeypatch.setattr("quant_trader.cli.CodexReviewer", FakeCodexReviewer)
    output = tmp_path / "analysis.json"
    result = CliRunner().invoke(app, [
        "agents", "analyze", "--ticker", "SPY", "--as-of", "2025-12-31",
        "--config", "configs/default.yaml", "--data-root", "data",
        "--output", str(output), "--llm-provider", "codex", "--dashboard",
    ])
    fake_dashboard = FakeDashboardRun.instances[-1]
    assert result.exit_code == 0
    assert fake_dashboard.started and fake_dashboard.stopped
    assert fake_dashboard.statuses[-1] == "completed"


def test_dashboard_requires_trading_agents_backtest(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    result = CliRunner().invoke(app, [
        "backtest", "--config", "configs/default.yaml", "--data-root", "data",
        "--output", str(output), "--dashboard",
    ])
    assert result.exit_code != 0
    assert "--dashboard requires --use-llm and --llm-workflow trading-agents" in result.output
```

Add tests for a TradingAgents backtest lifecycle, ineligible one-shot zero-call completion, server
startup failure before provider construction, and browser-open false continuing successfully.

- [ ] **Step 2: Run the CLI tests and verify `--dashboard` is unknown**

```bash
.venv/bin/pytest -q tests/unit/test_cli.py -k dashboard
```

Expected: tests fail because neither command accepts `--dashboard`.

- [ ] **Step 3: Add a small CLI lifecycle helper**

Keep server cleanup out of the command bodies:

```python
class _DashboardRun:
    def __init__(self, enabled: bool) -> None:
        self.state = DashboardState()
        self.server = DashboardServer(self.state) if enabled else None

    def start(self) -> None:
        if self.server is not None:
            typer.echo(f"Dashboard: {self.server.start()}", err=True)

    def finish(self, status: str) -> None:
        if self.server is None:
            return
        version = self.state.set_command_status(status)
        self.state.wait_until_seen(version, timeout_seconds=1.0)

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
```

If construction or `start()` raises `DashboardError`, print `Error: local dashboard could not
start` and exit before `_open_provider`.

- [ ] **Step 4: Connect `agents analyze --dashboard`**

Start `_DashboardRun` after loading/validating local config and context but before provider
construction. Pass `dashboard.state.publish` as `on_event` when enabled. Publish a completed state
for ineligible analysis with the existing reason and zero provider calls. On success, error, or
interrupt, call `finish` with `completed`, `failed`, or `stopped`, then always `close` and close the
MiniMax client.

- [ ] **Step 5: Connect `backtest --dashboard`**

Validate immediately after settings load:

```python
if dashboard and (
    not use_llm or llm_workflow is not LLMWorkflow.TRADING_AGENTS
):
    raise typer.BadParameter(
        "--dashboard requires --use-llm and --llm-workflow trading-agents"
    )
```

Start the dashboard before `_open_provider`, pass the event observer into every
`TradingAgentsReviewer`, publish final command status around the existing output write, and preserve
the default one-complete-workflow cap. Dashboard state must not be added to durable `run.json`.

- [ ] **Step 6: Document the exact commands and limitations**

Add to `README.md`:

```bash
quant-trader agents analyze --ticker SPY --as-of 2025-12-31 \
  --config configs/default.yaml --data-root data --output agent-run.json \
  --llm-provider minimax --dashboard

quant-trader backtest --config configs/default.yaml --data-root data --output run.json \
  --use-llm --llm-workflow trading-agents --dashboard
```

State that the page is loopback-only, temporary, observational, sanitized, and paper-only; refreshing
after the command exits is unsupported.

- [ ] **Step 7: Run CLI and full verification**

```bash
.venv/bin/pytest -q tests/unit/test_cli.py tests/unit/test_dashboard.py tests/unit/strategies/test_v2_events.py
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/quant-trader backtest --config configs/default.yaml --data-root data --output /tmp/quant-dashboard-verification.json
git diff --check
```

Expected: all tests pass, static checks report zero errors, the offline backtest writes the output,
and the diff check is empty.

- [ ] **Step 8: Commit and update PR #6**

```bash
git add README.md src/quant_trader/cli.py tests/unit/test_cli.py
git commit -m "feat: visualize live agent decisions"
git push
```

Confirm PR #6 points at the new head and remains open.

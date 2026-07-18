# Codex Reviewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an opt-in backtest use the user's local Codex login through `codex exec`, with a safe three-review default cap.

**Architecture:** Add a focused `CodexReviewer` adapter behind the existing `LLMReviewer` protocol. The CLI selects MiniMax or Codex, while the existing progress/fallback wrapper stays responsible for call caps and reporting.

**Tech Stack:** Python 3.12, `subprocess`, Typer, Pydantic message validation, Pytest, Ruff, MyPy

---

## File Map

- Create `src/quant_trader/llm/codex.py`: local Codex CLI adapter and sanitized provider errors.
- Create `tests/unit/llm/test_codex.py`: subprocess contract, parsing, and failure tests.
- Modify `src/quant_trader/llm/__init__.py`: export the new adapter.
- Modify `src/quant_trader/cli.py`: provider selection, Codex default cap, generic progress, and notes.
- Modify `tests/unit/test_cli.py`: provider wiring and cap/report behavior.
- Modify `README.md`: local CLI prerequisite and smoke command.

### Task 1: Implement the Codex CLI Adapter

**Files:**
- Create: `tests/unit/llm/test_codex.py`
- Create: `src/quant_trader/llm/codex.py`
- Modify: `src/quant_trader/llm/__init__.py`

- [ ] **Step 1: Write failing happy-path and command-safety tests**

Create tests that monkeypatch `quant_trader.llm.codex.subprocess.run`, call `check_available()` and
`complete()`, and assert the exact non-shell invocation:

```python
from pathlib import Path
from subprocess import CompletedProcess

from quant_trader.llm.base import ChatMessage
from quant_trader.llm.codex import CodexReviewer


def test_codex_uses_read_only_noninteractive_exec(monkeypatch) -> None:
    calls: list[tuple[list[str], object]] = []

    def run(args: list[str], **kwargs: object) -> CompletedProcess[bytes]:
        calls.append((args, kwargs.get("input")))
        if args == ["codex", "login", "status"]:
            return CompletedProcess(args, 0, stdout=b"", stderr=b"")
        output_path = Path(args[args.index("--output-last-message") + 1])
        output_path.write_text(
            '{"action":"maintain","overrides":{},"confidence":0.8}', encoding="utf-8"
        )
        return CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("quant_trader.llm.codex.subprocess.run", run)
    reviewer = CodexReviewer(timeout_seconds=12)
    reviewer.check_available()
    result = reviewer.complete((ChatMessage(role="user", content="review this"),))

    assert calls[0][0] == ["codex", "login", "status"]
    assert calls[1][0][:7] == [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--output-last-message"
    ]
    assert calls[1][0][-1] == "-"
    assert b"[USER]\nreview this" in (calls[1][1] or b"")
    assert result.startswith('{"action":"maintain"')
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `uv run --extra dev pytest tests/unit/llm/test_codex.py -q`

Expected: collection fails because `quant_trader.llm.codex` does not exist.

- [ ] **Step 3: Add failure and bounds tests**

Add parameterized tests for `FileNotFoundError`, `TimeoutExpired`, non-zero exit, a missing or empty
final-message file, and a response file larger than 64 KiB. Each must raise `CodexError`;
`str(error)` must be concise and must not contain the original prompt or captured response. Add a
broken-wrapper case returning exit code 1 and assert the message tells the user to repair the local
Codex CLI.

- [ ] **Step 4: Implement the minimal adapter**

Implement `src/quant_trader/llm/codex.py` with these concrete contracts and process settings:

```python
class CodexError(RuntimeError):
    """Sanitized local Codex process failure."""


class CodexReviewer:
    def __init__(self, executable: str = "codex", timeout_seconds: float = 120,
                 max_output_bytes: int = 65_536) -> None:
        if not executable.strip() or "\x00" in executable:
            raise ValueError("executable must be nonblank and contain no null byte")
        if timeout_seconds <= 0 or max_output_bytes < 1:
            raise ValueError("timeout and output bound must be positive")
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes

    def check_available(self) -> None:
        self._invoke([self._executable, "login", "status"], input_bytes=None, timeout=5)

    def complete(self, messages: Sequence[MessageInput]) -> str:
        canonical = canonical_messages(messages)
        prompt = "\n\n".join(
            f"[{message.role.upper()}]\n{message.content}" for message in canonical
        ).encode("utf-8")
        with TemporaryDirectory(prefix="quant-codex-") as directory:
            output_path = Path(directory) / "last-message.txt"
            self._invoke(
                [self._executable, "exec", "--ephemeral", "--sandbox", "read-only",
                 "--skip-git-repo-check", "--output-last-message", str(output_path), "-"],
                input_bytes=prompt,
                timeout=self._timeout_seconds,
            )
            try:
                with output_path.open("rb") as output_file:
                    raw = output_file.read(self._max_output_bytes + 1)
            except OSError:
                raise CodexError("Codex did not produce a final response") from None
        if len(raw) > self._max_output_bytes:
            raise CodexError("Codex response exceeded the allowed size")
        try:
            result = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise CodexError("Codex response was not valid UTF-8") from None
        if not result.strip():
            raise CodexError("Codex returned an empty response")
        return result

    def _invoke(self, args: list[str], *, input_bytes: bytes | None, timeout: float) -> None:
        try:
            completed = subprocess.run(
                args, input=input_bytes, stdin=DEVNULL if input_bytes is None else None,
                stdout=DEVNULL, stderr=DEVNULL, check=False, shell=False, timeout=timeout,
            )
        except FileNotFoundError:
            raise CodexError(
                "Codex CLI is unavailable; repair or reinstall it and run codex login"
            ) from None
        except subprocess.TimeoutExpired:
            raise CodexError("Codex review timed out") from None
        if completed.returncode != 0:
            raise CodexError(
                "Codex CLI exited unsuccessfully; repair the CLI and run codex login"
            )
```

Import `DEVNULL`, `Path`, `Sequence`, `TemporaryDirectory`, `subprocess`, `MessageInput`, and
`canonical_messages`. Do not include stdin, stdout, or stderr in exceptions.

Export `CodexError` and `CodexReviewer` from `src/quant_trader/llm/__init__.py`.

- [ ] **Step 5: Run adapter tests and static checks**

Run:

```bash
uv run --extra dev pytest tests/unit/llm/test_codex.py -q
uv run --extra dev ruff check src/quant_trader/llm tests/unit/llm/test_codex.py
uv run --extra dev mypy src/quant_trader/llm
```

Expected: all commands pass.

- [ ] **Step 6: Commit the adapter**

```bash
git add src/quant_trader/llm/codex.py src/quant_trader/llm/__init__.py tests/unit/llm/test_codex.py
git commit -m "feat: add local codex reviewer"
```

### Task 2: Add Provider Selection to Backtest CLI

**Files:**
- Modify: `src/quant_trader/cli.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing CLI selection tests**

Add a `LLMProvider(StrEnum)` expectation for `minimax` and `codex`. Patch `_frames`,
`run_backtest`, `buy_and_hold`, `MiniMaxReviewer`, and `CodexReviewer` with small fakes so the tests
do not read data or call a provider. Cover these assertions:

```python
assert "--llm-provider" in CliRunner().invoke(app, ["backtest", "--help"]).output

# --use-llm --llm-provider codex constructs CodexReviewer, calls check_available(),
# does not require MINIMAX_API_KEY, and gives _ProgressReviewer max_reviews=3.

# --llm-provider codex --llm-max-reviews 1 overrides the default cap.

# --llm-provider minimax retains the MINIMAX_API_KEY requirement and an omitted cap stays None.
```

Also update the existing `_ProgressReviewer` unit test to pass `provider_name="codex"` and assert
its messages say `Codex review 1 started` and `Codex review 1 completed`.

- [ ] **Step 2: Run CLI tests and verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_cli.py -q`

Expected: failures because `LLMProvider`, `CodexReviewer`, and provider-aware progress do not exist.

- [ ] **Step 3: Implement provider selection and reporting**

In `src/quant_trader/cli.py`:

```python
class LLMProvider(StrEnum):
    MINIMAX = "minimax"
    CODEX = "codex"
```

Add `llm_provider: LLMProvider = LLMProvider.MINIMAX` to `backtest`. Preserve `--use-llm` as the
required opt-in. For Codex, create `CodexReviewer()`, call `check_available()`, and use an effective
cap of three when the option is omitted. For MiniMax, preserve current API-key validation,
configuration, and unlimited omitted cap.

Change `_ProgressReviewer` to accept and expose `provider_name`. Use it in progress messages and in
the truncated output note. Catch `CodexError` in the CLI, print only its sanitized message, and exit
with code 1 without a traceback. Do not catch strategy or parsing defects.

- [ ] **Step 4: Run CLI tests and static checks**

Run:

```bash
uv run --extra dev pytest tests/unit/test_cli.py -q
uv run --extra dev ruff check src/quant_trader/cli.py tests/unit/test_cli.py
uv run --extra dev mypy src/quant_trader/cli.py
```

Expected: all commands pass.

- [ ] **Step 5: Commit CLI integration**

```bash
git add src/quant_trader/cli.py tests/unit/test_cli.py
git commit -m "feat: select codex for llm reviews"
```

### Task 3: Document and Verify the First Version

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update usage documentation**

Document the prerequisite `codex --version` and `codex login`, note that Codex mode uses the local
Codex account rather than `MINIMAX_API_KEY`, and add the bounded smoke command:

```bash
quant-trader backtest --config configs/default.yaml --data-root data --output run.json \
  --use-llm --llm-provider codex --llm-max-reviews 1
```

State that omitting `--llm-max-reviews` in Codex mode defaults to three reviews, and that the local
rules reviewer handles remaining review points with a clearly marked truncated note.

- [ ] **Step 2: Run the complete verification suite**

Run:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests
uv run --extra dev mypy src
uv run --extra dev quant-trader backtest --config configs/default.yaml --data-root data \
  --output /tmp/quant-codex-rules.json
```

Expected: all tests, lint, and types pass; the offline command writes a rules-only result containing
`rules_only` and `spy_buy_hold` without making any LLM request.

- [ ] **Step 3: Verify the local Codex failure is actionable**

Run: `uv run --extra dev quant-trader backtest --config configs/default.yaml --data-root data --output /tmp/quant-codex.json --use-llm --llm-provider codex --llm-max-reviews 1`

Expected on the currently broken installation: exit code 1 with the repair/login hint and no Python
traceback. If the CLI has been repaired by then, stop after one real review and verify the output
note identifies a truncated Codex run.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain codex-backed reviews"
```

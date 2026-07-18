import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_trader.backtest import MaintainReviewer
from quant_trader.cli import _CountingReviewer, _ProgressReviewer, _RejectReviewer, app
from quant_trader.dashboard import DashboardError
from quant_trader.data.validation import DataValidationError
from quant_trader.llm.base import ChatMessage
from quant_trader.llm.codex import CodexError


class FakeDashboardState:
    instances: list["FakeDashboardState"] = []

    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.preparations: list[tuple[str, str, str]] = []
        self.instances.append(self)

    def publish(self, event: object) -> None:
        return None

    def set_command_status(self, status: str, *, reason: str | None = None) -> int:
        self.statuses.append(status)
        return len(self.statuses)

    def prepare(self, ticker: str, as_of: str, provider: str) -> int:
        self.preparations.append((ticker, as_of, provider))
        return len(self.preparations)

    def wait_until_seen(self, version: int, *, timeout_seconds: float) -> bool:
        return True


class FakeDashboardServer:
    instances: list["FakeDashboardServer"] = []

    def __init__(self, state: FakeDashboardState) -> None:
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> str:
        self.started = True
        return "http://127.0.0.1:12345/token/"

    def stop(self) -> None:
        self.stopped = True


def _fake_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeDashboardState.instances.clear()
    FakeDashboardServer.instances.clear()
    monkeypatch.setattr("quant_trader.cli.DashboardState", FakeDashboardState)
    monkeypatch.setattr("quant_trader.cli.DashboardServer", FakeDashboardServer)


def test_cli_help_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0


@pytest.mark.parametrize("kind", ["finmem", "quanta-alpha", "alpha-arena"])
def test_experiment_run_commands_exist(kind: str) -> None:
    result = CliRunner().invoke(app, ["experiment", "run", kind, "--help"])

    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--data-root" in result.output
    assert "--output-dir" in result.output


def test_backtest_help_lists_trading_agents_workflow() -> None:
    result = CliRunner().invoke(app, ["backtest", "--help"], env={"COLUMNS": "160"})

    assert result.exit_code == 0
    assert "trading-agents" in result.output


def test_agents_analyze_ineligible_ticker_does_not_require_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    output = tmp_path / "analysis.json"

    result = CliRunner().invoke(
        app,
        [
            "agents",
            "analyze",
            "--ticker",
            "AAPL",
            "--as-of",
            "2023-01-03",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["eligible"] is False
    assert payload["provider_calls"] == 0


def test_agents_analyze_dashboard_runs_lifecycle_without_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    _fake_dashboard(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "agents",
            "analyze",
            "--ticker",
            "AAPL",
            "--as-of",
            "2023-01-03",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "analysis.json"),
            "--dashboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeDashboardServer.instances[-1].started is True
    assert FakeDashboardServer.instances[-1].stopped is True
    assert FakeDashboardState.instances[-1].statuses[-1] == "completed"
    assert FakeDashboardState.instances[-1].preparations == [
        ("AAPL", "2023-01-03", "MiniMax")
    ]


def test_dashboard_requires_trading_agents_backtest(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "run.json"),
            "--dashboard",
        ],
        env={"COLUMNS": "160"},
    )

    assert result.exit_code != 0
    assert "--dashboard requires --use-llm and --llm-workflow trading-agents" in result.output


def test_agent_event_stream_requires_trading_agents_backtest(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "run.json"),
            "--agent-events",
            str(tmp_path / "events.jsonl"),
        ],
        env={"COLUMNS": "160"},
    )

    assert result.exit_code != 0
    assert "--agent-events requires --use-llm" in result.output


def test_dashboard_startup_failure_happens_before_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenServer(FakeDashboardServer):
        def start(self) -> str:
            raise DashboardError("local dashboard could not start")

    provider_opened = False

    def open_provider(*args: object) -> object:
        nonlocal provider_opened
        provider_opened = True
        raise AssertionError("provider must not open")

    monkeypatch.setattr("quant_trader.cli.DashboardState", FakeDashboardState)
    monkeypatch.setattr("quant_trader.cli.DashboardServer", BrokenServer)
    monkeypatch.setattr("quant_trader.cli._open_provider", open_provider)

    result = CliRunner().invoke(
        app,
        [
            "agents",
            "analyze",
            "--ticker",
            "SPY",
            "--as-of",
            "2025-12-31",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "analysis.json"),
            "--llm-provider",
            "codex",
            "--dashboard",
        ],
    )

    assert result.exit_code == 1
    assert "local dashboard could not start" in result.output
    assert provider_opened is False


def test_agents_dashboard_closes_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class InterruptedCodex:
        def check_available(self) -> None:
            raise KeyboardInterrupt

    _fake_dashboard(monkeypatch)
    monkeypatch.setattr("quant_trader.cli.CodexReviewer", InterruptedCodex)

    result = CliRunner().invoke(
        app,
        [
            "agents",
            "analyze",
            "--ticker",
            "SPY",
            "--as-of",
            "2025-12-31",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "analysis.json"),
            "--llm-provider",
            "codex",
            "--dashboard",
        ],
    )

    assert result.exit_code == 130
    assert FakeDashboardState.instances[-1].statuses[-1] == "stopped"
    assert FakeDashboardServer.instances[-1].stopped is True


def test_dashboard_wait_and_stop_failures_do_not_change_valid_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenAuxiliaryState(FakeDashboardState):
        def wait_until_seen(self, version: int, *, timeout_seconds: float) -> bool:
            raise RuntimeError("browser disconnected")

    class BrokenStopServer(FakeDashboardServer):
        def stop(self) -> None:
            self.stopped = True
            raise RuntimeError("server shutdown failed")

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr("quant_trader.cli.DashboardState", BrokenAuxiliaryState)
    monkeypatch.setattr("quant_trader.cli.DashboardServer", BrokenStopServer)

    result = CliRunner().invoke(
        app,
        [
            "agents",
            "analyze",
            "--ticker",
            "AAPL",
            "--as-of",
            "2023-01-03",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "analysis.json"),
            "--dashboard",
        ],
    )

    assert result.exit_code == 0, result.output


def test_backtest_dashboard_closes_on_unexpected_simulation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProvider:
        def complete(self, messages: tuple[ChatMessage, ...]) -> str:
            return MaintainReviewer().complete(messages)

    _fake_dashboard(monkeypatch)
    monkeypatch.setattr(
        "quant_trader.cli._open_provider",
        lambda *args: (FakeProvider(), None, "Codex"),
    )
    monkeypatch.setattr(
        "quant_trader.cli.run_backtest",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulation bug")),
    )

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "run.json"),
            "--use-llm",
            "--llm-provider",
            "codex",
            "--llm-workflow",
            "trading-agents",
            "--dashboard",
        ],
    )

    assert result.exit_code != 0
    assert FakeDashboardState.instances[-1].statuses[-1] == "failed"
    assert FakeDashboardServer.instances[-1].stopped is True


def test_trading_agents_backtest_defaults_to_one_complete_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCodexReviewer:
        def check_available(self) -> None:
            return None

        def complete(self, messages: tuple[ChatMessage, ...]) -> str:
            return MaintainReviewer().complete(messages)

    class FakeTradingAgentsReviewer:
        calls = 0
        observed = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.traces: list[object] = []
            type(self).observed = callable(kwargs.get("on_event"))

        def complete(self, messages: tuple[ChatMessage, ...]) -> str:
            type(self).calls += 1
            return MaintainReviewer().complete(messages)

    monkeypatch.setattr("quant_trader.cli.CodexReviewer", FakeCodexReviewer)
    monkeypatch.setattr(
        "quant_trader.cli.TradingAgentsReviewer", FakeTradingAgentsReviewer
    )
    _fake_dashboard(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "agents.json"),
            "--use-llm",
            "--llm-provider",
            "codex",
            "--llm-workflow",
            "trading-agents",
            "--dashboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeTradingAgentsReviewer.calls == 1
    assert FakeTradingAgentsReviewer.observed is True
    assert FakeDashboardServer.instances[-1].stopped is True
    assert FakeDashboardState.instances[-1].statuses[-1] == "completed"


def test_data_sync_prints_concise_provider_error(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise DataValidationError("SPY: Yahoo Finance rate limited; wait a few minutes")

    monkeypatch.setattr("quant_trader.cli.SinaSource.fetch", fail)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "sync",
            "--config",
            "configs/default.yaml",
            "--start",
            "2023-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    assert result.exit_code == 1
    assert "Yahoo Finance rate limited" in result.output
    assert "Traceback" not in result.output


def test_data_sync_uses_sina_by_default(monkeypatch) -> None:
    calls: list[str] = []

    def fetch(_source: object, ticker: str, *args: object) -> object:
        calls.append(ticker)
        return object()

    monkeypatch.setattr("quant_trader.cli.SinaSource.fetch", fetch)
    monkeypatch.setattr(
        "quant_trader.cli.YFinanceSource.fetch",
        lambda *args: (_ for _ in ()).throw(AssertionError("Yahoo should not be called")),
    )
    monkeypatch.setattr("quant_trader.cli.ParquetMarketCache.write", lambda *args: None)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "sync",
            "--config",
            "configs/default.yaml",
            "--start",
            "2023-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]


def test_data_sync_can_explicitly_use_yahoo(monkeypatch) -> None:
    calls: list[str] = []

    def fetch(_source: object, ticker: str, *args: object) -> object:
        calls.append(ticker)
        return object()

    monkeypatch.setattr("quant_trader.cli.YFinanceSource.fetch", fetch)
    monkeypatch.setattr("quant_trader.cli.ParquetMarketCache.write", lambda *args: None)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "sync",
            "--source",
            "yahoo",
            "--config",
            "configs/default.yaml",
            "--start",
            "2023-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 9


def test_counting_reviewer_tracks_rules_only_review_count() -> None:
    reviewer = _CountingReviewer()

    first = reviewer.complete((ChatMessage(role="user", content="x"),))
    second = reviewer.complete((ChatMessage(role="user", content="y"),))

    assert reviewer.calls == 2
    assert '"action": "maintain"' in first
    assert '"action": "maintain"' in second


def test_progress_reviewer_stops_calling_provider_after_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: tuple[ChatMessage, ...]) -> str:
            self.calls += 1
            return _CountingReviewer().complete(messages)

    provider = Provider()
    reviewer = _ProgressReviewer(provider, max_reviews=1, provider_name="Codex")

    reviewer.complete((ChatMessage(role="user", content="x"),))
    fallback = reviewer.complete((ChatMessage(role="user", content="y"),))

    assert provider.calls == 1
    assert reviewer.real_calls == 1
    assert reviewer.truncated_calls == 1
    assert '"action": "maintain"' in fallback
    messages = capsys.readouterr().err
    assert "Codex review 1 started" in messages
    assert "Codex review 1 completed" in messages


def test_progress_reviewer_can_fail_closed_after_limit() -> None:
    reviewer = _ProgressReviewer(
        _CountingReviewer(),
        max_reviews=0,
        provider_name="MiniMax",
        fallback=_RejectReviewer(),
    )

    fallback = json.loads(
        reviewer.complete((ChatMessage(role="user", content="candidate"),))
    )

    assert fallback["action"] == "reject"
    assert fallback["weight_multiplier"] == 0
    assert reviewer.real_calls == 0


def test_codex_backtest_needs_no_minimax_key_and_defaults_to_three_reviews(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCodexReviewer:
        checked = False
        calls = 0

        def check_available(self) -> None:
            type(self).checked = True

        def complete(self, messages: tuple[ChatMessage, ...]) -> str:
            type(self).calls += 1
            return MaintainReviewer().complete(messages)

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr("quant_trader.cli.CodexReviewer", FakeCodexReviewer, raising=False)
    output = tmp_path / "codex.json"

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(output),
            "--use-llm",
            "--llm-provider",
            "codex",
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeCodexReviewer.checked
    assert FakeCodexReviewer.calls == 3
    payload = json.loads(output.read_text())
    assert payload["note"].startswith("LLM smoke run truncated")
    assert "first 3 reviews used Codex" in payload["note"]


def test_codex_backtest_honors_explicit_review_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCodexReviewer:
        calls = 0

        def check_available(self) -> None:
            return None

        def complete(self, messages: tuple[ChatMessage, ...]) -> str:
            type(self).calls += 1
            return MaintainReviewer().complete(messages)

    monkeypatch.setattr("quant_trader.cli.CodexReviewer", FakeCodexReviewer, raising=False)

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "codex.json"),
            "--use-llm",
            "--llm-provider",
            "codex",
            "--llm-max-reviews",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeCodexReviewer.calls == 1


def test_minimax_provider_still_requires_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "minimax.json"),
            "--use-llm",
            "--llm-provider",
            "minimax",
        ],
    )

    assert result.exit_code != 0
    assert "MINIMAX_API_KEY" in result.output


def test_codex_cli_error_is_actionable_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenCodexReviewer:
        def check_available(self) -> None:
            raise CodexError("Codex CLI is unavailable; repair it and run codex login")

    monkeypatch.setattr("quant_trader.cli.CodexReviewer", BrokenCodexReviewer, raising=False)

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            "configs/default.yaml",
            "--data-root",
            "data",
            "--output",
            str(tmp_path / "codex.json"),
            "--use-llm",
            "--llm-provider",
            "codex",
        ],
    )

    assert result.exit_code == 1
    assert "repair it and run codex login" in result.output
    assert "Traceback" not in result.output

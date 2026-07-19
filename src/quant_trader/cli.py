"""Safe paper-only command line interface."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from quant_trader.backtest import MaintainReviewer, buy_and_hold, run_backtest
from quant_trader.config import Settings, load_settings
from quant_trader.dashboard import DashboardError, DashboardServer, DashboardState
from quant_trader.data.cache import ParquetMarketCache
from quant_trader.data.sina_source import SinaSource
from quant_trader.data.validation import DataValidationError
from quant_trader.data.yfinance_source import YFinanceSource
from quant_trader.experiments.run import run_alpha_arena, run_finmem, run_quanta_alpha
from quant_trader.llm.base import LLMReviewer, MessageInput
from quant_trader.llm.codex import CodexError, CodexReviewer
from quant_trader.llm.minimax import MiniMaxError, MiniMaxReviewer
from quant_trader.llm.traex import TraexError, TraexReviewer
from quant_trader.paper import run_once
from quant_trader.report import write_report
from quant_trader.state import PaperState
from quant_trader.strategies.v2_multi_agent import (
    AgentEvent,
    AgentEventKind,
    TradingAgentsReviewer,
    load_external_context,
    prepare_analysis,
    reject_future_context,
)
from quant_trader.web import WebJobManager, WebPlatformServer

app = typer.Typer(help="Local web UI for safe US-equity research and paper simulation.")
data_app = typer.Typer(help="Manage the validated local market-data cache.")
paper_app = typer.Typer(help="Run one confirmed, paper-only cycle.")
agents_app = typer.Typer(help="Run bounded TradingAgents-style paper analysis.")
experiment_app = typer.Typer(help="Run reproducible paper strategy experiments.")
experiment_run_app = typer.Typer(help="Run one paper experiment.")
app.add_typer(data_app, name="data", hidden=True)
app.add_typer(paper_app, name="paper", hidden=True)
app.add_typer(agents_app, name="agents", hidden=True)
app.add_typer(experiment_app, name="experiment", hidden=True)
experiment_app.add_typer(experiment_run_app, name="run")


class MarketSource(StrEnum):
    SINA = "sina"
    YAHOO = "yahoo"


class LLMProvider(StrEnum):
    MINIMAX = "minimax"
    CODEX = "codex"
    TRAEX = "traex"


class LLMWorkflow(StrEnum):
    SINGLE = "single"
    TRADING_AGENTS = "trading-agents"


class LLMReviewSchedule(StrEnum):
    FIRST = "first"
    EVENLY = "evenly"
    CUSTOM = "custom"


@data_app.command("sync")
def data_sync(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    start: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    source: Annotated[MarketSource, typer.Option()] = MarketSource.SINA,
) -> None:
    """Download configured symbols into the local cache."""
    settings = load_settings(config)
    market_source = SinaSource() if source is MarketSource.SINA else YFinanceSource()
    cache = ParquetMarketCache(data_root)
    try:
        for ticker in settings.universe:
            cache.write(ticker, market_source.fetch(ticker, start.date(), end.date()))
            typer.echo(f"cached {ticker}")
    except DataValidationError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None


def _frames(
    data_root: Path,
    tickers: tuple[str, ...],
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    if start is not None and end is not None and start >= end:
        raise ValueError("start date must be before end date")
    cache = ParquetMarketCache(data_root)
    frames = {ticker: cache.read(ticker) for ticker in tickers}
    if start is None and end is None:
        return frames
    filtered: dict[str, pd.DataFrame] = {}
    for ticker, frame in frames.items():
        selected = frame
        if start is not None:
            selected = selected.loc[selected.index >= pd.Timestamp(start)]
        if end is not None:
            selected = selected.loc[selected.index < pd.Timestamp(end)]
        filtered[ticker] = selected
    return filtered


def _open_provider(
    settings: Settings, provider: LLMProvider, *, max_retries: int | None = None
) -> tuple[LLMReviewer, MiniMaxReviewer | None, str]:
    if provider is LLMProvider.CODEX:
        codex = CodexReviewer()
        codex.check_available()
        return codex, None, "Codex"
    if provider is LLMProvider.TRAEX:
        traex = TraexReviewer(model=settings.llm.traex_model)
        traex.check_available()
        return traex, None, "Trae X"
    key = settings.llm.api_key.get_secret_value()
    if not key:
        raise typer.BadParameter("MiniMax reviews require MINIMAX_API_KEY")
    client = MiniMaxReviewer(
        key,
        settings.llm.base_url,
        settings.llm.model,
        settings.llm.timeout_seconds,
        settings.llm.max_retries if max_retries is None else max_retries,
    )
    return client, client, "MiniMax"


def _write_json(output: Path, payload: object) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _agent_progress(role: object, status: str) -> None:
    role_name = getattr(role, "value", str(role))
    typer.echo(f"Agent {role_name} {status}.", err=True)


def _append_agent_event(output: Path, event: AgentEvent) -> None:
    """Append one sanitized structured event for an external local observer."""
    with output.open("a", encoding="utf-8") as destination:
        destination.write(event.model_dump_json() + "\n")


class _DashboardRun:
    def __init__(self, enabled: bool) -> None:
        self.state = DashboardState()
        self.server = DashboardServer(self.state) if enabled else None
        self._started = False

    @property
    def observer(self) -> Callable[[AgentEvent], None] | None:
        return self.state.publish if self.server is not None else None

    def start(self) -> None:
        if self.server is not None:
            typer.echo(f"Dashboard: {self.server.start()}", err=True)
            self._started = True

    def prepare(self, ticker: str, as_of: str, provider: str) -> None:
        if self.server is None or not self._started:
            return
        try:
            self.state.prepare(ticker, as_of, provider)
        except Exception as error:
            self._clear(error)

    def prepare_experiment(self, kind: str, run_id: str, provider: str) -> None:
        if self.server is None or not self._started:
            return
        try:
            self.state.prepare_experiment(kind, run_id, provider)
        except Exception as error:
            self._clear(error)

    def update_experiment(
        self, stage: str, status: str, payload: dict[str, object]
    ) -> None:
        if self.server is None or not self._started:
            return
        try:
            self.state.update_experiment(stage, status, payload)
        except Exception as error:
            self._clear(error)

    def finish(self, status: str, *, reason: str | None = None) -> None:
        if self.server is None or not self._started:
            return
        try:
            version = self.state.set_command_status(status, reason=reason)
            self.state.wait_until_seen(version, timeout_seconds=1.0)
        except Exception as error:
            self._clear(error)

    def close(self) -> None:
        if self.server is not None:
            try:
                self.server.stop()
            except Exception as error:
                self._clear(error)
            self._started = False

    @staticmethod
    def _clear(error: BaseException) -> None:
        if error.__traceback__ is not None:
            traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None


class _CountingReviewer:
    def __init__(self) -> None:
        self.calls = 0
        self._fallback = MaintainReviewer()

    def complete(self, messages: tuple[MessageInput, ...]) -> str:
        self.calls += 1
        return self._fallback.complete(messages)


class _RejectReviewer:
    def complete(self, messages: Sequence[MessageInput]) -> str:
        del messages
        return json.dumps(
            {
                "action": "reject",
                "weight_multiplier": 0,
                "confidence": 0,
                "thesis": "External review budget exhausted.",
                "risks": ["review_budget_exhausted"],
                "invalidation": "No position without an external review.",
                "input_anomalies": [],
            }
        )


class _ProgressReviewer:
    def __init__(
        self,
        reviewer: object,
        *,
        max_reviews: int | None,
        provider_name: str = "MiniMax",
        fallback: LLMReviewer | None = None,
    ) -> None:
        self.calls = 0
        self.real_calls = 0
        self.truncated_calls = 0
        self.provider_name = provider_name
        self._reviewer = reviewer
        self._fallback = fallback or MaintainReviewer()
        self._max_reviews = max_reviews
        self._review_indices: frozenset[int] | None = None

    @property
    def review_indices(self) -> tuple[int, ...] | None:
        return (
            tuple(sorted(self._review_indices))
            if self._review_indices is not None
            else None
        )

    def set_review_indices(self, indices: Sequence[int]) -> None:
        self._review_indices = frozenset(indices)

    def complete(self, messages: tuple[MessageInput, ...]) -> str:
        self.calls += 1
        if self._review_indices is not None and self.calls not in self._review_indices:
            self.truncated_calls += 1
            return self._fallback.complete(messages)
        if self._max_reviews is not None and self.real_calls >= self._max_reviews:
            self.truncated_calls += 1
            if self.truncated_calls == 1:
                typer.echo(
                    f"{self.provider_name} review limit reached; remaining reviews use local "
                    "rules-only replies.",
                    err=True,
                )
            return self._fallback.complete(messages)
        self.real_calls += 1
        typer.echo(f"{self.provider_name} review {self.real_calls} started...", err=True)
        result = self._reviewer.complete(messages)  # type: ignore[attr-defined]
        typer.echo(f"{self.provider_name} review {self.real_calls} completed.", err=True)
        return result


def _review_plan(
    total: int,
    maximum: int | None,
    schedule: LLMReviewSchedule,
    custom: str | None,
) -> tuple[int, ...] | None:
    if maximum is None:
        if schedule is not LLMReviewSchedule.FIRST or custom:
            raise ValueError("a review schedule requires --llm-max-reviews")
        return None
    count = min(maximum, total)
    if schedule is LLMReviewSchedule.CUSTOM:
        try:
            indices = tuple(int(item.strip()) for item in (custom or "").split(","))
        except ValueError:
            raise ValueError("custom review indices must be comma-separated integers") from None
        if not indices or len(indices) != maximum or len(set(indices)) != len(indices):
            raise ValueError("custom review indices must be unique and match max reviews")
        if any(index < 1 or index > total for index in indices):
            raise ValueError(f"custom review indices must be between 1 and {total}")
        return tuple(sorted(indices))
    if custom:
        raise ValueError("--llm-review-indices requires --llm-review-schedule custom")
    if count == 0:
        return ()
    if schedule is LLMReviewSchedule.FIRST:
        return tuple(range(1, count + 1))
    if count == 1:
        return ((total + 1) // 2,)
    return tuple(round(index * (total - 1) / (count - 1)) + 1 for index in range(count))


def _run_experiment_command(
    kind: str,
    config: Path,
    data_root: Path,
    output_dir: Path,
    llm_provider: LLMProvider | None,
    dashboard: bool,
    start: datetime | None = None,
    end: datetime | None = None,
    contestant_runs: tuple[Path, ...] = (),
) -> None:
    client: MiniMaxReviewer | None = None
    dashboard_run = _DashboardRun(dashboard)
    try:
        settings = load_settings(config)
        frames = _frames(data_root, settings.universe, start, end)
        dashboard_run.start()
        provider_name = "none"
        provider: LLMReviewer | None = None
        model = "none"
        if kind != "alpha-arena":
            assert llm_provider is not None
            provider, client, provider_name = _open_provider(
                settings, llm_provider, max_retries=0
            )
            model = (
                settings.llm.model
                if llm_provider is LLMProvider.MINIMAX
                else llm_provider.value
            )
        prepared = False

        def update(stage: str, status: str, payload: dict[str, object]) -> None:
            nonlocal prepared
            if not prepared:
                run_id = payload.get("run_id")
                if isinstance(run_id, str):
                    dashboard_run.prepare_experiment(kind, run_id, provider_name)
                    prepared = True
            dashboard_run.update_experiment(stage, status, payload)

        if kind == "finmem":
            assert provider is not None
            root = run_finmem(
                settings,
                frames,
                output_dir,
                provider,
                provider_name,
                model,
                lambda value: _ProgressReviewer(
                    value,
                    max_reviews=1,
                    provider_name=provider_name,
                    fallback=_RejectReviewer(),
                ),
                update if dashboard else None,
            )
        elif kind == "quanta-alpha":
            assert provider is not None
            root = run_quanta_alpha(
                settings,
                frames,
                output_dir,
                provider,
                provider_name,
                model,
                update if dashboard else None,
            )
        else:
            root = run_alpha_arena(
                settings,
                frames,
                output_dir,
                contestant_runs,
                update if dashboard else None,
            )
        typer.echo(str(root))
        dashboard_run.finish("completed")
    except KeyboardInterrupt:
        dashboard_run.finish("stopped")
        raise typer.Exit(code=130) from None
    except (
        CodexError,
        DashboardError,
        DataValidationError,
        MiniMaxError,
        TraexError,
        OSError,
        ValueError,
    ) as error:
        dashboard_run.finish("failed")
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    finally:
        if client is not None:
            client.close()
        dashboard_run.close()


@experiment_run_app.command("finmem")
def experiment_finmem(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    llm_provider: Annotated[LLMProvider, typer.Option()] = LLMProvider.MINIMAX,
    dashboard: Annotated[bool, typer.Option()] = False,
    start: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
    end: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
) -> None:
    """Run one-call, memory-aware paper backtest."""
    _run_experiment_command(
        "finmem", config, data_root, output_dir, llm_provider, dashboard, start, end
    )


@experiment_run_app.command("quanta-alpha")
def experiment_quanta_alpha(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    llm_provider: Annotated[LLMProvider, typer.Option()] = LLMProvider.MINIMAX,
    dashboard: Annotated[bool, typer.Option()] = False,
    start: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
    end: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
) -> None:
    """Mine a safe factor DSL with at most two provider calls."""
    _run_experiment_command(
        "quanta-alpha", config, data_root, output_dir, llm_provider, dashboard, start, end
    )


@experiment_run_app.command("alpha-arena")
def experiment_alpha_arena(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    contestant_run: Annotated[
        list[Path], typer.Option(exists=True, file_okay=False, help="Existing run directory.")
    ] = [],
    dashboard: Annotated[bool, typer.Option()] = False,
    start: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
    end: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
) -> None:
    """Compare existing artifacts without constructing an LLM provider."""
    _run_experiment_command(
        "alpha-arena",
        config,
        data_root,
        output_dir,
        None,
        dashboard,
        start,
        end,
        tuple(contestant_run),
    )


@app.command(hidden=True)
def backtest(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
    start: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
    end: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
    use_llm: Annotated[
        bool, typer.Option(help="Use external LLM reviews (MiniMax by default).")
    ] = False,
    llm_provider: Annotated[
        LLMProvider,
        typer.Option(help="Review provider: minimax requires an API key; codex uses local login."),
    ] = LLMProvider.MINIMAX,
    llm_workflow: Annotated[
        LLMWorkflow,
        typer.Option(help="Review workflow: one reviewer or a bounded multi-agent workflow."),
    ] = LLMWorkflow.SINGLE,
    context: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Optional point-in-time agent context JSON.",
        ),
    ] = None,
    llm_max_reviews: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Send only N selected reviews to the provider, then use local rules-only replies. "
                "Selection is controlled by --llm-review-schedule."
            ),
        ),
    ] = None,
    llm_review_schedule: Annotated[
        LLMReviewSchedule,
        typer.Option(
            help="Choose provider review opportunities: first, evenly, or custom."
        ),
    ] = LLMReviewSchedule.FIRST,
    llm_review_indices: Annotated[
        str | None,
        typer.Option(help="Comma-separated 1-based opportunity numbers for custom scheduling."),
    ] = None,
    dashboard: Annotated[
        bool,
        typer.Option(help="Open a local real-time TradingAgents decision dashboard."),
    ] = False,
    agent_events: Annotated[
        Path | None,
        typer.Option(help="Optional JSONL stream of sanitized TradingAgents events for local UIs."),
    ] = None,
) -> None:
    """Run cached chronological simulation (rules-only by default)."""
    settings = load_settings(config)
    if dashboard and (
        not use_llm or llm_workflow is not LLMWorkflow.TRADING_AGENTS
    ):
        raise typer.BadParameter(
            "--dashboard requires --use-llm and --llm-workflow trading-agents"
        )
    if agent_events is not None and (
        not use_llm or llm_workflow is not LLMWorkflow.TRADING_AGENTS
    ):
        raise typer.BadParameter(
            "--agent-events requires --use-llm and --llm-workflow trading-agents"
        )
    try:
        frames = _frames(data_root, settings.universe, start, end)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    reviewer = None
    client = None
    agent_reviewer: TradingAgentsReviewer | None = None
    dashboard_run = _DashboardRun(dashboard)
    if agent_events is not None:
        agent_events.parent.mkdir(parents=True, exist_ok=True)
        agent_events.write_text("", encoding="utf-8")

    def observe_agent(event: AgentEvent) -> None:
        observer = dashboard_run.observer
        if observer is not None:
            observer(event)
        if agent_events is not None:
            _append_agent_event(agent_events, event)
        if event.kind is AgentEventKind.ROLE_FAILED and event.report is not None:
            detail = event.report.input_anomalies[-1]
            role = event.role.value if event.role is not None else "unknown"
            typer.echo(f"Agent {role} error: {detail}", err=True)

    try:
        dashboard_run.start()
        if use_llm:
            external_context = (
                load_external_context(context)
                if llm_workflow is LLMWorkflow.TRADING_AGENTS
                else None
            )
            provider, client, provider_name = _open_provider(settings, llm_provider)
            max_reviews: int | None
            if llm_workflow is LLMWorkflow.TRADING_AGENTS:
                agent_reviewer = TradingAgentsReviewer(
                    provider,
                    provider_name=provider_name,
                    external_context=external_context,
                    on_progress=_agent_progress,
                    on_event=(observe_agent if dashboard or agent_events is not None else None),
                )
                provider = agent_reviewer
                max_reviews = 1 if llm_max_reviews is None else llm_max_reviews
            else:
                max_reviews = (
                    3
                    if llm_provider in {LLMProvider.CODEX, LLMProvider.TRAEX}
                    and llm_max_reviews is None
                    else llm_max_reviews
                )
            reviewer = _ProgressReviewer(
                provider, max_reviews=max_reviews, provider_name=provider_name
            )
        rules_counter = _CountingReviewer()
        rules_result = run_backtest(frames, settings, reviewer=rules_counter)  # type: ignore[arg-type]
        if use_llm:
            assert reviewer is not None
            plan = _review_plan(
                rules_counter.calls,
                max_reviews,
                llm_review_schedule,
                llm_review_indices,
            )
            if plan is not None:
                reviewer.set_review_indices(plan)
            typer.echo(
                f"{reviewer.provider_name} enabled: this backtest can request up to "
                f"{rules_counter.calls} reviews.",
                err=True,
            )
            if plan is not None:
                rendered = ", ".join(str(index) for index in plan)
                typer.echo(
                    f"{reviewer.provider_name} review plan: {len(plan)}/"
                    f"{rules_counter.calls} opportunities at indices {rendered}.",
                    err=True,
                )
        llm_result = (
            run_backtest(frames, settings, reviewer=reviewer)  # type: ignore[arg-type]
            if reviewer is not None
            else None
        )
    except KeyboardInterrupt:
        dashboard_run.finish("stopped")
        dashboard_run.close()
        raise typer.Exit(code=130) from None
    except (CodexError, DashboardError, MiniMaxError, TraexError, ValueError) as error:
        dashboard_run.finish("failed")
        dashboard_run.close()
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    except Exception:
        dashboard_run.finish("failed")
        dashboard_run.close()
        raise
    finally:
        if client is not None:
            client.close()
    try:
        benchmark = buy_and_hold(frames["SPY"], settings.paper.initial_cash)
        runs = {"rules_only": rules_result.to_dict(), "spy_buy_hold": benchmark.to_dict()}
        if llm_result is not None:
            runs["llm"] = llm_result.to_dict()
        note = "Paper simulation only."
        if isinstance(reviewer, _ProgressReviewer) and reviewer.truncated_calls:
            indices = reviewer.review_indices
            first = tuple(range(1, reviewer.real_calls + 1))
            if indices == first:
                note = (
                    "LLM smoke run truncated: only the first "
                    f"{reviewer.real_calls} reviews used {reviewer.provider_name}; remaining "
                    "reviews used local rules-only replies."
                )
            else:
                rendered = ", ".join(str(index) for index in (indices or ()))
                note = (
                    f"LLM scheduled run: {reviewer.real_calls} reviews used "
                    f"{reviewer.provider_name} at opportunity indices {rendered}; other reviews "
                    "used local rules-only replies."
                )
        if (
            llm_result is not None
            and not (isinstance(reviewer, _ProgressReviewer) and reviewer.truncated_calls)
            and llm_result.metrics()["total_return"] <= rules_result.metrics()["total_return"]
        ):
            note = "LLM run shows no proven gain over rules-only after costs."
        payload: dict[str, object] = {"runs": runs, "note": note}
        if agent_reviewer is not None and agent_reviewer.traces:
            payload["agent_traces"] = [
                trace.model_dump(mode="json") for trace in agent_reviewer.traces
            ]
        _write_json(output, payload)
        typer.echo(str(output))
        dashboard_run.finish("completed")
    except KeyboardInterrupt:
        dashboard_run.finish("stopped")
        raise typer.Exit(code=130) from None
    except Exception:
        dashboard_run.finish("failed")
        raise
    finally:
        dashboard_run.close()


@agents_app.command("analyze")
def agents_analyze(
    ticker: Annotated[str, typer.Option()],
    as_of: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
    context: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Optional point-in-time context JSON."),
    ] = None,
    llm_provider: Annotated[
        LLMProvider,
        typer.Option(
            help="Provider: minimax requires an API key; codex/traex use local login."
        ),
    ] = LLMProvider.MINIMAX,
    dashboard: Annotated[
        bool,
        typer.Option(help="Open a local real-time TradingAgents decision dashboard."),
    ] = False,
) -> None:
    """Analyze one eligible ticker once; never places an order."""
    client = None
    dashboard_run = _DashboardRun(dashboard)
    try:
        settings = load_settings(config)
        point_in_time = as_of.date()
        external_context = load_external_context(context)
        reject_future_context(external_context, ticker, point_in_time)
        prepared = prepare_analysis(
            _frames(data_root, settings.universe), settings, ticker, point_in_time
        )
        dashboard_run.start()
        dashboard_run.prepare(
            prepared.ticker,
            prepared.as_of.isoformat(),
            {LLMProvider.CODEX: "Codex", LLMProvider.TRAEX: "Trae X"}.get(
                llm_provider, "MiniMax"
            ),
        )
        if not prepared.eligible:
            _write_json(
                output,
                {
                    "workflow": LLMWorkflow.TRADING_AGENTS,
                    "ticker": prepared.ticker,
                    "as_of": prepared.as_of.isoformat(),
                    "eligible": False,
                    "provider_calls": 0,
                    "reason": prepared.reason,
                },
            )
            typer.echo(str(output))
            dashboard_run.finish("completed", reason=prepared.reason)
            return
        provider, client, provider_name = _open_provider(settings, llm_provider)
        reviewer = TradingAgentsReviewer(
            provider,
            provider_name=provider_name,
            external_context=external_context,
            on_progress=_agent_progress,
            on_event=dashboard_run.observer,
        )
        assert prepared.messages is not None
        reviewer.complete(prepared.messages)
        trace = reviewer.traces[-1]
        _write_json(
            output,
            {
                "workflow": LLMWorkflow.TRADING_AGENTS,
                "provider": provider_name,
                "ticker": prepared.ticker,
                "as_of": prepared.as_of.isoformat(),
                "eligible": True,
                "provider_calls": trace.provider_calls,
                "trace": trace.model_dump(mode="json"),
            },
        )
        typer.echo(str(output))
        dashboard_run.finish("completed")
    except KeyboardInterrupt:
        dashboard_run.finish("stopped")
        raise typer.Exit(code=130) from None
    except (CodexError, DashboardError, MiniMaxError, TraexError, ValueError) as error:
        dashboard_run.finish("failed")
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    except Exception:
        dashboard_run.finish("failed")
        raise
    finally:
        if client is not None:
            client.close()
        dashboard_run.close()


@paper_app.command("init")
def paper_init(db: Annotated[Path, typer.Option()]) -> None:
    """Create a version-1 paper SQLite database."""
    db.parent.mkdir(parents=True, exist_ok=True)
    state = PaperState(db)
    state.close()
    typer.echo(str(db))


@paper_app.command("status")
def paper_status(db: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    state = PaperState(db)
    try:
        typer.echo(json.dumps(state.status(), indent=2))
    finally:
        state.close()


@paper_app.command("run")
def paper_run(
    db: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    confirm: Annotated[bool, typer.Option(help="Required paper-execution confirmation.")] = False,
) -> None:
    """Run exactly one paper cycle from ./data; never sends broker orders."""
    if not confirm:
        raise typer.BadParameter("--confirm is required for paper execution")
    settings = load_settings(config)
    frames = _frames(Path("data"), settings.universe)
    state = PaperState(db)
    try:
        identifiers = run_once(state, frames, settings)  # type: ignore[arg-type]
    finally:
        state.close()
    typer.echo(json.dumps({"processed": identifiers}))


@app.command(hidden=True)
def report(
    run_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Render a self-contained HTML summary."""
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(run_json, output)
    typer.echo(str(output))


@app.command("web")
def web_platform(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("data"),
    output_root: Annotated[Path, typer.Option()] = Path("web-runs"),
    port: Annotated[int, typer.Option(min=0, max=65_535)] = 8000,
    workers: Annotated[int, typer.Option(min=1, max=4)] = 2,
    open_browser: Annotated[bool, typer.Option()] = True,
) -> None:
    """Open the local experiment website (research and paper simulation only)."""
    manager = WebJobManager(
        project_root=Path.cwd(),
        config=config,
        data_root=data_root,
        output_root=output_root,
        workers=workers,
    )
    server = WebPlatformServer(manager, port=port)
    try:
        server.serve(open_browser=open_browser)
    except KeyboardInterrupt:
        typer.echo("Web platform stopped.", err=True)

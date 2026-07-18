"""Safe paper-only command line interface."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from quant_trader.backtest import MaintainReviewer, buy_and_hold, run_backtest
from quant_trader.config import load_settings
from quant_trader.data.cache import ParquetMarketCache
from quant_trader.data.sina_source import SinaSource
from quant_trader.data.validation import DataValidationError
from quant_trader.data.yfinance_source import YFinanceSource
from quant_trader.llm.base import MessageInput
from quant_trader.llm.codex import CodexError, CodexReviewer
from quant_trader.llm.minimax import MiniMaxReviewer
from quant_trader.paper import run_once
from quant_trader.report import write_report
from quant_trader.state import PaperState

app = typer.Typer(help="Safe US-equity research and paper-trading tools (never live trading).")
data_app = typer.Typer(help="Manage the validated local market-data cache.")
paper_app = typer.Typer(help="Run one confirmed, paper-only cycle.")
app.add_typer(data_app, name="data")
app.add_typer(paper_app, name="paper")


class MarketSource(StrEnum):
    SINA = "sina"
    YAHOO = "yahoo"


class LLMProvider(StrEnum):
    MINIMAX = "minimax"
    CODEX = "codex"


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


def _frames(data_root: Path, tickers: tuple[str, ...]) -> dict[str, object]:
    cache = ParquetMarketCache(data_root)
    return {ticker: cache.read(ticker) for ticker in tickers}


class _CountingReviewer:
    def __init__(self) -> None:
        self.calls = 0
        self._fallback = MaintainReviewer()

    def complete(self, messages: tuple[MessageInput, ...]) -> str:
        self.calls += 1
        return self._fallback.complete(messages)


class _ProgressReviewer:
    def __init__(
        self, reviewer: object, *, max_reviews: int | None, provider_name: str = "MiniMax"
    ) -> None:
        self.calls = 0
        self.real_calls = 0
        self.truncated_calls = 0
        self.provider_name = provider_name
        self._reviewer = reviewer
        self._fallback = MaintainReviewer()
        self._max_reviews = max_reviews

    def complete(self, messages: tuple[MessageInput, ...]) -> str:
        self.calls += 1
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


@app.command()
def backtest(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
    use_llm: Annotated[
        bool, typer.Option(help="Use external LLM reviews (MiniMax by default).")
    ] = False,
    llm_provider: Annotated[
        LLMProvider,
        typer.Option(help="Review provider: minimax requires an API key; codex uses local login."),
    ] = LLMProvider.MINIMAX,
    llm_max_reviews: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Send only the first N reviews to the provider, then use local rules-only replies. "
                "Useful for API smoke tests."
            ),
        ),
    ] = None,
) -> None:
    """Run cached chronological simulation (rules-only by default)."""
    settings = load_settings(config)
    frames = _frames(data_root, settings.universe)
    reviewer = None
    client = None
    try:
        if use_llm:
            if llm_provider is LLMProvider.CODEX:
                codex = CodexReviewer()
                codex.check_available()
                max_reviews = 3 if llm_max_reviews is None else llm_max_reviews
                reviewer = _ProgressReviewer(
                    codex, max_reviews=max_reviews, provider_name="Codex"
                )
            else:
                key = settings.llm.api_key.get_secret_value()
                if not key:
                    raise typer.BadParameter("MiniMax reviews require MINIMAX_API_KEY")
                client = MiniMaxReviewer(
                    key,
                    settings.llm.base_url,
                    settings.llm.model,
                    settings.llm.timeout_seconds,
                    settings.llm.max_retries,
                )
                reviewer = _ProgressReviewer(
                    client, max_reviews=llm_max_reviews, provider_name="MiniMax"
                )
        rules_counter = _CountingReviewer()
        rules_result = run_backtest(frames, settings, reviewer=rules_counter)  # type: ignore[arg-type]
        if use_llm:
            assert reviewer is not None
            typer.echo(
                f"{reviewer.provider_name} enabled: this backtest can request up to "
                f"{rules_counter.calls} reviews.",
                err=True,
            )
        llm_result = (
            run_backtest(frames, settings, reviewer=reviewer)  # type: ignore[arg-type]
            if reviewer is not None
            else None
        )
    except CodexError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    finally:
        if client is not None:
            client.close()
    benchmark = buy_and_hold(frames["SPY"], settings.paper.initial_cash)  # type: ignore[arg-type]
    runs = {"rules_only": rules_result.to_dict(), "spy_buy_hold": benchmark.to_dict()}
    if llm_result is not None:
        runs["llm"] = llm_result.to_dict()
    note = "Paper simulation only."
    if isinstance(reviewer, _ProgressReviewer) and reviewer.truncated_calls:
        note = (
            "LLM smoke run truncated: only the first "
            f"{reviewer.real_calls} reviews used {reviewer.provider_name}; remaining reviews used "
            "local rules-only replies."
        )
    if (
        llm_result is not None
        and not (isinstance(reviewer, _ProgressReviewer) and reviewer.truncated_calls)
        and llm_result.metrics()["total_return"] <= rules_result.metrics()["total_return"]
    ):
        note = "LLM run shows no proven gain over rules-only after costs."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": runs, "note": note}, indent=2), encoding="utf-8")
    typer.echo(str(output))


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


@app.command()
def report(
    run_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Render a self-contained HTML summary."""
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(run_json, output)
    typer.echo(str(output))

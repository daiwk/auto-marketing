"""Safe paper-only command line interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from quant_trader.backtest import buy_and_hold, run_backtest
from quant_trader.config import load_settings
from quant_trader.data.cache import ParquetMarketCache
from quant_trader.data.yfinance_source import YFinanceSource
from quant_trader.llm.minimax import MiniMaxReviewer
from quant_trader.paper import run_once
from quant_trader.report import write_report
from quant_trader.state import PaperState

app = typer.Typer(help="Safe US-equity research and paper-trading tools (never live trading).")
data_app = typer.Typer(help="Manage the validated local market-data cache.")
paper_app = typer.Typer(help="Run one confirmed, paper-only cycle.")
app.add_typer(data_app, name="data")
app.add_typer(paper_app, name="paper")


@data_app.command("sync")
def data_sync(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    start: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    data_root: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Download configured symbols into the local cache."""
    settings = load_settings(config)
    source, cache = YFinanceSource(), ParquetMarketCache(data_root)
    for ticker in settings.universe:
        cache.write(ticker, source.fetch(ticker, start.date(), end.date()))
        typer.echo(f"cached {ticker}")


def _frames(data_root: Path, tickers: tuple[str, ...]) -> dict[str, object]:
    cache = ParquetMarketCache(data_root)
    return {ticker: cache.read(ticker) for ticker in tickers}


@app.command()
def backtest(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
    use_llm: Annotated[
        bool, typer.Option(help="Use MiniMax reviews; requires MINIMAX_API_KEY.")
    ] = False,
) -> None:
    """Run cached chronological simulation (rules-only by default)."""
    settings = load_settings(config)
    frames = _frames(data_root, settings.universe)
    reviewer = None
    client = None
    if use_llm:
        key = settings.llm.api_key.get_secret_value()
        if not key:
            raise typer.BadParameter("--use-llm requires MINIMAX_API_KEY")
        client = MiniMaxReviewer(
            key,
            settings.llm.base_url,
            settings.llm.model,
            settings.llm.timeout_seconds,
            settings.llm.max_retries,
        )
        reviewer = client
    try:
        rules_result = run_backtest(frames, settings)  # type: ignore[arg-type]
        llm_result = (
            run_backtest(frames, settings, reviewer=reviewer)  # type: ignore[arg-type]
            if reviewer is not None
            else None
        )
    finally:
        if client is not None:
            client.close()
    benchmark = buy_and_hold(frames["SPY"], settings.paper.initial_cash)  # type: ignore[arg-type]
    runs = {"rules_only": rules_result.to_dict(), "spy_buy_hold": benchmark.to_dict()}
    if llm_result is not None:
        runs["llm"] = llm_result.to_dict()
    note = "Paper simulation only."
    if (
        llm_result is not None
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

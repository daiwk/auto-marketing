"""Command-line scaffold for the paper-trading package."""

import typer

app = typer.Typer(help="Safe US-equity research and paper-trading tools.")


@app.callback()
def main() -> None:
    """Scaffold only; commands are intentionally not registered in V1."""

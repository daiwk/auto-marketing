"""Dependency-light JSON run report renderer."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_report(run_json: Path | str, output: Path | str) -> None:
    data: dict[str, Any] = json.loads(Path(run_json).read_text(encoding="utf-8"))
    rows = []
    for name, run in data["runs"].items():
        metrics = run["metrics"]
        rows.append(
            "<tr>"
            + f"<th>{html.escape(name)}</th>"
            + "".join(
                f"<td>{float(metrics[key]):.4f}</td>"
                for key in (
                    "total_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe",
                    "max_drawdown",
                    "trade_count",
                    "costs",
                )
            )
            + "</tr>"
        )
    labels = list(data["runs"]["rules_only"]["equity"])
    values = list(data["runs"]["rules_only"]["equity"].values())
    width, height = 800, 220
    minimum, maximum = min(values), max(values)
    span = maximum - minimum or 1
    points = " ".join(
        f"{index * width / max(len(values) - 1, 1):.1f},"
        f"{height - (value - minimum) / span * height:.1f}"
        for index, value in enumerate(values)
    )
    note = data.get("note", "Paper research only; no proven investment gain.")
    document = f"""<!doctype html><meta charset=utf-8><title>Paper trading report</title>
<style>
body{{font:14px system-ui;margin:2rem;max-width:1000px}}
table{{border-collapse:collapse}}
th,td{{padding:.5rem;border:1px solid #ccc;text-align:right}}
svg{{width:100%;height:220px}}
</style>
<h1>Paper trading report</h1><p>{html.escape(note)}</p>
<table><tr><th>Run</th><th>Total return</th><th>Annualized return</th>
<th>Annualized vol</th><th>Sharpe</th><th>Max drawdown</th><th>Trades</th>
<th>Costs</th></tr>{"".join(rows)}</table>
<h2>Rules-only equity</h2><svg viewBox="0 0 {width} {height}"
aria-label="Equity from {html.escape(labels[0])} to {html.escape(labels[-1])}">
<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{points}"/></svg>
<p><strong>Disclaimer:</strong> Paper simulation only. Not investment advice.
No live orders are supported.</p>"""
    Path(output).write_text(document, encoding="utf-8")

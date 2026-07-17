"""Minimal SQLite persistence for paper state and audit."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from quant_trader.execution import Account, Fill


class PaperState:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO meta VALUES ('schema_version', '1');
            CREATE TABLE IF NOT EXISTS decisions (
              decision_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, payload TEXT NOT NULL,
              review_metadata TEXT, processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS orders (
              decision_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, target_weight REAL NOT NULL,
              execution_date TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fills (
              id INTEGER PRIMARY KEY, decision_id TEXT UNIQUE NOT NULL, ticker TEXT NOT NULL,
              execution_date TEXT NOT NULL, shares REAL NOT NULL, price REAL NOT NULL,
              commission REAL NOT NULL, slippage REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS account_snapshots (
              id INTEGER PRIMARY KEY, as_of TEXT UNIQUE NOT NULL, cash REAL NOT NULL,
              high_water_mark REAL NOT NULL, positions TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS llm_cache (
              cache_key TEXT PRIMARY KEY, review_metadata TEXT NOT NULL);
            """
        )
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def processed(self, decision_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        return row is not None

    def cycle_processed(self, as_of: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM account_snapshots WHERE as_of = ?", (as_of,)
            ).fetchone()
            is not None
        )

    def save_cycle(
        self,
        *,
        as_of: str,
        decisions: list[dict[str, object]],
        fills: tuple[Fill, ...],
        account: Account,
    ) -> None:
        with self.transaction() as db:
            for decision in decisions:
                db.execute(
                    "INSERT INTO decisions(decision_id,ticker,payload,review_metadata) "
                    "VALUES(?,?,?,?)",
                    (
                        decision["decision_id"],
                        decision["ticker"],
                        json.dumps(decision),
                        json.dumps(decision.get("review_metadata")),
                    ),
                )
                db.execute(
                    "INSERT INTO orders VALUES(?,?,?,?)",
                    (
                        decision["decision_id"],
                        decision["ticker"],
                        decision["target_weight"],
                        decision["execution_date"],
                    ),
                )
            for fill in fills:
                db.execute(
                    "INSERT INTO fills(decision_id,ticker,execution_date,shares,price,"
                    "commission,slippage) VALUES(?,?,?,?,?,?,?)",
                    (
                        fill.decision_id,
                        fill.ticker,
                        fill.execution_date.isoformat(),
                        fill.shares,
                        fill.price,
                        fill.commission,
                        fill.slippage,
                    ),
                )
            db.execute(
                "INSERT OR REPLACE INTO account_snapshots"
                "(as_of,cash,high_water_mark,positions) VALUES(?,?,?,?)",
                (
                    as_of,
                    account.cash,
                    account.high_water_mark,
                    json.dumps(account.positions, sort_keys=True),
                ),
            )

    def latest_account(self, initial_cash: float = 100_000.0) -> Account:
        row = self.connection.execute(
            "SELECT cash,high_water_mark,positions FROM account_snapshots "
            "ORDER BY as_of DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return Account(initial_cash, {}, initial_cash)
        return Account(row["cash"], json.loads(row["positions"]), row["high_water_mark"])

    def status(self) -> dict[str, object]:
        snapshot = self.connection.execute(
            "SELECT as_of,cash,high_water_mark,positions FROM account_snapshots "
            "ORDER BY as_of DESC LIMIT 1"
        ).fetchone()
        return {
            "schema_version": 1,
            "decision_count": self.connection.execute("SELECT count(*) FROM decisions").fetchone()[
                0
            ],
            "fill_count": self.connection.execute("SELECT count(*) FROM fills").fetchone()[0],
            "account": dict(snapshot) if snapshot else None,
        }

    def close(self) -> None:
        self.connection.close()

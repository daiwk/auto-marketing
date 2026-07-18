import json
from pathlib import Path

from quant_trader.strategies.v5_alpha_arena.arena import AlphaArena, ArenaConfig

CONFIG = ArenaConfig(
    fingerprint="fp-1",
    universe=("AAPL", "MSFT"),
    start_date="2024-01-02",
    end_date="2024-01-05",
    initial_cash=100_000.0,
    cost_bps=5.0,
)


def _write_run(
    root: Path,
    name: str,
    *,
    total_return: float = 0.1,
    max_drawdown: float = -0.1,
    risk_violations: int = 0,
    fingerprint: str = "fp-1",
    invalid_json: bool = False,
) -> Path:
    run = root / name
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "universe": ["AAPL", "MSFT"],
                "start_date": "2024-01-02",
                "end_date": "2024-01-05",
                "initial_cash": 100_000.0,
                "cost_bps": 5.0,
                "result_file": "summary.json",
            }
        )
    )
    summary = {
        "status": "completed",
        "equity": {"2024-01-02": 100_000.0, "2024-01-05": 110_000.0},
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": 1.0,
        "costs": 12.5,
        "risk_violations": risk_violations,
        "actions": [
            {
                "date": "2024-01-05",
                "ticker": "AAPL",
                "action": "buy",
                "confidence": 0.7,
                "reason": "momentum",
            }
        ],
    }
    (run / "summary.json").write_text("{bad" if invalid_json else json.dumps(summary))
    return run


def test_ranks_completed_contestants_risk_first(tmp_path: Path) -> None:
    arena = AlphaArena(CONFIG)
    payload = arena.run(
        {
            "return_first": _write_run(tmp_path, "return", total_return=0.5, risk_violations=1),
            "safe": _write_run(tmp_path, "safe", total_return=0.01, risk_violations=0),
            "drawdown": _write_run(
                tmp_path, "drawdown", total_return=0.8, max_drawdown=-0.2, risk_violations=0
            ),
        }
    )

    rows = payload["leaderboard"]
    assert [row["name"] for row in rows[:3]] == ["safe", "drawdown", "return_first"]
    assert [row["rank"] for row in rows[:3]] == [1, 2, 3]


def test_bad_artifact_is_isolated_as_failed(tmp_path: Path) -> None:
    payload = AlphaArena(CONFIG).run(
        {
            "good": _write_run(tmp_path, "good"),
            "bad": _write_run(tmp_path, "bad", invalid_json=True),
        }
    )

    rows = {row["name"]: row for row in payload["leaderboard"]}
    assert rows["good"]["status"] == "completed"
    assert rows["bad"]["status"] == "failed"
    assert rows["bad"]["error_category"] == "invalid_json"
    assert rows["bad"]["rank"] is None


def test_config_mismatch_is_isolated(tmp_path: Path) -> None:
    payload = AlphaArena(CONFIG).run(
        {"wrong": _write_run(tmp_path, "wrong", fingerprint="other")}
    )

    row = next(row for row in payload["leaderboard"] if row["name"] == "wrong")
    assert row["status"] == "failed"
    assert row["error_category"] == "config_mismatch"


def test_missing_defaults_are_absent_and_payload_is_json_ready(tmp_path: Path) -> None:
    payload = AlphaArena(CONFIG).run({"custom": _write_run(tmp_path, "custom")})

    rows = {row["name"]: row for row in payload["leaderboard"]}
    assert {"rules", "trading-agents", "finmem", "quanta-alpha"} <= rows.keys()
    defaults = ("rules", "trading-agents", "finmem", "quanta-alpha")
    assert all(rows[name]["status"] == "absent" for name in defaults)
    assert set(payload) == {"leaderboard", "equity", "action_distribution", "costs", "risk_markers"}
    assert payload["action_distribution"] == {"buy": 1, "sell": 0, "hold": 0}
    assert json.loads(json.dumps(payload)) == payload

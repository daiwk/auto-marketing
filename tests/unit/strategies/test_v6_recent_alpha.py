import json

import pandas as pd

from quant_trader.strategies.v6_recent_alpha import AlphaAgentMiner, ChainOfAlphaMiner


def _panel(days: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day, stamp in enumerate(pd.bdate_range("2024-01-02", periods=days)):
        for number, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
            speed = number + 1
            close = 100.0 + day * speed + (day % 5) * number
            rows.append(
                {
                    "date": stamp,
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1_000.0 + 10 * number + day,
                    "returns": (speed + (day % 3) * 0.1) / 1_000,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class Reviewer:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def __call__(self, prompt: str) -> str:
        self.requests.append(json.loads(prompt))
        return json.dumps(self.responses.pop(0))


def test_alpha_agent_scores_originality_complexity_and_temporal_decay() -> None:
    reviewer = Reviewer(
        [
            {
                "candidates": [
                    {"hypothesis": "短期动量持续", "expression": "rank(delta(close,3))"},
                    {"hypothesis": "重复动量假设", "expression": "rank(delta(close,3))"},
                ]
            }
        ]
    )

    result = AlphaAgentMiner(reviewer, candidate_limit=2).mine(_panel())

    assert len(reviewer.requests) == 1
    assert result["paper"]["arxiv_id"] == "2502.16789"
    assert result["champion"]["frozen"] is True
    assert "test_ic" in result["champion"]
    first, second = result["candidates"]
    assert first["originality"] == 1
    assert second["originality"] == 0
    assert second["score"] < first["score"]
    assert first["temporal_decay"] >= 0


def test_chain_of_alpha_runs_generation_then_bounded_optimization() -> None:
    reviewer = Reviewer(
        [
            {
                "candidates": [
                    {"hypothesis": "量价趋势", "expression": "rank(delta(close,2))"}
                ]
            },
            {
                "candidates": [
                    {
                        "hypothesis": "对趋势降噪",
                        "expression": "rank(delta(rolling_mean(close,3),2))",
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "hypothesis": "加入波动标准化",
                        "expression": "rank(zscore(delta(close,2),3))",
                    }
                ]
            },
        ]
    )

    result = ChainOfAlphaMiner(
        reviewer, candidate_limit=1, optimization_rounds=2
    ).mine(_panel())

    assert [request["chain"] for request in reviewer.requests] == [
        "generation",
        "optimization",
        "optimization",
    ]
    assert result["paper"]["arxiv_id"] == "2508.06312"
    assert [round_["round"] for round_ in result["rounds"]] == [0, 1, 2]
    assert result["champion"]["frozen"] is True
    assert len(result["candidates"]) == 3


def test_recent_miners_reject_unbounded_or_invalid_settings() -> None:
    reviewer = Reviewer([])

    try:
        AlphaAgentMiner(reviewer, candidate_limit=9)
    except ValueError as error:
        assert "candidate_limit" in str(error)
    else:
        raise AssertionError("candidate limit should be bounded")

    try:
        ChainOfAlphaMiner(reviewer, optimization_rounds=4)
    except ValueError as error:
        assert "optimization_rounds" in str(error)
    else:
        raise AssertionError("optimization rounds should be bounded")

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trader.strategies.v7_auditable_benchmarks import (
    KTDFinBenchmark,
    OpenPMAuditor,
)


def _payload(*, leaked: bool = False) -> dict[str, object]:
    dates = pd.bdate_range("2025-01-02", periods=30)
    equity = {stamp.date().isoformat(): 100_000 + index * 150 for index, stamp in enumerate(dates)}
    gross = {stamp.date().isoformat(): 0.5 for stamp in dates}
    execution = dates[10]
    signal = execution if leaked else dates[9]
    return {
        "runs": {
            "rules_only": {
                "equity": equity,
                "gross_exposure": gross,
                "fills": [
                    {
                        "ticker": "SPY",
                        "shares": 10,
                        "price": 105,
                        "execution_date": execution.date().isoformat(),
                        "decision_id": f"candidate:{signal.strftime('%Y%m%d')}:digest",
                    }
                ],
                "metrics": {"costs": 2.5},
            }
        }
    }


def _frames() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=30)
    frames: dict[str, pd.DataFrame] = {}
    for offset, ticker in enumerate(("SPY", "QQQ", "IWM")):
        close = 100 + np.arange(len(dates)) * (0.2 + offset * 0.08)
        frames[ticker] = pd.DataFrame({"close": close}, index=dates)
    return frames


def test_openpm_emits_auditable_certificate_and_cost_curve() -> None:
    result = OpenPMAuditor(
        max_gross_exposure=0.8,
        max_drawdown=0.15,
        max_turnover=4,
    ).run(_payload())

    assert result["paper"]["arxiv_id"] == "2608.09988"
    assert result["contamination_certificate"]["passed"] is True
    assert result["constraint_report"]["passed"] is True
    assert [row["cost_bps"] for row in result["cost_sensitivity"]] == [0, 5, 10, 20, 50]


def test_openpm_detects_same_day_signal_leakage() -> None:
    result = OpenPMAuditor(
        max_gross_exposure=0.8,
        max_drawdown=0.15,
        max_turnover=4,
    ).run(_payload(leaked=True))

    assert result["contamination_certificate"]["passed"] is False
    assert "没有早于" in result["contamination_certificate"]["violations"][0]["reason"]


def test_ktd_fin_masks_identifiers_and_attributes_returns() -> None:
    result = KTDFinBenchmark().run(_payload(), _frames())

    assert result["paper"]["arxiv_id"] == "2605.28359"
    masking = result["masking_protocol"]
    assert masking["real_identifiers_exposed_in_sample"] is False
    serialized_sample = repr(masking["fully_masked_sample"])
    assert "SPY" not in serialized_sample
    assert "2025-" not in serialized_sample
    assert set(result["attribution"]["factor_exposures"]) == {"market", "growth", "size"}
    assert np.isfinite(result["attribution"]["annualized_selection_alpha"])

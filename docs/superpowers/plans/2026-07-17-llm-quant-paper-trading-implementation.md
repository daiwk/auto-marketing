# LLM Quant Paper Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible US-equity daily backtesting and paper-trading CLI in which deterministic rules propose positions, MiniMax may only reduce or reject them, and hard risk controls own final authority.

**Architecture:** Shared data, feature, portfolio, execution, risk, storage, and reporting modules sit below a versioned `Strategy` interface. `v1_rules_llm` produces immutable `SignalIntent` values from point-in-time snapshots; the same portfolio and execution code powers historical backtests and paper trading. All external inputs are cached and every decision is auditable.

**Tech Stack:** Python 3.12, uv, Pydantic 2, pydantic-settings, pandas, NumPy, yfinance, PyArrow/Parquet, httpx, SQLite, Typer, PyYAML, Jinja2, Plotly, pytest, pytest-cov, respx, Ruff, mypy.

---

## File map

```text
pyproject.toml                         package metadata, dependencies, tool config
.env.example                          secret-free MiniMax configuration template
configs/default.yaml                  versioned strategy and risk defaults
src/quant_trader/config.py            YAML + environment settings
src/quant_trader/core/models.py        immutable cross-module contracts
src/quant_trader/core/clock.py         trading-date helpers
src/quant_trader/data/base.py          market-data protocol
src/quant_trader/data/yfinance_source.py external data adapter
src/quant_trader/data/validation.py    OHLCV invariants and freshness checks
src/quant_trader/data/cache.py         Parquet persistence
src/quant_trader/features/technical.py indicators without look-ahead
src/quant_trader/features/snapshot.py  point-in-time feature construction
src/quant_trader/llm/base.py            LLM review protocol
src/quant_trader/llm/minimax.py         MiniMax OpenAI-compatible client
src/quant_trader/llm/parsing.py         strict response parsing and repair
src/quant_trader/llm/cache.py           deterministic response cache keys
src/quant_trader/strategies/base.py     strategy protocol
src/quant_trader/strategies/v1_rules_llm/rules.py deterministic candidate model
src/quant_trader/strategies/v1_rules_llm/prompt.py versioned prompt renderer
src/quant_trader/strategies/v1_rules_llm/strategy.py rules + LLM composition
src/quant_trader/strategies/v2_multi_agent/README.md reserved TradingAgents boundary
src/quant_trader/strategies/v3_factor_mining/README.md reserved QuantaAlpha boundary
src/quant_trader/risk/engine.py         portfolio-level hard constraints
src/quant_trader/portfolio/account.py   cash, positions, fills, NAV
src/quant_trader/execution/costs.py     commission/slippage model
src/quant_trader/execution/simulator.py next-open idempotent fills
src/quant_trader/storage/database.py    SQLite schema and transactions
src/quant_trader/storage/repositories.py audit/account persistence
src/quant_trader/backtest/engine.py     chronological event loop
src/quant_trader/backtest/benchmarks.py SPY and rules-only comparisons
src/quant_trader/backtest/walk_forward.py fixed time splits
src/quant_trader/reporting/metrics.py   performance calculations
src/quant_trader/reporting/html.py      JSON and HTML artifacts
src/quant_trader/paper/service.py       one idempotent paper-trading cycle
src/quant_trader/cli.py                 user-facing commands
tests/unit/                             focused deterministic tests
tests/integration/                      offline end-to-end tests
tests/fixtures/                         fixed OHLCV and LLM responses
```

### Task 1: Package scaffold, configuration, and core contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `configs/default.yaml`
- Create: `src/quant_trader/__init__.py`
- Create: `src/quant_trader/config.py`
- Create: `src/quant_trader/core/__init__.py`
- Create: `src/quant_trader/core/models.py`
- Create: `src/quant_trader/strategies/v2_multi_agent/README.md`
- Create: `src/quant_trader/strategies/v3_factor_mining/README.md`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/core/test_models.py`

- [ ] **Step 1: Write failing configuration and model tests**

```python
# tests/unit/test_config.py
from quant_trader.config import load_settings


def test_load_settings_uses_safe_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("universe: [SPY, QQQ]\n", encoding="utf-8")
    settings = load_settings(path)
    assert settings.universe == ["SPY", "QQQ"]
    assert settings.risk.max_position_weight == 0.15
    assert settings.paper.initial_cash == 100_000
    assert settings.llm.api_key.get_secret_value() == ""
```

```python
# tests/unit/core/test_models.py
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from quant_trader.core.models import LLMReview, ReviewAction, SignalIntent


def test_llm_review_cannot_increase_weight():
    with pytest.raises(ValidationError):
        LLMReview(
            action=ReviewAction.MAINTAIN,
            weight_multiplier=1.01,
            confidence=0.5,
            thesis="trend intact",
            risks=["volatility"],
            invalidation="close below SMA200",
            input_anomalies=[],
        )


def test_signal_execution_must_follow_signal_time():
    with pytest.raises(ValidationError):
        SignalIntent(
            decision_id="v1:SPY:2026-01-02",
            ticker="SPY",
            proposed_weight=0.1,
            signal_time=datetime(2026, 1, 2, 21, tzinfo=timezone.utc),
            earliest_execution_time=datetime(2026, 1, 2, 20, tzinfo=timezone.utc),
            stop_price=500,
            invalidation="close below stop",
            reason_codes=["trend"],
            strategy_version="v1_rules_llm",
            prompt_version="v1",
            llm_cache_key="abc",
        )
```

- [ ] **Step 2: Run the tests and verify import failures**

Run: `uv run pytest tests/unit/test_config.py tests/unit/core/test_models.py -v`

Expected: FAIL because `quant_trader.config` and `quant_trader.core.models` do not exist.

- [ ] **Step 3: Create package metadata and versioned defaults**

```toml
# pyproject.toml
[project]
name = "quant-trader"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "httpx>=0.27,<1", "jinja2>=3.1,<4", "numpy>=2,<3",
  "pandas>=2.2,<3", "plotly>=6,<7", "pyarrow>=17,<22",
  "pydantic>=2.8,<3", "pydantic-settings>=2.4,<3",
  "pyyaml>=6,<7", "typer>=0.12,<1", "yfinance>=0.2.65,<1",
]

[project.optional-dependencies]
dev = ["mypy>=1.11,<2", "pytest>=8,<10", "pytest-cov>=5,<8", "respx>=0.21,<1", "ruff>=0.6,<1"]

[project.scripts]
quant-trader = "quant_trader.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py312"
```

```yaml
# configs/default.yaml
universe: [SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, GOOGL, META]
paper: {initial_cash: 100000.0}
strategy:
  max_candidates: 4
  min_average_dollar_volume: 20000000.0
  target_volatility: 0.10
risk:
  max_position_weight: 0.15
  max_gross_exposure: 0.80
  min_cash_weight: 0.20
  reduce_drawdown: 0.10
  halt_drawdown: 0.15
  atr_multiple: 2.5
execution: {slippage_bps: 10.0, commission_bps: 1.0}
llm:
  base_url: https://api.minimax.io/v1
  model: MiniMax-M2.7
  prompt_version: v1
  timeout_seconds: 30.0
  max_retries: 2
```

```dotenv
# .env.example
MINIMAX_API_KEY=
MINIMAX_BASE_URL=https://api.minimax.io/v1
MINIMAX_MODEL=MiniMax-M2.7
```

```markdown
<!-- src/quant_trader/strategies/v2_multi_agent/README.md -->
# V2 Multi-Agent Boundary

This version will implement the shared `Strategy` protocol with market, bull, bear,
trader, and risk-review roles. It is not registered in the V1 CLI.
```

```markdown
<!-- src/quant_trader/strategies/v3_factor_mining/README.md -->
# V3 Factor-Mining Boundary

This version will produce validated factors through a restricted DSL and AST checks.
It is not registered in the V1 CLI.
```

- [ ] **Step 4: Implement strict settings and immutable contracts**

```python
# src/quant_trader/config.py
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class RiskSettings(BaseModel):
    max_position_weight: float = Field(0.15, gt=0, le=1)
    max_gross_exposure: float = Field(0.80, gt=0, le=1)
    min_cash_weight: float = Field(0.20, ge=0, lt=1)
    reduce_drawdown: float = Field(0.10, gt=0, lt=1)
    halt_drawdown: float = Field(0.15, gt=0, lt=1)
    atr_multiple: float = Field(2.5, gt=0)


class PaperSettings(BaseModel):
    initial_cash: float = Field(100_000, gt=0)


class StrategySettings(BaseModel):
    max_candidates: int = Field(4, gt=0)
    min_average_dollar_volume: float = Field(20_000_000, gt=0)
    target_volatility: float = Field(0.10, gt=0)


class ExecutionSettings(BaseModel):
    slippage_bps: float = Field(10, ge=0)
    commission_bps: float = Field(1, ge=0)


class LLMSettings(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.minimax.io/v1"
    model: str = "MiniMax-M2.7"
    prompt_version: str = "v1"
    timeout_seconds: float = Field(30, gt=0)
    max_retries: int = Field(2, ge=0, le=5)


class MiniMaxEnvironment(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    api_key: SecretStr = Field(SecretStr(""), validation_alias="MINIMAX_API_KEY")
    base_url: str | None = Field(None, validation_alias="MINIMAX_BASE_URL")
    model: str | None = Field(None, validation_alias="MINIMAX_MODEL")


class Settings(BaseModel):
    universe: list[str]
    paper: PaperSettings = PaperSettings()
    strategy: StrategySettings = StrategySettings()
    risk: RiskSettings = RiskSettings()
    execution: ExecutionSettings = ExecutionSettings()
    llm: LLMSettings = LLMSettings()


def load_settings(path: Path | str) -> Settings:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    llm = dict(raw.get("llm", {}))
    env = MiniMaxEnvironment()
    llm["api_key"] = env.api_key
    if env.base_url is not None:
        llm["base_url"] = env.base_url
    if env.model is not None:
        llm["model"] = env.model
    raw["llm"] = llm
    return Settings.model_validate(raw)
```

```python
# src/quant_trader/core/models.py
from datetime import date, datetime
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReviewAction(StrEnum):
    MAINTAIN = "maintain"
    REDUCE = "reduce"
    REJECT = "reject"


class LLMReview(BaseModel):
    model_config = ConfigDict(frozen=True)
    action: ReviewAction
    weight_multiplier: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    thesis: str = Field(min_length=1, max_length=1000)
    risks: list[str]
    invalidation: str = Field(min_length=1, max_length=500)
    input_anomalies: list[str]


class SignalIntent(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision_id: str
    ticker: str
    proposed_weight: float = Field(ge=0, le=1)
    signal_time: datetime
    earliest_execution_time: datetime
    stop_price: float = Field(gt=0)
    invalidation: str
    reason_codes: list[str]
    strategy_version: str
    prompt_version: str
    llm_cache_key: str

    @model_validator(mode="after")
    def execution_follows_signal(self):
        if self.earliest_execution_time <= self.signal_time:
            raise ValueError("execution must be after signal")
        return self


class ApprovedOrder(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision_id: str
    ticker: str
    target_weight: float = Field(ge=0, le=1)
    execution_date: date
    reason_codes: list[str]
```

- [ ] **Step 5: Run focused tests, lint, and commit**

Run: `uv sync --extra dev && uv run pytest tests/unit/test_config.py tests/unit/core/test_models.py -v && uv run ruff check .`

Expected: all tests PASS and Ruff reports no errors.

```bash
git add pyproject.toml .env.example configs src/quant_trader tests/unit
git commit -m "chore: scaffold quant trading package"
```

### Task 2: Validated point-in-time market data and Parquet cache

**Files:**
- Create: `src/quant_trader/data/__init__.py`
- Create: `src/quant_trader/data/base.py`
- Create: `src/quant_trader/data/validation.py`
- Create: `src/quant_trader/data/cache.py`
- Create: `src/quant_trader/data/yfinance_source.py`
- Test: `tests/unit/data/test_validation.py`
- Test: `tests/unit/data/test_cache.py`

- [ ] **Step 1: Write tests for OHLCV invariants and cache round trips**

```python
# tests/unit/data/test_validation.py
import pandas as pd
import pytest
from datetime import date
from quant_trader.data.validation import DataValidationError, assert_fresh, validate_ohlcv


def test_rejects_high_below_low():
    frame = pd.DataFrame(
        {"open": [10.0], "high": [9.0], "low": [11.0], "close": [10.0], "volume": [100]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    with pytest.raises(DataValidationError, match="price relationship"):
        validate_ohlcv(frame, "SPY")


def test_rejects_duplicate_dates(valid_ohlcv):
    duplicate = pd.concat([valid_ohlcv, valid_ohlcv.iloc[[-1]]])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_ohlcv(duplicate, "SPY")


def test_rejects_stale_market_date(valid_ohlcv):
    with pytest.raises(DataValidationError, match="latest market date"):
        assert_fresh(valid_ohlcv, date(2099, 1, 1), "SPY")
```

```python
# tests/unit/data/test_cache.py
import json
import pandas.testing as pdt
from quant_trader.data.cache import ParquetMarketCache


def test_cache_round_trip(tmp_path, valid_ohlcv):
    cache = ParquetMarketCache(tmp_path)
    cache.write("SPY", valid_ohlcv)
    pdt.assert_frame_equal(cache.read("SPY"), valid_ohlcv, check_freq=False)
    metadata = json.loads(cache.path_for("SPY").with_suffix(".json").read_text())
    assert metadata["ticker"] == "SPY"
    assert metadata["max_market_date"] == valid_ohlcv.index[-1].date().isoformat()
```

- [ ] **Step 2: Run tests and verify missing-module failures**

Run: `uv run pytest tests/unit/data -v`

Expected: FAIL because data modules do not exist.

- [ ] **Step 3: Implement the protocol, validation, and atomic cache**

```python
# src/quant_trader/data/base.py
from datetime import date
from typing import Protocol
import pandas as pd


class MarketDataSource(Protocol):
    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...
```

```python
# src/quant_trader/data/validation.py
import pandas as pd


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    pass


def validate_ohlcv(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DataValidationError(f"{ticker}: missing columns {sorted(missing)}")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise DataValidationError(f"{ticker}: dates are unordered or duplicate")
    values = frame[REQUIRED_COLUMNS]
    if values.isna().any().any() or (values <= 0).any().any():
        raise DataValidationError(f"{ticker}: non-positive or missing values")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
        (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
        raise DataValidationError(f"{ticker}: invalid price relationship")
    return frame[REQUIRED_COLUMNS].astype(float)


def assert_fresh(frame: pd.DataFrame, expected_market_date, ticker: str) -> None:
    latest = frame.index[-1].date()
    if latest != expected_market_date:
        raise DataValidationError(
            f"{ticker}: latest market date {latest} does not match {expected_market_date}"
        )
```

```python
# src/quant_trader/data/cache.py
from pathlib import Path
import os
import json
from datetime import datetime, timezone
import pandas as pd
from quant_trader.data.validation import validate_ohlcv


class ParquetMarketCache:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, ticker: str) -> Path:
        return self.root / "market" / f"{ticker.upper()}.parquet"

    def write(self, ticker: str, frame: pd.DataFrame,
              retrieved_at: datetime | None = None) -> None:
        clean = validate_ohlcv(frame, ticker)
        path = self.path_for(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.parquet")
        clean.to_parquet(temporary)
        os.replace(temporary, path)
        metadata = {
            "ticker": ticker.upper(),
            "retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat(),
            "max_market_date": clean.index[-1].date().isoformat(),
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )

    def read(self, ticker: str) -> pd.DataFrame:
        return validate_ohlcv(pd.read_parquet(self.path_for(ticker)), ticker)
```

- [ ] **Step 4: Implement the isolated yfinance adapter**

```python
# src/quant_trader/data/yfinance_source.py
from datetime import date
import pandas as pd
import yfinance as yf
from quant_trader.data.validation import validate_ohlcv


class YFinanceSource:
    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return validate_ohlcv(frame, ticker)
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/unit/data -v && uv run ruff check src/quant_trader/data tests/unit/data`

Expected: all data tests PASS.

```bash
git add src/quant_trader/data tests/unit/data tests/conftest.py
git commit -m "feat: add validated market data cache"
```

### Task 3: Technical features and deterministic candidate rules

**Files:**
- Create: `src/quant_trader/features/__init__.py`
- Create: `src/quant_trader/features/technical.py`
- Create: `src/quant_trader/features/snapshot.py`
- Create: `src/quant_trader/strategies/__init__.py`
- Create: `src/quant_trader/strategies/base.py`
- Create: `src/quant_trader/strategies/v1_rules_llm/__init__.py`
- Create: `src/quant_trader/strategies/v1_rules_llm/rules.py`
- Test: `tests/unit/features/test_technical.py`
- Test: `tests/unit/strategies/test_v1_rules.py`

- [ ] **Step 1: Write no-look-ahead and candidate-ranking tests**

```python
# tests/unit/features/test_technical.py
import pandas.testing as pdt
from quant_trader.features.technical import technical_features


def test_future_prices_do_not_change_past_features(valid_ohlcv):
    original = technical_features(valid_ohlcv)
    changed = valid_ohlcv.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 10
    revised = technical_features(changed)
    pdt.assert_series_equal(original.iloc[-2], revised.iloc[-2])
```

```python
# tests/unit/strategies/test_v1_rules.py
from quant_trader.strategies.v1_rules_llm.rules import rank_candidates


def test_rank_candidates_filters_ineligible_and_limits_count(feature_rows):
    ranked = rank_candidates(feature_rows, max_candidates=2, min_dollar_volume=20_000_000)
    assert [candidate.ticker for candidate in ranked] == ["SPY", "QQQ"]
    assert all(candidate.base_weight <= 0.15 for candidate in ranked)
```

- [ ] **Step 2: Run tests and verify failures**

Run: `uv run pytest tests/unit/features tests/unit/strategies/test_v1_rules.py -v`

Expected: FAIL because feature and rule functions do not exist.

- [ ] **Step 3: Implement indicators using past and current rows only**

```python
# src/quant_trader/features/technical.py
import numpy as np
import pandas as pd


def technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    previous_close = close.shift(1)
    true_range = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    result = frame.copy()
    result["sma_200"] = close.rolling(200, min_periods=200).mean()
    result["return_20"] = close.pct_change(20)
    result["return_60"] = close.pct_change(60)
    result["return_120"] = close.pct_change(120)
    result["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    result["atr_14"] = true_range.rolling(14).mean()
    result["average_dollar_volume_20"] = (close * frame["volume"]).rolling(20).mean()
    return result
```

- [ ] **Step 4: Implement explicit eligibility and scoring**

```python
# src/quant_trader/strategies/v1_rules_llm/rules.py
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Candidate:
    ticker: str
    score: float
    annualized_volatility: float
    atr_14: float
    close: float
    base_weight: float = 0.0


def rank_candidates(rows: Mapping[str, Mapping[str, float]], max_candidates: int,
                    min_dollar_volume: float, target_volatility: float = 0.10) -> list[Candidate]:
    eligible: list[Candidate] = []
    for ticker, row in rows.items():
        if not (row["close"] > row["sma_200"] and row["return_20"] > 0 and
                row["return_60"] > 0 and
                row["average_dollar_volume_20"] >= min_dollar_volume and
                row["volatility_20"] > 0):
            continue
        momentum = 0.2 * row["return_20"] + 0.5 * row["return_60"] + 0.3 * row["return_120"]
        eligible.append(Candidate(ticker, momentum / row["volatility_20"],
                                  row["volatility_20"], row["atr_14"], row["close"]))
    selected = sorted(eligible, key=lambda item: item.score, reverse=True)[:max_candidates]
    inverse_vol_sum = sum(1 / item.annualized_volatility for item in selected)
    raw = [0.8 * (1 / item.annualized_volatility) / inverse_vol_sum for item in selected]
    estimated_volatility = sum(
        (weight * item.annualized_volatility) ** 2
        for weight, item in zip(raw, selected, strict=True)
    ) ** 0.5
    scale = min(1.0, target_volatility / estimated_volatility) if estimated_volatility else 0
    return [Candidate(**{**item.__dict__, "base_weight": min(0.15, weight * scale)})
            for item, weight in zip(selected, raw, strict=True)]
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/unit/features tests/unit/strategies/test_v1_rules.py -v`

Expected: all feature and rule tests PASS.

```bash
git add src/quant_trader/features src/quant_trader/strategies tests/unit/features tests/unit/strategies
git commit -m "feat: add point-in-time candidate rules"
```

### Task 4: MiniMax provider, strict parser, and deterministic cache

**Files:**
- Create: `src/quant_trader/llm/__init__.py`
- Create: `src/quant_trader/llm/base.py`
- Create: `src/quant_trader/llm/cache.py`
- Create: `src/quant_trader/llm/parsing.py`
- Create: `src/quant_trader/llm/minimax.py`
- Create: `src/quant_trader/strategies/v1_rules_llm/prompt.py`
- Test: `tests/unit/llm/test_cache.py`
- Test: `tests/unit/llm/test_parsing.py`
- Test: `tests/unit/llm/test_minimax.py`

- [ ] **Step 1: Write cache, parser, and HTTP contract tests**

```python
# tests/unit/llm/test_parsing.py
from quant_trader.core.models import ReviewAction
from quant_trader.llm.parsing import parse_review


def test_parse_review_accepts_json_fenced_by_model():
    review = parse_review('```json\n{"action":"reduce","weight_multiplier":0.5,'
                          '"confidence":0.6,"thesis":"trend","risks":["vol"],'
                          '"invalidation":"SMA break","input_anomalies":[]}\n```')
    assert review.action is ReviewAction.REDUCE
    assert review.weight_multiplier == 0.5
```

```python
# tests/unit/llm/test_minimax.py
import httpx
import respx
from quant_trader.llm.minimax import MiniMaxReviewer


@respx.mock
def test_minimax_sends_bearer_token_and_returns_content():
    route = respx.post("https://api.minimax.io/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    client = MiniMaxReviewer("secret", "https://api.minimax.io/v1", "MiniMax-M2.7", 1, 0)
    assert client.complete([{"role": "user", "content": "review"}]) == "{}"
    assert route.calls[0].request.headers["authorization"] == "Bearer secret"
```

- [ ] **Step 2: Run tests and verify missing-module failures**

Run: `uv run pytest tests/unit/llm -v`

Expected: FAIL because LLM modules do not exist.

- [ ] **Step 3: Implement canonical cache keys and strict parsing**

```python
# src/quant_trader/llm/cache.py
import hashlib
import json


def review_cache_key(model: str, prompt_version: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"model": model, "prompt_version": prompt_version,
                          "messages": messages}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
```

```python
# src/quant_trader/llm/parsing.py
import json
from quant_trader.core.models import LLMReview


def parse_review(content: str) -> LLMReview:
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    return LLMReview.model_validate(json.loads(stripped))
```

- [ ] **Step 4: Implement bounded HTTP retries without logging secrets**

```python
# src/quant_trader/llm/minimax.py
import time
import httpx


class MiniMaxReviewer:
    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout_seconds: float, max_retries: int):
        self.model = model
        self.max_retries = max_retries
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds,
                                   headers={"Authorization": f"Bearer {api_key}"})

    def complete(self, messages: list[dict[str, str]]) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post("/chat/completions", json={
                    "model": self.model, "messages": messages,
                    "temperature": 0.1, "max_completion_tokens": 1200,
                })
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.HTTPStatusError):
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
```

- [ ] **Step 5: Verify retry/error scenarios and commit**

Run: `uv run pytest tests/unit/llm -v && uv run ruff check src/quant_trader/llm tests/unit/llm`

Expected: normal response, timeout retry, terminal failure, invalid JSON, and cache tests all PASS.

```bash
git add src/quant_trader/llm src/quant_trader/strategies/v1_rules_llm/prompt.py tests/unit/llm
git commit -m "feat: add constrained MiniMax reviewer"
```

### Task 5: Compose V1 rules and LLM reviews into signal intents

**Files:**
- Create: `src/quant_trader/strategies/base.py`
- Create: `src/quant_trader/features/snapshot.py`
- Create: `src/quant_trader/strategies/v1_rules_llm/strategy.py`
- Test: `tests/unit/strategies/test_v1_strategy.py`

- [ ] **Step 1: Write tests proving the LLM cannot add or enlarge positions**

```python
# tests/unit/strategies/test_v1_strategy.py
def test_review_can_only_reduce_candidate_weight(strategy, market_snapshot, maintain_review):
    intents = strategy.generate(market_snapshot, reviews={"SPY": maintain_review})
    assert intents[0].ticker == "SPY"
    assert intents[0].proposed_weight <= market_snapshot.candidates["SPY"].base_weight


def test_rejected_candidate_has_zero_target(strategy, market_snapshot, reject_review):
    intents = strategy.generate(market_snapshot, reviews={"SPY": reject_review})
    assert intents[0].proposed_weight == 0


def test_unknown_review_ticker_is_ignored(strategy, market_snapshot, maintain_review):
    intents = strategy.generate(market_snapshot, reviews={"UNLISTED": maintain_review})
    assert all(intent.ticker != "UNLISTED" for intent in intents)
```

- [ ] **Step 2: Run tests and verify failures**

Run: `uv run pytest tests/unit/strategies/test_v1_strategy.py -v`

Expected: FAIL because `V1RulesLLMStrategy.generate` does not exist.

- [ ] **Step 3: Implement the strategy protocol and conservative composition**

```python
# src/quant_trader/strategies/base.py
from typing import Protocol
from quant_trader.core.models import SignalIntent


class Strategy(Protocol):
    version: str
    def generate(self, snapshot, reviews) -> list[SignalIntent]: ...
```

```python
# src/quant_trader/strategies/v1_rules_llm/strategy.py
from datetime import datetime
from quant_trader.core.models import ReviewAction, SignalIntent


class V1RulesLLMStrategy:
    version = "v1_rules_llm"

    def __init__(self, prompt_version: str):
        self.prompt_version = prompt_version

    def generate(self, snapshot, reviews) -> list[SignalIntent]:
        intents = []
        for ticker, candidate in snapshot.candidates.items():
            review = reviews.get(ticker)
            multiplier = 0.0 if review is None or review.action is ReviewAction.REJECT else review.weight_multiplier
            target = min(candidate.base_weight, candidate.base_weight * multiplier)
            intents.append(SignalIntent(
                decision_id=f"{self.version}:{ticker}:{snapshot.as_of.date().isoformat()}",
                ticker=ticker, proposed_weight=target, signal_time=snapshot.as_of,
                earliest_execution_time=snapshot.next_open,
                stop_price=candidate.close - 2.5 * candidate.atr_14,
                invalidation=review.invalidation if review else "LLM review unavailable",
                reason_codes=["rules_candidate", review.action.value if review else "review_failed"],
                strategy_version=self.version, prompt_version=self.prompt_version,
                llm_cache_key=snapshot.review_cache_keys.get(ticker, ""),
            ))
        return intents
```

- [ ] **Step 4: Add one repair attempt and fail-closed orchestration test**

```python
# append to tests/unit/strategies/test_v1_strategy.py
from quant_trader.core.models import ReviewAction
from quant_trader.strategies.v1_rules_llm.strategy import review_candidate


class StubReviewer:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return next(self.responses)


def test_review_candidate_repairs_once():
    reviewer = StubReviewer(["not-json", '{"action":"reduce","weight_multiplier":0.5,'
        '"confidence":0.5,"thesis":"trend","risks":[],"invalidation":"break",'
        '"input_anomalies":[]}'])
    result = review_candidate(reviewer, [{"role": "user", "content": "review"}])
    assert reviewer.calls == 2
    assert result.action is ReviewAction.REDUCE


def test_review_candidate_fails_closed_after_repair():
    reviewer = StubReviewer(["bad", "still bad"])
    result = review_candidate(reviewer, [{"role": "user", "content": "review"}])
    assert reviewer.calls == 2
    assert result.action is ReviewAction.REJECT
    assert result.weight_multiplier == 0
```

```python
# append to src/quant_trader/strategies/v1_rules_llm/strategy.py
from pydantic import ValidationError
from quant_trader.core.models import LLMReview
from quant_trader.llm.parsing import parse_review


def review_candidate(reviewer, messages: list[dict[str, str]]) -> LLMReview:
    first = reviewer.complete(messages)
    try:
        return parse_review(first)
    except (ValueError, ValidationError):
        repair = reviewer.complete([
            *messages,
            {"role": "assistant", "content": first},
            {"role": "user", "content": "Return only one valid LLMReview JSON object."},
        ])
        try:
            return parse_review(repair)
        except (ValueError, ValidationError):
            return LLMReview(
                action=ReviewAction.REJECT, weight_multiplier=0, confidence=0,
                thesis="review unavailable", risks=["invalid model output"],
                invalidation="review unavailable", input_anomalies=["parse failure"],
            )
```

Run: `uv run pytest tests/unit/strategies/test_v1_strategy.py -v`

Expected: both repair and fail-closed cases PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/features/snapshot.py src/quant_trader/strategies tests/unit/strategies
git commit -m "feat: compose rules and llm signal intents"
```

### Task 6: Portfolio accounting, costs, simulator, and SQLite audit trail

**Files:**
- Create: `src/quant_trader/portfolio/__init__.py`
- Create: `src/quant_trader/portfolio/account.py`
- Create: `src/quant_trader/execution/__init__.py`
- Create: `src/quant_trader/execution/costs.py`
- Create: `src/quant_trader/execution/simulator.py`
- Create: `src/quant_trader/storage/__init__.py`
- Create: `src/quant_trader/storage/database.py`
- Create: `src/quant_trader/storage/repositories.py`
- Test: `tests/unit/portfolio/test_account.py`
- Test: `tests/unit/execution/test_simulator.py`
- Test: `tests/unit/storage/test_database.py`

- [ ] **Step 1: Write cash conservation and duplicate-order tests**

```python
# tests/unit/execution/test_simulator.py
from datetime import date
from quant_trader.core.models import ApprovedOrder


def test_buy_fill_conserves_account_value(simulator, account):
    order = ApprovedOrder(decision_id="d1", ticker="SPY", target_weight=0.5,
                          execution_date=date(2026, 1, 5), reason_codes=["trend"])
    fill = simulator.execute(order, open_price=100, account=account)
    assert account.cash + account.positions["SPY"].quantity * fill.price + fill.total_cost == 100_000


def test_duplicate_decision_id_is_not_filled_twice(simulator, account):
    order = ApprovedOrder(decision_id="d1", ticker="SPY", target_weight=0.1,
                          execution_date=date(2026, 1, 5), reason_codes=["trend"])
    assert simulator.execute(order, 100, account) is not None
    assert simulator.execute(order, 100, account) is None
```

- [ ] **Step 2: Run tests and verify failures**

Run: `uv run pytest tests/unit/portfolio tests/unit/execution tests/unit/storage -v`

Expected: FAIL because portfolio, execution, and storage modules do not exist.

- [ ] **Step 3: Implement account and cost math using fractional paper shares**

```python
# src/quant_trader/execution/costs.py
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    slippage_bps: float = 10
    commission_bps: float = 1

    def fill_price(self, open_price: float, is_buy: bool) -> float:
        direction = 1 if is_buy else -1
        return open_price * (1 + direction * self.slippage_bps / 10_000)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps / 10_000
```

```python
# src/quant_trader/portfolio/account.py
from dataclasses import dataclass, field


@dataclass
class Position:
    quantity: float = 0
    last_price: float = 0
    highest_close: float = 0
    average_cost: float = 0


@dataclass
class Account:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    high_water_mark: float = 0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(position.quantity * prices.get(ticker, position.last_price)
                               for ticker, position in self.positions.items())
```

```python
# src/quant_trader/execution/simulator.py
from dataclasses import dataclass
from datetime import datetime, timezone
from quant_trader.portfolio.account import Position


@dataclass(frozen=True)
class Fill:
    decision_id: str
    ticker: str
    quantity: float
    price: float
    commission: float
    realized_pnl: float
    filled_at: datetime

    @property
    def total_cost(self) -> float:
        return self.commission


class Simulator:
    def __init__(self, costs, repository=None):
        self.costs = costs
        self.repository = repository
        self.executed: set[str] = set()

    def execute(self, order, open_price: float, account):
        if order.decision_id in self.executed:
            return None
        current = account.positions.get(order.ticker, Position())
        equity = account.cash + sum(
            position.quantity * (open_price if ticker == order.ticker else position.last_price)
            for ticker, position in account.positions.items()
        )
        target_quantity = equity * order.target_weight / open_price
        delta = target_quantity - current.quantity
        price = self.costs.fill_price(open_price, delta >= 0)
        commission = self.costs.commission(delta * price)
        if delta > 0 and delta * price + commission > account.cash:
            delta = max(0, (account.cash - commission) / price)
        realized_pnl = max(-delta, 0) * (price - current.average_cost)
        new_quantity = current.quantity + delta
        average_cost = (
            (current.quantity * current.average_cost + delta * price) / new_quantity
            if delta > 0 and new_quantity > 0 else current.average_cost
        )
        account.cash -= delta * price + commission
        account.positions[order.ticker] = Position(
            quantity=new_quantity, last_price=price,
            highest_close=max(current.highest_close, price), average_cost=average_cost,
        )
        fill = Fill(order.decision_id, order.ticker, delta, price, commission, realized_pnl,
                    datetime.now(timezone.utc))
        if self.repository is None or self.repository.save_fill_once(order, fill):
            self.executed.add(order.decision_id)
            return fill
        return None
```

- [ ] **Step 4: Implement SQLite schema and transactional idempotency**

```sql
-- executed by src/quant_trader/storage/database.py
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, ticker TEXT NOT NULL,
  signal_time TEXT NOT NULL, input_json TEXT NOT NULL, raw_llm_output TEXT,
  review_json TEXT, risk_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_cache(
  cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  input_json TEXT NOT NULL, raw_output TEXT NOT NULL, review_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders(
  order_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL UNIQUE, status TEXT NOT NULL,
  target_weight REAL NOT NULL, execution_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills(
  fill_id TEXT PRIMARY KEY, order_id TEXT NOT NULL UNIQUE, ticker TEXT NOT NULL,
  quantity REAL NOT NULL, price REAL NOT NULL, commission REAL NOT NULL,
  realized_pnl REAL NOT NULL,
  filled_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS account_snapshots(
  run_id TEXT NOT NULL, as_of TEXT NOT NULL, cash REAL NOT NULL,
  equity REAL NOT NULL, positions_json TEXT NOT NULL,
  PRIMARY KEY(run_id, as_of)
);
```

```python
# src/quant_trader/storage/database.py
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, path: Path, schema: str):
        self.path = path
        self.schema = schema

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(self.schema)

    @contextmanager
    def transaction(self):
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
```

```python
# src/quant_trader/storage/repositories.py
import json
import sqlite3
import uuid


class ExecutionRepository:
    def __init__(self, database):
        self.database = database

    def save_fill_once(self, order, fill) -> bool:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO orders(order_id, decision_id, status, target_weight, execution_date) "
                "VALUES (?, ?, 'filled', ?, ?)",
                (str(uuid.uuid4()), order.decision_id, order.target_weight,
                 order.execution_date.isoformat()),
            )
            if cursor.rowcount == 0:
                return False
            order_id = connection.execute(
                "SELECT order_id FROM orders WHERE decision_id = ?", (order.decision_id,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO fills(fill_id, order_id, ticker, quantity, price, commission, realized_pnl, filled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), order_id, fill.ticker, fill.quantity, fill.price,
                 fill.commission, fill.realized_pnl, fill.filled_at.isoformat()),
            )
            return True


class LLMCacheRepository:
    def __init__(self, database):
        self.database = database

    def get(self, cache_key: str):
        with sqlite3.connect(self.database.path) as connection:
            row = connection.execute(
                "SELECT raw_output, review_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return None if row is None else {"raw_output": row[0], "review": json.loads(row[1])}

    def put(self, cache_key, model, prompt_version, input_payload, raw_output, review) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                (cache_key, model, prompt_version,
                 json.dumps(input_payload, sort_keys=True), raw_output,
                 review.model_dump_json()),
            )
```

- [ ] **Step 5: Verify invariants and commit**

Run: `uv run pytest tests/unit/portfolio tests/unit/execution tests/unit/storage -v`

Expected: accounting conservation, costs, transaction rollback, and duplicate order tests PASS.

```bash
git add src/quant_trader/portfolio src/quant_trader/execution src/quant_trader/storage tests/unit/portfolio tests/unit/execution tests/unit/storage
git commit -m "feat: add auditable paper execution ledger"
```

### Task 7: Hard risk engine and circuit breaker

**Files:**
- Create: `src/quant_trader/risk/__init__.py`
- Create: `src/quant_trader/risk/engine.py`
- Test: `tests/unit/risk/test_engine.py`

- [ ] **Step 1: Write risk-invariant tests**

```python
# tests/unit/risk/test_engine.py
def test_caps_position_and_gross_exposure(risk_engine, account, intents):
    orders = risk_engine.approve(intents, account, prices={"SPY": 100, "QQQ": 100})
    assert all(order.target_weight <= 0.15 for order in orders)
    assert sum(order.target_weight for order in orders) <= 0.80


def test_ten_percent_drawdown_halves_new_targets(risk_engine, account, one_intent):
    account.high_water_mark = 100_000
    account.cash = 90_000
    order = risk_engine.approve([one_intent], account, prices={})[0]
    assert order.target_weight == one_intent.proposed_weight * 0.5


def test_fifteen_percent_drawdown_halts_new_risk(risk_engine, account, one_intent):
    account.high_water_mark = 100_000
    account.cash = 85_000
    assert risk_engine.approve([one_intent], account, prices={})[0].target_weight == 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/unit/risk/test_engine.py -v`

Expected: FAIL because `RiskEngine` does not exist.

- [ ] **Step 3: Implement fail-closed risk approval**

```python
# src/quant_trader/risk/engine.py
from datetime import date
from quant_trader.core.models import ApprovedOrder


class RiskEngine:
    def __init__(self, max_position: float = 0.15, max_gross: float = 0.80,
                 reduce_drawdown: float = 0.10, halt_drawdown: float = 0.15):
        self.max_position = max_position
        self.max_gross = max_gross
        self.reduce_drawdown = reduce_drawdown
        self.halt_drawdown = halt_drawdown
        self.halted = False

    def approve(self, intents, account, prices, execution_date: date | None = None):
        equity = account.equity(prices)
        high = max(account.high_water_mark, equity)
        account.high_water_mark = high
        drawdown = 0 if high == 0 else 1 - equity / high
        if drawdown >= self.halt_drawdown:
            self.halted = True
        factor = 0 if self.halted else 0.5 if drawdown >= self.reduce_drawdown else 1
        remaining = self.max_gross
        orders = []
        for intent in sorted(intents, key=lambda item: item.ticker):
            target = min(intent.proposed_weight * factor, self.max_position, remaining)
            remaining -= target
            orders.append(ApprovedOrder(
                decision_id=intent.decision_id, ticker=intent.ticker,
                target_weight=max(0, target),
                execution_date=execution_date or intent.earliest_execution_time.date(),
                reason_codes=[*intent.reason_codes, f"drawdown:{drawdown:.4f}"],
            ))
        return orders

    def liquidation_orders(self, account, execution_date: date) -> list[ApprovedOrder]:
        if not self.halted:
            return []
        return [ApprovedOrder(
            decision_id=f"circuit-breaker:{ticker}:{execution_date.isoformat()}",
            ticker=ticker, target_weight=0, execution_date=execution_date,
            reason_codes=["circuit_breaker_liquidation"],
        ) for ticker, position in account.positions.items() if position.quantity > 0]

    def trailing_stop_orders(self, account, closes, atr_values,
                             execution_date: date) -> list[ApprovedOrder]:
        orders = []
        for ticker, position in account.positions.items():
            close = closes[ticker]
            position.highest_close = max(position.highest_close, close)
            stop = position.highest_close - 2.5 * atr_values[ticker]
            if position.quantity > 0 and close <= stop:
                orders.append(ApprovedOrder(
                    decision_id=f"trailing-stop:{ticker}:{execution_date.isoformat()}",
                    ticker=ticker, target_weight=0, execution_date=execution_date,
                    reason_codes=["trailing_stop"],
                ))
        return orders

    def reset_circuit_breaker(self) -> None:
        self.halted = False
```

- [ ] **Step 4: Add property-style boundary cases and verify**

```python
# append to tests/unit/risk/test_engine.py
import pytest
from datetime import date


@pytest.mark.parametrize("weight", [0, 0.15, 0.16, 1.0])
@pytest.mark.parametrize("drawdown", [0, 0.0999, 0.10, 0.1499, 0.15])
def test_risk_boundaries_never_exceed_limits(risk_engine, account, intent_factory,
                                             weight, drawdown):
    account.high_water_mark = 100_000
    account.cash = 100_000 * (1 - drawdown)
    orders = risk_engine.approve([intent_factory(weight=weight) for _ in range(9)], account, prices={})
    assert all(0 <= order.target_weight <= 0.15 for order in orders)
    assert sum(order.target_weight for order in orders) <= 0.80


def test_halt_is_latched_until_explicit_reset(risk_engine, invested_account):
    invested_account.high_water_mark = 100_000
    risk_engine.approve([], invested_account, {"SPY": 80})
    exits = risk_engine.liquidation_orders(invested_account, date(2026, 1, 5))
    assert exits[0].target_weight == 0
    risk_engine.reset_circuit_breaker()
    assert risk_engine.liquidation_orders(invested_account, date(2026, 1, 6)) == []


def test_trailing_stop_uses_highest_close(risk_engine, invested_account):
    invested_account.positions["SPY"].highest_close = 110
    exits = risk_engine.trailing_stop_orders(
        invested_account, {"SPY": 100}, {"SPY": 3}, date(2026, 1, 5)
    )
    assert exits[0].reason_codes == ["trailing_stop"]
```

Run: `uv run pytest tests/unit/risk/test_engine.py -v`

Expected: all boundary tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/risk tests/unit/risk
git commit -m "feat: enforce portfolio risk circuit breakers"
```

### Task 8: Chronological backtest, benchmarks, and walk-forward splits

**Files:**
- Create: `src/quant_trader/core/clock.py`
- Create: `src/quant_trader/backtest/__init__.py`
- Create: `src/quant_trader/backtest/engine.py`
- Create: `src/quant_trader/backtest/benchmarks.py`
- Create: `src/quant_trader/backtest/walk_forward.py`
- Test: `tests/unit/backtest/test_engine.py`
- Test: `tests/unit/backtest/test_walk_forward.py`

- [ ] **Step 1: Write tests proving next-open execution and fixed splits**

```python
# tests/unit/backtest/test_engine.py
def test_signal_from_close_fills_only_at_next_open(backtest_fixture):
    result = backtest_fixture.run()
    fill = result.fills[0]
    assert fill.signal_date.isoformat() == "2026-01-02"
    assert fill.execution_date.isoformat() == "2026-01-05"
    assert fill.reference_price == backtest_fixture.bars.loc["2026-01-05", "open"]
```

```python
# tests/unit/backtest/test_walk_forward.py
from datetime import date
from quant_trader.backtest.walk_forward import default_periods


def test_default_periods_do_not_overlap():
    periods = default_periods(date(2026, 7, 17))
    assert periods.development.end < periods.validation.start
    assert periods.validation.end < periods.test.start
    assert periods.test.end == date(2026, 7, 16)
```

- [ ] **Step 2: Run tests and verify failures**

Run: `uv run pytest tests/unit/backtest -v`

Expected: FAIL because backtest modules do not exist.

- [ ] **Step 3: Implement the event order explicitly**

```python
# src/quant_trader/backtest/engine.py
from quant_trader.core.clock import is_week_end


class BacktestEngine:
    def run(self, dates):
        for index, trading_date in enumerate(dates):
            self.execution.fill_pending_at_open(trading_date)
            self.portfolio.mark_to_close(trading_date)
            close_prices = self.features.close_prices(trading_date)
            self.risk.approve([], self.portfolio.account, close_prices)
            if index + 1 < len(dates):
                next_date = dates[index + 1].date()
                exits = self.risk.trailing_stop_orders(
                    self.portfolio.account, close_prices,
                    self.features.atr_values(trading_date), next_date,
                )
                exits += self.risk.liquidation_orders(self.portfolio.account, next_date)
                self.execution.queue(exits)
            if is_week_end(dates, index):
                snapshot = self.features.build(as_of=trading_date)
                intents = self.strategy.generate_from_snapshot(snapshot)
                orders = self.risk.approve(intents, self.portfolio.account,
                                           close_prices)
                self.execution.queue(orders)
            self.storage.save_account_snapshot(trading_date, self.portfolio.account)
        return self.result()
```

```python
# src/quant_trader/core/clock.py
def is_week_end(dates, index: int) -> bool:
    if index == len(dates) - 1:
        return True
    current = dates[index].isocalendar()
    following = dates[index + 1].isocalendar()
    return (current.year, current.week) != (following.year, following.week)
```

The test fixtures supply concrete fake objects for every dependency. Do not allow strategy code to access bars after `trading_date`; the feature store receives a frame sliced through that date.

- [ ] **Step 4: Implement exact periods and three comparison runs**

```python
# src/quant_trader/backtest/walk_forward.py
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Period:
    start: date
    end: date


@dataclass(frozen=True)
class ValidationPeriods:
    development: Period
    validation: Period
    test: Period


def default_periods(run_date: date) -> ValidationPeriods:
    return ValidationPeriods(
        Period(date(2016, 1, 1), date(2021, 12, 31)),
        Period(date(2022, 1, 1), date(2023, 12, 31)),
        Period(date(2024, 1, 1), run_date - timedelta(days=1)),
    )
```

```python
# src/quant_trader/backtest/benchmarks.py
def run_spy_buy_hold(engine_factory, start, end):
    return engine_factory("spy_buy_hold").run_between(start, end)


def run_rules_only(engine_factory, start, end):
    return engine_factory("v1_rules_only").run_between(start, end)


def run_rules_llm(engine_factory, start, end):
    return engine_factory("v1_rules_llm").run_between(start, end)


def run_comparison(engine_factory, start, end):
    return {
        "spy_buy_hold": run_spy_buy_hold(engine_factory, start, end),
        "rules_only": run_rules_only(engine_factory, start, end),
        "rules_llm": run_rules_llm(engine_factory, start, end),
    }
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/unit/backtest -v`

Expected: chronological fill, no-look-ahead spy, benchmark-shape, and split tests PASS.

```bash
git add src/quant_trader/backtest tests/unit/backtest
git commit -m "feat: add chronological backtest comparisons"
```

### Task 9: Metrics and self-contained reports

**Files:**
- Create: `src/quant_trader/reporting/__init__.py`
- Create: `src/quant_trader/reporting/metrics.py`
- Create: `src/quant_trader/reporting/html.py`
- Test: `tests/unit/reporting/test_metrics.py`
- Test: `tests/unit/reporting/test_html.py`

- [ ] **Step 1: Write metric and disclosure tests**

```python
# tests/unit/reporting/test_metrics.py
import pandas as pd
from quant_trader.reporting.metrics import calculate_metrics, calculate_trade_metrics


def test_max_drawdown_and_return_metrics():
    equity = pd.Series([100.0, 110.0, 88.0, 99.0], index=pd.date_range("2026-01-01", periods=4))
    metrics = calculate_metrics(equity)
    assert metrics.max_drawdown == -0.20
```

```python
# tests/unit/reporting/test_html.py
def test_report_marks_llm_without_proven_gain(report_builder, comparison_without_gain):
    html = report_builder.render(comparison_without_gain)
    assert "LLM 层无已证实增益" in html
    assert all(label in html for label in ["SPY 买入持有", "纯规则", "规则 + MiniMax"])
```

- [ ] **Step 2: Run tests and verify failures**

Run: `uv run pytest tests/unit/reporting -v`

Expected: FAIL because reporting modules do not exist.

- [ ] **Step 3: Implement metrics with explicit annualization**

```python
# src/quant_trader/reporting/metrics.py
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float


@dataclass(frozen=True)
class TradeMetrics:
    turnover: float
    trade_count: int
    win_rate: float
    average_realized_pnl: float
    cost_ratio: float


def calculate_metrics(equity: pd.Series) -> Metrics:
    returns = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    volatility = returns.std(ddof=1) * np.sqrt(252)
    downside = returns[returns < 0].std(ddof=1) * np.sqrt(252)
    drawdown = equity / equity.cummax() - 1
    maximum = float(drawdown.min())
    return Metrics(float(total), float(annual_return), float(volatility),
                   float(annual_return / volatility) if volatility else 0,
                   float(annual_return / downside) if downside else 0,
                   maximum, float(annual_return / abs(maximum)) if maximum else 0)


def calculate_trade_metrics(fills, average_equity: float) -> TradeMetrics:
    notionals = [abs(fill.quantity * fill.price) for fill in fills]
    realized = [fill.realized_pnl for fill in fills if fill.realized_pnl != 0]
    total_notional = sum(notionals)
    return TradeMetrics(
        turnover=total_notional / average_equity if average_equity else 0,
        trade_count=len(fills),
        win_rate=sum(value > 0 for value in realized) / len(realized) if realized else 0,
        average_realized_pnl=float(np.mean(realized)) if realized else 0,
        cost_ratio=sum(fill.commission for fill in fills) / total_notional if total_notional else 0,
    )
```

- [ ] **Step 4: Render JSON plus embedded Plotly HTML**

```python
# src/quant_trader/reporting/html.py
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import plotly.graph_objects as go
from quant_trader.reporting.metrics import calculate_metrics


LABELS = {
    "spy_buy_hold": "SPY 买入持有",
    "rules_only": "纯规则",
    "rules_llm": "规则 + MiniMax",
}


@dataclass(frozen=True)
class ReportArtifacts:
    json_path: Path
    html_path: Path


class ReportBuilder:
    def render(self, comparison) -> str:
        metrics = {name: calculate_metrics(result.equity) for name, result in comparison.items()}
        rules = metrics["rules_only"]
        llm = metrics["rules_llm"]
        proven = llm.total_return > rules.total_return and llm.max_drawdown >= rules.max_drawdown
        figure = go.Figure(layout={"title": "Equity"})
        drawdown_figure = go.Figure(layout={"title": "Drawdown"})
        monthly_figure = go.Figure(layout={"title": "Monthly Returns"})
        exposure_figure = go.Figure(layout={"title": "Gross Exposure"})
        for name in ["spy_buy_hold", "rules_only", "rules_llm"]:
            figure.add_scatter(x=comparison[name].equity.index, y=comparison[name].equity,
                               name=LABELS[name])
            drawdown = comparison[name].equity / comparison[name].equity.cummax() - 1
            monthly = comparison[name].equity.resample("ME").last().pct_change().dropna()
            drawdown_figure.add_scatter(x=drawdown.index, y=drawdown, name=LABELS[name])
            monthly_figure.add_bar(x=monthly.index, y=monthly, name=LABELS[name])
            exposure_figure.add_scatter(x=comparison[name].gross_exposure.index,
                                        y=comparison[name].gross_exposure, name=LABELS[name])
        disclosure = "" if proven else "<strong>LLM 层无已证实增益</strong>"
        rows = "".join(f"<tr><td>{LABELS[name]}</td><td>{metrics[name].total_return:.2%}</td>"
                       f"<td>{metrics[name].max_drawdown:.2%}</td></tr>"
                       for name in ["spy_buy_hold", "rules_only", "rules_llm"])
        charts = "".join(chart.to_html(full_html=False, include_plotlyjs=index == 0)
                         for index, chart in enumerate([
                             figure, drawdown_figure, monthly_figure, exposure_figure]))
        return f"<html><body>{disclosure}<table>{rows}</table>{charts}</body></html>"

    def write(self, comparison, output_dir: Path) -> ReportArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            name: {
                "performance": asdict(calculate_metrics(result.equity)),
                "trades": asdict(calculate_trade_metrics(result.fills, float(result.equity.mean()))),
            }
            for name, result in comparison.items()
        }
        json_path = output_dir / "summary.json"
        html_path = output_dir / "report.html"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                             encoding="utf-8")
        html_path.write_text(self.render(comparison), encoding="utf-8")
        return ReportArtifacts(json_path, html_path)
```

Run: `uv run pytest tests/unit/reporting -v`

Expected: metric values, JSON schema, chart labels, and no-gain disclosure tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trader/reporting tests/unit/reporting
git commit -m "feat: report strategy performance and llm value"
```

### Task 10: Paper service and Typer CLI

**Files:**
- Create: `src/quant_trader/paper/__init__.py`
- Create: `src/quant_trader/paper/service.py`
- Create: `src/quant_trader/cli.py`
- Test: `tests/unit/paper/test_service.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write CLI safety and idempotency tests**

```python
# tests/unit/test_cli.py
from typer.testing import CliRunner
from quant_trader.cli import app


runner = CliRunner()


def test_paper_run_requires_confirmation(tmp_path):
    result = runner.invoke(app, ["paper", "run", "--db", str(tmp_path / "paper.db")])
    assert result.exit_code == 2
    assert "--confirm" in result.output


def test_cli_has_no_live_trading_command():
    result = runner.invoke(app, ["--help"])
    assert "broker" not in result.output.lower()
    assert "live" not in result.output.lower()
```

```python
# tests/unit/paper/test_service.py
def test_same_market_date_is_processed_once(paper_service):
    first = paper_service.run_once(as_of="2026-01-02")
    second = paper_service.run_once(as_of="2026-01-02")
    assert first.status == "completed"
    assert second.status == "already_completed"
```

- [ ] **Step 2: Run tests and verify failures**

Run: `uv run pytest tests/unit/paper tests/unit/test_cli.py -v`

Expected: FAIL because paper service and CLI do not exist.

- [ ] **Step 3: Implement one atomic paper cycle**

```python
# src/quant_trader/paper/service.py
class PaperService:
    def run_once(self, as_of):
        run_key = f"paper:{as_of}"
        if self.repository.run_exists(run_key):
            return PaperRunResult(status="already_completed", run_id=run_key)
        with self.repository.transaction():
            data = self.market_data.load_through(as_of)
            self.market_data.assert_fresh(data, as_of)
            self.execution.fill_pending_at_open(as_of)
            snapshot = self.features.build(data, as_of)
            intents = self.strategy.generate_from_snapshot(snapshot)
            orders = self.risk.approve(intents, self.account, snapshot.close_prices)
            self.execution.queue(orders)
            self.repository.save_run(run_key, snapshot, intents, orders)
        return PaperRunResult(status="completed", run_id=run_key)
```

- [ ] **Step 4: Wire exact CLI commands**

```python
# src/quant_trader/cli.py
from pathlib import Path
import typer

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer()
paper_app = typer.Typer()
risk_app = typer.Typer()
app.add_typer(data_app, name="data")
app.add_typer(paper_app, name="paper")
app.add_typer(risk_app, name="risk")


@paper_app.command("run")
@cli_errors
def paper_run(db: Path = typer.Option(...), confirm: bool = typer.Option(False, "--confirm")):
    if not confirm:
        raise typer.BadParameter("paper run requires --confirm")
    service = build_paper_service(db)
    result = service.run_once(current_market_date())
    typer.echo(result.model_dump_json())
```

```python
# append to src/quant_trader/cli.py
from quant_trader.application import build_application
from quant_trader.config import load_settings
from quant_trader.data.validation import DataValidationError
from functools import wraps
import httpx
import sqlite3


def cli_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (DataValidationError, httpx.HTTPError, sqlite3.DatabaseError) as error:
            typer.echo(f"error: {type(error).__name__}: {error}", err=True)
            raise typer.Exit(code=1) from error
    return wrapped


def application(config: Path, db: Path):
    return build_application(load_settings(config), Path("data"), db)


@data_app.command("sync")
@cli_errors
def data_sync(config: Path = typer.Option(Path("configs/default.yaml"))):
    result = application(config, Path("data/research.db")).data.sync()
    typer.echo(result.model_dump_json())


@app.command("backtest")
@cli_errors
def backtest(strategy: str = typer.Option("v1_rules_llm"),
             config: Path = typer.Option(Path("configs/default.yaml"))):
    result = application(config, Path("data/research.db")).backtest.run(strategy)
    typer.echo(result.run_id)


@paper_app.command("init")
@cli_errors
def paper_init(db: Path = typer.Option(Path("data/paper.db")),
               config: Path = typer.Option(Path("configs/default.yaml"))):
    application(config, db).paper.initialize()
    typer.echo("initialized")


@paper_app.command("status")
@cli_errors
def paper_status(db: Path = typer.Option(Path("data/paper.db")),
                 config: Path = typer.Option(Path("configs/default.yaml"))):
    typer.echo(application(config, db).paper.status().model_dump_json())


@risk_app.command("reset-circuit-breaker")
@cli_errors
def reset_circuit_breaker(db: Path = typer.Option(Path("data/paper.db")),
                          config: Path = typer.Option(Path("configs/default.yaml")),
                          confirm: bool = typer.Option(False, "--confirm")):
    if not confirm:
        raise typer.BadParameter("risk reset requires --confirm")
    application(config, db).risk.reset_circuit_breaker()
    typer.echo("reset")


@app.command("report")
@cli_errors
def report(run_id: str = typer.Option(...),
           config: Path = typer.Option(Path("configs/default.yaml"))):
    artifacts = application(config, Path("data/research.db")).reports.write_run(run_id)
    typer.echo(str(artifacts.html_path))
```

- [ ] **Step 5: Verify CLI help and commit**

Run: `uv run pytest tests/unit/paper tests/unit/test_cli.py -v && uv run quant-trader --help`

Expected: tests PASS; help lists only research, backtest, report, paper, and risk commands.

```bash
git add src/quant_trader/paper src/quant_trader/cli.py tests/unit/paper tests/unit/test_cli.py
git commit -m "feat: add safe paper trading cli"
```

### Task 11: Offline end-to-end fixture and full quality gate

**Files:**
- Create: `tests/fixtures/ohlcv/SPY.parquet`
- Create: `tests/fixtures/ohlcv/QQQ.parquet`
- Create: `tests/fixtures/llm/reviews.json`
- Create: `tests/integration/test_offline_workflow.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing end-to-end acceptance test**

```python
# tests/integration/test_offline_workflow.py
def test_offline_backtest_paper_and_report_workflow(tmp_path, fixture_market_data, fixture_reviews):
    application = build_test_application(tmp_path, fixture_market_data, fixture_reviews)
    backtest = application.backtest.run("v1_rules_llm")
    assert backtest.fills
    assert all(fill.execution_time > fill.signal_time for fill in backtest.fills)
    assert max(backtest.exposures) <= 0.80

    first = application.paper.run_once("2026-01-30")
    second = application.paper.run_once("2026-01-30")
    assert first.status == "completed"
    assert second.status == "already_completed"

    artifacts = application.reports.write(backtest.run_id)
    assert artifacts.json_path.exists()
    assert artifacts.html_path.exists()
    assert "SPY 买入持有" in artifacts.html_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the test and observe the missing application composition root**

Run: `uv run pytest tests/integration/test_offline_workflow.py -v`

Expected: FAIL because `build_test_application` or one of its concrete dependencies is not wired.

- [ ] **Step 3: Add a composition root and deterministic fixtures**

```python
# src/quant_trader/application.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from quant_trader.data.yfinance_source import YFinanceSource
from quant_trader.llm.minimax import MiniMaxReviewer


@dataclass(frozen=True)
class Application:
    data: Any
    backtest: Any
    paper: Any
    risk: Any
    reports: Any


def build_application(settings, data_root: Path, db_path: Path) -> Application:
    reviewer = MiniMaxReviewer(
        settings.llm.api_key.get_secret_value(), settings.llm.base_url,
        settings.llm.model, settings.llm.timeout_seconds, settings.llm.max_retries,
    )
    return assemble_application(settings, data_root, db_path, YFinanceSource(), reviewer)


def build_test_application(data_root: Path, db_path: Path, market_source, reviewer,
                           settings) -> Application:
    return assemble_application(settings, data_root, db_path, market_source, reviewer)
```

`assemble_application` is a single explicit constructor function in the same file. It creates
`ParquetMarketCache`, `Database`, `RiskEngine`, `CostModel`, `Simulator`,
`V1RulesLLMStrategy`, `BacktestEngine`, `PaperService`, and `ReportBuilder`, then returns them
in `Application`. It must not branch on production versus test mode; only the injected market
source and reviewer differ.

```python
# tests/fixtures/build_market_data.py
from pathlib import Path
import numpy as np
import pandas as pd


def write_fixture(path: Path, start_price: float) -> None:
    index = pd.bdate_range("2024-01-02", periods=320)
    close = start_price * np.cumprod(np.full(len(index), 1.0005))
    frame = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": np.full(len(index), 5_000_000.0),
    }, index=index)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


if __name__ == "__main__":
    write_fixture(Path("tests/fixtures/ohlcv/SPY.parquet"), 400)
    write_fixture(Path("tests/fixtures/ohlcv/QQQ.parquet"), 300)
```

Run `uv run python tests/fixtures/build_market_data.py` once and commit both Parquet files.
Create `tests/fixtures/llm/reviews.json` with valid maintain and reduce reviews keyed by the
canonical hashes asserted in `tests/unit/llm/test_cache.py`. The integration test monkeypatches
`socket.socket.connect` to raise `AssertionError("network forbidden in offline test")`, proving
neither Yahoo nor MiniMax is contacted.

- [ ] **Step 4: Document setup, commands, and safety limits**

Write this exact quick-start section to `README.md`:

````markdown
## Quick start

```bash
uv sync --extra dev
cp .env.example .env
uv run quant-trader data sync --config configs/default.yaml
uv run quant-trader backtest --strategy v1_rules_llm --config configs/default.yaml
uv run quant-trader paper init --db data/paper.db
uv run quant-trader paper run --db data/paper.db --confirm
```

This project is research software. It has no broker adapter, does not place live orders,
and fails closed when market data, model output, or risk checks are invalid.
````

- [ ] **Step 5: Run the complete verification suite**

Run:

```bash
uv run ruff check .
uv run mypy src/quant_trader
uv run pytest --cov=quant_trader --cov-report=term-missing
uv run quant-trader --help
```

Expected: Ruff and mypy report no errors; all tests PASS; coverage is at least 85%; CLI help contains no live/broker command.

- [ ] **Step 6: Perform a secret and scope audit**

Run:

```bash
git grep -n -E 'MINIMAX_API_KEY=.+|Bearer [A-Za-z0-9_-]{12,}' -- ':!.env.example'
git grep -n -E 'submit_live|place_live|broker_order' -- src tests
git status --short
```

Expected: the first two commands print nothing; status contains only intended implementation and documentation changes.

- [ ] **Step 7: Commit the completed V1**

```bash
git add README.md src tests configs pyproject.toml uv.lock .env.example
git commit -m "feat: complete llm paper trading v1"
```

### Task 12: Manual MiniMax smoke test without trading side effects

**Files:**
- Modify: `README.md`
- Create: `scripts/smoke_minimax.py`
- Test: `tests/unit/test_smoke_minimax.py`

- [ ] **Step 1: Write a test that forbids order-capable imports in the smoke script**

```python
# tests/unit/test_smoke_minimax.py
from pathlib import Path


def test_smoke_script_cannot_import_execution_or_paper_modules():
    source = Path("scripts/smoke_minimax.py").read_text(encoding="utf-8")
    assert "quant_trader.execution" not in source
    assert "quant_trader.paper" not in source
    assert "quant_trader.portfolio" not in source
```

- [ ] **Step 2: Run the test and verify the script is missing**

Run: `uv run pytest tests/unit/test_smoke_minimax.py -v`

Expected: FAIL with `FileNotFoundError` for `scripts/smoke_minimax.py`.

- [ ] **Step 3: Implement a review-only smoke script**

```python
# scripts/smoke_minimax.py
import json
from quant_trader.config import load_settings
from quant_trader.llm.minimax import MiniMaxReviewer
from quant_trader.llm.parsing import parse_review


def main() -> None:
    settings = load_settings("configs/default.yaml")
    reviewer = MiniMaxReviewer(
        settings.llm.api_key.get_secret_value(), settings.llm.base_url,
        settings.llm.model, settings.llm.timeout_seconds, settings.llm.max_retries,
    )
    content = reviewer.complete([{"role": "user", "content": json.dumps({
        "instruction": "Return only a valid LLMReview JSON object. Reject this synthetic candidate.",
        "ticker": "TEST", "close": 100, "sma_200": 110,
    })}])
    print(parse_review(content).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify offline safety, then optionally call MiniMax**

Run: `uv run pytest tests/unit/test_smoke_minimax.py -v`

Expected: PASS.

With `MINIMAX_API_KEY` set locally, run: `uv run python scripts/smoke_minimax.py`.

Expected: one valid `LLMReview` JSON object with `weight_multiplier` between 0 and 1; no database, order, fill, or account file is created.

- [ ] **Step 5: Commit the smoke-test utility**

```bash
git add scripts/smoke_minimax.py tests/unit/test_smoke_minimax.py README.md
git commit -m "test: add side-effect-free minimax smoke check"
```

## Spec coverage check

| Approved design requirement | Implemented and verified by |
| --- | --- |
| Versioned strategy directories and shared contracts | Tasks 1, 3, 5 |
| Point-in-time adjusted daily data, metadata, and freshness | Task 2 |
| Exact trend/momentum/volatility candidate rules | Task 3 |
| MiniMax OpenAI-compatible API, repair, cache, and fail-closed behavior | Tasks 4, 5, 6, 12 |
| 15% position, 80% gross, 10% target volatility, drawdown and trailing-stop controls | Tasks 3, 7 |
| Next-open fills, 10 bps slippage, 1 bp commission, and idempotency | Tasks 6, 8 |
| SQLite decisions, cache, orders, fills, account snapshots, and transactions | Task 6 |
| SPY, rules-only, and rules+LLM historical comparisons | Tasks 8, 9 |
| Fixed development/validation/test periods and chronological execution | Task 8 |
| HTML/JSON metrics, charts, trade statistics, and no-gain disclosure | Task 9 |
| Confirmed paper-only CLI and explicit circuit-breaker reset | Task 10 |
| Offline end-to-end workflow, secret scan, lint, typing, and coverage | Task 11 |
| No broker adapter or live-order path | Tasks 10, 11, 12 |

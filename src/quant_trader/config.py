"""Application settings loaded from YAML with explicit MiniMax environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaperSettings(_StrictModel):
    initial_cash: float = Field(default=100_000, gt=0)


class StrategySettings(_StrictModel):
    max_candidates: int = Field(default=4, ge=1)
    min_average_dollar_volume: float = Field(default=20_000_000, ge=0)
    target_volatility: float = Field(default=0.10, gt=0, le=1)


class RiskSettings(_StrictModel):
    max_position_weight: float = Field(default=0.15, gt=0, le=1)
    max_gross_exposure: float = Field(default=0.80, gt=0, le=1)
    min_cash_weight: float = Field(default=0.20, ge=0, le=1)
    reduce_drawdown: float = Field(default=0.10, gt=0, le=1)
    halt_drawdown: float = Field(default=0.15, gt=0, le=1)
    atr_multiple: float = Field(default=2.5, gt=0)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> RiskSettings:
        if self.min_cash_weight + self.max_gross_exposure > 1:
            raise ValueError("min_cash_weight must be compatible with max_gross_exposure")
        if self.reduce_drawdown >= self.halt_drawdown:
            raise ValueError("reduce_drawdown must be less than halt_drawdown")
        return self


class ExecutionSettings(_StrictModel):
    slippage_bps: float = Field(default=10, ge=0)
    commission_bps: float = Field(default=1, ge=0)


class LLMSettings(_StrictModel):
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str = "https://api.minimax.io/v1"
    model: str = "MiniMax-M2.7"
    prompt_version: str = "v1"
    timeout_seconds: float = Field(default=30, gt=0)
    retries: int = Field(default=2, ge=0)


class Settings(_StrictModel):
    universe: tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"
    )
    paper: PaperSettings = Field(default_factory=PaperSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


def load_settings(path: Path | str) -> Settings:
    """Load YAML settings, applying only MiniMax variables present in the environment."""
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text())
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError("settings YAML must contain a mapping")
            data = loaded

    llm = dict(data.get("llm", {}))
    overrides = {
        "MINIMAX_API_KEY": "api_key",
        "MINIMAX_BASE_URL": "base_url",
        "MINIMAX_MODEL": "model",
    }
    for environment_name, setting_name in overrides.items():
        if environment_name in os.environ:
            llm[setting_name] = os.environ[environment_name]
    if llm:
        data = {**data, "llm": llm}
    return Settings.model_validate(data)

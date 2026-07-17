"""Application settings loaded from YAML with explicit MiniMax environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from quant_trader.validation import StrictInteger, StrictNumber, USEquityTicker


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaperSettings(_StrictModel):
    initial_cash: StrictNumber = Field(default=100_000, gt=0)


class StrategySettings(_StrictModel):
    max_candidates: StrictInteger = Field(default=4, ge=1)
    min_average_dollar_volume: StrictNumber = Field(default=20_000_000, ge=0)
    target_volatility: StrictNumber = Field(default=0.10, gt=0, le=1)


class RiskSettings(_StrictModel):
    max_position_weight: StrictNumber = Field(default=0.15, gt=0, le=1)
    max_gross_exposure: StrictNumber = Field(default=0.80, gt=0, le=1)
    min_cash_weight: StrictNumber = Field(default=0.20, ge=0, le=1)
    reduce_drawdown: StrictNumber = Field(default=0.10, gt=0, le=1)
    halt_drawdown: StrictNumber = Field(default=0.15, gt=0, le=1)
    atr_multiple: StrictNumber = Field(default=2.5, gt=0)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> RiskSettings:
        if self.min_cash_weight + self.max_gross_exposure > 1:
            raise ValueError("min_cash_weight must be compatible with max_gross_exposure")
        if self.reduce_drawdown >= self.halt_drawdown:
            raise ValueError("reduce_drawdown must be less than halt_drawdown")
        return self


class ExecutionSettings(_StrictModel):
    slippage_bps: StrictNumber = Field(default=10, ge=0)
    commission_bps: StrictNumber = Field(default=1, ge=0)


class LLMSettings(_StrictModel):
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str = "https://api.minimax.io/v1"
    model: str = "MiniMax-M2.7"
    prompt_version: str = "v1"
    timeout_seconds: StrictNumber = Field(default=30, gt=0)
    max_retries: StrictInteger = Field(default=2, ge=0, le=5)

    @field_validator("base_url", mode="before")
    @classmethod
    def validate_base_url(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("base_url must be a nonblank http or https URL")
        normalized = value.strip()
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be a nonblank http or https URL")
        return normalized

    @field_validator("model", "prompt_version", mode="before")
    @classmethod
    def strip_required_label(cls, value: Any) -> str:
        if not isinstance(value, str) or not (normalized := value.strip()):
            raise ValueError("value must be a nonempty string")
        return normalized


class Settings(_StrictModel):
    universe: tuple[USEquityTicker, ...] = (
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

    raw_llm = data.get("llm", {})
    if not isinstance(raw_llm, dict):
        raise ValueError("llm settings must be a mapping")
    llm = cast(dict[str, Any], raw_llm).copy()
    if "api_key" in llm:
        raise ValueError("llm.api_key is environment-only; use MINIMAX_API_KEY")
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

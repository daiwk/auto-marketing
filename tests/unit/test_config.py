from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_trader.config import LLMSettings, load_settings


def test_load_settings_uses_safe_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.yaml")

    assert settings.paper.initial_cash == 100_000
    assert settings.llm.api_key.get_secret_value() == ""
    assert settings.llm.base_url == "https://api.minimax.io/v1"


def test_explicit_minimax_environment_variables_override_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("llm:\n  base_url: https://yaml.example/v1\n  model: yaml-model\n")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("MINIMAX_MODEL", "env-model")

    settings = load_settings(config)

    assert settings.llm.api_key.get_secret_value() == "test-key"
    assert settings.llm.base_url == "https://env.example/v1"
    assert settings.llm.model == "env-model"


def test_load_settings_rejects_incompatible_risk_weights(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("risk:\n  min_cash_weight: 0.3\n  max_gross_exposure: 0.8\n")

    with pytest.raises(ValidationError, match="min_cash_weight"):
        load_settings(config)


def test_load_settings_rejects_unordered_drawdown_thresholds(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("risk:\n  reduce_drawdown: 0.15\n  halt_drawdown: 0.15\n")

    with pytest.raises(ValidationError, match="reduce_drawdown"):
        load_settings(config)


def test_llm_settings_rejects_more_than_five_retries() -> None:
    assert LLMSettings(max_retries=5).max_retries == 5
    with pytest.raises(ValidationError):
        LLMSettings(max_retries=6)

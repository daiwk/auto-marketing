from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_trader.config import LLMSettings, load_settings


def test_load_settings_uses_safe_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.yaml")

    assert settings.paper.initial_cash == 100_000
    assert settings.llm.api_key.get_secret_value() == ""
    assert settings.llm.base_url == "https://api.minimax.io/v1"
    assert isinstance(settings.paper.initial_cash, float)


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


def test_settings_rejects_api_key_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("llm:\n  api_key: should-not-be-here\n")

    with pytest.raises(ValueError, match="environment"):
        load_settings(config)


def test_settings_redacts_environment_api_key_in_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "not-for-logs")

    settings = load_settings(tmp_path / "missing.yaml")

    assert "not-for-logs" not in repr(settings.model_dump())
    assert "not-for-logs" not in settings.model_dump_json()


@pytest.mark.parametrize("llm_value", [None, [], "not-a-mapping"])
def test_settings_rejects_non_mapping_llm_yaml(tmp_path: Path, llm_value: object) -> None:
    config = tmp_path / "settings.yaml"
    if llm_value is None:
        config.write_text("llm: null\n")
    elif isinstance(llm_value, list):
        config.write_text("llm: []\n")
    else:
        config.write_text("llm: not-a-mapping\n")

    with pytest.raises(ValueError, match="llm.*mapping"):
        load_settings(config)


def test_settings_validates_llm_endpoint_and_labels(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "llm:\n  base_url: ftp://example.com\n  model: '   '\n  prompt_version: '   '\n"
    )

    with pytest.raises(ValidationError):
        load_settings(config)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://exa mple.com",
        "https://example..com",
        "https://example.com:invalid",
        "https://user:password@example.com",
        "https://example.com?debug=true",
        "https://example.com#fragment",
    ],
)
def test_settings_rejects_unsafe_or_malformed_llm_base_urls(tmp_path: Path, base_url: str) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(f"llm:\n  base_url: {base_url}\n")

    with pytest.raises(ValidationError):
        load_settings(config)


def test_settings_normalizes_llm_base_url_without_trailing_slash(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("llm:\n  base_url: https://api.example.com/v1/\n")

    assert load_settings(config).llm.base_url == "https://api.example.com/v1"


@pytest.mark.parametrize(
    "yaml_body",
    [
        "paper:\n  initial_cash: '100000'\n",
        "risk:\n  max_position_weight: true\n",
        "strategy:\n  target_volatility: '0.1'\n",
        "execution:\n  slippage_bps: true\n",
        "llm:\n  timeout_seconds: '30'\n",
        "llm:\n  max_retries: true\n",
    ],
)
def test_settings_rejects_boolean_and_numeric_string_safety_values(
    tmp_path: Path, yaml_body: str
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(yaml_body)

    with pytest.raises(ValidationError):
        load_settings(config)


def test_settings_accepts_yaml_integer_values_for_float_thresholds(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "paper:\n  initial_cash: 100000\nstrategy:\n  min_average_dollar_volume: 20000000\n"
    )

    settings = load_settings(config)

    assert settings.paper.initial_cash == 100_000
    assert settings.strategy.min_average_dollar_volume == 20_000_000


@pytest.mark.parametrize(
    "yaml_body",
    ["paper:\n  initial_cash: .inf\n", "execution:\n  slippage_bps: .nan\n"],
)
def test_settings_rejects_non_finite_numeric_yaml_values(tmp_path: Path, yaml_body: str) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(yaml_body)

    with pytest.raises(ValidationError):
        load_settings(config)


def test_settings_uses_shared_ticker_validation_for_universe(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("universe: [brk.b, BF-B]\n")

    assert load_settings(config).universe == ("BRK.B", "BF-B")

    config.write_text("universe: ['$SPY']\n")
    with pytest.raises(ValidationError):
        load_settings(config)


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

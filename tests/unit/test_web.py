from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from pydantic import ValidationError

from quant_trader.web import (
    WebJobManager,
    WebMode,
    WebParameters,
    WebPlatformServer,
    WebProvider,
    WebRunRequest,
)


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = iter(("Agent market started.",))
        self.returncode: int | None = None

    def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15


def _manager(tmp_path: Path, commands: list[list[str]]) -> WebJobManager:
    config = tmp_path / "config.yaml"
    config.write_text("universe: [SPY]\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()

    def process(command: Sequence[str], cwd: Path) -> FakeProcess:
        del cwd
        argv = list(command)
        commands.append(argv)
        if "backtest" in argv:
            output = Path(argv[argv.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        "runs": {
                            "rules_only": {
                                "metrics": {
                                    "total_return": 0.12,
                                    "max_drawdown": -0.04,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            if "--agent-events" in argv:
                event_output = Path(argv[argv.index("--agent-events") + 1])
                event_output.write_text(
                    json.dumps(
                        {
                            "kind": "role_started",
                            "ticker": "SPY",
                            "as_of": "2025-12-31",
                            "provider": "Codex",
                            "role": "market_analyst",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
        else:
            mode = argv[argv.index("run") + 1]
            output = Path(argv[argv.index("--output-dir") + 1]) / f"{mode}-test"
            detail = output / mode.replace("-", "_")
            detail.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"kind": mode}), encoding="utf-8")
            (detail / "result.json").write_text(
                json.dumps({"metrics": {"total_return": 0.1}}), encoding="utf-8"
            )
        return FakeProcess()

    return WebJobManager(
        project_root=tmp_path,
        config=config,
        data_root=data,
        output_root=tmp_path / "runs",
        process_factory=process,
        workers=1,
    )


def test_request_rejects_incompatible_provider() -> None:
    with pytest.raises(ValidationError, match="requires MiniMax, Codex, or Trae X"):
        WebRunRequest(mode=WebMode.FINMEM, provider=WebProvider.RULES)


def test_traex_is_allowed_for_llm_modes() -> None:
    request = WebRunRequest(mode=WebMode.FINMEM, provider=WebProvider.TRAEX)
    assert request.provider is WebProvider.TRAEX


def test_background_rules_job_collects_logs_and_result(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    manager = _manager(tmp_path, commands)
    try:
        run_id = manager.submit(WebRunRequest(mode=WebMode.RULES, provider=WebProvider.RULES))
        run = manager.wait(run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert run["result"]["runs"]["rules_only"]["metrics"]["total_return"] == 0.12
        assert any(event["kind"] == "log" for event in run["events"])
        assert "backtest" in commands[0]
        assert "--use-llm" not in commands[0]
    finally:
        manager.close()


def test_trading_agents_command_is_bounded(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    manager = _manager(tmp_path, commands)
    try:
        run_id = manager.submit(
            WebRunRequest(
                mode=WebMode.TRADING_AGENTS,
                provider=WebProvider.CODEX,
                max_reviews=2,
            )
        )
        run = manager.wait(run_id)
        assert run is not None and run["status"] == "completed"
        command = commands[0]
        assert command[command.index("--llm-provider") + 1] == "codex"
        assert command[command.index("--llm-max-reviews") + 1] == "2"
        assert "--agent-events" in command
        assert run["agent_events"][0]["role"] == "market_analyst"
    finally:
        manager.close()


def test_run_parameters_create_isolated_safe_config(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    manager = _manager(tmp_path, commands)
    try:
        defaults = manager.defaults()
        parameters = WebParameters.model_validate(
            {
                **defaults,
                "universe": ["SPY", "QQQ"],
                "initial_cash": 250000,
                "target_volatility": 0.12,
            }
        )
        run_id = manager.submit(
            WebRunRequest(
                mode=WebMode.RULES,
                provider=WebProvider.RULES,
                parameters=parameters,
            )
        )
        run = manager.wait(run_id)
        assert run is not None
        config_path = Path(commands[0][commands[0].index("--config") + 1])
        assert config_path.parent.name == run_id
        config_text = config_path.read_text(encoding="utf-8")
        assert "initial_cash: 250000" in config_text
        assert "target_volatility: 0.12" in config_text
        assert "api_key" not in config_text
        assert run["parameters"]["universe"] == ["SPY", "QQQ"]
    finally:
        manager.close()


def test_token_protected_http_api_submits_run(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    manager = _manager(tmp_path, commands)
    server = WebPlatformServer(manager, port=0)
    url = server.start()
    try:
        with urlopen(url, timeout=2) as response:
            assert "Quant Trader Lab" in response.read().decode()
        with urlopen(url + "api/config", timeout=2) as response:
            assert json.loads(response.read())["parameters"]["universe"] == ["SPY"]
        request = Request(
            url + "api/runs",
            data=json.dumps({"mode": "rules", "provider": "rules"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            run_id = json.loads(response.read())["id"]
        run = manager.wait(run_id)
        assert run is not None and run["status"] == "completed"
        with pytest.raises(HTTPError) as denied:
            urlopen(url.replace(f"/{server.token}/", "/wrong/"), timeout=2)
        assert denied.value.code == 404
    finally:
        server.close()


def test_web_page_contains_agent_board_and_equity_chart() -> None:
    from quant_trader.web_template import WEB_HTML

    assert 'id="agentRoles"' in WEB_HTML
    assert "market_analyst:'市场分析师'" in WEB_HTML
    assert 'id="equityChart"' in WEB_HTML
    assert 'id="chartTooltip"' in WEB_HTML
    assert "agent_events" in WEB_HTML
    assert "rules_only:'规则策略'" in WEB_HTML
    assert "repeating-linear-gradient" in WEB_HTML
    assert '<option value="traex">本地 Trae X</option>' in WEB_HTML
    assert 'id="targetVolatility"' in WEB_HTML
    assert 'id="targetVolatility" type="number" min="0.01" max="100" step="any"' in WEB_HTML
    assert 'id="initialCash" type="number" min="1" step="any"' in WEB_HTML
    assert "parameters:parameters()" in WEB_HTML

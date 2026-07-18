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
    with pytest.raises(ValidationError, match="requires MiniMax or Codex"):
        WebRunRequest(mode=WebMode.FINMEM, provider=WebProvider.RULES)


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

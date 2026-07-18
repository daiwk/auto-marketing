from __future__ import annotations

from datetime import date

import httpx
import pytest

from quant_trader.core.models import LLMReview, ReviewAction
from quant_trader.dashboard import DashboardError, DashboardServer, DashboardState
from quant_trader.dashboard_template import DASHBOARD_HTML
from quant_trader.strategies.v2_multi_agent import (
    AgentEvent,
    AgentEventKind,
    ReportStatus,
    RoleName,
    RoleReport,
    Stance,
)


def _event(kind: AgentEventKind, **values: object) -> AgentEvent:
    return AgentEvent(
        kind=kind,
        ticker="SPY",
        as_of=date(2025, 12, 31),
        provider="MiniMax",
        **values,
    )


def _report(role: RoleName, status: ReportStatus = ReportStatus.AVAILABLE) -> RoleReport:
    return RoleReport(
        role=role,
        status=status,
        stance=Stance.BULLISH if status is ReportStatus.AVAILABLE else Stance.NEUTRAL,
        confidence=0.7 if status is ReportStatus.AVAILABLE else 0,
        summary="Point-in-time sanitized summary.",
        evidence=("Momentum is positive.",),
        risks=("Volatility may rise.",),
    )


def _review() -> LLMReview:
    return LLMReview(
        action=ReviewAction.REDUCE,
        weight_multiplier=0.5,
        confidence=0.6,
        thesis="Reduce exposure after risk review.",
        risks=("Volatility may rise.",),
        invalidation="Momentum improves.",
    )


def test_dashboard_state_projects_workflow_events_and_versions() -> None:
    state = DashboardState()

    state.publish(_event(AgentEventKind.WORKFLOW_STARTED))
    first = state.snapshot()
    state.publish(
        _event(
            AgentEventKind.ROLE_COMPLETED,
            role=RoleName.MARKET,
            report=_report(RoleName.MARKET),
        )
    )
    state.publish(
        _event(
            AgentEventKind.ROLE_SKIPPED,
            role=RoleName.SENTIMENT,
            report=_report(RoleName.SENTIMENT, ReportStatus.UNAVAILABLE),
        )
    )
    state.publish(_event(AgentEventKind.WORKFLOW_COMPLETED, final_review=_review()))
    final = state.snapshot()

    assert first["version"] == 1
    assert final["version"] == 4
    assert final["workflow_count"] == 1
    assert final["workflow"]["ticker"] == "SPY"  # type: ignore[index]
    roles = final["workflow"]["roles"]  # type: ignore[index]
    assert roles[RoleName.MARKET.value]["status"] == "completed"  # type: ignore[index]
    assert roles[RoleName.SENTIMENT.value]["status"] == "skipped"  # type: ignore[index]
    assert final["workflow"]["final_review"]["action"] == "reduce"  # type: ignore[index]


def test_failed_role_keeps_the_workflow_failed_after_completion() -> None:
    state = DashboardState()
    state.publish(_event(AgentEventKind.WORKFLOW_STARTED))
    failed = _report(RoleName.MARKET, ReportStatus.FAILED)
    state.publish(
        _event(
            AgentEventKind.ROLE_FAILED,
            role=RoleName.MARKET,
            report=failed,
        )
    )
    state.publish(_event(AgentEventKind.WORKFLOW_COMPLETED, final_review=_review()))

    workflow = state.snapshot()["workflow"]

    assert workflow["status"] == "failed"  # type: ignore[index]
    assert workflow["failure_role"] == RoleName.MARKET.value  # type: ignore[index]


def test_dashboard_server_exposes_only_tokenized_fixed_routes() -> None:
    state = DashboardState()
    opened: list[str] = []
    server = DashboardServer(state, browser_open=lambda url: not opened.append(url))
    url = server.start()
    try:
        page = httpx.get(url)
        snapshot = httpx.get(f"{url}state")
        forbidden = httpx.get(url.replace(server.token, "wrong-token"))
        traversal = httpx.get(f"{url}../../README.md")
    finally:
        server.stop()

    assert opened == [url]
    assert page.status_code == 200
    assert snapshot.status_code == 200
    assert forbidden.status_code == 404
    assert traversal.status_code == 404
    assert page.headers["Content-Security-Policy"].startswith("default-src 'none'")
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    assert snapshot.json()["command_status"] == "preparing"


def test_state_endpoint_acknowledges_the_rendered_version() -> None:
    state = DashboardState()
    version = state.set_command_status("completed")
    server = DashboardServer(state, browser_open=lambda _url: False)
    url = server.start()
    try:
        assert state.wait_until_seen(version, timeout_seconds=0.01) is False
        assert httpx.get(f"{url}state").status_code == 200
        assert state.wait_until_seen(version, timeout_seconds=0.1) is True
    finally:
        server.stop()


def test_partial_server_start_closes_bound_socket_and_sanitizes_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundServer:
        server_address = ("127.0.0.1", 12345)
        closed = False

        def serve_forever(self) -> None:
            return None

        def server_close(self) -> None:
            self.closed = True

    class BrokenThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("thread detail must not leak")

    bound = BoundServer()
    monkeypatch.setattr("quant_trader.dashboard.ThreadingHTTPServer", lambda *args: bound)
    monkeypatch.setattr("quant_trader.dashboard.threading.Thread", BrokenThread)

    with pytest.raises(DashboardError, match="local dashboard could not start"):
        DashboardServer(DashboardState(), browser_open=lambda _url: False).start()

    assert bound.closed is True


def test_dashboard_template_uses_safe_local_rendering() -> None:
    lowered = DASHBOARD_HTML.lower()

    assert "innerhtml" not in lowered
    assert "https://" not in lowered
    assert "http://" not in lowered
    assert "textcontent" in lowered
    assert "fetch('state'" in lowered
    assert "selectedmanually" in lowered


def test_dashboard_state_projects_experiment_snapshots() -> None:
    state = DashboardState()

    state.prepare_experiment("finmem", "run-123", "Codex")
    state.update_experiment(
        "backtest",
        "running",
        {
            "calls": 1,
            "memory": {"short": [], "mid": [], "long": []},
            "decision": {"action": "maintain", "memory_ids": []},
        },
    )

    snapshot = state.snapshot()
    assert snapshot["mode"] == "experiment"
    assert snapshot["command_status"] == "running"
    assert snapshot["experiment"] == {
        "kind": "finmem",
        "run_id": "run-123",
        "provider": "Codex",
        "stage": "backtest",
        "status": "running",
        "payload": {
            "calls": 1,
            "memory": {"short": [], "mid": [], "long": []},
            "decision": {"action": "maintain", "memory_ids": []},
        },
    }


def test_dashboard_template_contains_all_experiment_views() -> None:
    lowered = DASHBOARD_HTML.lower()

    assert "finmem" in lowered
    assert "quanta-alpha" in lowered
    assert "alpha-arena" in lowered
    assert "candidate" in lowered
    assert "leaderboard" in lowered
    assert "innerhtml" not in lowered

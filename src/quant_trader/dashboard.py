"""Loopback-only runtime dashboard for sanitized multi-agent events."""

from __future__ import annotations

import copy
import json
import secrets
import threading
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from quant_trader.dashboard_template import DASHBOARD_HTML
from quant_trader.strategies.v2_multi_agent import AgentEvent, AgentEventKind, RoleName

_ROLE_ORDER = tuple(role.value for role in RoleName)
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
}


class DashboardError(RuntimeError):
    """Sanitized local dashboard startup failure."""


def _waiting_roles() -> dict[str, dict[str, object]]:
    return {role: {"status": "waiting", "report": None} for role in _ROLE_ORDER}


class DashboardState:
    """Thread-safe JSON projection of the latest workflow."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0
        self._seen_version = 0
        self._snapshot: dict[str, object] = {
            "version": 0,
            "command_status": "preparing",
            "workflow_count": 0,
            "workflow": None,
        }

    def _advance(self) -> int:
        self._version += 1
        self._snapshot["version"] = self._version
        self._condition.notify_all()
        return self._version

    def publish(self, event: AgentEvent) -> None:
        with self._condition:
            self._project(event)
            self._advance()

    def _project(self, event: AgentEvent) -> None:
        if event.kind is AgentEventKind.WORKFLOW_STARTED:
            self._snapshot["command_status"] = "running"
            self._snapshot["workflow"] = {
                "ticker": event.ticker,
                "as_of": event.as_of.isoformat(),
                "provider": event.provider,
                "status": "running",
                "active_role": None,
                "roles": _waiting_roles(),
                "proposal": None,
                "final_review": None,
            }
            return
        workflow = self._snapshot.get("workflow")
        if not isinstance(workflow, dict):
            return
        roles = workflow["roles"]
        assert isinstance(roles, dict)
        if event.role is not None:
            item = roles[event.role.value]
            assert isinstance(item, dict)
            statuses = {
                AgentEventKind.ROLE_STARTED: "running",
                AgentEventKind.ROLE_COMPLETED: "completed",
                AgentEventKind.ROLE_SKIPPED: "skipped",
                AgentEventKind.ROLE_FAILED: "failed",
            }
            item["status"] = statuses[event.kind]
            item["report"] = (
                event.report.model_dump(mode="json") if event.report is not None else None
            )
            workflow["active_role"] = (
                event.role.value if event.kind is AgentEventKind.ROLE_STARTED else None
            )
        elif event.kind is AgentEventKind.TRADER_COMPLETED:
            workflow["proposal"] = event.proposal.model_dump(mode="json")  # type: ignore[union-attr]
            roles[RoleName.TRADER.value] = {"status": "completed", "report": None}
        elif event.kind is AgentEventKind.FINAL_COMPLETED:
            workflow["final_review"] = event.final_review.model_dump(  # type: ignore[union-attr]
                mode="json"
            )
            roles[RoleName.PORTFOLIO_MANAGER.value] = {"status": "completed", "report": None}
        elif event.kind is AgentEventKind.WORKFLOW_COMPLETED:
            workflow["status"] = "completed"
            workflow["active_role"] = None
            workflow["final_review"] = event.final_review.model_dump(  # type: ignore[union-attr]
                mode="json"
            )
            count = self._snapshot["workflow_count"]
            assert isinstance(count, int)
            self._snapshot["workflow_count"] = count + 1

    def set_command_status(self, status: str, *, reason: str | None = None) -> int:
        if status not in {"preparing", "running", "completed", "failed", "stopped"}:
            raise ValueError("invalid dashboard command status")
        with self._condition:
            self._snapshot["command_status"] = status
            self._snapshot["reason"] = reason
            return self._advance()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return copy.deepcopy(self._snapshot)

    def mark_seen(self, version: int) -> None:
        with self._condition:
            self._seen_version = max(self._seen_version, version)
            self._condition.notify_all()

    def wait_until_seen(self, version: int, *, timeout_seconds: float) -> bool:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative")
        with self._condition:
            return self._condition.wait_for(
                lambda: self._seen_version >= version, timeout=timeout_seconds
            )


class DashboardServer:
    """Serve one fixed dashboard and read-only state endpoint on loopback."""

    def __init__(
        self,
        state: DashboardState,
        *,
        browser_open: Callable[[str], object] = webbrowser.open,
    ) -> None:
        self.state = state
        self.token = secrets.token_urlsafe(32)
        self._browser_open = browser_open
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        state = self.state
        page_path = f"/{self.token}/"
        state_path = f"/{self.token}/state"

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                for name, value in _SECURITY_HEADERS.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == page_path:
                    self._respond(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                    return
                if self.path == state_path:
                    snapshot = state.snapshot()
                    body = json.dumps(snapshot, ensure_ascii=False).encode()
                    self._respond(200, body, "application/json; charset=utf-8")
                    version = snapshot["version"]
                    assert isinstance(version, int)
                    state.mark_seen(version)
                    return
                self._respond(404, b"not found", "text/plain; charset=utf-8")

            def log_message(self, format: str, *args: Any) -> None:
                return None

        return Handler

    def start(self) -> str:
        if self._server is not None:
            raise DashboardError("local dashboard is already running")
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
        except OSError:
            raise DashboardError("local dashboard could not start") from None
        self._server, self._thread = server, thread
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/{self.token}/"
        try:
            self._browser_open(url)
        except Exception:
            pass
        return url

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2)

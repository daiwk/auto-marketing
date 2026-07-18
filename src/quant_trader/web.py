"""Local, token-protected web platform for bounded paper experiments."""

from __future__ import annotations

import copy
import json
import secrets
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_trader.web_template import WEB_HTML

_MAX_BODY_BYTES = 64 * 1024
_MAX_EVENTS = 1_000
_MAX_LINE_CHARS = 2_000
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


class WebMode(StrEnum):
    RULES = "rules"
    TRADING_AGENTS = "trading-agents"
    FINMEM = "finmem"
    QUANTA_ALPHA = "quanta-alpha"
    ALPHA_ARENA = "alpha-arena"


class WebProvider(StrEnum):
    RULES = "rules"
    MINIMAX = "minimax"
    CODEX = "codex"


class WebRunRequest(BaseModel):
    """Strict user-selectable parameters; paths and credentials are server-owned."""

    model_config = ConfigDict(extra="forbid")

    mode: WebMode
    provider: WebProvider
    max_reviews: int = Field(default=1, ge=1, le=10)
    contestant_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def validate_combination(self) -> WebRunRequest:
        if self.mode is WebMode.RULES and self.provider is not WebProvider.RULES:
            raise ValueError("rules mode requires the rules provider")
        if self.mode in {WebMode.TRADING_AGENTS, WebMode.FINMEM, WebMode.QUANTA_ALPHA}:
            if self.provider not in {WebProvider.MINIMAX, WebProvider.CODEX}:
                raise ValueError("this mode requires MiniMax or Codex")
        if self.mode is WebMode.ALPHA_ARENA and self.provider is not WebProvider.RULES:
            raise ValueError("Alpha Arena is an artifact-only rules run")
        if self.mode is not WebMode.ALPHA_ARENA and self.contestant_ids:
            raise ValueError("contestants are only valid for Alpha Arena")
        if len(set(self.contestant_ids)) != len(self.contestant_ids):
            raise ValueError("contestant ids must be unique")
        return self


class ProcessLike(Protocol):
    @property
    def stdout(self) -> Iterable[str] | None: ...

    @property
    def returncode(self) -> int | None: ...

    def wait(self) -> int: ...

    def terminate(self) -> None: ...


ProcessFactory = Callable[[Sequence[str], Path], ProcessLike]


def _default_process(command: Sequence[str], cwd: Path) -> ProcessLike:
    return subprocess.Popen(  # noqa: S603 - command is constructed from strict enums
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=False,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


class WebJobManager:
    """Run a small number of controlled CLI jobs and expose sanitized snapshots."""

    def __init__(
        self,
        *,
        project_root: Path,
        config: Path,
        data_root: Path,
        output_root: Path,
        process_factory: ProcessFactory = _default_process,
        workers: int = 2,
    ) -> None:
        if workers < 1 or workers > 4:
            raise ValueError("workers must be from 1 to 4")
        self.project_root = project_root.resolve()
        self.config = config.resolve()
        self.data_root = data_root.resolve()
        self.output_root = output_root.resolve()
        if not self.config.is_file() or not self.data_root.is_dir():
            raise ValueError("config and data root must exist")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._process_factory = process_factory
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quant-web")
        self._condition = threading.Condition()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, ProcessLike] = {}
        self._closed = False

    def submit(self, request: WebRunRequest) -> str:
        with self._condition:
            if self._closed:
                raise RuntimeError("web runner is closed")
            for contestant_id in request.contestant_ids:
                contestant = self._jobs.get(contestant_id)
                if contestant is None or contestant["status"] not in {"completed", "partial"}:
                    raise ValueError("contestants must reference completed web runs")
                if contestant["mode"] not in {
                    WebMode.FINMEM.value,
                    WebMode.QUANTA_ALPHA.value,
                }:
                    raise ValueError("Alpha Arena currently accepts FinMem or QuantaAlpha runs")
                if not contestant.get("artifact_root"):
                    raise ValueError("contestant does not have a reusable artifact")
            run_id = uuid4().hex
            run_dir = self.output_root / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            self._jobs[run_id] = {
                "id": run_id,
                "mode": request.mode.value,
                "provider": request.provider.value,
                "max_reviews": request.max_reviews,
                "contestant_ids": list(request.contestant_ids),
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "artifact_root": None,
                "events": [],
                "result": None,
                "error": None,
                "run_dir": str(run_dir),
            }
            self._event_locked(run_id, "queued", "任务已进入后台队列")
            self._persist_locked(run_id)
            self._executor.submit(self._run, run_id, request)
            return run_id

    def list_runs(self) -> list[dict[str, Any]]:
        with self._condition:
            jobs = sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)
            return [self._public(item, include_events=False) for item in jobs]

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self._jobs.get(run_id)
            return self._public(job, include_events=True) if job is not None else None

    def wait(self, run_id: str, timeout: float = 10) -> dict[str, Any] | None:
        """Wait for one job to reach a terminal state; primarily useful to callers/tests."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    run_id not in self._jobs
                    or self._jobs[run_id]["status"] in {"completed", "failed", "partial"}
                ),
                timeout=timeout,
            )
            job = self._jobs.get(run_id)
            return self._public(job, include_events=True) if job is not None else None

    def _command(self, run_id: str, request: WebRunRequest) -> list[str]:
        run_dir = Path(self._jobs[run_id]["run_dir"])
        base = [
            sys.executable,
            "-m",
            "quant_trader",
        ]
        common = [
            "--config",
            str(self.config),
            "--data-root",
            str(self.data_root),
        ]
        if request.mode is WebMode.RULES:
            return [*base, "backtest", *common, "--output", str(run_dir / "run.json")]
        if request.mode is WebMode.TRADING_AGENTS:
            return [
                *base,
                "backtest",
                *common,
                "--output",
                str(run_dir / "run.json"),
                "--use-llm",
                "--llm-provider",
                request.provider.value,
                "--llm-workflow",
                "trading-agents",
                "--llm-max-reviews",
                str(request.max_reviews),
            ]
        experiment = [
            *base,
            "experiment",
            "run",
            request.mode.value,
            *common,
            "--output-dir",
            str(run_dir / "artifacts"),
        ]
        if request.mode in {WebMode.FINMEM, WebMode.QUANTA_ALPHA}:
            return [*experiment, "--llm-provider", request.provider.value]
        for contestant_id in request.contestant_ids:
            artifact = self._jobs[contestant_id]["artifact_root"]
            experiment.extend(("--contestant-run", str(artifact)))
        return experiment

    def _run(self, run_id: str, request: WebRunRequest) -> None:
        monitor_stop = threading.Event()
        monitor: threading.Thread | None = None
        with self._condition:
            job = self._jobs[run_id]
            job["status"] = "running"
            job["started_at"] = _now()
            self._event_locked(run_id, "running", "后台进程已启动")
            self._persist_locked(run_id)
        try:
            process = self._process_factory(self._command(run_id, request), self.project_root)
            with self._condition:
                self._processes[run_id] = process
            if request.mode in {
                WebMode.FINMEM,
                WebMode.QUANTA_ALPHA,
                WebMode.ALPHA_ARENA,
            }:
                monitor = threading.Thread(
                    target=self._monitor_artifact_events,
                    args=(run_id, monitor_stop),
                    daemon=True,
                )
                monitor.start()
            if process.stdout is not None:
                for line in process.stdout:
                    cleaned = line.strip()
                    if cleaned:
                        with self._condition:
                            self._event_locked(run_id, "log", cleaned[:_MAX_LINE_CHARS])
            exit_code = process.wait()
            monitor_stop.set()
            if monitor is not None:
                monitor.join(timeout=1)
            result, artifact_root = self._collect_result(run_id, request)
            with self._condition:
                job = self._jobs[run_id]
                job["exit_code"] = exit_code
                job["finished_at"] = _now()
                job["artifact_root"] = str(artifact_root) if artifact_root else None
                job["result"] = result
                partial = (
                    isinstance(result, dict)
                    and isinstance(result.get("summary"), dict)
                    and result["summary"].get("status") == "partial"
                )
                job["status"] = (
                    "partial"
                    if exit_code == 0 and partial
                    else "completed"
                    if exit_code == 0
                    else "failed"
                )
                if exit_code != 0:
                    job["error"] = "后台命令执行失败，请查看过程日志"
                self._event_locked(run_id, job["status"], "任务执行结束")
                self._persist_locked(run_id)
                self._condition.notify_all()
        except Exception:
            with self._condition:
                job = self._jobs[run_id]
                job["status"] = "failed"
                job["finished_at"] = _now()
                job["error"] = "后台任务发生内部错误"
                self._event_locked(run_id, "failed", job["error"])
                self._persist_locked(run_id)
                self._condition.notify_all()
        finally:
            monitor_stop.set()
            if monitor is not None and monitor.is_alive():
                monitor.join(timeout=1)
            with self._condition:
                self._processes.pop(run_id, None)

    def _monitor_artifact_events(self, run_id: str, stop: threading.Event) -> None:
        """Mirror durable experiment stages into the website while a CLI job runs."""
        root = Path(self._jobs[run_id]["run_dir"]) / "artifacts"
        seen: set[tuple[str, int]] = set()
        while not stop.wait(0.25):
            for path in root.glob("*/events.jsonl"):
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError):
                    continue
                for line in lines:
                    try:
                        event = json.loads(line)
                        sequence = int(event["sequence"])
                        stage = str(event["stage"])
                        message = str(event["message"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    key = (path.parent.name, sequence)
                    if key in seen:
                        continue
                    seen.add(key)
                    with self._condition:
                        self._event_locked(run_id, "stage", f"{stage} · {message}")

    def _collect_result(
        self, run_id: str, request: WebRunRequest
    ) -> tuple[object | None, Path | None]:
        run_dir = Path(self._jobs[run_id]["run_dir"])
        if request.mode in {WebMode.RULES, WebMode.TRADING_AGENTS}:
            return _read_json(run_dir / "run.json"), run_dir
        artifact_parent = run_dir / "artifacts"
        candidates = (
            sorted(
                (path for path in artifact_parent.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if artifact_parent.is_dir()
            else []
        )
        if not candidates:
            return None, None
        artifact = candidates[0]
        module_files = {
            WebMode.FINMEM: artifact / "finmem" / "result.json",
            WebMode.QUANTA_ALPHA: artifact / "quanta_alpha" / "result.json",
            WebMode.ALPHA_ARENA: artifact / "alpha_arena" / "result.json",
        }
        return {
            "summary": _read_json(artifact / "summary.json"),
            "details": _read_json(module_files[request.mode]),
        }, artifact

    def _event_locked(self, run_id: str, kind: str, message: str) -> None:
        events = self._jobs[run_id]["events"]
        events.append({"at": _now(), "kind": kind, "message": message})
        if len(events) > _MAX_EVENTS:
            del events[: len(events) - _MAX_EVENTS]

    def _persist_locked(self, run_id: str) -> None:
        job = self._public(self._jobs[run_id], include_events=True)
        path = Path(self._jobs[run_id]["run_dir"]) / "web-run.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _public(job: dict[str, Any], *, include_events: bool) -> dict[str, Any]:
        value = copy.deepcopy(job)
        value.pop("run_dir", None)
        if not include_events:
            value.pop("events", None)
            value.pop("result", None)
        return value

    def close(self) -> None:
        with self._condition:
            self._closed = True
            processes = tuple(self._processes.values())
        for process in processes:
            try:
                process.terminate()
            except OSError:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)


class WebPlatformServer:
    """Serve the fixed UI and JSON API on a loopback-only capability URL."""

    def __init__(
        self,
        manager: WebJobManager,
        *,
        port: int = 8000,
        browser_open: Callable[[str], object] = webbrowser.open,
    ) -> None:
        if not 0 <= port <= 65_535:
            raise ValueError("port must be from 0 to 65535")
        self.manager = manager
        self.port = port
        self.token = secrets.token_urlsafe(24)
        self._browser_open = browser_open
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        manager = self.manager
        prefix = f"/{self.token}/"

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                for name, value in _SECURITY_HEADERS.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, payload: object) -> None:
                self._respond(
                    status,
                    json.dumps(payload, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )

            def do_GET(self) -> None:  # noqa: N802
                if self.path == prefix:
                    self._respond(200, WEB_HTML.encode(), "text/html; charset=utf-8")
                    return
                if self.path == f"{prefix}api/runs":
                    self._json(200, {"runs": manager.list_runs()})
                    return
                run_prefix = f"{prefix}api/runs/"
                if self.path.startswith(run_prefix):
                    run_id = self.path[len(run_prefix) :]
                    if not run_id or "/" in run_id:
                        self._json(404, {"error": "not found"})
                        return
                    job = manager.get(run_id)
                    self._json(200, job) if job is not None else self._json(
                        404, {"error": "not found"}
                    )
                    return
                self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != f"{prefix}api/runs":
                    self._json(404, {"error": "not found"})
                    return
                raw_length = self.headers.get("Content-Length", "")
                if not raw_length.isdigit() or not 0 < int(raw_length) <= _MAX_BODY_BYTES:
                    self._json(400, {"error": "invalid request size"})
                    return
                try:
                    payload = json.loads(self.rfile.read(int(raw_length)))
                    request = WebRunRequest.model_validate(payload)
                    run_id = manager.submit(request)
                except (json.JSONDecodeError, UnicodeError, ValueError) as error:
                    self._json(400, {"error": str(error)[:500]})
                    return
                self._json(202, {"id": run_id})

            def log_message(self, format: str, *args: Any) -> None:
                return None

        return Handler

    def _bind(self) -> tuple[ThreadingHTTPServer, str]:
        if self._server is not None:
            raise RuntimeError("web platform is already running")
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        actual_port = self._server.server_address[1]
        url = f"http://127.0.0.1:{actual_port}/{self.token}/"
        return self._server, url

    def start(self, *, open_browser: bool = False) -> str:
        """Start in a daemon thread and return the capability URL."""
        server, url = self._bind()
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        if open_browser:
            try:
                self._browser_open(url)
            except Exception:
                pass
        return url

    def serve(self, *, open_browser: bool = True) -> str:
        server, url = self._bind()
        if open_browser:
            try:
                self._browser_open(url)
            except Exception:
                pass
        print(f"Quant Trader Web: {url}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
            self._server = None
            self.manager.close()
        return url

    def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            if self._thread is not None:
                server.shutdown()
            server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.manager.close()

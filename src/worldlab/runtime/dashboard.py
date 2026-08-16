"""Dependency-free, read-only HTTP dashboard for a runtime event buffer."""

from __future__ import annotations

import json
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Optional, Sequence, cast
from urllib.parse import parse_qs, urlsplit

from worldlab.data import (
    EnvironmentStepped,
    EpisodeEnded,
    EpisodeStarted,
    PolicyActed,
    RuntimeErrorEvent,
    RuntimeEvent,
    RuntimeSnapshot,
    TransitionCommitted,
)

from .tracing import TraceSource


class DashboardServer:
    """Serve a live snapshot and event history without mutating the run."""

    def __init__(
        self,
        source: TraceSource,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        poll_interval_s: float = 1.0,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if port < 0 or port > 65535:
            raise ValueError("port must be between 0 and 65535")
        if poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        self.source = source
        self.host = host
        self.poll_interval_s = float(poll_interval_s)
        self._httpd = ThreadingHTTPServer(
            (host, port),
            _make_handler(source, self.poll_interval_s),
        )
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="worldlab-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            self._httpd.server_close()
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)
        self._thread = None

    def __enter__(self) -> "DashboardServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def _make_handler(source: TraceSource, poll_interval_s: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/":
                    self._send_bytes(
                        200,
                        _render_dashboard_html(source.snapshot, poll_interval_s).encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                elif parsed.path == "/healthz":
                    self._send_json(200, {"status": "ok"})
                elif parsed.path == "/api/status":
                    self._send_json(200, _snapshot_payload(source.snapshot))
                elif parsed.path == "/api/events":
                    self._send_json(200, _events_payload(source, parse_qs(parsed.query)))
                else:
                    self._send_json(404, {"error": f"unknown path {parsed.path}"})
            except (KeyError, ValueError) as error:
                self._send_json(400, {"error": str(error)})

        def do_POST(self) -> None:
            self._send_json(405, {"error": "dashboard is read-only"})

        def _send_json(self, code: int, payload: Mapping[str, Any]) -> None:
            self._send_bytes(
                code,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_bytes(self, code: int, data: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if data:
                self.wfile.write(data)

    return Handler


def _events_payload(source: TraceSource, query: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    after = _query_int(query, "after", 0)
    limit = min(max(_query_int(query, "limit", 100), 1), 500)
    events = source.since(after) if after > 0 else source.events
    return {
        "events": [_event_payload(event) for event in events[-limit:]],
        "latest_sequence": source.snapshot.sequence,
    }


def _render_dashboard_html(snapshot: RuntimeSnapshot, poll_interval_s: float = 1.0) -> str:
    """Inject a server-side first paint so the page is useful before JS polls."""

    payload = _snapshot_payload(snapshot)
    snapshot_text = json.dumps(
        {
            "status": payload["status"],
            "phase": payload["phase"],
            "observation": payload["observation"],
            "action": payload["action"],
            "next_observation": payload["next_observation"],
            "reward": payload["reward"],
            "terminated": payload["terminated"],
            "truncated": payload["truncated"],
            "info": payload["info"],
        },
        ensure_ascii=False,
        indent=2,
    )
    error_text = (
        f"{payload['error_type']}: {payload['error_message']}"
        if payload["error_message"]
        else "暂无错误"
    )
    replacements = {
        "__INITIAL_STATUS__": str(payload["status"]),
        "__INITIAL_EPISODE__": (
            "—" if payload["episode_index"] is None else str(payload["episode_index"])
        ),
        "__INITIAL_STEP__": str(payload["step_index"]),
        "__INITIAL_REWARD__": f"{float(payload['total_reward']):.3f}",
        "__INITIAL_EVENT__": payload["last_event"] or "—",
        "__INITIAL_SNAPSHOT__": snapshot_text,
        "__INITIAL_ERROR__": error_text,
        "__POLL_INTERVAL_MS__": str(max(1, int(poll_interval_s * 1000.0))),
    }
    rendered = DASHBOARD_HTML
    for token, value in replacements.items():
        rendered = rendered.replace(token, html.escape(value, quote=False))
    return rendered


def _query_int(query: Mapping[str, Sequence[str]], name: str, default: int) -> int:
    values = query.get(name)
    if not values:
        return default
    return int(values[0])


def _snapshot_payload(snapshot: RuntimeSnapshot) -> dict[str, Any]:
    return {
        "status": snapshot.status.value,
        "sequence": snapshot.sequence,
        "timestamp_s": snapshot.timestamp_s,
        "monotonic_s": snapshot.monotonic_s,
        "episode_index": snapshot.episode_index,
        "step_index": snapshot.step_index,
        "total_reward": snapshot.total_reward,
        "reward": snapshot.reward,
        "terminated": snapshot.terminated,
        "truncated": snapshot.truncated,
        "last_event": snapshot.last_event.value if snapshot.last_event else None,
        "phase": snapshot.phase.value if snapshot.phase else None,
        "observation": _safe_value(snapshot.observation),
        "action": _safe_value(snapshot.action),
        "next_observation": _safe_value(snapshot.next_observation),
        "info": _safe_value(snapshot.info),
        "error_type": snapshot.error_type,
        "error_message": snapshot.error_message,
    }


def _event_payload(event: RuntimeEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence": event.sequence,
        "kind": event.kind.value,
        "timestamp_s": event.timestamp_s,
        "monotonic_s": event.monotonic_s,
        "episode_index": event.episode_index,
        "step_index": event.step_index,
        "duration_s": event.duration_s,
    }
    if isinstance(event, EpisodeStarted):
        payload.update({"observation": _safe_value(event.observation), "info": _safe_value(event.info)})
    elif isinstance(event, PolicyActed):
        payload.update({"action": _safe_value(event.action), "policy_info": _safe_value(event.policy_info)})
    elif isinstance(event, EnvironmentStepped):
        payload.update(
            {
                "observation": _safe_value(event.observation),
                "reward": event.reward,
                "terminated": event.terminated,
                "truncated": event.truncated,
                "info": _safe_value(event.info),
            }
        )
    elif isinstance(event, TransitionCommitted):
        payload.update({"reward": event.transition.reward, "total_reward": event.total_reward})
    elif isinstance(event, EpisodeEnded):
        payload.update(
            {
                "total_reward": event.result.total_reward,
                "length": event.result.length,
                "terminated": event.result.terminated,
                "truncated": event.result.truncated,
            }
        )
    elif isinstance(event, RuntimeErrorEvent):
        payload.update(
            {
                "phase": event.phase.value,
                "error_type": event.error_type,
                "message": event.message,
            }
        )
    return payload


def _safe_value(value: object, *, limit: int = 180) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item, limit=limit) for key, item in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        items = [_safe_value(item, limit=limit) for item in value[:20]]
        return items + ([f"... ({len(value) - 20} more)"] if len(value) > 20 else [])
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return f"<{type(value).__name__} shape={tuple(shape)} dtype={dtype}>"
    text = repr(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WorldLab Runtime Dashboard</title>
<style>
:root { --bg:#f4f6f8; --card:#fff; --ink:#17202a; --muted:#65717d; --line:#dce2e8;
        --blue:#2563eb; --green:#15803d; --red:#b91c1c; --amber:#b45309; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:24px; }
header { display:flex; align-items:center; gap:12px; border-bottom:1px solid var(--line); padding-bottom:16px; }
h1 { margin:0; font-size:22px; } .readonly { color:var(--muted); font-size:12px; border:1px solid var(--line); border-radius:999px; padding:3px 9px; }
.status { margin-left:auto; display:flex; gap:8px; align-items:center; font-weight:600; }
.dot { width:10px; height:10px; border-radius:50%; background:#94a3b8; } .status.running .dot { background:var(--green); }
.status.completed .dot { background:var(--blue); } .status.failed .dot { background:var(--red); }
.grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:18px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
.value { margin-top:4px; font:600 20px ui-monospace,SFMono-Regular,Consolas,monospace; word-break:break-word; }
.wide { margin-top:14px; } .wide h2 { margin:0 0 10px; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.columns { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
pre { margin:0; white-space:pre-wrap; word-break:break-word; font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; }
.error { color:var(--red); } .ok { color:var(--green); }
table { width:100%; border-collapse:collapse; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }
th,td { padding:6px 5px; border-bottom:1px solid #eef1f4; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:500; } td.num { text-align:right; } td.msg { max-width:480px; word-break:break-word; }
@media(max-width:760px) { .grid,.columns { grid-template-columns:1fr 1fr; } }
@media(max-width:480px) { .grid,.columns { grid-template-columns:1fr; } .status { margin-left:0; } header { flex-wrap:wrap; } }
</style>
</head>
<body><main class="wrap">
<header><h1>WorldLab Runtime</h1><span class="readonly">只读面板</span><span id="status" class="status"><i class="dot"></i><span>__INITIAL_STATUS__</span></span></header>
<section class="grid">
  <div class="card"><div class="label">episode</div><div id="episode" class="value">__INITIAL_EPISODE__</div></div>
  <div class="card"><div class="label">step</div><div id="step" class="value">__INITIAL_STEP__</div></div>
  <div class="card"><div class="label">total reward</div><div id="reward" class="value">__INITIAL_REWARD__</div></div>
  <div class="card"><div class="label">last event</div><div id="event" class="value">__INITIAL_EVENT__</div></div>
</section>
<section class="wide columns">
  <div class="card"><h2>当前快照</h2><pre id="snapshot">__INITIAL_SNAPSHOT__</pre></div>
  <div class="card"><h2>错误诊断</h2><pre id="error">__INITIAL_ERROR__</pre></div>
</section>
<section class="wide card"><h2>事件时间线</h2><table><thead><tr><th>#</th><th>事件</th><th>episode/step</th><th>耗时</th><th>摘要</th></tr></thead><tbody id="events"><tr><td colspan="5">等待事件…</td></tr></tbody></table></section>
<p style="color:var(--muted);font-size:12px;text-align:center">只读轮询：/api/status · /api/events · /healthz</p>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "—").replace(/[&<>"]/g, ch => ({
  "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;"
}[ch]));
function renderStatus(s) {
  const el = $("status"); el.className = "status " + s.status; el.querySelector("span:last-child").textContent = s.status;
  $("episode").textContent = s.episode_index ?? "—"; $("step").textContent = s.step_index;
  $("reward").textContent = Number(s.total_reward || 0).toFixed(3); $("event").textContent = s.last_event || "—";
  $("snapshot").textContent = JSON.stringify({status:s.status, phase:s.phase, observation:s.observation, action:s.action, next_observation:s.next_observation, reward:s.reward, terminated:s.terminated, truncated:s.truncated, info:s.info}, null, 2);
  $("error").innerHTML = s.error_message ? '<span class="error">' + esc(s.error_type + ": " + s.error_message) + '</span>' : '<span class="ok">暂无错误</span>';
}
function renderEvents(items) {
  $("events").innerHTML = items.length ? items.slice().reverse().map(e => '<tr><td class="num">' + e.sequence + '</td><td>' + esc(e.kind) + '</td><td>' + e.episode_index + '/' + e.step_index + '</td><td>' + Number(e.duration_s).toFixed(6) + 's</td><td class="msg">' + esc(e.message || e.observation || e.action || (e.reward !== undefined ? 'reward=' + e.reward : '')) + '</td></tr>').join('') : '<tr><td colspan="5">等待事件…</td></tr>';
}
function renderUnreachable(error) {
  const el = $("status");
  el.className = "status failed";
  el.querySelector("span:last-child").textContent = "unreachable";
  $("snapshot").textContent = "无法连接 dashboard 服务。请确认启动命令仍在运行。";
  $("error").innerHTML = '<span class="error">' + esc(String(error)) + '</span>';
}
async function getJson(path) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch(path, {cache:'no-store', signal:controller.signal});
    if (!response.ok) throw new Error(path + ' HTTP ' + response.status);
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}
async function tick() {
  try {
    const s = await getJson('/api/status');
    const e = await getJson('/api/events?limit=100');
    renderStatus(s);
    renderEvents(e.events);
  } catch (error) {
    renderUnreachable(error);
  }
}
tick(); setInterval(tick, __POLL_INTERVAL_MS__);
</script></body></html>"""

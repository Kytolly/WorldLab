"""Panel view for a WorldLab :class:`TraceSource`.

The UI only reads immutable runtime events.  It never calls an environment,
agent, or simulator method, so it can be replaced by a remote event reader
without changing the view code.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from io import BytesIO
from typing import Any, Callable, Mapping

import numpy as np
import panel as pn
from PIL import Image

from worldlab import TraceSource
from worldlab.data import (
    EnvironmentStepped,
    RuntimeEvent,
    RuntimeEventKind,
    RuntimeSnapshot,
    SIMULATION_CHUNK_INDEX,
    SIMULATION_FRAMES,
    SIMULATION_MODEL_LATENCY_S,
    SIMULATION_STATE,
)


_CSS = """
:host { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.wl-shell { max-width: 1440px; margin: 0 auto; color: #17202a; }
.wl-header { display: flex; align-items: baseline; gap: 12px; border-bottom: 1px solid #d9e0e7; padding: 8px 0 12px; }
.wl-title { font-size: 20px; font-weight: 650; }
.wl-subtitle { color: #667085; font-size: 12px; }
.wl-status { margin-left: auto; font-weight: 600; font-size: 13px; }
.wl-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: #98a2b3; }
.wl-dot.running { background: #12b76a; }
.wl-dot.completed { background: #1570ef; }
.wl-dot.failed { background: #d92d20; }
.wl-metrics { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin: 14px 0; }
.wl-metric { border: 1px solid #d9e0e7; padding: 10px 12px; background: #fff; }
.wl-metric-label { color: #667085; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
.wl-metric-value { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 20px; margin-top: 3px; }
.wl-section { border-top: 1px solid #d9e0e7; padding-top: 12px; margin-top: 14px; }
.wl-section h2 { color: #475467; font-size: 13px; margin: 0 0 8px; }
.wl-frame { width: 100%; min-height: 220px; object-fit: contain; background: #f2f4f7; border: 1px solid #d9e0e7; }
.wl-table-wrap { overflow: auto; max-height: 260px; border: 1px solid #d9e0e7; background: #fff; }
.wl-table { border-collapse: collapse; width: 100%; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; white-space: nowrap; }
.wl-table th { position: sticky; top: 0; background: #f8fafc; color: #667085; font-weight: 600; text-align: right; }
.wl-table th:first-child, .wl-table td:first-child { text-align: left; }
.wl-table th, .wl-table td { padding: 4px 7px; border-bottom: 1px solid #eef2f6; }
.wl-empty { color: #667085; border: 1px dashed #d9e0e7; padding: 30px; text-align: center; }
.wl-events { max-height: 300px; }
@media (max-width: 850px) { .wl-metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); } }
"""


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def _as_matrix(value: Any) -> np.ndarray[Any, Any] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0 or array.dtype.kind not in "biufc":
        return None
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def _latest_step(source: TraceSource) -> EnvironmentStepped[Any, Any] | None:
    for event in reversed(source.events):
        if isinstance(event, EnvironmentStepped):
            return event
    return None


def _frame_png(frames: Any) -> bytes | None:
    """Convert the latest ``(T, C, V, H, W)`` frame to a tiled PNG."""

    try:
        array = np.asarray(frames, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.ndim != 5 or array.shape[0] == 0 or array.shape[1] != 3:
        return None
    frame = np.clip(array[-1], 0.0, 1.0)
    views = np.transpose(frame, (1, 2, 3, 0))
    tiled = np.concatenate(
        [(view * 255.0).round().astype(np.uint8) for view in views],
        axis=1,
    )
    output = BytesIO()
    Image.fromarray(tiled, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _matrix_table(title: str, value: Any, *, max_rows: int = 32) -> str:
    matrix = _as_matrix(value)
    if matrix is None:
        return f'<div class="wl-section"><h2>{escape(title)}</h2><div class="wl-empty">等待数据</div></div>'
    rows, columns = matrix.shape
    shown_rows = min(rows, max_rows)
    headers = "".join(f"<th>d{index}</th>" for index in range(columns))
    body: list[str] = []
    for row_index, row in enumerate(matrix[:shown_rows]):
        values = "".join(f"<td>{_format_number(item)}</td>" for item in row)
        body.append(f"<tr><td>t{row_index}</td>{values}</tr>")
    suffix = f"<div class=\"wl-subtitle\">显示 {shown_rows}/{rows} 行</div>" if rows > shown_rows else ""
    return (
        f'<div class="wl-section"><h2>{escape(title)}</h2>'
        f'<div class="wl-table-wrap"><table class="wl-table"><thead><tr><th>step</th>{headers}'
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>{suffix}</div>"
    )


def _event_summary(event: RuntimeEvent) -> str:
    if isinstance(event, EnvironmentStepped):
        chunk = event.info.get(SIMULATION_CHUNK_INDEX, "-")
        latency = event.info.get(SIMULATION_MODEL_LATENCY_S)
        return f"reward={_format_number(event.reward)} chunk={chunk} model={_format_number(latency, 4)}s"
    if event.kind is RuntimeEventKind.POLICY_ACTED:
        return "action emitted"
    if event.kind is RuntimeEventKind.TRANSITION_COMMITTED:
        return "transition committed"
    if event.kind is RuntimeEventKind.RUNTIME_ERROR:
        return "runtime error"
    return str(event.kind.value).replace("_", " ")


def _timeline_table(source: TraceSource) -> str:
    events = source.events[-80:]
    if not events:
        return '<div class="wl-empty">等待运行事件</div>'
    rows = []
    for event in reversed(events):
        rows.append(
            "<tr>"
            f"<td>{event.sequence}</td><td>{escape(event.kind.value)}</td>"
            f"<td>{event.episode_index}/{event.step_index}</td>"
            f"<td>{event.duration_s:.4f}s</td><td>{escape(_event_summary(event))}</td>"
            "</tr>"
        )
    return (
        '<div class="wl-table-wrap wl-events"><table class="wl-table"><thead><tr>'
        "<th>#</th><th>event</th><th>episode/step</th><th>duration</th><th>summary</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _metrics(snapshot: RuntimeSnapshot, step: EnvironmentStepped[Any, Any] | None) -> str:
    info: Mapping[str, Any] = step.info if step is not None else snapshot.info
    chunk = info.get(SIMULATION_CHUNK_INDEX, "-")
    latency = info.get(SIMULATION_MODEL_LATENCY_S)
    values = (
        ("status", snapshot.status.value),
        ("episode / step", f"{snapshot.episode_index if snapshot.episode_index is not None else '-'} / {snapshot.step_index}"),
        ("total reward", _format_number(snapshot.total_reward)),
        ("chunk", chunk),
        ("model latency", f"{_format_number(latency, 4)} s"),
    )
    cards = "".join(
        f'<div class="wl-metric"><div class="wl-metric-label">{escape(str(label))}</div>'
        f'<div class="wl-metric-value">{escape(str(value))}</div></div>'
        for label, value in values
    )
    return f'<div class="wl-metrics">{cards}</div>'


@dataclass
class PanelDashboard:
    """Live read-only Panel dashboard backed by a ``TraceSource``."""

    source: TraceSource
    poll_interval_ms: int = 500

    def __post_init__(self) -> None:
        if self.poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be greater than zero")
        pn.extension(raw_css=[_CSS])
        self._header = pn.pane.HTML(sizing_mode="stretch_width")  # type: ignore[no-untyped-call]
        self._metrics = pn.pane.HTML(sizing_mode="stretch_width")  # type: ignore[no-untyped-call]
        self._frame = pn.pane.PNG(height=300, sizing_mode="stretch_width")  # type: ignore[no-untyped-call]
        self._tables = pn.pane.HTML(sizing_mode="stretch_width")  # type: ignore[no-untyped-call]
        self._timeline = pn.pane.HTML(sizing_mode="stretch_width")  # type: ignore[no-untyped-call]
        self._view = pn.Column(
            self._header,
            self._metrics,
            pn.Row(self._frame, sizing_mode="stretch_width"),
            self._tables,
            pn.pane.HTML('<div class="wl-section"><h2>事件时间线</h2></div>'),  # type: ignore[no-untyped-call]
            self._timeline,
            sizing_mode="stretch_width",
            css_classes=["wl-shell"],
        )
        self.refresh()
        self._periodic = pn.state.add_periodic_callback(
            self.refresh,
            period=self.poll_interval_ms,
            start=True,
        )

    def panel(self) -> pn.Column:
        """Return the Panel view to pass to ``pn.serve``."""

        return self._view

    def refresh(self) -> None:
        snapshot = self.source.snapshot
        step = _latest_step(self.source)
        status = snapshot.status.value
        self._header.object = (
            '<div class="wl-header"><span class="wl-title">WorldLab Runtime</span>'
            '<span class="wl-subtitle">read-only Panel</span>'
            f'<span class="wl-status"><span class="wl-dot {escape(status)}"></span>{escape(status)}</span></div>'
        )
        self._metrics.object = _metrics(snapshot, step)
        frames = step.info.get(SIMULATION_FRAMES, step.info.get("frames")) if step else None
        png = _frame_png(frames)
        if png is None:
            self._frame.object = None
            self._frame.alt_text = "等待 synthetic frames"
        else:
            self._frame.object = png
            self._frame.alt_text = "latest generated multi-view frame"
        action = step.action if step else None
        state = step.info.get(SIMULATION_STATE, step.info.get("state")) if step else None
        self._tables.object = (
            '<div class="wl-section"><h2>动作序列</h2></div>'
            + _matrix_table("action", action)
            + _matrix_table("predicted state", state)
        )
        self._timeline.object = _timeline_table(self.source)

    def stop(self) -> None:
        """Stop periodic refresh callbacks when the host owns the lifecycle."""

        self._periodic.stop()  # type: ignore[no-untyped-call]


def create_panel(source: TraceSource, *, poll_interval_ms: int = 500) -> pn.Column:
    """Create a live Panel view inside the current Panel session."""

    return PanelDashboard(source, poll_interval_ms=poll_interval_ms).panel()


def create_panel_app(
    source: TraceSource,
    *,
    poll_interval_ms: int = 500,
) -> Callable[[], pn.Column]:
    """Return a session factory suitable for :func:`panel.serve`.

    Panel periodic callbacks belong to a browser session document.  Delaying
    dashboard construction until Panel opens that document keeps live refresh
    attached to the page instead of to the process-global bootstrap document.
    """

    def app() -> pn.Column:
        return create_panel(source, poll_interval_ms=poll_interval_ms)

    return app

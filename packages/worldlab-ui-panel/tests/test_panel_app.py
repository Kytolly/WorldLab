from __future__ import annotations

import numpy as np
import panel as pn

from worldlab import EventBuffer, load_config, run_configured_demo
from worldlab_ui_panel.app import (
    PanelDashboard,
    _frame_png,
    _latest_step,
    create_panel_app,
)


def test_panel_dashboard_reads_runtime_events_and_frames() -> None:
    source = EventBuffer(max_events=64)
    config = load_config(
        overrides=[
            "environment.task.goal=2",
            "observability.dashboard.enabled=false",
        ]
    )
    run_configured_demo(config, trace=source)

    dashboard = PanelDashboard(source, poll_interval_ms=1000)
    try:
        assert isinstance(dashboard.panel(), pn.Column)
        step = _latest_step(source)
        assert step is not None
        frames = step.info["worldlab.simulation.frames"]
        assert _frame_png(frames) is not None
    finally:
        dashboard.stop()


def test_frame_encoder_rejects_non_video_values() -> None:
    assert _frame_png(np.zeros((3, 4), dtype=np.float32)) is None


def test_panel_application_constructs_dashboard_lazily(monkeypatch) -> None:
    source = EventBuffer(max_events=8)
    constructed: list[tuple[EventBuffer, int]] = []
    view = pn.Column("session view")

    class StubDashboard:
        def __init__(self, received_source: EventBuffer, poll_interval_ms: int) -> None:
            constructed.append((received_source, poll_interval_ms))

        def panel(self) -> pn.Column:
            return view

    monkeypatch.setattr("worldlab_ui_panel.app.PanelDashboard", StubDashboard)
    app = create_panel_app(source, poll_interval_ms=250)

    assert constructed == []
    assert app() is view
    assert constructed == [(source, 250)]

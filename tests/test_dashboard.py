from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from worldlab import DashboardServer, EnvironmentLoop, EventBuffer, build_demo, run_demo
from worldlab.__main__ import main


def _get(url: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(url, timeout=3) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()


def test_read_only_dashboard_exposes_snapshot_and_events() -> None:
    source = EventBuffer(max_events=32)
    with DashboardServer(source, port=0) as dashboard:
        status, content_type, html = _get(dashboard.url)
        assert status == 200
        assert "text/html" in content_type
        assert "只读面板".encode("utf-8") in html

        status, _, health = _get(f"http://127.0.0.1:{dashboard.port}/healthz")
        assert status == 200
        assert json.loads(health)["status"] == "ok"

        _, _, initial_status = _get(f"http://127.0.0.1:{dashboard.port}/api/status")
        assert json.loads(initial_status)["status"] == "idle"

        env, agent = build_demo(goal=2)
        with EnvironmentLoop(env, agent, trace=source) as loop:
            result = loop.run_episode(seed=0)

        _, _, status_body = _get(f"http://127.0.0.1:{dashboard.port}/api/status")
        snapshot = json.loads(status_body)
        assert result.total_reward == 2.0
        assert snapshot["status"] == "completed"
        assert snapshot["sequence"] == 8
        assert snapshot["step_index"] == 2
        assert snapshot["terminated"] is True
        assert snapshot["total_reward"] == 2.0

        _, _, rendered_html = _get(dashboard.url)
        assert b"completed" in rendered_html
        assert b"sequence" in rendered_html
        assert b"__INITIAL_" not in rendered_html
        assert b"'\"':\"&quot;\"" in rendered_html
        assert b'""":' not in rendered_html

        _, _, events_body = _get(
            f"http://127.0.0.1:{dashboard.port}/api/events?after=6&limit=10"
        )
        events = json.loads(events_body)
        assert [event["sequence"] for event in events["events"]] == [7, 8]
        assert events["latest_sequence"] == 8

        request = urllib.request.Request(
            f"http://127.0.0.1:{dashboard.port}/api/status",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        assert error.value.code == 405


def test_dashboard_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="port"):
        DashboardServer(EventBuffer(), port=65536)

    with pytest.raises(ValueError, match="poll_interval_s"):
        DashboardServer(EventBuffer(), poll_interval_s=0.0)


def test_dashboard_observes_an_intermediate_running_step() -> None:
    source = EventBuffer(max_events=32)
    results: list[object] = []

    def run() -> None:
        results.append(run_demo(goal=2, seed=0, trace=source, step_delay=0.2))

    with DashboardServer(source, port=0) as dashboard:
        worker = threading.Thread(target=run)
        worker.start()
        time.sleep(0.05)

        _, _, status_body = _get(f"http://127.0.0.1:{dashboard.port}/api/status")
        snapshot = json.loads(status_body)
        assert snapshot["status"] == "running"
        assert snapshot["sequence"] >= 3

        worker.join(timeout=3.0)
        assert not worker.is_alive()
        assert len(results) == 1

        _, _, final_body = _get(f"http://127.0.0.1:{dashboard.port}/api/status")
        assert json.loads(final_body)["status"] == "completed"


def test_cli_starts_dashboard_without_enabling_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldlab",
            "--goal",
            "1",
            "--dashboard",
            "--dashboard-port",
            "0",
            "--dashboard-seconds",
            "0",
            "--dashboard-step-delay",
            "0",
        ],
    )

    assert main() == 0

    output = capsys.readouterr().out
    assert "WorldLab dashboard: http://127.0.0.1:" in output

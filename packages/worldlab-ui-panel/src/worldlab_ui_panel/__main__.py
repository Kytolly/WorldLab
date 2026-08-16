"""Run the optional Panel dashboard with a deterministic synthetic demo."""

from __future__ import annotations

import argparse
import threading

import panel as pn

from worldlab import EventBuffer, load_config, run_configured_demo

from .app import PanelDashboard


def main() -> int:
    parser = argparse.ArgumentParser(description="WorldLab Panel dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--goal", type=int, default=8)
    parser.add_argument("--step-delay", type=float, default=0.6)
    parser.add_argument("--poll-interval-ms", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="open a browser window")
    args = parser.parse_args()
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.goal <= 0:
        parser.error("--goal must be greater than zero")
    if args.step_delay < 0.0:
        parser.error("--step-delay must be non-negative")

    source = EventBuffer(max_events=max(128, args.goal * 8))

    def run() -> None:
        config = load_config(
            overrides=[
                f"environment.task.goal={args.goal}",
                f"runtime.step_delay_s={args.step_delay}",
                "observability.dashboard.enabled=false",
            ]
        )
        run_configured_demo(config, trace=source)

    threading.Thread(target=run, name="worldlab-panel-demo", daemon=True).start()
    dashboard = PanelDashboard(source, poll_interval_ms=args.poll_interval_ms)
    print(f"WorldLab Panel: http://{args.host}:{args.port}", flush=True)
    pn.serve(
        dashboard.panel(),
        address=args.host,
        port=args.port,
        websocket_origin=[f"{args.host}:{args.port}", f"localhost:{args.port}"],
        show=args.show,
        title="WorldLab Runtime",
    )  # type: ignore[no-untyped-call]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

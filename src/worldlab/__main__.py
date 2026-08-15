"""Command-line entry point for the built-in WorldLab demo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .acceptance import run_acceptance
from .config import ConfigError, load_config
from .demo import run_configured_demo, run_demo
from .runtime import DashboardServer, TraceRecorder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the built-in WorldLab demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="load a layered YAML configuration through OmegaConf",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="apply an OmegaConf dotlist override, for example rollout.chunk_size=1",
    )
    parser.add_argument("--model", choices=("counter",), default="counter")
    parser.add_argument("--goal", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="run deterministic end-to-end acceptance checks and exit",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print the closed-loop runtime timeline and diagnosis",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="serve the read-only runtime dashboard while the demo is observable",
    )
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument(
        "--dashboard-step-delay",
        type=float,
        default=0.6,
        help="seconds between demo transitions while dashboard is enabled",
    )
    parser.add_argument(
        "--dashboard-seconds",
        type=float,
        default=30.0,
        help="seconds to keep the dashboard alive after the demo (0 = no wait)",
    )
    args = parser.parse_args()

    if args.config is not None:
        if args.acceptance:
            parser.error("--acceptance cannot be combined with --config")
        try:
            config = load_config(args.config, overrides=args.override)
        except ConfigError as error:
            parser.error(str(error))
        values = config
        trace_config = values.observability.trace
        trace = (
            TraceRecorder(max_events=int(trace_config.max_events))
            if bool(trace_config.enabled)
            else None
        )
        dashboard_config = values.observability.dashboard
        dashboard = (
            DashboardServer(
                trace,
                host=str(dashboard_config.host),
                port=int(dashboard_config.port),
                poll_interval_s=float(dashboard_config.poll_interval_s),
            )
            if trace is not None and bool(dashboard_config.enabled)
            else None
        )
        if dashboard is not None:
            dashboard.start()
            print(f"WorldLab dashboard: {dashboard.url}", flush=True)
        try:
            result = run_configured_demo(config, trace=trace)
        except Exception:
            if dashboard is not None:
                dashboard.stop()
            if trace is not None:
                print("closed_loop_trace")
                print(trace.format_timeline())
            raise
        finally:
            if dashboard is not None and float(dashboard_config.keep_alive_s) == 0.0:
                dashboard.stop()
        print("WorldLab configured demo")
        print(f"total_reward={result.total_reward}")
        print(f"length={result.length}")
        print(f"terminated={result.terminated}")
        print(f"truncated={result.truncated}")
        if trace is not None:
            print("closed_loop_trace")
            print(trace.format_timeline())
        if dashboard is not None and float(dashboard_config.keep_alive_s) > 0.0:
            print(
                "Dashboard remains available for "
                f"{float(dashboard_config.keep_alive_s):.1f}s"
            )
            try:
                time.sleep(float(dashboard_config.keep_alive_s))
            finally:
                dashboard.stop()
        return 0

    if args.dashboard_seconds < 0.0:
        parser.error("--dashboard-seconds must be non-negative")
    if args.dashboard_step_delay < 0.0:
        parser.error("--dashboard-step-delay must be non-negative")
    if args.acceptance:
        if args.dashboard:
            parser.error("--acceptance cannot be combined with --dashboard")
        if args.model != "counter" or args.random_policy:
            parser.error("--acceptance requires the deterministic counter demo")
        report = run_acceptance(goal=args.goal, seed=args.seed)
        print(report.format())
        return 0 if report.passed else 1

    trace = TraceRecorder(max_events=4096) if args.dashboard else None
    if args.trace and trace is None:
        trace = TraceRecorder()
    dashboard = (
        DashboardServer(trace, port=args.dashboard_port)
        if trace is not None and args.dashboard
        else None
    )
    if dashboard is not None:
        dashboard.start()
        print(f"WorldLab dashboard: {dashboard.url}", flush=True)

    try:
        result = run_demo(
            model=args.model,
            goal=args.goal,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            random_policy=args.random_policy,
            trace=trace,
            step_delay=args.dashboard_step_delay if args.dashboard else 0.0,
        )
    except Exception:
        if dashboard is not None:
            dashboard.stop()
        if trace is not None:
            print("closed_loop_trace")
            print(trace.format_timeline())
        raise
    finally:
        if dashboard is not None and args.dashboard_seconds == 0.0:
            dashboard.stop()

    print("WorldLab demo")
    print(f"total_reward={result.total_reward}")
    print(f"length={result.length}")
    print(f"terminated={result.terminated}")
    print(f"truncated={result.truncated}")
    if trace is not None and args.trace:
        print("closed_loop_trace")
        print(trace.format_timeline())
    if dashboard is not None and args.dashboard_seconds > 0.0:
        print(f"Dashboard remains available for {args.dashboard_seconds:.1f}s")
        try:
            time.sleep(args.dashboard_seconds)
        finally:
            dashboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

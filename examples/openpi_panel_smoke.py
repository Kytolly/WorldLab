"""End-to-end smoke test: OpenPI service -> ExampleWorldModel -> Panel.

This connects to an already-running OpenPI-compatible service. It bridges the
current array-based ExampleEnvironment observation to the structured
OpenPIObservation used by the optional OpenPI integration.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

import numpy as np
import panel as pn
from numpy.typing import NDArray

from worldlab import (
    EnvironmentLoop,
    EventBuffer,
    LoopConfig,
    Policy,
    PolicyAgent,
    build_configured_demo,
    load_config,
)
from worldlab.data import PolicyOutput
from worldlab.data.runtime import RuntimeErrorEvent, RuntimePhase
from worldlab_ui_panel import create_panel_app
from worldlab_openpi import OpenPIObservation, OpenPIPolicy


Array = NDArray[Any]


class _ExampleOpenPIAdapter(Policy[Any, Array]):
    """Bridge ExampleEnvironment arrays to the structured OpenPI observation."""

    def __init__(self, delegate: OpenPIPolicy, frame_shape: tuple[int, ...], task: str) -> None:
        self.delegate = delegate
        self.frame_shape = frame_shape
        self.task = task
        self._last_frame: Array | None = None

    def reset(self, *, seed: Optional[int] = None) -> None:
        del seed
        self.delegate.reset()
        self._last_frame = None

    def act(
        self,
        observation: Any,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[Array]:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape == self.frame_shape:
            self._last_frame = value
            state = np.zeros(16, dtype=np.float32)
        elif value.ndim == 2 and value.shape[1] == 16:
            state = value[-1].astype(np.float32, copy=False)
            frames = info.get("frames")
            if frames is not None:
                self._last_frame = np.asarray(frames, dtype=np.float32)[-1]
        else:
            raise ValueError(f"unexpected ExampleEnvironment observation shape {value.shape}")

        if self._last_frame is None:
            raise RuntimeError("OpenPI bridge has no current frame")
        frame = np.clip(self._last_frame, 0.0, 1.0)
        images = {
            name: (frame[:, index].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            for index, name in enumerate(("head", "left_wrist", "right_wrist"))
        }
        return self.delegate.act(
            OpenPIObservation(images=images, state=state, task=self.task),
            info=info,
            deterministic=deterministic,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="WorldLab OpenPI + Panel smoke test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5018)
    parser.add_argument("--policy-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--goal", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--step-delay", type=float, default=0.5)
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for the OpenPI WebSocket port before reporting failure",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.goal <= 0 or args.chunk_size <= 0:
        parser.error("goal and chunk-size must be greater than zero")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be greater than zero")

    config = load_config(
        overrides=[
            f"rollout.chunk_size={args.chunk_size}",
            f"environment.task.goal={args.goal}",
            f"runtime.step_delay_s={args.step_delay}",
            "observability.dashboard.enabled=false",
        ]
    )
    environment, _, options = build_configured_demo(config)
    model_frame_shape = tuple(int(value) for value in environment.observation_space.shapes[0])
    source = EventBuffer(max_events=max(256, args.goal * 8))

    loop_config = LoopConfig(training=False, deterministic=True, validate_spaces=True)

    def connect_policy(url: str, timeout_s: float) -> OpenPIPolicy:
        parsed = urlsplit(url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise ValueError(
                "--policy-url must use ws:// or wss://, for example "
                "ws://127.0.0.1:8000; HTTP requests produce OpenPI's 426 response"
            )
        result: list[OpenPIPolicy] = []
        errors: list[Exception] = []

        def create_client() -> None:
            try:
                result.append(OpenPIPolicy(url, action_horizon=args.chunk_size))
            except Exception as error:
                errors.append(error)

        # The official client retries connection-refused forever. Keep that
        # wait off Panel's event loop, but let the UI report a bounded failure.
        connector = threading.Thread(
            target=create_client,
            name="worldlab-openpi-connect",
            daemon=True,
        )
        connector.start()
        connector.join(timeout_s)
        if connector.is_alive():
            raise TimeoutError(
                f"OpenPI WebSocket handshake for {url} did not complete "
                f"within {timeout_s:.1f}s"
            )
        if errors:
            raise errors[0]
        if not result:
            raise RuntimeError("OpenPI client exited without a policy instance")
        return result[0]

    def run_loop() -> None:
        try:
            print(f"Waiting for OpenPI WebSocket: {args.policy_url}", flush=True)
            openpi = connect_policy(args.policy_url, args.connect_timeout)
            agent = PolicyAgent(
                _ExampleOpenPIAdapter(openpi, model_frame_shape, "synthetic OpenPI task")
            )
            print("OpenPI handshake completed; starting rollout", flush=True)
        except Exception as error:
            print(f"OpenPI connection failed: {error}", flush=True)
            traceback.print_exc()
            source.record(
                RuntimeErrorEvent(
                    sequence=1,
                    timestamp_s=time.time(),
                    monotonic_s=time.perf_counter(),
                    episode_index=0,
                    step_index=0,
                    duration_s=0.0,
                    phase=RuntimePhase.AGENT_RESET,
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback=traceback.format_exc(),
                )
            )
            environment.close()
            return
        try:
            with EnvironmentLoop(
                environment,
                agent,
                config=loop_config,
                trace=source,
            ) as loop:
                loop.run_episode(seed=0, options=options)
        finally:
            environment.close()

    threading.Thread(target=run_loop, name="worldlab-openpi-smoke", daemon=True).start()
    print(f"OpenPI service: {args.policy_url}", flush=True)
    print(f"WorldLab Panel: http://{args.host}:{args.port}", flush=True)
    pn.serve(
        create_panel_app(source, poll_interval_ms=200),
        address=args.host,
        port=args.port,
        websocket_origin=[f"{args.host}:{args.port}", f"localhost:{args.port}"],
        show=args.show,
        title="WorldLab OpenPI Smoke",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

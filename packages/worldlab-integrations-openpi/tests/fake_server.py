"""Test-only OpenPI-compatible WebSocket server."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy  # type: ignore[import-untyped]
from websockets.asyncio.server import Server, serve


FakePolicy = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class FakeOpenPIServer:
    """A real localhost WebSocket fixture for protocol acceptance tests."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        action_horizon: int = 4,
        action_dim: int = 16,
        action_layout: str = "policy",
        policy: FakePolicy | None = None,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if port < 0 or port > 65535:
            raise ValueError("port must be between 0 and 65535")
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        self.host = host
        self._requested_port = port
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.action_layout = action_layout
        self._policy = policy or self._default_policy
        self._requests: list[Mapping[str, Any]] = []
        self._requests_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: Server | None = None
        self._port: int | None = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("fake server has not been started")
        return self._port

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def requests(self) -> tuple[Mapping[str, Any], ...]:
        with self._requests_lock:
            return tuple(self._requests)

    def start(self, *, timeout_s: float = 5.0) -> "FakeOpenPIServer":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("fake OpenPI server did not start")
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        self._server = None

    def __enter__(self) -> "FakeOpenPIServer":
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()

    async def _handle(self, websocket: Any) -> None:
        metadata = {
            "name": "worldlab-test-openpi",
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "action_layout": self.action_layout,
        }
        await websocket.send(msgpack_numpy.packb(metadata))
        async for message in websocket:
            if not isinstance(message, (bytes, bytearray)):
                await websocket.send("expected binary msgpack payload")
                continue
            payload = msgpack_numpy.unpackb(bytes(message))
            if not isinstance(payload, Mapping):
                await websocket.send("payload must decode to a mapping")
                continue
            with self._requests_lock:
                self._requests.append(dict(payload))
            try:
                response = self._policy(payload)
                await websocket.send(msgpack_numpy.packb(dict(response)))
            except Exception as error:  # pragma: no cover - defensive server path
                await websocket.send(f"fake policy error: {type(error).__name__}: {error}")

    async def _serve(self) -> None:
        async with serve(self._handle, self.host, self._requested_port) as server:
            self._server = server
            sockets = server.sockets
            if not sockets:
                raise RuntimeError("fake OpenPI server has no listening socket")
            self._port = int(next(iter(sockets)).getsockname()[1])
            self._ready.set()
            await asyncio.get_running_loop().run_in_executor(None, self._stop.wait)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        finally:
            self._ready.set()

    def _default_policy(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        del payload
        return {
            "actions": np.zeros(
                (self.action_horizon, self.action_dim),
                dtype=np.float32,
            ),
            "server_timing": {"infer_ms": 0.0},
        }

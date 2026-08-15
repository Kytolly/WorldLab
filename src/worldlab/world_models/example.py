"""Deterministic synthetic chunk-level World Model for v0.2.1."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from worldlab.data import WorldModelContext, WorldModelStepResult

from .base import WorldModel


Array = NDArray[Any]


class ExampleWorldModel(WorldModel[int, Array, Array]):
    """Generate action-conditioned synthetic multi-view frames."""

    def __init__(
        self,
        *,
        chunk_size: int,
        num_views: int = 3,
        frame_height: int = 32,
        frame_width: int = 32,
        state_dim: int = 16,
        seed: int = 0,
        noise_scale: float = 0.01,
    ) -> None:
        for name, value in (
            ("chunk_size", chunk_size),
            ("num_views", num_views),
            ("frame_height", frame_height),
            ("frame_width", frame_width),
            ("state_dim", state_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative")

        self.chunk_size = int(chunk_size)
        self.num_views = int(num_views)
        self.frame_height = int(frame_height)
        self.frame_width = int(frame_width)
        self.state_dim = int(state_dim)
        self.seed = int(seed)
        self.noise_scale = float(noise_scale)
        self._rng = np.random.default_rng(self.seed)
        self._first_frame: Optional[Array] = None
        self._trajectory: Optional[Array] = None
        self._c2w: Optional[Array] = None
        self._intrinsic: Optional[Array] = None
        self._extrinsic: Optional[Array] = None
        self._task = ""
        self._chunk_index = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ExampleWorldModel":
        if "chunk_size" not in config:
            raise ValueError("ExampleWorldModel config requires chunk_size")
        return cls(
            chunk_size=int(config["chunk_size"]),
            num_views=int(config.get("num_views", 3)),
            frame_height=int(config.get("frame_height", 32)),
            frame_width=int(config.get("frame_width", 32)),
            state_dim=int(config.get("state_dim", 16)),
            seed=int(config.get("seed", 0)),
            noise_scale=float(config.get("noise_scale", 0.01)),
        )

    @property
    def frame_shape(self) -> tuple[int, int, int, int]:
        return (3, self.num_views, self.frame_height, self.frame_width)

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    def initialize(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> WorldModelContext[int, Array]:
        self._rng = np.random.default_rng(self.seed if seed is None else seed)
        self._chunk_index = 0
        self._first_frame = None
        self._trajectory = None
        self._c2w = None
        self._intrinsic = None
        self._extrinsic = None
        self._task = ""

        conditions = options or {}
        self._load_camera_conditions(
            conditions.get("intrinsic"),
            conditions.get("extrinsic"),
        )
        self._load_first_frame(conditions.get("first_frame"))
        self._load_trajectory_conditions(
            conditions.get("trajectory"),
            conditions.get("c2w"),
        )
        self._task = str(conditions.get("task", ""))
        assert self._first_frame is not None
        return WorldModelContext(
            context=0,
            state=self._first_frame.copy(),
            info={
                "chunk_size": self.chunk_size,
                "task": self._task,
                "frame_shape": self.frame_shape,
            },
        )

    def sample_step(
        self,
        context: int,
        action_chunk: Array,
    ) -> WorldModelStepResult[int, Array]:
        del context
        self._ensure_ready()
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.state_dim:
            raise ValueError(
                f"action_chunk must have shape (L, {self.state_dim}), got {actions.shape}"
            )
        if actions.shape[0] <= 0 or actions.shape[0] > self.chunk_size:
            raise ValueError(
                "action_chunk length must be between 1 and chunk_size "
                f"({self.chunk_size}), got {actions.shape[0]}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("action_chunk must contain only finite values")

        padded_actions = self._pad_actions(actions)
        action_signal = np.tanh(padded_actions.mean(axis=1)).astype(np.float32)
        ramp = np.linspace(0.0, 0.25, self.chunk_size, dtype=np.float32)
        noise = self._rng.normal(
            loc=0.0,
            scale=self.noise_scale,
            size=(self.chunk_size, *self.frame_shape),
        ).astype(np.float32)
        assert self._first_frame is not None
        frames = (
            self._first_frame[None]
            + ramp[:, None, None, None, None]
            + 0.05 * action_signal[:, None, None, None, None]
            + noise
        )
        frames = np.clip(frames, 0.0, 1.0).astype(np.float32)
        self._chunk_index += 1
        return WorldModelStepResult(
            context=self._chunk_index,
            state=padded_actions.copy(),
            frames=frames,
            info={
                "chunk_index": self._chunk_index,
                "action_length": int(actions.shape[0]),
                "output_length": self.chunk_size,
                "task": self._task,
            },
        )

    def close(self) -> None:
        self._first_frame = None
        self._trajectory = None
        self._c2w = None
        self._intrinsic = None
        self._extrinsic = None

    def _load_camera_conditions(self, intrinsic: Any, extrinsic: Any) -> None:
        if intrinsic is None:
            return
        intrinsic_array = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic_array.shape != (self.num_views, 3, 3):
            raise ValueError(
                "intrinsic must have shape "
                f"({self.num_views}, 3, 3), got {intrinsic_array.shape}"
            )
        if not np.isfinite(intrinsic_array).all():
            raise ValueError("intrinsic must contain only finite values")
        self._intrinsic = intrinsic_array.copy()
        if extrinsic is None:
            return
        extrinsic_array = np.asarray(extrinsic, dtype=np.float32)
        if extrinsic_array.shape != (self.num_views, 4, 4):
            raise ValueError(
                "extrinsic must have shape "
                f"({self.num_views}, 4, 4), got {extrinsic_array.shape}"
            )
        if not np.isfinite(extrinsic_array).all():
            raise ValueError("extrinsic must contain only finite values")
        self._extrinsic = extrinsic_array.copy()

    def _load_first_frame(self, first_frame: Any) -> None:
        if first_frame is None:
            raise ValueError("initialize options require first_frame")
        frame = np.asarray(first_frame, dtype=np.float32)
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"first_frame must have shape {self.frame_shape}, got {frame.shape}"
            )
        if not np.isfinite(frame).all() or np.any(frame < 0.0) or np.any(frame > 1.0):
            raise ValueError("first_frame must be finite and within [0, 1]")
        self._first_frame = frame.copy()

    def _load_trajectory_conditions(self, trajectory: Any, c2w: Any) -> None:
        if trajectory is None or c2w is None:
            raise ValueError("initialize options require trajectory and c2w")
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim != 5 or traj.shape[0] != 3:
            raise ValueError(
                f"trajectory must have shape (3, V, T, H, W), got {traj.shape}"
            )
        if traj.shape[1] != self.num_views or traj.shape[2] <= 0:
            raise ValueError("trajectory has an invalid view or time dimension")
        if traj.shape[3:] != (self.frame_height, self.frame_width):
            raise ValueError(
                "trajectory spatial shape must be "
                f"({self.frame_height}, {self.frame_width}), got {traj.shape[3:]}"
            )
        if not np.isfinite(traj).all() or np.any(traj < 0.0) or np.any(traj > 1.0):
            raise ValueError("trajectory must be finite and within [0, 1]")
        poses = np.asarray(c2w, dtype=np.float32)
        if poses.shape != (self.num_views, traj.shape[2], 4, 4):
            raise ValueError(
                "c2w must have shape "
                f"({self.num_views}, {traj.shape[2]}, 4, 4), got {poses.shape}"
            )
        if not np.isfinite(poses).all():
            raise ValueError("c2w must contain only finite values")
        self._trajectory = traj.copy()
        self._c2w = poses.copy()

    def _ensure_ready(self) -> None:
        if self._first_frame is None:
            raise RuntimeError("initialize options require first_frame")
        if self._trajectory is None or self._c2w is None:
            raise RuntimeError("initialize options require trajectory and c2w")

    def _pad_actions(self, actions: Array) -> Array:
        if actions.shape[0] == self.chunk_size:
            return actions.copy()
        padding = np.repeat(
            actions[-1:],
            self.chunk_size - actions.shape[0],
            axis=0,
        )
        return np.concatenate((actions, padding), axis=0)

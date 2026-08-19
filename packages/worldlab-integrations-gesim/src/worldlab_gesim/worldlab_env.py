"""WorldLab adapters for the GE-Sim world-model backend.

The backend may be the author's local ``WorldModel`` or the author's HTTP
``WorldModelClient``. This module owns the shared episode setup, conditioning,
chunk handling, and conversion to WorldLab ``Simulation*`` results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, Protocol, TypeVar

import numpy as np

from worldlab.core import Simulator, Space, Task
from worldlab.data import (
    SIMULATION_CHUNK_INDEX,
    SIMULATION_FRAMES,
    SIMULATION_STATE,
    SimulationReset,
    SimulationStep,
)
from worldlab.envs import WorldEnvironment

from gesim.action_chunk import compress_action_chunk
from gesim.episode import EpisodeBundle
from gesim.types import ACTION_DIM, Observation, frame_to_view_images, wm_state_to_policy_state

from .provider import GESIM_TASK_INFO

if TYPE_CHECKING:
    from gesim.conditioning.policy_band import PolicyBandRenderer


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")


class GESimWorldModelBackend(Protocol):
    """Author-compatible model/client surface consumed by the simulator."""

    chunk_size: int

    def reset(self) -> None:
        ...

    def set_camera_params(
        self, intrinsic: np.ndarray, extrinsic: np.ndarray | None = None
    ) -> None:
        ...

    def set_episode_data(self, first_frame: np.ndarray) -> None:
        ...

    def set_episode_traj(self, traj: np.ndarray, c2w: np.ndarray) -> None:
        ...

    def step(self, actions: np.ndarray) -> Any:
        ...


class GESimSimulatorAdapter(Simulator[Observation, np.ndarray]):
    """Run a GE-Sim backend through the WorldLab ``Simulator`` contract.

    ``backend`` can be either a local author model or a network client. The
    simulator, rather than the environment or task, owns GE-Sim-specific
    episode initialization and trajectory-band conditioning.
    """

    def __init__(
        self,
        backend: GESimWorldModelBackend,
        *,
        chunk_size: int | None = None,
        compress_actions: bool = True,
    ) -> None:
        self.backend = backend
        inferred_size = int(getattr(backend, "chunk_size", 25))
        self.chunk_size = int(chunk_size or inferred_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        self.compress_actions = bool(compress_actions)
        self.bundle: EpisodeBundle | None = None
        self.task = ""
        self.conditioning = "action"
        self._band_renderer: Any = None
        self._chunk_index = 0
        self._last_frames: np.ndarray | None = None
        self._closed = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> SimulationReset[Observation]:
        del seed
        self._ensure_open()
        values = dict(options or {})
        episode = values.get("episode")
        if episode is None:
            raise ValueError("reset options must contain an 'episode' bundle")
        conditioning = values.get("conditioning", "action")
        if conditioning not in ("action", "episode"):
            raise ValueError("conditioning must be 'action' or 'episode'")

        bundle = episode if isinstance(episode, EpisodeBundle) else EpisodeBundle.load(episode)
        task = values.get("task", bundle.task)
        if not isinstance(task, str) or not task.strip():
            raise ValueError("reset option 'task' must be a non-empty string")

        self.bundle = bundle
        self.task = task
        self.conditioning = conditioning
        self._chunk_index = 0
        self._last_frames = None
        self.backend.reset()
        set_task = getattr(self.backend, "set_task", None)
        if callable(set_task):
            set_task(task)
        self.backend.set_camera_params(bundle.intrinsic, bundle.initial_extrinsic)
        self.backend.set_episode_data(bundle.first_frame)

        if conditioning == "episode":
            from gesim.conditioning.band import render_band_from_bundle

            band, c2w = render_band_from_bundle(bundle)
            self.backend.set_episode_traj(np.asarray(band), c2w)
            self._band_renderer = None
        else:
            from gesim.conditioning.policy_band import PolicyBandRenderer

            self._band_renderer = PolicyBandRenderer(bundle)

        observation = bundle.first_observation(task)
        return SimulationReset(
            state=observation,
            info={
                GESIM_TASK_INFO: task,
                "gesim.conditioning": conditioning,
                SIMULATION_STATE: observation.state.copy(),
            },
        )

    def step(self, action: np.ndarray) -> SimulationStep[Observation]:
        self._ensure_open()
        if self.bundle is None:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(action, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] == 0:
            raise ValueError(f"action must have shape (L, {ACTION_DIM}) with L > 0")
        if self.compress_actions and actions.shape[0] > self.chunk_size:
            actions = compress_action_chunk(actions, self.chunk_size)

        frame_parts: list[np.ndarray] = []
        state_parts: list[np.ndarray] = []
        for start in range(0, actions.shape[0], self.chunk_size):
            chunk = actions[start : start + self.chunk_size]
            if self._band_renderer is not None:
                band, c2w = self._band_renderer.render(chunk)
                self.backend.set_episode_traj(np.asarray(band), c2w)
            prediction = self.backend.step(chunk)
            frames, state = _unpack_prediction(prediction)
            count = min(chunk.shape[0], frames.shape[0])
            if count == 0:
                raise RuntimeError("GE-Sim world model returned no frames")
            frame_parts.append(frames[:count])
            if state is not None:
                state_parts.append(state[:count])

        frames = np.concatenate(frame_parts, axis=0)
        predicted_state = np.concatenate(state_parts, axis=0) if state_parts else None
        if predicted_state is not None and predicted_state.shape[-1] >= ACTION_DIM:
            next_state = wm_state_to_policy_state(predicted_state[-1])
        else:
            next_state = wm_state_to_policy_state(actions[-1])
        observation = Observation(
            images=frame_to_view_images(frames[-1]),
            state=next_state,
            task=self.task,
        )
        info: dict[str, Any] = {
            SIMULATION_FRAMES: frames,
            SIMULATION_CHUNK_INDEX: self._chunk_index,
            GESIM_TASK_INFO: self.task,
            "gesim.action_shape": tuple(actions.shape),
        }
        if predicted_state is not None:
            info[SIMULATION_STATE] = predicted_state
        self._last_frames = frames
        self._chunk_index += 1
        return SimulationStep(state=observation, info=info)

    def render(self) -> Any:
        return self._last_frames

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("simulator is closed")


class GESimWorldModelEnv(
    WorldEnvironment[Observation, ObservationT, np.ndarray],
    Generic[ObservationT],
):
    """Compose a simulator and a task without owning GE-Sim services."""

    def __init__(
        self,
        simulator: Simulator[Observation, np.ndarray] | GESimWorldModelBackend,
        task: Task[Observation, ObservationT, np.ndarray],
        *,
        observation_space: Space[ObservationT] | None = None,
        action_space: Space[np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            _coerce_simulator(simulator),
            task,
            observation_space=observation_space,  # type: ignore[arg-type]
            action_space=action_space,  # type: ignore[arg-type]
        )

    @property
    def task_instruction(self) -> str:
        instruction = getattr(self.task, "instruction", "")
        return str(instruction)


class _LegacySimulator(Simulator[Any, Any]):
    """Compatibility shim for pre-v0.3.6 WorldLab-facing clients."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        return self.client.reset(seed=seed, options=options)

    def step(self, action: Any):
        return self.client.step(action)

    def close(self) -> None:
        self.client.close()


def _coerce_simulator(
    value: Simulator[Observation, np.ndarray] | GESimWorldModelBackend,
) -> Simulator[Any, Any]:
    if isinstance(value, Simulator):
        return value
    required_backend = ("set_camera_params", "set_episode_data", "set_episode_traj")
    if all(callable(getattr(value, name, None)) for name in required_backend):
        return GESimSimulatorAdapter(value)
    return _LegacySimulator(value)


def _unpack_prediction(prediction: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(prediction, "frames") and hasattr(prediction, "state"):
        frames = prediction.frames
        state = prediction.state
    elif isinstance(prediction, tuple) and len(prediction) == 2:
        frames, state = prediction
    else:
        raise TypeError("GE-Sim backend step must return (frames, state) or an object with both")
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 5 or frames.shape[1] != 3:
        raise ValueError(f"backend frames must have shape (T, 3, V, H, W), got {frames.shape}")
    return frames, None if state is None else np.asarray(state, dtype=np.float32)


# Compatibility name retained for callers of the previous integration API.
GESimWorldModelSimulator = GESimSimulatorAdapter

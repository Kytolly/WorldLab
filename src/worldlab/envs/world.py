"""Environment composed from a simulator and an RL task."""

from __future__ import annotations

from typing import Any, Generic, Mapping, Optional, TypeVar, cast

from worldlab.core import Environment, Simulator, Space, Task
from worldlab.data import ResetResult, StepResult


StateT = TypeVar("StateT")
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
_UNSET = object()


class WorldEnvironment(Environment[ObservationT, ActionT], Generic[StateT, ObservationT, ActionT]):
    def __init__(
        self,
        simulator: Simulator[StateT, ActionT],
        task: Task[StateT, ObservationT, ActionT],
        *,
        observation_space: Space[ObservationT],
        action_space: Space[ActionT],
    ) -> None:
        self.simulator = simulator
        self.task = task
        self.observation_space = observation_space
        self.action_space = action_space
        self._state: object = _UNSET
        self._active = False
        self._closed = False

    @property
    def state(self) -> StateT:
        if not self._active or self._state is _UNSET:
            raise RuntimeError("environment has no active episode")
        return cast(StateT, self._state)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> ResetResult[ObservationT]:
        self._ensure_open()
        simulation = self.simulator.reset(seed=seed, options=options)
        self._state = simulation.state
        self._active = True
        return self.task.reset(simulation)

    def step(self, action: ActionT) -> StepResult[ObservationT]:
        self._ensure_open()
        previous_state = self.state
        simulation = self.simulator.step(action)
        result = self.task.step(previous_state, action, simulation)
        self._state = simulation.state
        if result.done:
            self._active = False
        return result

    def render(self) -> Any:
        self._ensure_open()
        return self.simulator.render()

    def close(self) -> None:
        if self._closed:
            return
        self.simulator.close()
        self._closed = True
        self._active = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")


# Compatibility alias for callers migrating from v0.3.x.
SimulatorEnvironment = WorldEnvironment

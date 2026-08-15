"""Stateful simulator runtime backed by a high-level WorldModel."""

from __future__ import annotations

from typing import Any, Generic, Mapping, Optional, TypeVar, cast

from worldlab.core import Simulator, WorldModel
from worldlab.data import SimulationReset, SimulationStep


ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
_UNSET = object()


class WorldModelSimulator(Simulator[StateT, ActionT], Generic[ContextT, StateT, ActionT]):
    """Own episode state while delegating generation to a WorldModel."""

    def __init__(self, model: WorldModel[ContextT, StateT, ActionT]) -> None:
        self.model = model
        self._context: object = _UNSET
        self._state: object = _UNSET
        self._active = False
        self._closed = False

    @property
    def context(self) -> ContextT:
        if not self._active or self._context is _UNSET:
            raise RuntimeError("simulator has not been reset")
        return cast(ContextT, self._context)

    @property
    def state(self) -> StateT:
        if not self._active or self._state is _UNSET:
            raise RuntimeError("simulator has not been reset")
        return cast(StateT, self._state)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> SimulationReset[StateT]:
        self._ensure_open()
        result = self.model.initialize(seed=seed, options=options)
        self._context = result.context
        self._state = result.state
        self._active = True
        return SimulationReset(result.state, result.info)

    def step(self, action: ActionT) -> SimulationStep[StateT]:
        self._ensure_open()
        prediction = self.model.sample_step(self.context, action)
        self._context = prediction.context
        self._state = prediction.state
        return SimulationStep(prediction.state, prediction.info)

    def render(self) -> Any:
        self._ensure_open()
        return self.model.render(self.context, self.state)

    def close(self) -> None:
        if self._closed:
            return
        self.model.close()
        self._closed = True
        self._active = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("simulator is closed")

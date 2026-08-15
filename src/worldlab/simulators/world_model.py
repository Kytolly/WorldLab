"""Use a learned world model as a stateful WorldLab simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar, cast

from worldlab.core import Simulator
from worldlab.data import SimulationReset, SimulationStep


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
_UNSET = object()


class WorldModel(ABC, Generic[StateT, ActionT]):
    """A transition model that does not own episode state."""

    @abstractmethod
    def initial_state(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> SimulationReset[StateT]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: StateT, action: ActionT) -> SimulationStep[StateT]:
        raise NotImplementedError

    def render(self, state: StateT) -> Any:
        return None

    def close(self) -> None:
        return None


class WorldModelSimulator(Simulator[StateT, ActionT]):
    """Adds episode state and lifecycle checks to a :class:`WorldModel`."""

    def __init__(self, model: WorldModel[StateT, ActionT]) -> None:
        self.model = model
        self._state: object = _UNSET
        self._active = False
        self._closed = False

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
        result = self.model.initial_state(seed=seed, options=options)
        self._state = result.state
        self._active = True
        return result

    def step(self, action: ActionT) -> SimulationStep[StateT]:
        self._ensure_open()
        result = self.model.predict(self.state, action)
        self._state = result.state
        return result

    def render(self) -> Any:
        self._ensure_open()
        return self.model.render(self.state)

    def close(self) -> None:
        if self._closed:
            return
        self.model.close()
        self._closed = True
        self._active = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("simulator is closed")

"""Stateful simulator contract used by WorldLab environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.data import SimulationReset, SimulationStep


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class Simulator(ABC, Generic[StateT, ActionT]):
    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> SimulationReset[StateT]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActionT) -> SimulationStep[StateT]:
        raise NotImplementedError

    def render(self) -> Any:
        return None

    def close(self) -> None:
        return None

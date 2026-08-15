"""Framework-neutral reinforcement-learning environment contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, Protocol, TypeVar, runtime_checkable

from worldlab.data import ResetResult, SimulationReset, SimulationStep, StepResult


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
StateT = TypeVar("StateT")
SpaceValueT_co = TypeVar("SpaceValueT_co", covariant=True)


@runtime_checkable
class Space(Protocol[SpaceValueT_co]):
    def sample(self) -> SpaceValueT_co:
        ...

    def contains(self, value: Any) -> bool:
        ...

    def seed(self, seed: Optional[int] = None) -> Any:
        ...


class Environment(ABC, Generic[ObservationT, ActionT]):
    observation_space: Space[ObservationT]
    action_space: Space[ActionT]

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> ResetResult[ObservationT]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActionT) -> StepResult[ObservationT]:
        raise NotImplementedError

    def render(self) -> Any:
        return None

    def close(self) -> None:
        return None

    @property
    def unwrapped(self) -> "Environment[ObservationT, ActionT]":
        return self


class Task(ABC, Generic[StateT, ObservationT, ActionT]):
    """Maps simulator state evolution to an RL task."""

    @abstractmethod
    def reset(self, simulation: SimulationReset[StateT]) -> ResetResult[ObservationT]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        previous_state: StateT,
        action: ActionT,
        simulation: SimulationStep[StateT],
    ) -> StepResult[ObservationT]:
        raise NotImplementedError

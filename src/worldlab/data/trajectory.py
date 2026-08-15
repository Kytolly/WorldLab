"""Immutable episode trajectory container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterator, Tuple, TypeVar

from .interaction import Transition


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Trajectory(Generic[ObservationT, ActionT]):
    transitions: Tuple[Transition[ObservationT, ActionT], ...]

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[Transition[ObservationT, ActionT]]:
        return iter(self.transitions)

    @property
    def total_reward(self) -> float:
        return sum(item.reward for item in self.transitions)

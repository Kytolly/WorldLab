"""A minimal typed transition batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterator, Tuple, TypeVar

from .interaction import Transition


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class TransitionBatch(Generic[ObservationT, ActionT]):
    transitions: Tuple[Transition[ObservationT, ActionT], ...]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("a transition batch cannot be empty")

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[Transition[ObservationT, ActionT]]:
        return iter(self.transitions)

"""Agent interaction contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.data import EpisodeResult, PolicyOutput, Transition


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class Agent(ABC, Generic[ObservationT, ActionT]):
    def reset(self, *, seed: Optional[int] = None) -> None:
        return None

    @abstractmethod
    def act(
        self,
        observation: ObservationT,
        *,
        info: Mapping[str, Any],
        training: bool,
        deterministic: bool,
    ) -> PolicyOutput[ActionT]:
        raise NotImplementedError

    def observe(self, transition: Transition[ObservationT, ActionT]) -> None:
        return None

    def end_episode(
        self,
        result: EpisodeResult[ObservationT],
        *,
        training: bool,
    ) -> None:
        return None

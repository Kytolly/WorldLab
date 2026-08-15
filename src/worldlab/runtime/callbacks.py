"""Environment-loop observation hooks."""

from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar

from worldlab.data import EpisodeResult, Transition


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class LoopCallback(Generic[ObservationT, ActionT]):
    def on_episode_start(
        self,
        episode_index: int,
        observation: ObservationT,
        info: Mapping[str, Any],
    ) -> None:
        return None

    def on_step(
        self,
        episode_index: int,
        step_index: int,
        transition: Transition[ObservationT, ActionT],
    ) -> None:
        return None

    def on_episode_end(self, result: EpisodeResult[ObservationT]) -> None:
        return None

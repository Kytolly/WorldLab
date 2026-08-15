"""Callback that stores transitions in a replay buffer."""

from __future__ import annotations

from typing import Generic, TypeVar

from worldlab.buffers import Buffer
from worldlab.data import Transition

from .callbacks import LoopCallback


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class ReplayCollector(LoopCallback[ObservationT, ActionT], Generic[ObservationT, ActionT]):
    def __init__(self, buffer: Buffer[Transition[ObservationT, ActionT]]) -> None:
        self.buffer = buffer

    def on_step(
        self,
        episode_index: int,
        step_index: int,
        transition: Transition[ObservationT, ActionT],
    ) -> None:
        del episode_index, step_index
        self.buffer.add(transition)

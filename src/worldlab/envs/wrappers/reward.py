"""Reward-transforming environment wrapper."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from typing import TypeVar

from worldlab.data import StepResult

from .base import EnvironmentWrapper


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class RewardWrapper(EnvironmentWrapper[ObservationT, ActionT]):
    @abstractmethod
    def reward(self, reward: float) -> float:
        raise NotImplementedError

    def step(self, action: ActionT) -> StepResult[ObservationT]:
        result = self.env.step(action)
        return replace(result, reward=float(self.reward(result.reward)))

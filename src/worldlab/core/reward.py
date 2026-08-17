"""Reward signal contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from worldlab.data.task import RewardResult


ResetContextT = TypeVar("ResetContextT")
EvaluatedContextT = TypeVar("EvaluatedContextT")


class RewardProvider(ABC, Generic[ResetContextT, EvaluatedContextT]):
    def reset(self, context: ResetContextT) -> None:
        return None

    @abstractmethod
    def compute(self, context: EvaluatedContextT) -> RewardResult:
        raise NotImplementedError

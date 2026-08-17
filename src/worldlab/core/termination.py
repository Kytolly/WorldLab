"""Termination signal contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from worldlab.data.task import TerminationResult


ResetContextT = TypeVar("ResetContextT")
StepContextT = TypeVar("StepContextT")


class TerminationProvider(ABC, Generic[ResetContextT, StepContextT]):
    def reset(self, context: ResetContextT) -> None:
        return None

    @abstractmethod
    def compute(self, context: StepContextT) -> TerminationResult:
        raise NotImplementedError

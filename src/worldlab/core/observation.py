"""Observation signal contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from worldlab.data.task import ObservationResult


ResetContextT = TypeVar("ResetContextT")
StepContextT = TypeVar("StepContextT")
ObservationT = TypeVar("ObservationT")


class ObservationProvider(ABC, Generic[ResetContextT, StepContextT, ObservationT]):
    @abstractmethod
    def reset(self, context: ResetContextT) -> ObservationResult[ObservationT]:
        raise NotImplementedError

    @abstractmethod
    def compute(self, context: StepContextT) -> ObservationResult[ObservationT]:
        raise NotImplementedError

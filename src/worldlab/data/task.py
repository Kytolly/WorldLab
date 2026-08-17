"""Task-level data contracts for composable signals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Generic, Mapping, TypeVar

from ._immutable import freeze_mapping
from .simulation import SimulationReset, SimulationStep


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class TaskResetContext(Generic[StateT]):
    simulation: SimulationReset[StateT]
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))


@dataclass(frozen=True)
class TaskStepContext(Generic[StateT, ActionT]):
    previous_state: StateT
    action: ActionT
    simulation: SimulationStep[StateT]
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))


@dataclass(frozen=True)
class ObservationResult(Generic[ObservationT]):
    observation: ObservationT
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))


@dataclass(frozen=True)
class TerminationResult:
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))
        if not isinstance(self.terminated, bool):
            raise ValueError("terminated must be a bool")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a bool")

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class EvaluatedStepContext(Generic[StateT, ActionT, ObservationT]):
    step: TaskStepContext[StateT, ActionT]
    observation: ObservationResult[ObservationT]
    termination: TerminationResult
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))


@dataclass(frozen=True)
class RewardResult:
    value: float
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))
        self._validate_scalar(self.value, "value")

    @staticmethod
    def _validate_scalar(value: Any, name: str) -> None:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"{name} must be a finite scalar")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite scalar")

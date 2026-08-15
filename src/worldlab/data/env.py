"""Environment-facing result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Generic, Mapping, TypeVar


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class ResetResult(Generic[ObservationT]):
    observation: ObservationT
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult(Generic[ObservationT]):
    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.reward, Real) or isinstance(self.reward, bool):
            raise ValueError("reward must be a finite scalar")
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be a finite scalar")

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

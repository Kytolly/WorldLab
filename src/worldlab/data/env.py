"""Environment-facing result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

"""Framework-neutral simulator outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class SimulationReset(Generic[StateT]):
    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationStep(Generic[StateT]):
    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)

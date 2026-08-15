"""Context and prediction values exchanged with a WorldModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar


ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class WorldModelContext(Generic[ContextT, StateT]):
    """Opaque model context plus the simulator-facing current state."""

    context: ContextT
    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldModelPrediction(Generic[ContextT, StateT]):
    """One stochastic or deterministic model transition."""

    context: ContextT
    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)

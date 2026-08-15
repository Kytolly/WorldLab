"""Data contracts exchanged with a WorldModel."""

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
class WorldModelStepResult(Generic[ContextT, StateT]):
    """One model transition or action-chunk prediction.

    State may be a chunk-level sequence with the same leading length as
    frames. Frames are optional so scalar toy models remain valid.
    """

    context: ContextT
    state: StateT
    frames: Any = None
    info: Mapping[str, Any] = field(default_factory=dict)

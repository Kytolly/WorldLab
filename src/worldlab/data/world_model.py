"""Data contracts exchanged with a WorldModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

import numpy as np

from ._immutable import freeze_mapping


ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class WorldModelContext(Generic[ContextT, StateT]):
    """Opaque model context plus the simulator-facing current state."""

    context: ContextT
    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))


@dataclass(frozen=True)
class WorldModelStepResult(Generic[ContextT, StateT]):
    """One model transition or model-specific prediction.

    ``output`` is intentionally opaque: a video model, latent model, or
    symbolic model may choose its own output type. Concrete models own any
    shape or modality-specific validation. Reward belongs to the Task.
    """

    context: ContextT
    state: StateT
    output: Any = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))
        self.validate()

    def validate(self) -> None:
        """Validate generic numeric state without interpreting model semantics."""

        _validate_numeric_value(self.state, "state")


def _validate_numeric_value(value: Any, name: str) -> None:
    if value is None:
        return
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be array-like") from error
    if array.dtype.kind in "biufc" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

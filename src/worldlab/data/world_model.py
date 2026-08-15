"""Data contracts exchanged with a WorldModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

import numpy as np


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

    ``frames`` is optional for scalar toy models, while ``state`` must share
    its leading sequence length when frames are present. Action is an input
    to the model rather than a model output, and reward belongs to the Task.
    """

    context: ContextT
    state: StateT
    frames: Any = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, *, chunk_size: int | None = None) -> None:
        """Validate model outputs without interpreting task reward semantics."""

        _validate_numeric_value(self.state, "state")
        if self.frames is not None:
            frames = _as_array(self.frames, "frames")
            if frames.ndim == 0:
                raise ValueError("frames must have a leading sequence dimension")
            if chunk_size is not None and frames.shape[0] != chunk_size:
                raise ValueError(
                    f"frames leading dimension must equal chunk_size ({chunk_size}), "
                    f"got {frames.shape[0]}"
                )
            state = _as_optional_array(self.state)
            if state is not None and state.ndim > 0 and state.shape[0] != frames.shape[0]:
                raise ValueError(
                    "state and frames must have the same leading sequence length"
                )


def _as_array(value: Any, name: str) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be array-like") from error
    if array.dtype.kind not in "biufc":
        raise ValueError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_optional_array(value: Any) -> np.ndarray[Any, Any] | None:
    if value is None:
        return None
    return np.asarray(value)


def _validate_numeric_value(value: Any, name: str) -> None:
    if value is None:
        return
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be array-like") from error
    if array.dtype.kind in "biufc" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

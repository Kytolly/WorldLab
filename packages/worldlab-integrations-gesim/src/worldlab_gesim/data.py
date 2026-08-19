"""GE-Sim task and frame conversion data helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .base import RewardResult


@dataclass(frozen=True)
class GESimTaskContext:
    """Task information required by the GE-Sim World Judge."""

    instruction: str
    subtask_caption: str | None = None
    task_id: str | None = None
    episode_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_text(self.instruction, "instruction")
        for name, value in (
            ("subtask_caption", self.subtask_caption),
            ("task_id", self.task_id),
            ("episode_id", self.episode_id),
        ):
            if value is not None:
                _validate_text(value, name)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def head_view_frames(frames: np.ndarray) -> np.ndarray:
    """Extract author-compatible uint8 head-camera frames from WorldLab output."""

    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 5 or array.shape[1] != 3 or array.shape[2] < 1:
        raise ValueError(
            "frames must have shape (T, 3, V, H, W), "
            f"got {array.shape}"
        )
    head = np.clip(array[:, :, 0], 0.0, 1.0)
    return (head.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return MappingProxyType(dict(value))


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


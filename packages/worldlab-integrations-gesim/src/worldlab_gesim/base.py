"""Author-compatible GE-Sim reward boundary.

The types in this module intentionally remain outside ``worldlab.data``.
WorldLab consumes the adapted scalar reward; GE-Sim keeps its per-frame
success output and optional progress field here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class RewardResult:
    """Per-frame GE-Sim scores for one generated video chunk."""

    success: np.ndarray
    progress: np.ndarray | None = None

    def __post_init__(self) -> None:
        success = _as_scores(self.success, "success")
        progress = None if self.progress is None else _as_scores(self.progress, "progress")
        if progress is not None and progress.shape != success.shape:
            raise ValueError(
                "success and progress must have the same shape, "
                f"got {success.shape} and {progress.shape}"
            )
        if success.size == 0:
            raise ValueError("success must not be empty")
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "progress", progress)


@runtime_checkable
class RewardClient(Protocol):
    """Protocol implemented by a GE-Sim World Judge."""

    def evaluate(self, head_frames: np.ndarray, task: str) -> RewardResult:
        """Score ``(T, H, W, 3)`` uint8 head-camera frames."""
        ...


def _as_scores(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (T,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()

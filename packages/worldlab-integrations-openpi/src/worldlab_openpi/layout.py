"""Action layout validation and conversion for OpenPI and GE-style models."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


ACTION_DIM = 16


class ActionLayout(str, Enum):
    """Known 16-D dual-arm action layouts."""

    POLICY = "policy"  # [L7_arm, R7_arm, L_grip, R_grip]
    WORLD_MODEL = "world_model"  # [L7_arm, L_grip, R7_arm, R_grip]


def validate_actions(actions: Any, *, action_dim: int = ACTION_DIM) -> NDArray[np.float32]:
    """Normalize a policy response to ``(horizon, action_dim)`` float32."""

    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != action_dim or array.shape[0] == 0:
        raise ValueError(
            f"expected actions (horizon, {action_dim}) with horizon > 0, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("actions must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def convert_actions(
    actions: Any,
    *,
    source: ActionLayout,
    target: ActionLayout,
) -> NDArray[np.float32]:
    """Convert a validated dual-arm action sequence between known layouts."""

    array = validate_actions(actions)
    if source is target:
        return array.copy()
    if {source, target} != {ActionLayout.POLICY, ActionLayout.WORLD_MODEL}:
        raise ValueError(f"unsupported action layout conversion: {source} -> {target}")
    output = np.empty_like(array)
    if source is ActionLayout.POLICY and target is ActionLayout.WORLD_MODEL:
        output[:, 0:7] = array[:, 0:7]
        output[:, 7] = array[:, 14]
        output[:, 8:15] = array[:, 7:14]
        output[:, 15] = array[:, 15]
    else:
        output[:, 0:7] = array[:, 0:7]
        output[:, 7:14] = array[:, 8:15]
        output[:, 14] = array[:, 7]
        output[:, 15] = array[:, 15]
    return output

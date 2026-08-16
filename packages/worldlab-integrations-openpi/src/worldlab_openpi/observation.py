"""WorldLab observation to OpenPI payload conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


DEFAULT_IMAGE_KEYS: dict[str, str] = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.hand_left",
    "right_wrist": "observation.images.hand_right",
}
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_PROMPT_KEY = "prompt"


@dataclass(frozen=True)
class OpenPIObservation:
    """Canonical visual observation accepted by :class:`OpenPIPolicy`.

    Images are HWC RGB arrays. State is a flat float32 vector in the policy's
    configured layout. The adapter does not assume a particular task schema.
    """

    images: Mapping[str, NDArray[Any]]
    state: NDArray[np.float32]
    task: str

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.ndim != 1 or state.size == 0:
            raise ValueError(f"state must be a non-empty 1-D vector, got {state.shape}")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")


def build_openpi_payload(
    observation: OpenPIObservation,
    *,
    image_keys: Mapping[str, str] | None = None,
    state_key: str = DEFAULT_STATE_KEY,
    prompt_key: str = DEFAULT_PROMPT_KEY,
) -> dict[str, Any]:
    """Build the flat mapping expected by OpenPI's policy server."""

    if not state_key or not prompt_key:
        raise ValueError("state_key and prompt_key must not be empty")
    keys = dict(image_keys or DEFAULT_IMAGE_KEYS)
    payload: dict[str, Any] = {
        state_key: np.ascontiguousarray(observation.state, dtype=np.float32),
        prompt_key: observation.task,
    }
    for view, key in keys.items():
        if view not in observation.images:
            raise KeyError(
                f"observation is missing view {view!r}; "
                f"available views: {sorted(observation.images)}"
            )
        if not key:
            raise ValueError(f"payload key for view {view!r} must not be empty")
        image = np.asarray(observation.images[view])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"image {view!r} must be HWC RGB with shape (H, W, 3), got {image.shape}"
            )
        if image.dtype.kind not in "biufc":
            raise ValueError(f"image {view!r} must contain numeric values")
        payload[key] = np.ascontiguousarray(image)
    return payload

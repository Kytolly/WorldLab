"""WorldLab adapter around the GE-Sim OpenPI policy backend."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import numpy as np

from gesim.policies.base import Policy as GESimPolicy
from gesim.policies.openpi import OpenPIPolicy as GESimOpenPIPolicy
from gesim.types import Observation
from worldlab.core import Policy
from worldlab.data import PolicyOutput


class OpenPIPolicy(Policy[Observation, np.ndarray]):
    """Adapt a GE-Sim OpenPI backend to the WorldLab ``Policy`` contract.

    GE-Sim's world model and policy already share ``gesim.types.Observation``.
    Keeping that type at this boundary lets generated observations flow into
    OpenPI without an intermediate WorldLab-specific data object.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        action_horizon: int | None = None,
        image_keys: Mapping[str, str] | None = None,
        state_key: str = "observation.state",
        prompt_key: str = "prompt",
        backend: GESimPolicy | None = None,
    ) -> None:
        if action_horizon is not None and action_horizon <= 0:
            raise ValueError("action_horizon must be greater than zero")
        if backend is not None and url is not None:
            raise ValueError("pass either url or backend, not both")
        if backend is None:
            if url is None:
                raise ValueError("url is required when backend is not provided")
            backend = GESimOpenPIPolicy(
                url,
                action_horizon=action_horizon,
                image_keys=dict(image_keys) if image_keys is not None else None,
                state_key=state_key,
                prompt_key=prompt_key,
            )
        self.backend = backend
        self._action_horizon = action_horizon

    def reset(self, *, seed: Optional[int] = None) -> None:
        del seed
        self.backend.reset()

    def infer(
        self,
        observation: Observation,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[np.ndarray]:
        del info, deterministic
        started_at = time.perf_counter()
        actions = np.asarray(self.backend.infer(observation), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 16:
            raise ValueError(f"OpenPI backend must return (T, 16), got {actions.shape}")
        if self._action_horizon is not None:
            actions = actions[: self._action_horizon]
            if actions.shape[0] == 0:
                raise ValueError("OpenPI backend returned no action rows")
        return PolicyOutput(
            action=np.ascontiguousarray(actions),
            info={"openpi.model_latency_s": time.perf_counter() - started_at},
        )

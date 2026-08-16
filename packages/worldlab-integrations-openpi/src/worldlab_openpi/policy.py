"""WorldLab Policy adapter backed by an OpenPI server."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from worldlab.core import Policy
from worldlab.data import PolicyOutput

from .layout import ActionLayout, convert_actions, validate_actions
from .observation import (
    DEFAULT_IMAGE_KEYS,
    DEFAULT_PROMPT_KEY,
    DEFAULT_STATE_KEY,
    OpenPIObservation,
    build_openpi_payload,
)


class OpenPIPolicy(Policy[OpenPIObservation, NDArray[np.float32]]):
    """Policy adapter for the official OpenPI WebSocket server.

    OpenPI receives state in the configured policy layout. The returned action
    sequence is converted from ``server_action_layout`` to the fixed
    GE/world-model layout before it reaches the WorldLab runtime.
    """

    def __init__(
        self,
        url: str,
        *,
        action_horizon: int | None = None,
        server_action_layout: ActionLayout = ActionLayout.WORLD_MODEL,
        image_keys: Mapping[str, str] | None = None,
        state_key: str = DEFAULT_STATE_KEY,
        prompt_key: str = DEFAULT_PROMPT_KEY,
    ) -> None:
        if action_horizon is not None and action_horizon <= 0:
            raise ValueError("action_horizon must be greater than zero")
        try:
            from openpi_client.websocket_client_policy import (  # type: ignore[import-untyped]
                WebsocketClientPolicy,
            )
        except ImportError as error:
            raise ImportError(
                "openpi-client is not installed; install "
                '"worldlab-integrations-openpi" or "worldlab[openpi]"'
            ) from error
        self._client = WebsocketClientPolicy(host=url)
        self._action_horizon = action_horizon
        self._server_layout = server_action_layout
        self._image_keys = dict(image_keys or DEFAULT_IMAGE_KEYS)
        self._state_key = state_key
        self._prompt_key = prompt_key

    def reset(self, *, seed: Optional[int] = None) -> None:
        del seed
        self._client.reset()

    def act(
        self,
        observation: OpenPIObservation,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[NDArray[np.float32]]:
        del info, deterministic
        payload = build_openpi_payload(
            observation,
            image_keys=self._image_keys,
            state_key=self._state_key,
            prompt_key=self._prompt_key,
        )
        started_at = time.perf_counter()
        response = self._client.infer(payload)
        latency_s = time.perf_counter() - started_at
        if "actions" not in response:
            raise KeyError(
                f"OpenPI response is missing 'actions'; got keys {sorted(response)}"
            )
        actions = validate_actions(response["actions"])
        if self._action_horizon is not None:
            actions = actions[: self._action_horizon]
            if actions.shape[0] == 0:
                raise ValueError("OpenPI response is shorter than action_horizon")
        actions = convert_actions(
            actions,
            source=self._server_layout,
            target=ActionLayout.WORLD_MODEL,
        )
        return PolicyOutput(
            action=actions,
            info={
                "openpi.model_latency_s": latency_s,
                "openpi.server_action_layout": self._server_layout.value,
            },
        )

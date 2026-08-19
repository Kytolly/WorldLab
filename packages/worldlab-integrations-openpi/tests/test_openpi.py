from __future__ import annotations

import numpy as np
import pytest

from gesim.types import Observation
from worldlab.data import PolicyOutput
from worldlab_openpi import OpenPIPolicy


class _PolicyBackend:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def infer(self, observation) -> np.ndarray:
        assert observation.task == "open the drawer"
        assert set(observation.images) == {"head", "left_wrist", "right_wrist"}
        return np.ones((2, 16), dtype=np.float32)


def _observation() -> Observation:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    return Observation(
        images={"head": image, "left_wrist": image, "right_wrist": image},
        state=np.zeros(16, dtype=np.float32),
        task="open the drawer",
    )


def test_openpi_adapter_uses_injected_backend() -> None:
    backend = _PolicyBackend()
    policy = OpenPIPolicy(backend=backend, action_horizon=1)

    policy.reset()
    output = policy.infer(_observation(), info={}, deterministic=True)

    assert isinstance(output, PolicyOutput)
    assert output.action.shape == (1, 16)
    assert policy.backend is backend
    assert backend.reset_count == 1


def test_openpi_adapter_rejects_invalid_backend_actions() -> None:
    class InvalidBackend(_PolicyBackend):
        def infer(self, observation):
            return np.zeros((2, 7), dtype=np.float32)

    policy = OpenPIPolicy(backend=InvalidBackend())
    with pytest.raises(ValueError, match="backend must return \\(T, 16\\)"):
        policy.infer(_observation(), info={})

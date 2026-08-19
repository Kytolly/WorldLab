from __future__ import annotations

from typing import Any

import numpy as np

from gesim.types import Observation
from worldlab.core import ObservationProvider, Simulator, TerminationProvider
from worldlab.data import (
    ObservationResult,
    SimulationReset,
    SimulationStep,
    TerminationResult,
)
from worldlab.runtime import LoopConfig
from worldlab_gesim import GESimTask, GESimWorldModelEnv, make_gesim_loop
from worldlab_openpi import OpenPIPolicy


def _observation() -> Observation:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    return Observation(
        images={"head": image, "left_wrist": image, "right_wrist": image},
        state=np.zeros(16, dtype=np.float32),
        task="open the drawer",
    )


class _WorldModelSimulator(Simulator[Observation, np.ndarray]):
    def __init__(self) -> None:
        self.last_action: np.ndarray | None = None

    def reset(self, *, seed=None, options=None) -> SimulationReset[Observation]:
        del seed, options
        return SimulationReset(_observation())

    def step(self, action: np.ndarray) -> SimulationStep[Observation]:
        self.last_action = action
        return SimulationStep(_observation())


class _ObservationProvider(ObservationProvider):
    def reset(self, context: Any) -> ObservationResult[Observation]:
        return ObservationResult(context.simulation.state)

    def compute(self, context: Any) -> ObservationResult[Observation]:
        return ObservationResult(context.simulation.state)


class _OneStepTermination(TerminationProvider):
    def compute(self, context: Any) -> TerminationResult:
        del context
        return TerminationResult(terminated=False, truncated=True)


class _OpenPIBackend:
    def __init__(self) -> None:
        self.observation: Observation | None = None

    def reset(self) -> None:
        self.observation = None

    def infer(self, observation: Observation) -> np.ndarray:
        self.observation = observation
        return np.ones((1, 16), dtype=np.float32)


def test_world_model_observation_flows_directly_into_openpi_backend() -> None:
    simulator = _WorldModelSimulator()
    task = GESimTask(
        instruction="open the drawer",
        observation=_ObservationProvider(),
        termination=_OneStepTermination(),
    )
    env = GESimWorldModelEnv(
        simulator,
        task,
        observation_space=None,
        action_space=None,
    )
    backend = _OpenPIBackend()
    policy = OpenPIPolicy(backend=backend)
    loop = make_gesim_loop(
        env=env,
        policy=policy,
        config=LoopConfig(training=False, validate_spaces=False),
    )

    result = loop.run_episode()

    assert result.length == 1
    assert isinstance(backend.observation, Observation)
    assert backend.observation.task == "open the drawer"
    assert simulator.last_action is not None
    assert simulator.last_action.shape == (1, 16)

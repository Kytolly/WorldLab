"""Chunk-level synthetic environment used by the v0.2.1 config demo."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from worldlab.core import ArraySpace, Environment, Task
from worldlab.data import (
    SIMULATION_OUTPUT,
    ResetResult,
    SimulationReset,
    SimulationStep,
    StepResult,
)
from worldlab.simulators import WorldModelSimulator
from worldlab.world_models import ExampleWorldModel

from .simulator_env import SimulatorEnvironment
from .wrappers import TimeLimitWrapper


Array = NDArray[Any]


class ExampleTask(Task[Array, Array, Array]):
    """Define reward and termination semantics for the Example environment."""

    def __init__(self, goal: int) -> None:
        if goal <= 0:
            raise ValueError("goal must be greater than zero")
        self.goal = int(goal)
        self._steps = 0

    def reset(self, simulation: SimulationReset[Array]) -> ResetResult[Array]:
        self._steps = 0
        return ResetResult(simulation.state, {"goal": self.goal, **dict(simulation.info)})

    def step(
        self,
        previous_state: Array,
        action: Array,
        simulation: SimulationStep[Array],
    ) -> StepResult[Array]:
        del previous_state
        self._steps += 1
        info = {"goal": self.goal, "chunk_step": self._steps, **dict(simulation.info)}
        info["action"] = action
        info["frames"] = simulation.info.get(SIMULATION_OUTPUT)
        return StepResult(
            observation=simulation.state,
            reward=1.0,
            terminated=self._steps >= self.goal,
            truncated=False,
            info=info,
        )


class ExampleEnvironment(SimulatorEnvironment[Array, Array, Array]):
    """Compose ExampleWorldModel's simulator with ExampleTask."""

    def __init__(self, model: ExampleWorldModel, *, goal: int) -> None:
        state_shape = (model.chunk_size, model.state_dim)
        super().__init__(
            WorldModelSimulator(model, chunk_size=model.chunk_size),
            ExampleTask(goal),
            observation_space=ArraySpace((model.frame_shape, state_shape), dtype=np.float32),
            action_space=ArraySpace(
                (model.chunk_size, model.action_dim),
                dtype=np.float32,
            ),
        )


def make_example_environment(
    model: ExampleWorldModel,
    *,
    goal: int,
    max_episode_steps: Optional[int] = None,
) -> Environment[Array, Array]:
    """Build a simulator-backed environment with fixed chunk-level spaces."""

    environment: Environment[Array, Array] = ExampleEnvironment(model, goal=goal)
    if max_episode_steps is not None:
        environment = TimeLimitWrapper(environment, max_episode_steps)
    return environment

"""Toy task and factory used by the built-in WorldLab demo."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from worldlab.core import DiscreteSpace, Environment, FrameSpace, Task
from worldlab.data import ResetResult, SimulationReset, SimulationStep, StepResult
from worldlab.simulators import (
    CounterWorldModel,
    RandomFrameWorldModel,
    WorldModelSimulator,
)

from .simulator_env import SimulatorEnvironment
from .wrappers import TimeLimitWrapper


class GoalTask(Task[int, int, int]):
    """Reward one point per step until an integer goal is reached."""

    def __init__(self, goal: int) -> None:
        if goal <= 0:
            raise ValueError("goal must be greater than zero")
        self.goal = goal

    def reset(self, simulation: SimulationReset[int]) -> ResetResult[int]:
        return ResetResult(simulation.state, {"goal": self.goal})

    def step(
        self,
        previous_state: int,
        action: int,
        simulation: SimulationStep[int],
    ) -> StepResult[int]:
        del previous_state, action
        return StepResult(
            observation=simulation.state,
            reward=1.0,
            terminated=simulation.state >= self.goal,
            truncated=False,
            info={"state": simulation.state, "goal": self.goal},
        )


def make_counter_environment(
    *,
    goal: int = 3,
    max_episode_steps: Optional[int] = None,
) -> Environment[int, int]:
    """Build the dependency-free default Environment."""

    env: Environment[int, int] = SimulatorEnvironment(
        WorldModelSimulator(CounterWorldModel()),
        GoalTask(goal),
        observation_space=DiscreteSpace(max(goal + 1, 2)),
        action_space=DiscreteSpace(2),
    )
    if max_episode_steps is not None:
        env = TimeLimitWrapper(env, max_episode_steps)
    return env


class RandomFrameTask(Task[tuple[int, ...], tuple[int, ...], int]):
    """A never-terminating frame task intended for TimeLimitWrapper."""

    def reset(
        self,
        simulation: SimulationReset[tuple[int, ...]],
    ) -> ResetResult[tuple[int, ...]]:
        return ResetResult(simulation.state, {"task": "random_frame"})

    def step(
        self,
        previous_state: tuple[int, ...],
        action: int,
        simulation: SimulationStep[tuple[int, ...]],
    ) -> StepResult[tuple[int, ...]]:
        del previous_state, action
        return StepResult(
            observation=simulation.state,
            reward=1.0,
            terminated=False,
            truncated=False,
            info={"task": "random_frame", **simulation.info},
        )


def make_random_frame_environment(
    *,
    frame_size: int = 8,
    max_episode_steps: int = 3,
) -> Environment[tuple[int, ...], int]:
    """Build an environment backed by the stochastic frame model."""

    env: Environment[tuple[int, ...], int] = SimulatorEnvironment(
        WorldModelSimulator(RandomFrameWorldModel(frame_size)),
        RandomFrameTask(),
        observation_space=FrameSpace(frame_size),
        action_space=DiscreteSpace(2),
    )
    return TimeLimitWrapper(env, max_episode_steps)

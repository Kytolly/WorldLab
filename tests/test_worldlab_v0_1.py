from __future__ import annotations

import random
from typing import Any, Mapping, Optional

import pytest

from worldlab import (
    EnvironmentLoop,
    LoopConfig,
    ReplayBuffer,
    ResetResult,
    SimulationReset,
    SimulationStep,
    SimulatorEnvironment,
    StepResult,
    Task,
    TimeLimitWrapper,
    Transition,
    WorldModel,
    WorldModelContext,
    WorldModelPrediction,
    WorldModelSimulator,
)
from worldlab.agents import PolicyAgent
from worldlab.data import EpisodeResult, PolicyOutput
from worldlab.policies import CallablePolicy
from worldlab.runtime import Evaluator, ReplayCollector


class DiscreteSpace:
    def __init__(self, size: int) -> None:
        self.size = size
        self._random = random.Random()

    def sample(self) -> int:
        return self._random.randrange(self.size)

    def contains(self, value: Any) -> bool:
        return isinstance(value, int) and 0 <= value < self.size

    def seed(self, seed: Optional[int] = None) -> None:
        self._random.seed(seed)


class CounterWorldModel(WorldModel[int, int, int]):
    def initialize(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> WorldModelContext[int, int]:
        del seed
        state = int((options or {}).get("start", 0))
        return WorldModelContext(context=state, state=state)

    def sample_step(self, context: int, action: int) -> WorldModelPrediction[int, int]:
        state = context + action
        return WorldModelPrediction(context=state, state=state)


class GoalTask(Task[int, int, int]):
    def __init__(self, goal: int) -> None:
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
            info={"state": simulation.state},
        )


class RecordingPolicyAgent(PolicyAgent[int, int]):
    def __init__(self) -> None:
        super().__init__(
            CallablePolicy(lambda observation, info, deterministic: 1)
        )
        self.transitions: list[Transition[int, int]] = []
        self.episodes: list[tuple[EpisodeResult[int], bool]] = []

    def observe(self, transition: Transition[int, int]) -> None:
        self.transitions.append(transition)

    def end_episode(self, result: EpisodeResult[int], *, training: bool) -> None:
        self.episodes.append((result, training))


def make_environment(goal: int = 3) -> SimulatorEnvironment[int, int, int]:
    return SimulatorEnvironment(
        WorldModelSimulator(CounterWorldModel()),
        GoalTask(goal),
        observation_space=DiscreteSpace(100),
        action_space=DiscreteSpace(2),
    )


def test_world_model_simulator_runs_with_a_regular_policy() -> None:
    agent = RecordingPolicyAgent()
    result = EnvironmentLoop(make_environment(), agent).run_episode(seed=7)

    assert result.total_reward == 3.0
    assert result.length == 3
    assert result.terminated is True
    assert result.truncated is False
    assert result.final_observation == 3
    assert len(agent.transitions) == 3
    assert agent.episodes == [(result, True)]


def test_time_limit_is_an_environment_level_truncation() -> None:
    env = TimeLimitWrapper(make_environment(goal=99), max_episode_steps=2)
    result = EnvironmentLoop(env, RecordingPolicyAgent()).run_episode()

    assert result.terminated is False
    assert result.truncated is True
    assert result.length == 2
    assert result.final_info["worldlab.time_limit_reached"] is True


def test_replay_collector_stores_transitions() -> None:
    buffer: ReplayBuffer[Transition[int, int]] = ReplayBuffer(10, seed=0)
    collector = ReplayCollector(buffer)

    EnvironmentLoop(
        make_environment(),
        RecordingPolicyAgent(),
        callbacks=[collector],
    ).run_episode()

    assert len(buffer) == 3
    assert len(buffer.sample(2)) == 2


def test_evaluator_does_not_call_agent_observe() -> None:
    agent = RecordingPolicyAgent()
    result = Evaluator(make_environment(goal=2), agent).evaluate(episodes=2, seed=10)

    assert result.mean_reward == 2.0
    assert result.mean_length == 2.0
    assert agent.transitions == []
    assert [training for _, training in agent.episodes] == [False, False]


def test_runtime_safety_limit_is_distinct_from_environment_time_limit() -> None:
    config = LoopConfig(safety_max_steps=1)
    result = EnvironmentLoop(
        make_environment(goal=99),
        RecordingPolicyAgent(),
        config=config,
    ).run_episode()

    assert result.truncated is True
    assert result.final_info["worldlab.runtime.safety_limit_reached"] is True


def test_environment_rejects_step_after_episode_end() -> None:
    env = make_environment(goal=1)
    env.reset()
    env.step(1)

    with pytest.raises(RuntimeError, match="active episode"):
        env.step(1)


def test_spaces_are_validated_by_default() -> None:
    policy = CallablePolicy[int, int](lambda observation, info, deterministic: 4)

    with pytest.raises(ValueError, match="action outside action_space"):
        EnvironmentLoop(make_environment(), PolicyAgent(policy)).run_episode()

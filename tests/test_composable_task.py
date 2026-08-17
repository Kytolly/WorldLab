from __future__ import annotations

import pytest

from worldlab.core import DiscreteSpace
from worldlab.data import (
    EvaluatedStepContext,
    SimulationReset,
    SimulationStep,
    TaskResetContext,
    TaskStepContext,
    TerminationResult,
)
from worldlab.envs import ComposableTask, SimulatorEnvironment
from worldlab.managers import (
    ObservationManager,
    ObservationTerm,
    ObservationTermSpec,
    RewardManager,
    RewardTerm,
    RewardTermSpec,
    TerminationManager,
    TerminationTerm,
    TerminationTermSpec,
)
from worldlab.simulators import WorldModelSimulator
from worldlab.world_models import CounterWorldModel


class StateObservationTerm(
    ObservationTerm[TaskResetContext[int], TaskStepContext[int, int], int]
):
    def reset(self, context: TaskResetContext[int]) -> int:
        return context.simulation.state

    def compute(self, context: TaskStepContext[int, int]) -> int:
        return context.simulation.state


class GoalTerminationTerm(
    TerminationTerm[TaskResetContext[int], TaskStepContext[int, int]]
):
    def __init__(self, goal: int) -> None:
        self.goal = goal

    def compute(self, context: TaskStepContext[int, int]) -> TerminationResult:
        return TerminationResult(context.simulation.state >= self.goal, False)


class StepRewardTerm(
    RewardTerm[TaskResetContext[int], EvaluatedStepContext[int, int, dict[str, dict[str, int]]]]
):
    def compute(
        self,
        context: EvaluatedStepContext[int, int, dict[str, dict[str, int]]],
    ) -> float:
        del context
        return 1.0


def _make_task(goal: int = 2) -> ComposableTask[int, dict[str, dict[str, int]], int]:
    observation = ObservationManager(
        {
            "state": ObservationTermSpec(
                StateObservationTerm(),
                DiscreteSpace(10),
                group="policy",
            )
        }
    )
    termination = TerminationManager(
        {"goal": TerminationTermSpec(GoalTerminationTerm(goal))}
    )
    reward = RewardManager({"step": RewardTermSpec(StepRewardTerm())})
    return ComposableTask(observation, termination, reward)


def test_composable_task_preserves_fixed_signal_order_and_info_layers() -> None:
    task = _make_task()

    reset = task.reset(SimulationReset(0, {"world": "counter"}))
    step = task.step(0, 1, SimulationStep(1, {"chunk": 0}))

    assert reset.observation == {"policy": {"state": 0}}
    assert reset.info["world"] == "counter"
    assert "worldlab.task.observation" in reset.info
    assert step.observation == {"policy": {"state": 1}}
    assert step.reward == 1.0
    assert step.terminated is False
    assert step.truncated is False
    assert step.info["chunk"] == 0
    assert step.info["worldlab.task.observation"][
        "worldlab.observation.terms.policy.state.computed"
    ] is True
    assert step.info["worldlab.task.reward"]["worldlab.reward.total"] == 1.0


def test_composable_task_returns_terminal_observation_and_environment_stops() -> None:
    task = _make_task(goal=1)
    env = SimulatorEnvironment(
        WorldModelSimulator(CounterWorldModel()),
        task,
        observation_space=task.observation.space,
        action_space=DiscreteSpace(2),
    )

    try:
        reset = env.reset(options={"start": 0})
        result = env.step(1)

        assert reset.observation == {"policy": {"state": 0}}
        assert result.observation == {"policy": {"state": 1}}
        assert result.terminated is True
        assert result.truncated is False
        with pytest.raises(RuntimeError, match="active episode"):
            env.step(1)
    finally:
        env.close()

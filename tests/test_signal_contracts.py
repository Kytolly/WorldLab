from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from worldlab.core import ObservationProvider, RewardProvider, TerminationProvider
from worldlab.data import (
    EvaluatedStepContext,
    ObservationResult,
    RewardResult,
    SimulationReset,
    SimulationStep,
    TaskResetContext,
    TaskStepContext,
    TerminationResult,
)


class DummyObservationProvider(
    ObservationProvider[TaskResetContext[int], TaskStepContext[int, int], int]
):
    def __init__(self) -> None:
        self.reset_calls: list[TaskResetContext[int]] = []
        self.step_calls: list[TaskStepContext[int, int]] = []

    def reset(self, context: TaskResetContext[int]) -> ObservationResult[int]:
        self.reset_calls.append(context)
        return ObservationResult(context.simulation.state, {"phase": "reset"})

    def compute(self, context: TaskStepContext[int, int]) -> ObservationResult[int]:
        self.step_calls.append(context)
        return ObservationResult(context.simulation.state, {"phase": "step"})


class DummyRewardProvider(
    RewardProvider[TaskResetContext[int], EvaluatedStepContext[int, int, int]]
):
    def __init__(self) -> None:
        self.reset_calls: list[TaskResetContext[int]] = []
        self.step_calls: list[EvaluatedStepContext[int, int, int]] = []

    def reset(self, context: TaskResetContext[int]) -> None:
        self.reset_calls.append(context)

    def compute(self, context: EvaluatedStepContext[int, int, int]) -> RewardResult:
        self.step_calls.append(context)
        return RewardResult(1.0, {"phase": "reward"})


class DummyTerminationProvider(
    TerminationProvider[TaskResetContext[int], TaskStepContext[int, int]]
):
    def __init__(self) -> None:
        self.reset_calls: list[TaskResetContext[int]] = []
        self.step_calls: list[TaskStepContext[int, int]] = []

    def reset(self, context: TaskResetContext[int]) -> None:
        self.reset_calls.append(context)

    def compute(self, context: TaskStepContext[int, int]) -> TerminationResult:
        self.step_calls.append(context)
        return TerminationResult(False, True, {"phase": "termination"})


def test_task_contract_objects_are_immutable() -> None:
    simulation_reset = SimulationReset(7, {"seed": 1})
    simulation_step = SimulationStep(8, {"chunk": 0})
    reset_context = TaskResetContext(simulation_reset, {"task": "demo"})
    step_context = TaskStepContext(7, 1, simulation_step, {"task": "demo"})
    observation = ObservationResult(9, {"kind": "policy"})
    termination = TerminationResult(True, False, {"kind": "done"})
    evaluated = EvaluatedStepContext(step_context, observation, termination, {"task": "demo"})
    reward = RewardResult(2.5, {"task": "demo"})

    with pytest.raises(FrozenInstanceError):
        reset_context.info = {}
    with pytest.raises(TypeError):
        reset_context.info["x"] = 1
    with pytest.raises(FrozenInstanceError):
        reward.value = 0.0
    with pytest.raises(TypeError):
        reward.info["task"] = "other"
    with pytest.raises(FrozenInstanceError):
        evaluated.info = {}


def test_reward_result_rejects_non_finite_scalars() -> None:
    with pytest.raises(ValueError, match="value"):
        RewardResult(float("inf"))

    RewardResult(0.0, {"diagnostic": float("nan")})


def test_provider_lifecycle_can_be_reset_repeatedly() -> None:
    reset_context = TaskResetContext(SimulationReset(0, {"seed": 1}), {"task": "demo"})
    step_context = TaskStepContext(0, 1, SimulationStep(1, {"chunk": 0}), {"task": "demo"})
    evaluated_context = EvaluatedStepContext(
        step_context,
        ObservationResult(1, {"phase": "step"}),
        TerminationResult(False, False, {}),
        {"task": "demo"},
    )

    observation = DummyObservationProvider()
    reward = DummyRewardProvider()
    termination = DummyTerminationProvider()

    observation.reset(reset_context)
    observation.reset(reset_context)
    observation.compute(step_context)
    reward.reset(reset_context)
    reward.reset(reset_context)
    reward.compute(evaluated_context)
    termination.reset(reset_context)
    termination.reset(reset_context)
    termination.compute(step_context)

    assert len(observation.reset_calls) == 2
    assert len(observation.step_calls) == 1
    assert len(reward.reset_calls) == 2
    assert len(reward.step_calls) == 1
    assert len(termination.reset_calls) == 2
    assert len(termination.step_calls) == 1

from __future__ import annotations

import pytest

from worldlab.core import ArraySpace, DiscreteSpace
from worldlab.data import (
    EvaluatedStepContext,
    ObservationResult,
    SimulationReset,
    SimulationStep,
    TaskResetContext,
    TaskStepContext,
    TerminationResult,
)
from worldlab.managers import (
    ManagerTermError,
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


class ValueObservationTerm(ObservationTerm[TaskResetContext[int], TaskStepContext[int, int], int]):
    def __init__(self, value: int, calls: list[str], name: str) -> None:
        self.value = value
        self.calls = calls
        self.name = name

    def reset(self, context: TaskResetContext[int]) -> int:
        del context
        self.calls.append(f"reset:{self.name}")
        return self.value

    def compute(self, context: TaskStepContext[int, int]) -> int:
        del context
        self.calls.append(f"compute:{self.name}")
        return self.value


class ConstantRewardTerm(RewardTerm[TaskResetContext[int], EvaluatedStepContext[int, int, int]]):
    def __init__(self, value: float, calls: list[str], name: str) -> None:
        self.value = value
        self.calls = calls
        self.name = name

    def reset(self, context: TaskResetContext[int]) -> None:
        del context
        self.calls.append(f"reset:{self.name}")

    def compute(self, context: EvaluatedStepContext[int, int, int]) -> float:
        del context
        self.calls.append(f"compute:{self.name}")
        return self.value


class FlagTerminationTerm(
    TerminationTerm[TaskResetContext[int], TaskStepContext[int, int]]
):
    def __init__(self, result: TerminationResult, calls: list[str], name: str) -> None:
        self.result = result
        self.calls = calls
        self.name = name

    def reset(self, context: TaskResetContext[int]) -> None:
        del context
        self.calls.append(f"reset:{self.name}")

    def compute(self, context: TaskStepContext[int, int]) -> TerminationResult:
        del context
        self.calls.append(f"compute:{self.name}")
        return self.result


def _contexts() -> tuple[
    TaskResetContext[int],
    TaskStepContext[int, int],
    EvaluatedStepContext[int, int, int],
]:
    reset = TaskResetContext(SimulationReset(0))
    step = TaskStepContext(0, 1, SimulationStep(1))
    evaluated = EvaluatedStepContext(
        step,
        ObservationResult(1),
        TerminationResult(False, False),
    )
    return reset, step, evaluated


def test_observation_manager_preserves_order_groups_and_space() -> None:
    calls: list[str] = []
    manager = ObservationManager(
        {
            "state": ObservationTermSpec(
                ValueObservationTerm(2, calls, "state"),
                DiscreteSpace(4),
                group="policy",
            ),
            "critic_state": ObservationTermSpec(
                ValueObservationTerm(3, calls, "critic_state"),
                DiscreteSpace(4),
                group="critic",
            ),
        }
    )
    reset, step, _ = _contexts()

    reset_result = manager.reset(reset)
    result = manager.compute(step)

    assert calls == [
        "reset:state",
        "reset:critic_state",
        "compute:state",
        "compute:critic_state",
    ]
    assert result.observation == {
        "policy": {"state": 2},
        "critic": {"critic_state": 3},
    }
    assert reset_result.observation == result.observation
    assert manager.space.contains(result.observation)
    assert result.info["worldlab.observation.terms.policy.state.computed"] is True


def test_disabled_observation_term_is_not_executed() -> None:
    calls: list[str] = []
    manager = ObservationManager(
        {
            "active": ObservationTermSpec(
                ValueObservationTerm(1, calls, "active"), DiscreteSpace(2)
            ),
            "disabled": ObservationTermSpec(
                ValueObservationTerm(1, calls, "disabled"),
                DiscreteSpace(2),
                enabled=False,
            ),
        }
    )
    reset, step, _ = _contexts()

    manager.reset(reset)
    result = manager.compute(step)

    assert calls == ["reset:active", "compute:active"]
    assert result.observation == {"policy": {"active": 1}}


def test_reward_manager_applies_weights_and_skips_zero_weight_terms() -> None:
    calls: list[str] = []
    manager = RewardManager(
        {
            "progress": RewardTermSpec(ConstantRewardTerm(2.0, calls, "progress"), 0.5),
            "bonus": RewardTermSpec(ConstantRewardTerm(3.0, calls, "bonus"), 2.0),
            "disabled": RewardTermSpec(ConstantRewardTerm(99.0, calls, "disabled"), 0.0),
        }
    )
    reset, _, evaluated = _contexts()

    manager.reset(reset)
    result = manager.compute(evaluated)

    assert result.value == pytest.approx(7.0)
    assert calls == [
        "reset:progress",
        "reset:bonus",
        "compute:progress",
        "compute:bonus",
    ]
    assert result.info["worldlab.reward.terms.progress.raw"] == 2.0
    assert result.info["worldlab.reward.terms.progress.weighted"] == 1.0
    assert result.info["worldlab.reward.total"] == 7.0


def test_termination_manager_ors_terminated_and_truncated_separately() -> None:
    calls: list[str] = []
    manager = TerminationManager(
        {
            "goal": TerminationTermSpec(
                FlagTerminationTerm(TerminationResult(True, False), calls, "goal")
            ),
            "timeout": TerminationTermSpec(
                FlagTerminationTerm(TerminationResult(False, True), calls, "timeout")
            ),
        }
    )
    reset, step, _ = _contexts()

    manager.reset(reset)
    result = manager.compute(step)

    assert result.terminated is True
    assert result.truncated is True
    assert result.done is True
    assert calls == [
        "reset:goal",
        "reset:timeout",
        "compute:goal",
        "compute:timeout",
    ]
    assert result.info["worldlab.termination.terms.goal.terminated"] is True
    assert result.info["worldlab.termination.terms.timeout.truncated"] is True


def test_manager_term_errors_include_location() -> None:
    class BrokenReward(RewardTerm[TaskResetContext[int], EvaluatedStepContext[int, int, int]]):
        def compute(self, context: EvaluatedStepContext[int, int, int]) -> float:
            del context
            raise ValueError("bad reward")

    _, _, evaluated = _contexts()
    manager = RewardManager({"distance": RewardTermSpec(BrokenReward())})

    with pytest.raises(ManagerTermError, match="reward/term=distance"):
        manager.compute(evaluated)

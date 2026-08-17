"""Task implementation that composes observation, termination, and reward providers."""

from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar

from worldlab.core import (
    ObservationProvider,
    RewardProvider,
    Task,
    TerminationProvider,
)
from worldlab.data import (
    EvaluatedStepContext,
    ObservationResult,
    RewardResult,
    ResetResult,
    SimulationReset,
    SimulationStep,
    StepResult,
    TaskResetContext,
    TaskStepContext,
    TerminationResult,
)


StateT = TypeVar("StateT")
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class ComposableTask(Task[StateT, ObservationT, ActionT], Generic[StateT, ObservationT, ActionT]):
    """Compose the three signal providers behind the legacy ``Task`` API.

    The fixed step order is observation, termination, evaluated context, and
    reward. Simulator metadata remains at the top level of ``info``; provider
    diagnostics are grouped under ``worldlab.task.*`` namespaces.
    """

    def __init__(
        self,
        observation: ObservationProvider[
            TaskResetContext[StateT],
            TaskStepContext[StateT, ActionT],
            ObservationT,
        ],
        termination: TerminationProvider[
            TaskResetContext[StateT],
            TaskStepContext[StateT, ActionT],
        ],
        reward: RewardProvider[
            TaskResetContext[StateT],
            EvaluatedStepContext[StateT, ActionT, ObservationT],
        ],
    ) -> None:
        self.observation = observation
        self.termination = termination
        self.reward = reward

    def reset(self, simulation: SimulationReset[StateT]) -> ResetResult[ObservationT]:
        context = TaskResetContext(simulation, dict(simulation.info))
        observation = self.observation.reset(context)
        if not isinstance(observation, ObservationResult):
            raise TypeError("observation provider reset must return ObservationResult")
        self.termination.reset(context)
        self.reward.reset(context)
        return ResetResult(
            observation.observation,
            _merge_info(
                simulation.info,
                observation=observation.info,
            ),
        )

    def step(
        self,
        previous_state: StateT,
        action: ActionT,
        simulation: SimulationStep[StateT],
    ) -> StepResult[ObservationT]:
        step_context = TaskStepContext(
            previous_state,
            action,
            simulation,
            dict(simulation.info),
        )
        observation = self.observation.compute(step_context)
        if not isinstance(observation, ObservationResult):
            raise TypeError("observation provider compute must return ObservationResult")

        termination = self.termination.compute(step_context)
        if not isinstance(termination, TerminationResult):
            raise TypeError("termination provider compute must return TerminationResult")

        evaluated = EvaluatedStepContext(
            step_context,
            observation,
            termination,
            dict(simulation.info),
        )
        reward = self.reward.compute(evaluated)
        if not isinstance(reward, RewardResult):
            raise TypeError("reward provider compute must return RewardResult")

        return StepResult(
            observation=observation.observation,
            reward=reward.value,
            terminated=termination.terminated,
            truncated=termination.truncated,
            info=_merge_info(
                simulation.info,
                observation=observation.info,
                termination=termination.info,
                reward=reward.info,
            ),
        )


def _merge_info(
    simulation_info: Mapping[str, Any],
    *,
    observation: Mapping[str, Any] | None = None,
    termination: Mapping[str, Any] | None = None,
    reward: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge diagnostics while preserving legacy simulator info keys."""

    info = dict(simulation_info)
    namespaces = (
        ("observation", observation),
        ("termination", termination),
        ("reward", reward),
    )
    for name, values in namespaces:
        if values:
            info[f"worldlab.task.{name}"] = dict(values)
    return info

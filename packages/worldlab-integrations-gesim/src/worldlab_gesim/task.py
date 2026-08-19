"""Typed GE-Sim task boundary for the author's string task interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from worldlab.core import ObservationProvider, RewardProvider, TerminationProvider
from worldlab.data import EvaluatedStepContext, RewardResult
from worldlab.envs import ComposableTask

from .provider import GESimRewardClientProvider
from .reward import GESimRewardAdapter


SubtaskResolver = Callable[[EvaluatedStepContext[Any, Any, Any]], str]


class GESimTask(ComposableTask[Any, Any, Any]):
    """WorldLab task with a typed instruction and optional sub-task resolver."""

    def __init__(
        self,
        *,
        instruction: str,
        observation: ObservationProvider[Any, Any, Any],
        termination: TerminationProvider[Any, Any],
        reward: RewardProvider[Any, Any] | None = None,
        world_judge: Any | None = None,
        subtask_resolver: SubtaskResolver | None = None,
        reward_adapter: GESimRewardAdapter | None = None,
    ) -> None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("GESim task instruction must be non-empty")
        self.instruction = instruction
        self.subtask_resolver = subtask_resolver
        if reward is not None and world_judge is not None:
            raise ValueError("pass either reward or world_judge, not both")
        if world_judge is not None:
            reward = GESimRewardClientProvider(
                reward_client=world_judge,
                task_resolver=self.resolve_task,
                adapter=reward_adapter,
            )
        super().__init__(observation, termination, reward or _DisabledRewardProvider())

    def resolve_task(self, context: EvaluatedStepContext[Any, Any, Any]) -> str:
        if self.subtask_resolver is None:
            return self.instruction
        task = self.subtask_resolver(context)
        if not isinstance(task, str) or not task.strip():
            raise ValueError("subtask_resolver must return a non-empty string")
        return task

    def __str__(self) -> str:
        return self.instruction


class _DisabledRewardProvider:
    def reset(self, context: Any) -> None:
        return None

    def compute(self, context: Any) -> RewardResult:
        return RewardResult(0.0, {"gesim.reward.enabled": False})
